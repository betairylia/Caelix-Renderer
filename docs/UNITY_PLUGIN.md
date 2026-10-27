# Unity plugin bridge

Status: design, 2026-10-15. Titania branch `fable/vulkan-native-plugin`, Editor and player on
Vulkan (owner decision). The Unity-side renderers stay as they are; the plugin is a third choice
on the renderer asset and the comparison target.

## Shape

```
Caelix-Renderer/plugin/unity/      C++: caelix_unity.dll, links caelix_renderer.dll
  unity_plugin.cpp                 UnityPluginLoad/Unload, IUnityGraphicsVulkanV2, device events
  unity_bridge.cpp                 the plugin's own C API for C#, packet ring, render-thread work
  include/caelix_unity.h           that C API
Caelix/Runtime/Plugins/x86_64/     caelix_unity.dll + caelix_renderer.dll (copied by a CMake step;
                                   path from CAELIX_UNITY_PLUGIN_DIR, default off)
Caelix/Runtime/Rendering/Native/   C#: CaelixNativeRenderer (component), CaelixNativeBridge
                                   (P/Invoke), CaelixNativeRendererFeature + pass
```

Two DLLs, not one: `caelix_renderer.dll` keeps its C ABI unchanged and the Rust host keeps
loading it; `caelix_unity.dll` is the only code that knows Unity.

## Finding (2026-10-15): Unity 6000.5 Vulkan has no ray tracing of its own

`supportsRayTracing` and `supportsInlineRayTracing` are false on Vulkan, Unity creates its
instance as Vulkan 1.1 and leaves the driver's `VK_KHR_acceleration_structure` /
`VK_KHR_ray_query` / `VK_KHR_buffer_device_address` disabled, and Unity.dll carries none of
those extension names. The existing Caelix renderers therefore cannot draw on Vulkan. Owner
decision: the Editor stays on Vulkan; comparisons with the Unity renderers flip the API to
D3D12 and restart. For the plugin this means the intercept wraps BOTH `vkCreateInstance`
(raise `VkApplicationInfo::apiVersion` to 1.3 so the core 1.2/1.3 entry points volk loads
exist) and `vkCreateDevice` (append our extensions and chain our features2 structs; keep
everything Unity asked for). The plugin's `.dll.meta` must set the PluginImporter
"Load on startup" (`isPreloaded`) so `UnityPluginLoad` runs before Unity's device exists.

## Device: intercept, then adopt

`UnityPluginLoad` fetches `IUnityGraphicsVulkanV2` and calls `InterceptInitialization` before
`kUnityGfxDeviceEventInitialize` (the plugin must be in the project before the Editor starts
its Vulkan device; a native plugin inside a package is loaded at startup, which is early
enough; this is the first thing the skeleton step verifies). The intercept callback returns a
`vkGetInstanceProcAddr` that wraps `vkCreateDevice`: it appends the extensions from
`caelix_get_device_requirements` and enables the features from the bitmask on the
`VkDeviceCreateInfo` chain, leaving everything Unity asked for in place. `vkCreateInstance`
is wrapped the same way only if an instance extension turns out to be needed (none today).

On `kUnityGfxDeviceEventInitialize` (render thread) the plugin reads `UnityVulkanInstance`
(instance, physical device, device, graphics queue, queue family, `getInstanceProcAddr`) and
calls `caelix_device_adopt`. `kUnityGfxDeviceEventShutdown` destroys renderer, scene, device in
that order after `vkDeviceWaitIdle` through the queue Unity exposes.

## Threads: everything through packets

The ABI requires external synchronization per scene. Unity's render thread runs plugin events
while the main thread continues into the next frame, so no scene call may come from C#
directly. Rule: C# only fills packets; the render thread applies them.

```
C# main thread, per frame:
  packet = ring.Begin(frame)                       3-deep ring of NativeArray-backed packets
  packet.options  = render options, camera (basis, projection, jitter px), targets (native
                    texture pointers of colour/depth/motion/albedo/normal RTs), encoding
  packet.entities = for each EntityView: guid, transform, alive flag
  packet.bricks   = up to BudgetBytes of changed bricks, COPIED from storage by a Burst job:
                    Removed changes -> keys; Updated/Added -> keys + 512 block values read
                    through TryBindBrick; remainder of the change list carried to the next frame
  packet.materials = the material table when it changed (first frame)
  CommandBuffer.IssuePluginEventAndData(render_callback, kEventRender, packet.ptr)

plugin, render thread, event callback:
  state = CommandRecordingState(DontCare)          command buffer, currentFrameNumber, safeFrameNumber
  caelix_scene_recycle(safe); apply entity add/remove/transform; upload_bricks with `consumed`
  (unconsumed bricks are re-queued into the NEXT packet by the plugin, not by C#);
  caelix_scene_flush(frame); AccessTexture for each target -> VkImage + layout;
  caelix_renderer_render(frame, targets, camera, options)
```

Copying bricks costs about 1 KB each. The citadel's initial 1.1 million bricks stream over
frames under the byte budget, the same way the Rust host streams them. Changes during play
are small. Packet memory is freed when its frame number is at or below `safeFrameNumber`.

Why copy instead of pointing at storage: the render thread lags the main thread by up to a
frame, and the server tick mutates storage in between. The phase rule holds only on the main
thread.

## Targets and present

C# allocates RenderTextures with `enableRandomWrite`: colour RGBA16F, depth R32F, motion
RGBA16F, albedo RGBA8, normal RGBA16F. `AccessTexture(nativeTexturePtr, subresource,
VK_IMAGE_LAYOUT_GENERAL, COMPUTE stage, SHADER_WRITE, PipelineBarrier)` gives the `VkImage`
and the current layout; after `caelix_renderer_render` the images are in GENERAL, and the plugin
tells Unity through a second `AccessTexture` with `ObserveOnly` or leaves GENERAL for Unity's
own barrier on next use (the skeleton step finds which one Unity needs; the header says access
calls invalidate the recording state, so they come before `CommandRecordingState`).

The existing `CaelixPresentPass` then draws the colour RT with the depth RT as camera depth, in
its two stages (opaque before skybox, edges after). Depth: the plugin's depth target is
`clip.z / clip.w` of the projection C# passes, which is Unity's GPU projection
(`GL.GetGPUProjectionMatrix`, reverse Z on Vulkan), so the camera depth is exact.

Motion vectors: the plugin writes `previous - current` in pixels. C# writes Unity's
`_MotionVectorTexture` convention through a tiny conversion in the present pass (sign and
pixel-to-UV scale) only when TAA or STP is enabled; NRD inside the plugin uses the pixel form.

Jitter: when URP TAA/STP is on, `camera.projectionMatrix` carries the jitter; C# extracts it
(elements `m02`, `m12` scaled to pixels) and passes it as the camera's `jitter`. The plugin
offsets the primary rays by it and hands it to NRD. Without TAA the jitter is zero.

## C# component and feature

`CaelixNativeRenderer` (MonoBehaviour, sibling of `CaelixRayQueryRenderer`): ticked by
`CaelixHost` like the other scene renderers, owns the packet ring, the target RTs, and the
material table (from the generated `VoxelMaterials` data for block-ID, nothing for packed).
`CaelixNativeRendererFeature` adds one pass that issues the plugin event and one present pass.
Options mirror the Rust host's `RenderSettings` so the Inspector shows mode, bounces, denoise,
sun and sky.

## Steps

- **P1 skeleton.** Plugin DLLs built and copied into the package; interception adds our
  extensions (verified by `vkEnumerateDeviceExtensionProperties` on Unity's device and by a
  `caelix_device_adopt` that returns OK); the event callback renders the sky clear into the
  colour RT; the present pass shows it in the Game view. Check: Editor on Vulkan, no validation
  layer available inside Unity, so the check is the console plus `caelix` logs routed to
  `Debug.Log` through `caelix_set_log_callback`.
- **P2 scene feed.** Entities and bricks through packets; the citadel streams in; the primary
  albedo view matches the Rust host's screenshot at the benchmark pose (same pose, same size).
- **P3 lighting.** Path tracing + NRD + glass, present with depth, TAA on. Comparison numbers
  next to the Unity renderer in the same scene: GPU frame time from the frame debugger /
  profiler recipe in the notes.

## P2 as built (2026-10-15)

Packet version 2 (`plugin/unity/include/caelix_unity.h`, mirrored in `CaelixNativeBridge.cs`): one
unmanaged block per ring slot = the fixed `CaelixUnityPacket` (368 B: v1 fields + a 48 B scene
section of counts and byte offsets) followed by `CaelixUnityEntity[]` (48 B: guid, position,
rotation, flags add 1 / remove 2 / transform 4, 0 = reference), `CaelixUnityBrickRemove[]` (16 B),
`CaelixUnityUpload[]` (16 B: entity index, count, keys offset, blocks offset), the material entries
(full table `CaelixMaterial[]` or partial `CaelixUnityMaterialEntry[]`, 52 B) and the brick data
(keys, then 16-byte aligned 1024 B blocks). Scene flag `RESET` (remove every entity, drop pending,
built-in materials) is set on the first scene packet of a component: the plugin's scene outlives
Play sessions.

Differences from the design above: the render thread reads the packet IN PLACE and publishes
`last_consumed_sequence` after the scene records are applied (no 64 MiB copy); the library copies
bricks into its staging ring and the plugin copies only the remainder into its pending queue. An
encoding change is handled by the release event, which now receives the following packet as its
data. `CaelixNativeRenderer.Tick` is called by `CaelixHost` after the renderers that read the same
change lists (the serialized `nativeRenderer` field, or `CaelixNativeRenderer.Active`); it fills the
scene section, and `TryPrepareFrame` adds camera, options and target to that slot.

Trap: slot reuse depends on sequence numbers, and a domain reload resets the C# counter while the
plugin keeps its last consumed sequence. `TryPrepareFrame` continues above the plugin's value;
without that every slot looked consumed and C# overwrote packets the render thread was reading
(fingerprint: "a record names an entity the scene does not have", invalid material entries).

Display: the primary-mode debug views are written as they are into the RGBA16F target, which Unity
treats as linear and encodes to sRGB on output, so the albedo view looks brighter than in the Rust
host (same values: 51 dB after decoding). The lighting modes write linear values and are correct.

## P3 as built (2026-10-15)

Packet version 3 (448 B): `CaelixUnityPacket` gained the optional targets `depth` (R32F), `motion`
(RGBA16F) and `albedo` (RGBA8 UNORM, a = coverage) after `color`, and `CaelixCamera` gained `jitter`
(152 B). The plugin accesses each present target like the colour target; a target that
`AccessTexture` refuses, without STORAGE usage or of another size is dropped with a warning.

Library additions (ABI version 2, no bump):
- `CAELIX_RENDER_FLAG_FLIP_Y` (bit 4): every HOST target write goes to row `height - 1 - y`, and the
  host motion's y is negated (also the motion debug view). Ray generation, the random stream, the
  jitter and the internal targets (NRD's inputs) stay unflipped, so a flipped render is the plain
  render mirrored, bit for bit (`trace_test` "flip y": colour, albedo, normal, depth, view_z,
  motion, path trace composite, glass checkerboard + cross resolve). The Unity component always sets
  it; the present pass has no flip any more. The Rust host does not set it (its overview screenshot
  is byte-identical to step 5).
- `CaelixCamera::jitter` (pixels, x right, y down; the sample position is `px + 0.5 + jitter`). The
  motion target is the surface motion: previous pixel minus the jittered sample position, both
  through the unjittered pinholes, so a static camera writes motion 0 for any jitter. NRD gets it
  as `cameraJitter` / `cameraJitterPrev` (clamped to [-0.5, 0.5]); before P3 both were 0.

Unity side:
- The camera is filled in the feature's `RecordRenderGraph` (it was already), from
  `UniversalCameraData.GetProjectionMatrix()`, which carries URP's TAA jitter as
  `Translate(o) * P`. Jitter in library pixels: `((P'02 - P02) * w / 2, -(P'12 - P12) * h / 2)` with
  `P = camera.projectionMatrix`. Sign test (static camera, TAA, 60+ frames): this sign gives a sharp
  image, the opposite sign a blurred one (Laplacian variance 169 vs 124 on the castle). TAA off:
  jitter exactly 0.
- `CaelixCamera::projection`: rows 0 and 1 from the unjittered GL projection (clip y up, which NRD
  expects), rows 2 and 3 from `GL.GetGPUProjectionMatrix(P, true)` (Unity's device depth, reversed Z
  on Vulkan), column 2 negated for the library's +z-forward view.
- Present (shader `Hidden/Caelix/NativePresent`, `Caelix/Runtime/Resources`): stage 0 at
  `AfterRenderingOpaques` draws coverage >= 0.99 (coverage = `albedo.a` in every mode) with
  `SV_Depth` into camera colour + depth (ZTest LEqual, reversed by Unity); stage 1 at
  `AfterRenderingSkybox + 1` blends 0.01 <= coverage < 0.99 (`Blend One OneMinusSrcAlpha`, no depth
  write). The primary mode (and the path trace motion view) decodes sRGB before output so Unity's
  own encode shows the stored value once.
- Motion vectors: with TAA requested the stage-0 pass declares `ScriptableRenderPassInput.Motion`;
  a third pass at `AfterRenderingSkybox + 1` (after URP's motion vector pass in every depth copy
  schedule TAA allows) writes `resourceData.motionVectorColor` where coverage >= 0.5 and the voxel
  depth passes the test against the camera depth. URP's convention
  (`CalcNdcMotionVectorFromCsPositions`): `(posNDC - prevPosNDC) * 0.5`, y negated with
  `UNITY_UV_STARTS_AT_TOP`, i.e. current UV minus previous UV with v up, unjittered. Ours (flipped
  rows, y up, previous - current in pixels): `mv = -motion / size`. URP's TAA reads the live camera
  depth (`resourceData.cameraDepth`), which stage 0 wrote, for its depth dilation.
- Options: every Rust host setting is on the component (plus `primarySunDirection`, the Rust host's
  primary-mode default). The reference mode restarts when the camera (jitter excluded) or an option
  changes, every lighting mode on a mode change and on `ResetAccumulation()` (context menu).
- Setup: menu `Caelix/Native Renderer/Setup Scene` (package) adds the component to the `CaelixHost`
  object, sets encoding Packed on a new component, assigns `Assets/Caelix/titania-materials.json`
  when it exists, links `CaelixHost.nativeRenderer`, marks the scene dirty and raises
  `CaelixNativeRendererSetup.SceneSetUp`. Titania's `Assets/Editor/NativeRendererPreference.cs`
  handles that event and stores the native choice of "graphics/renderer"
  (`PlayerOptions.NativeRendererChoiceIndex`: 3 with the GI feature present).

Traps met: the ray query renderer component still runs on Vulkan and floods the console with
`AddInstance failed` (Unity has no RT there); disable it for native sessions. URP post-processing
with no Volume still changes the primary view (28.7 dB vs 52.2 dB with post-processing off on the
camera), so primary comparisons run with post-processing off. TAA needs post-processing on: for the
TAA captures the Global Volume was disabled instead.

## Views as built (2026-10-15)

Packet version 4 (still 448 B): the last field of the scene section, `pad`, is now `view`, an index
below `CAELIX_UNITY_MAX_VIEWS` (4; C# `CaelixNativeBridge.MaxViews`). The plugin keeps one scene
and one renderer per view (`State::renderers[]`). Reason: a renderer recreates its internal
targets (and loses its NRD history) whenever its target size changes, and a primary-mode render
marks its NRD history invalid, so a Game view and a Scene view on one renderer would destroy each
other's history every frame.

Plugin (`unity_bridge.cpp`): `EnsureScene(encoding)` creates the shared scene where the render
event used to create both objects; `EnsureRenderer(view)` creates the view's renderer lazily, just
before `caelix_renderer_render` ("renderer created for view N"). The scene records of a packet are
applied, flushed and consumed whatever its view: a view out of range or a failed renderer creation
fails only the render (`Fail` + `PublishSceneStats`). The status GPU timings are those of view 0.
The release event releases the cached target views of every renderer, and on an encoding change
destroys every renderer and the scene. Plugin frames stay one counter per render event, so a Unity
frame with two views is two plugin frames; `SafePluginFrame` already pops every entry of a Unity
frame at once.

C# (`CaelixNativeRenderer`, `CaelixNativeRendererFeature`): view 0 is the Game view, view 1 the
Scene view. The component has a `sceneView` setting: `Off` (not drawn), `Lit` (default: the Game
view's mode and debug view, own denoiser history) and `Primary` (albedo debug view, no denoiser).
Each view has its own four targets ("Caelix Native Colour (view 1)", ...), its own reset state
(previous camera, options and mode) and its own reset request; `ResetAccumulation()` resets every
view. The target generation stays ONE counter for all views, because one release event releases
the views of every renderer; per-view generations would make the plugin warn about a generation
change without a release. The scene section that `Tick` filled goes with whichever view prepares
first in the frame; the other view gets a free slot with an empty scene section. The frame seed
and the jitter warning stay shared; `LastJitter` follows view 0. Scene view cameras have no TAA,
so the motion pass does not run for them. The packet ring grew from 3 to 6 slots: every view
issues one packet per frame and the render thread may be a frame behind, so two views need about
four slots in flight.

Several Scene views no longer share view 1: since packet version 5 every camera has its own view
(see "Packet v5 as built" below).

Quiet logs: a `verboseLogging` setting (default off) gates the three informational C# logs: the
encoding change in `Tick`, "stream complete after N frames" (the stream statistics still run) and
"N material entries from ...". Warnings, errors, the periodic status log (`statusLogInterval`) and
the one-time plugin status line stay as they were.

## Packet v5 as built (2026-10-18)

Packet version 5 (`plugin/unity/include/caelix_unity.h` is the contract; `unity_bridge.cpp` the
render-thread side). Every packet starts with the 24-byte `CaelixUnityPacketHeader` (size, version,
kind, sequence). The plugin rejects a packet whose size, version or kind differs from what it was
compiled with (logged, rate limited, counted in `failed_events`) and does not publish its sequence.
Sequences are per kind: `last_consumed_frame_sequence`, `last_consumed_view_sequence`,
`last_consumed_release_sequence`.

Three plugin events (`ReserveEventIDRange(3)`: frame, render = frame + 1, release = frame + 2;
`status.frame_event_id` and the two others):

- **Frame event**, `CaelixUnityFramePacket` (80 B + records), once per Unity frame from the
  per-frame tick, before any camera renders. Order: session reset (`CAELIX_UNITY_FRAME_RESET`),
  entity records in list order, brick removes in runs per entity, uploads, publish the sequence,
  then the flush. Same configuration as the render event (outside a render pass, no queue access,
  `EnsurePreviousFrameSubmission | ModifiesCommandBuffersState`).
- **Render event**, `CaelixUnityViewPacket` (488 B; 504 B since the DLSS step grew
  `CaelixRenderOptions` to 192 B, the C# mirror must follow), once per view per Unity frame from
  the view's camera pass. Accesses the targets, renders the view's renderer (created lazily), publishes the
  sequence. A render event before the session's first frame packet publishes its sequence and
  renders nothing (one warning).
- **Release event**, `CaelixUnityReleasePacket` (32 B), with graphics queue access (flush, sync
  worker threads) and `vkQueueWaitIdle` when there is work. Flags: `TARGET_VIEWS` releases the
  cached image views of every renderer, `VIEW` destroys one view's renderer, `SCENE` destroys every
  renderer and the scene and forgets the entity map (implies `TARGET_VIEWS`). A NULL packet is an
  error now.

Sessions: the scene's encoding, staging size and material entries are fixed for a session. C#
starts a session with a release event (`SCENE`) followed by a frame packet with
`CAELIX_UNITY_FRAME_RESET`; that packet creates the scene (`CaelixSceneDesc::encoding`,
`staging_bytes = staging_mib << 20`, 0 = library default) and applies the material entries
(`caelix_scene_set_material_entries`, 52-byte `CaelixMaterialEntry`, over the built-in table). C#
ends the session with another `SCENE` release when its component is disabled, so the plugin holds
no scene between Play sessions. A reset that finds a scene (the release was skipped) logs an
error, removes every entity, restores the built-in materials and applies the entries; the scene
keeps its encoding and ring. A frame packet without a reset and without a scene fails (sequence
published). The plugin no longer has a material mode, a per-packet encoding or an encoding-change
path in the release event.

Frame numbers come straight from Unity: the flush and every render of a Unity frame use
`frame_number = currentFrameNumber`, `safe_frame_number = safeFrameNumber`. The plugin frame
counter and its mapping table are gone. The scene is flushed at most once per Unity frame
(`last_flushed_frame`); a second frame event in the same Unity frame applies its records, logs a
warning and leaves them for the next flush. A `frame <= safe` frame is an error (no flush, no
render). The library allows one render per renderer per frame number, which is why every camera
has its own view.

Upload sizing: the plugin has no queue for bricks the staging ring refuses. At scene creation it
reads `CaelixSceneStats::upload_bricks_free` of the empty ring and publishes a quarter of it as
`upload_bricks_cap` (frames in flight keep their staging space until they complete); after every
flush it publishes `upload_bricks_free`. C# sizes a frame packet's uploads from that value minus
the bricks of the frame packets it issued since, capped at `upload_bricks_cap`. Bricks the library
still refuses are dropped, counted in `bricks_dropped` and logged (rate limited); C# treats a
growing count as an error and sends the world again.

Views: `CAELIX_UNITY_MAX_VIEWS` is 8, one per camera. `live_views` is the bitmask of the views
that have a renderer. A `VIEW` release destroys the renderer of a camera that is gone.

Sky cubemap: the view packet's `sky` target (a Cubemap or a cube RenderTexture with mips, no sun
disk) is observed with `AccessTexture(..., kUnityVulkanResourceAccess_ObserveOnly)`, which records
nothing and does not change the layout. The plugin accepts an image with 6 layers, a square face,
at least one mip and the layout `SHADER_READ_ONLY_OPTIMAL` or `GENERAL`, and fills
`options.sky = {image, format, layout, face size, mip count}` in its copy of the packet's options.
Any other image is dropped for the frame (warning, at most 8 times, with the reason) and the
procedural sky is used.

Timings: `CaelixUnityStatus` carries the GPU timings (`flush_ms`, `trace_ms`, `composite_ms`,
`denoise_ms`, `timings_frame`) of the view whose packet sets `CAELIX_UNITY_VIEW_REPORT_TIMINGS`, and
`timings_view` names that view. The frame event no longer reads timings.

Status (280 B): the three packet sizes as compiled, the three event ids, the per-kind consumed
sequences, `last_unity_frame` / `last_safe_frame` / `last_flushed_frame`, counters `frames`
(frame events that flushed), `renders`, `releases`, `failed_events`, `scene_resets`, the scene
statistics after the newest flush (plus `staging_bytes`, `upload_bricks_free`), `bricks_dropped`,
`upload_bricks_cap` and `live_views`. Removed: `bricks_pending_plugin`, `scene_packets`,
`last_plugin_frame`.

## Packet v6 as built (2026-10-20): DLSS upscalers

Packet version 6 makes the library's upscalers (DLSS Super Resolution and Ray Reconstruction,
`docs/DLSS.md`) usable from Unity. The packets keep their layout apart from the view packet's
embedded `CaelixRenderOptions` (176 -> 192 bytes: `upscaler`, `upscale_quality`, `upscale_preset`,
`upscale_flags`), so the view packet is 504 bytes.

Status (280 -> 536 bytes). `pad1` (offset 164) became `upscaler_extensions`, `pad2` (offset 276)
became `upscale_ms` (the `CaelixGpuTimings::upscale_ms` of the `REPORT_TIMINGS` view), and
`CaelixUnityUpscalerStatus upscalers[8]` (32 bytes each) starts at offset 280:

| Offset in the entry | Field | Meaning |
| --- | --- | --- |
| 0 | `upscaler` | the key of the view's newest query: `CaelixUpscaler` (0 = no query yet) |
| 4 | `quality` | `CaelixUpscaleQuality` of the query |
| 8, 12 | `output_width`, `output_height` | the colour target size of the query |
| 16 | `supported` | `CaelixUpscalerQueryResult::supported` |
| 20 | `reason` | `CaelixUpscalerUnsupportedReason` (`BAD_ARGUMENT` also when the query call failed) |
| 24, 28 | `render_width`, `render_height` | the size of the geometry targets while upscaling |

The render thread writes an entry whole under a mutex, and `caelix_unity_status` copies the array
under the same mutex, so a reader never pairs the key of one query with the result of another.

One-frame handshake. A view packet in `CAELIX_MODE_PATH_TRACE` with `options.upscaler != NONE`:

1. The plugin builds the key (upscaler, quality, colour image width and height) and, when it
   differs from the view's cached key, calls `caelix_upscaler_query` (the first query of the device
   creates the NGX runtime) and publishes the entry.
2. Unsupported: `options.upscaler` is cleared for this frame, the frame renders at the colour size
   (warning once per view and key).
3. Supported: when every present geometry target (depth, motion, albedo) of the packet declares
   the render size, the frame is upscaled and those targets must have the render size; otherwise
   `options.upscaler` is cleared for this frame and the geometry targets are accepted at the
   colour size (an info line once per view and key: the normal first frame of a key). The first
   upscaled frame of a key logs "view v: DLSS SR quality q preset p upscales WxH -> WxH".
4. `AccessOptionalTarget` takes the expected size (render size when upscaling, colour size
   otherwise); a target of another size is dropped with "it is WxH, the frame expects WxH".

The other modes ignore the upscaler (the plugin clears the field). The log lines of the upscaler
decision have their own rate limit (the first 10, then one in 600).

Vulkan extensions. `vkCreateInstance` adds the names of `caelix_upscaler_instance_extensions` that
the loader offers (`vkEnumerateInstanceExtensionProperties` through the loader's
`vkGetInstanceProcAddr`); a missing one is logged and skipped, and a failed `vkCreateInstance` with
the additions retries with Unity's create info (as before for the api version). `vkCreateDevice`
asks `caelix_upscaler_device_extensions(instance, physical_device)` after the requirement check and
adds the names the device offers. They are optional: missing ones are logged and skipped (NGX then
does not start and the query reports `NO_NGX`). `upscaler_extensions` is 1 only when every name was
enabled and the device was created with the plugin's create info (the retry path and the "lacks a
Caelix requirement" path leave it 0). On the RTX 5080 / 610.88 the lists are instance
`VK_KHR_get_physical_device_properties2`, device `VK_NVX_binary_import`,
`VK_NVX_image_view_handle`, `VK_KHR_buffer_device_address`, `VK_KHR_push_descriptor`.

C# (`CaelixNativeRenderer`, section "Upscaler" of the Inspector, live in Play Mode): `upscaler`
(None, DlssSuperResolution, DlssRayReconstruction), `upscaleQuality` (Dlaa, Quality, Balanced,
Performance, UltraPerformance; default Quality), `upscalePreset` (0..15, 0 = NGX default; SR J 10,
K 11, L 12, M 13; RR D 4, E 5, F 6), `upscaleAutoExposure` (SR only). Properties `UpscalerKind`,
`UpscaleQualityLevel`, `UpscalePreset`, `UpscaleAutoExposure`. Per view frame
(`ResolveUpscaler` in `CaelixNativeRenderer.Views.cs`):

- The request is none unless the view is in the path trace mode (a Scene view in `Primary` mode
  gets none) and `upscaler` is set.
- The status entry of the view with the request's key and `supported = 1` gives the render size:
  depth, motion and albedo are allocated at `render_width x render_height`, the colour at the
  camera size (`CaelixNativeViewTargets.Ensure(colour w, h, geometry w, h, view)` recreates only the
  targets whose size changed; each packet target carries its own texture size). Before that entry
  arrives the geometry targets have the camera size and the plugin renders that frame without the
  upscaler. An entry with the key and `supported = 0` makes the request none (one warning per view
  and key with the reason); C# then sends no upscaler, so the plugin keeps the entry.
- Jitter: while the request is not none, URP's jittered projection is ignored and the jitter is
  `caelix_jitter_halton(Time.frameCount, caelix_upscaler_jitter_phases(upscaler, render_width,
  camera width))` (render width 0 while unknown = ratio 1: 8 phases for SR, 32 for RR), through the
  P/Invoke class `CaelixNativeLibrary` (`caelix_renderer.dll`). `LastJitter` follows it.
- A camera with `UniversalAdditionalCameraData.antialiasing == TemporalAntiAliasing` while a
  request is active logs once "DLSS replaces TAA; set the anti-aliasing of camera ... to None";
  the frame still renders.
- `ResetAccumulation()` stays the reset path (`reset_accumulation` also drops the upscaler's
  history). `CAELIX_UPSCALE_FLAG_RESET_HISTORY` is not used by C#.
- The feature and the present shader are unchanged: the shader point-samples every target by UV,
  so render-size depth, motion and albedo scale by themselves, and the motion pass divides by the
  motion texture's own dimensions (`GetDimensions`).
- The status log (`statusLogInterval`) adds `upscale` to the GPU timings, `upscaler extensions`
  and, per live view with a query, "view v: upscaler u quality q WxH -> WxH (reason r)".

## Packet v7 as built (2026-10-27)

Packet version 7 mirrors the library's material compaction, voxel id, light tree and light NEE
ABI in the plugin and in C#. Sizes (the plugin's `static_assert`s in `unity_bridge.cpp` and the C#
constants in `CaelixNativeBridge.cs`, checked again by `Check()` with `Marshal.SizeOf` and
`Marshal.OffsetOf`):

| Struct | v6 | v7 | Change |
| --- | --- | --- | --- |
| `CaelixMaterial` | 48 | 112 | `pad[2]` replaced by `sss_model`, `sss_probability`, `sss_albedo[3]`, `sss_anisotropy`, `sss_steps`, `sss_params[8]`, `texture_set`, `connect_group`, `reserved` |
| `CaelixMaterialEntry` | 52 | 116 | follows the material |
| `CaelixRenderOptions` | 192 | 208 | `voxel_id_tmax` (offset 192) + `reserved_voxel_id[3]` |
| `CaelixUnityFramePacket` | 80 | 96 | `CaelixLightOptions light` at offset 80 |
| `CaelixUnityViewPacket` | 504 | 560 | options grew 16; `CaelixUnityTarget voxel_id` at 520, `CaelixLightNeeOptions light_nee` at 544 |
| `CaelixUnityStatus` | 536 | 568 | `light_bricks` (536), `light_gpu_ms`, `light_enabled`, `light_nee_running`, `voxel_id_views` (556), `pad1`; 4 bytes of tail padding to the 8-byte alignment (C and C# agree) |

Apply points:

- Light tree options (session config, read with `CAELIX_UNITY_FRAME_RESET` only):
  `caelix_scene_set_light_options(scene, &packet.light)` right after `caelix_scene_create`, before
  the material entries and every record of the packet (the ABI wants them set before uploads). The
  "reset while a scene exists" branch applies them too (the library refinalizes resident bricks).
  A failure is a scene error (logged); the session goes on. The "session reset:" line prints them.
- Light NEE options (per view, per frame): after `EnsureRenderer(view)` and before
  `caelix_renderer_render`, `caelix_renderer_set_light_nee_options` runs when `packet.light_nee`
  differs from what the view's renderer has (`g_rt.light_nee_applied[view]`) or that record is not
  valid (a new renderer; cleared when the view's renderer or the scene is destroyed). A failure
  counts as a failed event (`Fail`, logged) and the frame renders with the previous options.
  After a successful render, `caelix_renderer_get_light_nee_stats().running` sets bit `view` of
  `light_nee_running`.
- Voxel id target: `AccessOptionalTarget` at the COLOUR target's size in every case (also while
  the view upscales: the library writes it at the output size), plus a format check (R32_UINT or
  R32G32_UINT; another format is dropped with the same rate-limited warning as the other dropped
  targets). Bit `view` of `voxel_id_views` says whether the newest render event passed it.
- Light stats: `PublishSceneStats` (after every flush) copies `CaelixLightStats::light_bricks`,
  `gpu_ms` and `enabled`; a scene release zeroes them.
- The first-render log line adds "voxel id yes/no, light NEE <enabled>".

C# (`CaelixNativeRenderer`, Inspector sections "Light Tree and Light NEE" and "Voxel Id"):

| Field | Default | Scope | Meaning |
| --- | --- | --- | --- |
| `lightTree` | on | session | builds the light BVH over emissive voxels |
| `lightMinEmission` | 0 | session | light voxel threshold before the emission scale; 0 = library default 20 |
| `lightHeapMiB` | 0 | session | initial size of each light heap buffer; 0 = library default 16 MiB |
| `lightNee` | on | live, per view | light NEE in the lighting modes (needs the tree) |
| `lightNeeMis` | on | live, per view | balance heuristic; off = `CAELIX_LIGHT_NEE_FLAG_NO_MIS` |
| `voxelIdTarget` | on | live, per view | one R32_UINT target per view at the camera size |
| `voxelIdTMax` | 0 | live | hits farther than this are misses in the voxel id target and view; 0 = no limit |

Properties `LightTree`, `LightMinEmission`, `LightHeapMiB`, `LightNee`, `LightNeeMis`,
`VoxelIdTarget`, `VoxelIdTMax`, and `VoxelIdHandle(view)`. The session fixes the three light tree
values in `BeginSession` and sends them in every frame packet (the plugin reads them with the reset
only). A change during a session is logged once ("apply at the next session") and is NOT reverted
(unlike the encoding and the material JSON). The "session started" log line and the periodic
status line (`light tree`, `light bricks`, `light gpu ms`, `nee running`, `voxel id views`) show
them. `DebugView.VoxelId = 11` (primary mode) mirrors `CAELIX_VIEW_VOXEL_ID`.

`CaelixNativeViewTargets.Ensure(colour w, h, geometry w, h, view, voxelId)` allocates the voxel id
target at the colour size when `voxelIdTarget` is on and releases it otherwise (both bump the target
generation, so a release event precedes the next render). The feature imports it into the render
graph when the view has it: the render pass declares it as a write, the present and motion passes
as a read, and `BindTargets` sets the global texture `_CaelixNativeVoxelId` (shader property id
`VoxelIdTexId`) next to the other four. The present shader does not read it; it is there for a
later post-process of the camera.

Material JSON (`CaelixNativeMaterialJson`): optional keys `sss_model` (clamped 0..255),
`sss_probability` (0..1), `sss_albedo` (3 values, each clamped 0..1, else 0), `sss_anisotropy`
(-1..1), `sss_steps` (0..255), `sss_params` (the first 8 values; a longer array warns once per
asset), `texture_set` and `connect_group` (0..65535); `reserved` = 0. An absent key is 0.

Not done in this step: no Editor validation (the plugin and the C# assembly compile, the sizes
match; nothing ran in Unity), and the voxel id texture has no consumer yet. Light NEE options do
not take part in the reference mode's reset check (both settings converge to the same image).

## Open questions for the owner

1. (answered) Loading time: "Load on startup" on the plugin importer; P1 verifies it.
2. (answered by packet v6) Should the plugin's targets be camera-sized or allow a render scale now
   (for the later upscalers)? The colour target stays camera-sized; the geometry targets take the
   upscaler's render size (see "Packet v6 as built").
