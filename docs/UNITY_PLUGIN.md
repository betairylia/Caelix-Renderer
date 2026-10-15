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

Known limitation: several Scene views share view 1. When their sizes differ, the view's targets
are recreated at every change of size, which costs a release event (queue idle) per frame.

Quiet logs: a `verboseLogging` setting (default off) gates the three informational C# logs: the
encoding change in `Tick`, "stream complete after N frames" (the stream statistics still run) and
"N material entries from ...". Warnings, errors, the periodic status log (`statusLogInterval`) and
the one-time plugin status line stay as they were.

## Open questions for the owner

1. (answered) Loading time: "Load on startup" on the plugin importer; P1 verifies it.
2. Should the plugin's targets be camera-sized or allow a render scale now (for the later
   upscalers)? Proposal: camera-sized in P1-P3; a scale factor is one line later.
