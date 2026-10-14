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

## Open questions for the owner

1. (answered) Loading time: "Load on startup" on the plugin importer; P1 verifies it.
2. Should the plugin's targets be camera-sized or allow a render scale now (for the later
   upscalers)? Proposal: camera-sized in P1-P3; a scale factor is one line later.
