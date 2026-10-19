# DLSS upscalers (NGX) in the library

Status: library phase as built (2026-10-20). The Rust host and the Unity plugin do not use the
upscalers yet (later phases); the plugin only copies the feature DLLs and accepts the larger
`CaelixRenderOptions` (see "Unity plugin" below).

The renderer can hand its path-traced frame to NVIDIA DLSS:

- **DLSS Super Resolution (SR)**, `CAELIX_UPSCALER_DLSS_SR`: NRD stays the denoiser (when
  `denoise = 1`); DLSS upscales and anti-aliases the composited colour.
- **DLSS Ray Reconstruction (RR)**, `CAELIX_UPSCALER_DLSS_RR`: replaces NRD; DLSS denoises the
  raw 1 spp colour with guide images and upscales it.

Both run behind a small generic interface (`lib/src/render/upscaler.h`), are chosen per render
call through the C ABI and can be removed at build time (`CAELIX_DLSS=OFF`).

## Build

| CMake option | Default | Effect |
| --- | --- | --- |
| `CAELIX_DLSS` | `ON` on Windows, `OFF` elsewhere | Fetches the NGX SDK, compiles `dlss_upscaler.cpp`, links `ngx::ngx`, copies the feature DLLs. `OFF`: no NGX code at all (the file compiles to nothing, `CAELIX_DLSS=0`). |
| `CAELIX_DLSS_DEV_DLL` | `OFF` | Copies the development DLLs of `lib/Windows_x86_64/dev` (watermark, debug overlay `CTRL+ALT+F12`) instead of the release DLLs. Never ship them. |

- `cmake/Dependencies.cmake`: `FetchContent_Declare(ngx GIT_REPOSITORY https://github.com/NVIDIA/DLSS.git GIT_TAG v310.9.16 GIT_SHALLOW TRUE GIT_SUBMODULES "")`
  (the repository has no CMakeLists.txt, MakeAvailable only populates it; its
  NVIDIAImageScaling submodule is not fetched). The shallow clone needs about 2.3 GB of disk.
- `ngx::ngx`: IMPORTED STATIC, `IMPORTED_LOCATION` = `lib/Windows_x86_64/x64/nvsdk_ngx_d.lib`
  (dynamic CRT), `IMPORTED_LOCATION_DEBUG` = `nvsdk_ngx_d_dbg.lib`, SYSTEM include directory
  `include/`. The linker asked for no additional system library.
- `lib/CMakeLists.txt`: `CAELIX_DLSS=$<BOOL:...>` compile definition; a post-build step copies
  `nvngx_dlss.dll` and `nvngx_dlssd.dll` from `CAELIX_NGX_DLL_DIR` next to
  `caelix_renderer.dll` (the tests run from there). `plugin/unity/CMakeLists.txt` adds both DLLs
  to its CopyPlugin list when `CAELIX_UNITY_PLUGIN_DIR` is set.

## Architecture

```
CaelixDevice (abi/caelix.cpp)
 ├─ caelix::Device
 ├─ RenderKernels           + encode set layout, caelix_upscale_encode pipeline
 └─ NgxRuntime (lazy)       one NGX SDK instance per adopted device (CAELIX_DLSS only)
CaelixRenderer
 └─ Renderer
     ├─ upscaler_ : std::unique_ptr<Upscaler>   (DlssUpscaler)
     ├─ upscale_out_ : GpuImage                 RGBA16F, output size
     └─ retired_upscalers_                      until their last frame is safe
```

- `upscaler.h` (generic, no NGX types): `UpscalerKind`, `UpscalerDesc` (kind, quality, preset,
  render and output size, depth inverted, auto exposure; any change creates a new instance),
  `UpscalerImage`, `UpscalerInputs`, `class Upscaler { desc(); Evaluate(cmd, inputs); failed(); }`.
  `renderer.h` sees only this header and a forward declaration of `NgxRuntime`.
- `dlss_upscaler.h/.cpp` (whole body under `#if CAELIX_DLSS`; only `.cpp` files include the
  header, only the `.cpp` includes the SDK headers):
  - `NgxRuntime`: `NVSDK_NGX_VULKAN_Init_with_ProjectID` with the fixed project id
    `d8ec5ae2-3625-4c4a-ba69-224a61105aad` (engine type CUSTOM, engine version
    `CAELIX_VERSION_STRING`), application data path `%TEMP%\caelix-ngx`, feature DLL path list =
    the directory of `caelix_renderer.dll` (`GetModuleHandleExW(FROM_ADDRESS)`), the host's
    `vkGetInstanceProcAddr` (volk's) and the `vkGetDeviceProcAddr` it returns, a log callback;
    then `NVSDK_NGX_VULKAN_GetCapabilityParameters`. Throws `NgxError` on failure. Owned by
    `CaelixDevice` (destroyed first: renderers are gone, the device is idle); destructor:
    destroy the capability map, `NVSDK_NGX_VULKAN_Shutdown1(device)`. `Query` reads
    `NVSDK_NGX_VULKAN_GetFeatureRequirements`, the capability map (`*_Available`,
    `*_NeedsUpdatedDriver`, `*_MinDriverVersionMajor/Minor`, `*_FeatureInitResult`) and
    `NGX_DLSS_GET_OPTIMAL_SETTINGS` / `NGX_DLSSD_GET_OPTIMAL_SETTINGS`.
  - `DlssUpscaler`: one feature (`NGX_VULKAN_CREATE_DLSS_EXT1` / `NGX_VULKAN_CREATE_DLSSD_EXT1`,
    node masks 1) created in the frame's command buffer, its own parameter map
    (`NVSDK_NGX_VULKAN_AllocateParameters`, the preset hint of the chosen quality mode set before
    creation when `preset != 0`), and for SR without auto exposure a 1 x 1 R32F exposure image
    cleared to 1.0 in the same command buffer and kept in `SHADER_READ_ONLY_OPTIMAL`.
    `Evaluate` records the barriers, wraps the images (`NVSDK_NGX_Create_ImageView_Resource_VK`,
    mip 0, layer 0, read-write only for the output) and calls `NGX_VULKAN_EVALUATE_DLSS_EXT` /
    `NGX_VULKAN_EVALUATE_DLSSD_EXT`. An NGX failure is logged once and sets `failed()`.
  - The extension queries (`UpscalerInstanceExtensions`, `UpscalerDeviceExtensions`): the
    union of `NVSDK_NGX_VULKAN_GetFeatureInstance/DeviceExtensionRequirements` for
    `NVSDK_NGX_Feature_SuperSampling` and `NVSDK_NGX_Feature_RayReconstruction`, copied into
    library-owned strings.
- `renderer_upscale.cpp`: `QueryUpscaler` (cached per (upscaler, quality, output size); logs
  once per key when unavailable), `EnsureUpscaler` (like `EnsureDenoiser`: a new desc retires
  the old upscaler and `upscale_out_`, creates both again; an `NgxError` sets
  `upscaler_failed_` until the desc changes), `GetUpscalerInfo`. The device's NGX runtime
  reaches the renderer through a `std::function<NgxRuntime*()>` given to `Renderer::Create`.

## Data flow

Path trace mode, colour target present, `upscaler != NONE`. `size` = render size (the query's
optimal size), `output_size` = the colour target's size.

1. Validation: the colour target is the output size (at least 64 x 32); every other present
   target has the render size (else `CAELIX_ERROR_INVALID_ARGUMENT`, the log names the expected
   size); an unavailable upscaler returns `CAELIX_ERROR_UNSUPPORTED` before anything is
   recorded.
2. `EnsureInternalTargets(render size)`, NRD at render size for SR (`denoise` as before;
   never for RR), `EnsureUpscaler` (records the feature creation at the first frame of a desc).
3. Trace (render size). Writes the internal device depth every lighting frame; with RR the four
   guide images.
4. SR: NRD as before.
5. Composite (and the cross resolve when the delta checkerboard runs, SR only) with
   `fc.upscale = 1`: the final linear, exposed colour goes to the internal `upscale_in`
   (RGBA16F, alpha = coverage, unflipped, no sRGB, no validation overlay) instead of the colour
   target. The motion debug view likewise.
6. Upscale timer begin. `Evaluate`:
   - SR inputs: colour `upscale_in`, depth `depth`, motion `motion` (RGBA16F, xy used),
     exposure (1 x 1 = 1.0, or the AutoExposure flag), output `upscale_out_`.
   - RR adds diffuse albedo, specular albedo, normal + roughness (`InRoughnessMode = Packed`,
     the same image for `pInNormals` and `pInRoughness`), specular hit distance, the
     world-to-view matrix (`WorldToView`, as for NRD) and the view-to-clip matrix
     (`CaelixCamera::projection` as it is).
   - Common: jitter (`CaelixCamera::jitter`, clamped to [-0.5, 0.5]), render subrect = render
     size, MV scale 1, pre-exposure 1, exposure scale 1, frame time (steady clock between
     evaluates), reset.
7. Encode kernel (`shaders/render/caelix_upscale_encode.hlsl`, output size, 8 x 8): colour
   target[p] = (sRGB for a UNORM target of `src[p * src_size / output_size]`, coverage of the
   internal albedo at `p * size / output_size`), flip-Y on the output height. Source:
   `upscale_out_`, or `upscale_in` (nearest upscale) when the upscaler failed. Upscale timer
   end: `CaelixGpuTimings::upscale_ms` = evaluate + encode.

Reset (`UpscalerInputs::reset`): `CAELIX_UPSCALE_FLAG_RESET_HISTORY`, the first frame of a new
upscaler (any desc change), `reset_accumulation`, or a render call without a previous camera.

Fallback: an `NgxError` at creation or a failed evaluate logs the error, records the frame with
the encode kernel reading `upscale_in` (nearest upscale of the render-size colour), returns
`CAELIX_ERROR_VULKAN` from that call, and `CaelixUpscalerInfo::failed = 1` with `CAELIX_OK`
afterwards until the upscaler options change (like NRD).

## Internal targets added

| Target | Format | Content |
| --- | --- | --- |
| `depth` | R32F | device depth of the host projection (miss = `miss_depth`); the upscaler's depth. Written in every lighting frame. |
| `rr_diffuse_albedo` | RGBA8 | albedo * (1 - metallic) times the delta chain's transmission; sky 0.5. RR only. |
| `rr_specular_albedo` | RGBA8 | `EnvBRDFApprox2(lerp(0.04, albedo, metallic), roughness^2, N.V)` times the transmission; sky 0. RR only. |
| `rr_normal_roughness` | RGBA16F | world normal xyz, linear roughness (1 - smoothness) w; sky 0. RR only. |
| `rr_hit_dist` | R16F | hit distance of the first bounce segment; sky and no segment 65504. RR only. |
| `upscale_in` | RGBA16F | the composite's (or cross resolve's) linear colour; the upscaler's colour input. |

All are render size; the RR images exist in every lighting frame (cheap) and are written only
when `fc.upscaler == 2`. The trace set grew to bindings 7..18 for the internal targets (sky 19,
sampler 20; `CreateSetLayout` handles up to 32 bindings); the composite set has `upscale_in` at
binding 9 (sky 10, sampler 11); the encode set is a fourth set per frame slot. Frame constants:
544 bytes (`output_size` 520, `upscale_src_size` 528, `upscaler` 536, `upscale` 540).

## ABI contract (summary; `include/caelix/caelix.h` is the truth)

- `CaelixRenderOptions` (176 -> 192 bytes): `upscaler` (176), `upscale_quality` (180),
  `upscale_preset` (184, raw NGX preset value, 0 = default), `upscale_flags` (188:
  `CAELIX_UPSCALE_FLAG_RESET_HISTORY`, `CAELIX_UPSCALE_FLAG_AUTO_EXPOSURE`).
- `CaelixGpuTimings` (24 -> 32 bytes): `upscale_ms` at 24.
- `caelix_upscaler_query(device, desc, out)`: `CaelixUpscalerQueryDesc` (16 bytes),
  `CaelixUpscalerQueryResult` (40 bytes: supported, reason, optimal / min / max render size,
  minimum driver). `CAELIX_OK` also when unsupported; `INVALID_ARGUMENT` for null or unknown
  enums. `NONE` is supported at the output size. Reasons: `NOT_BUILT`, `NO_NGX`,
  `FEATURE_UNAVAILABLE`, `DRIVER_TOO_OLD`, `QUALITY_UNAVAILABLE`, `BAD_ARGUMENT` (output below
  64 x 32).
- `caelix_upscaler_instance_extensions` / `caelix_upscaler_device_extensions(instance,
  physical_device, ...)`: what NGX needs. Measured on the RTX 5080 / 610.88: instance
  `VK_KHR_get_physical_device_properties2`; device `VK_NVX_binary_import`,
  `VK_NVX_image_view_handle`, `VK_KHR_buffer_device_address`, `VK_KHR_push_descriptor`. Without
  the device extensions NGX does not start (`NO_NGX`). The test device (`tests/test_device.cpp`)
  enables them when present.
- `caelix_upscaler_jitter_phases(upscaler, render_width, output_width)` = 8 * ceil((output /
  render)^2), at least 32 for RR; `caelix_jitter_halton(index, phases, out)` = Halton(2, 3)
  point (index % phases) + 1 minus 0.5.
- `caelix_renderer_get_upscaler_info` -> `CaelixUpscalerInfo` (36 bytes).

## SDK facts the code relies on

- Motion vectors: render-resolution pixels, top-left origin, y down, value added to the current
  pixel gives the previous pixel: exactly our `motion` (previous - current). MV scale 1,
  `MVLowRes` set, `MVJittered` not set (our motion has no jitter).
- Jitter: render pixels, same axes as the MVs, in [-0.5, 0.5]: `CaelixCamera::jitter`.
- Depth: device depth of the host projection; `DepthInverted` when the projection is reverse Z,
  detected per call (depth at view z = 2 near below the depth at z = near). RR: hardware depth
  (`InUseHWDepth = HW`) with the same image.
- Colour: linear HDR (`IsHDR`), already multiplied by `exposure`, so the fixed exposure is 1.0.
  RR ignores exposure and sharpness.
- RR guides: as in the table above; matrices "row major, left multiplication" = our column-major
  column-vector memory layout, passed as they are.
- RR guide 3.5: no checkerboard rendering, so the library forces
  `CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD` off while RR is active.
- Optimal render sizes on the RTX 5080 for 1920x1080 (both SR and RR): DLAA 1920x1080, Quality
  1280x720, Balanced 1114x626, Performance 960x540, Ultra Performance 640x360. Defaults chosen
  by NGX 310.9.16 (from its log): SR presets K (DLAA, Quality, Balanced), M (Performance), L
  (Ultra Performance); RR preset D.

## Vulkan state rules

- Inputs need `VK_IMAGE_USAGE_SAMPLED_BIT` (every internal target has STORAGE | SAMPLED); the
  output needs STORAGE and **TRANSFER_DST**: DLSS clears its output with
  `vkCmdClearColorImage` ("fast UAV clear"; the validation layer reported the missing usage).
- Before the evaluate: one barrier with a global compute-write -> all-commands-read memory
  barrier, every input `GENERAL -> SHADER_READ_ONLY_OPTIMAL`, the output `GENERAL -> GENERAL`
  after the previous compute accesses. After it (also when it failed): every input back to
  `GENERAL` for the next compute work, the output all-commands-write -> compute read.
- NGX changes the command buffer state: the encode kernel binds its pipeline, set and push
  constant after the evaluate. The frame constants are written to the slot's buffer after the
  whole frame is recorded (the GPU reads them at execution), so the encode source size follows
  the evaluate's result.
- Feature creation needs a recording command buffer: it happens inside the render call that
  first uses a desc. Feature release (`NVSDK_NGX_VULKAN_ReleaseFeature`) waits until the
  feature's last frame is safe (retired list), or happens at renderer destruction (device idle).
- The NGX API is not thread safe: the ABI thread rule already serializes every call of one
  device (query, render, destroy).

## Logging

The log callback forwards NGX lines to `CAELIX_LOG*`: a line marked `[error]` / `error:` keeps
the error level, `[warning]` / `warning:` the warning level, every other line is debug output.
`MinimumLoggingLevel = ON` and `DisableOtherLoggingSinks = false`, so NGX also writes its own
files to `%TEMP%\caelix-ngx` (`nvngx.log`, `nvngx_dlss_310_9_16.log`,
`nvngx_dlssd_310_9_16.log`, `nvsdk_ngx.log`) **and prints its log to the process's stdout**
(visible in the test output; in Unity it would land in `Editor.log`). `utils/ngx_log_on.reg` of
the SDK enables the driver-side log as well.

## Known approximations

- RR specular hit distance: the hit distance of the first bounce segment of the lobe the path
  sampled (diffuse or specular), not always a specular segment. Pixels whose path sampled the
  diffuse lobe report a diffuse distance.
- RR guides of glass: the guides describe the surface where the delta chain stopped (primary
  surface replacement), with the chain's transmission in the albedos; rough glass (shaded
  stochastically) gets diffuse 0 and specular = transmission. No specular motion vectors.
- Debug views are upscaled like the colour: NRD's validation overlay is not drawn, the motion
  view is not sRGB-encoded (as without an upscaler), the coverage (alpha) is the nearest
  render-size coverage, not upscaled.
- The fallback after an NGX error is a nearest upscale (no filtering).
- The render size is always the query's optimal size (no dynamic resolution).

## Unity plugin

`CaelixUnityViewPacket` embeds `CaelixRenderOptions`, so it grew from 488 to 504 bytes; the
plugin's `static_assert` and `caelix_unity.h` say so. The C# mirror (Caelix repository,
`CaelixNativeBridge.cs`) must grow by the four new fields before the plugin works in Unity
again: its `Check()` size comparison fails until then. The plugin passes the fields through
unchanged (zero = no upscaler). The plugin does not enable the NGX extensions in its
`vkCreateDevice` interception yet, and it drops targets of another size than the colour target;
both are plugin-phase work.

## Validation (2026-10-20, RTX 5080, driver 610.88, Vulkan validation layer on)

`ctest --test-dir build -C Release`: 7/7 passed (the six existing tests unchanged, plus
`caelix_dlss_test`). `CAELIX_DLSS=OFF` (separate build directory): configures, builds without
warnings, 7/7 passed (the DLSS GPU cases skip with `NOT_BUILT`; the render call with an
upscaler returns `CAELIX_ERROR_UNSUPPORTED`).

`caelix_dlss_test` (6 cases, 6940 assertions, no validation message, no library warning):

- Query at 1920x1080: the sizes listed under "SDK facts"; DLAA = output size.
- Jitter helpers: Halton points, period, phase counts (32 for SR 960 -> 1920, 32 for RR
  1280 -> 1920, 8 for 1920 -> 1920).
- SR Quality 640x360 (render 427x240), NRD on, 12 jittered frames: floor mean 0.7856 against
  the 256-frame reference 0.7899 (ratio 0.9945); upscale 0.233 ms, denoise 0.062 ms at that
  size.
- RR Quality 640x360 (render 427x240), `denoise = 1` and the checkerboard flag set (both
  ignored: no NRD instance), 12 frames: floor mean 0.7837 (ratio 0.9921); upscale 0.33 ms.
- Arguments: a 640x360 depth target with SR Quality -> `INVALID_ARGUMENT` (427x240 accepted);
  48x24 output -> `INVALID_ARGUMENT`; unknown upscaler / quality -> `INVALID_ARGUMENT`; Quality
  -> Performance recreates the feature (info 320x180); upscaler off renders at 640x360 again.
- Cost at 1920x1080 Quality (render 1280x720), mean of 8 frames:

| Upscaler | upscale_ms (evaluate + encode) | trace_ms | denoise_ms |
| --- | --- | --- | --- |
| DLSS SR | 0.49 | 0.25 | 0.33 (NRD at 1280x720) |
| DLSS RR | 1.42 - 1.50 | 0.25 | 0 |

(The RR guide's table lists 1.02 ms for an RTX 5080 at 1920x1080 in Performance mode.)

Not tested: the fallback path after an NGX error (no way to provoke one in the test), reverse-Z
projections with DLSS (`depth_inverted` is computed but only normal Z was rendered), auto
exposure, preset hints other than 0, image quality beyond the floor-mean smoke check.
