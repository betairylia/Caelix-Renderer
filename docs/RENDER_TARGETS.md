# Render targets and their VRAM (measured 2026-10-28)

This document lists every image that the renderer reads or writes per view: the host targets of
`CaelixRenderTargets`, the library's internal targets, the NRD pools, the upscaler output and the
DLSS memory. It gives their formats, sizes and VRAM at 1920x1080, 2560x1440 and 3840x2160. (The
request said "2140p"; this document reads it as 3840x2160, 4K UHD.)

Source state: commit `b34452d` (`main`). Hardware: NVIDIA GeForce RTX 5080, driver 610.88, Vulkan
1.4.341. Upscaler: DLSS Quality, the render size that `caelix_upscaler_query` reports.

Number labels:

- **[log]**: measured, read from a host run log (resolution and mode given; files in "Raw data").
- **[drv]**: measured, `vkGetImageMemoryRequirements().size` of an image with the same format,
  size, optimal tiling and usage, on the same GPU and driver (`image_reqs.py`). VMA allocates
  exactly this size per image (`GpuImage::allocation_bytes` = `VmaAllocationInfo::size`). Check:
  the [drv] sums of the NRD pools are equal to the NRD log lines at all five sizes (table 3).
- **[calc]**: computed, width x height x bytes per pixel. A real allocation adds the driver's tile
  padding: for example RGBA16F at 1920x1080 needs 17,694,720 B = 1920 x 1152 x 8 [drv]. The
  overhead per image is 6.7 % at 1280x720, 1920x1080 and 2560x1440, 6.7 .. 8.0 % at 1707x960 and
  0.7 % at 3840x2160 [drv].

MiB = 2^20 bytes, one decimal.

## 1. Totals per resolution

Table 1: VRAM of the render targets of ONE view, [drv] (in brackets: [calc]). Render sizes for DLSS
Quality [log]: 1280x720, 1707x960, 2560x1440.

| Output | Internal targets at output size | NRD at output size (perm + trans) | Native path, NRD (internal + NRD) | DLSS SR path, without NGX (internal + NRD at render size + `upscale_out`) | DLSS RR path, without NGX (internal at render size + `upscale_out`) | Unity host targets, native | Unity host targets, DLSS Quality |
|---|---|---|---|---|---|---|---|
| 1920x1080 | 248.9 (233.3) | 147.7 (138.4) | 396.6 (371.8) | 193.1 (181.1) | 127.5 (119.5) | 75.9 (71.2) | 52.5 (49.2) |
| 2560x1440 | 442.5 (414.8) | 262.5 (246.1) | 705.0 (661.0) | 344.7 (321.9) | 227.2 (212.5) | 135.0 (126.6) | 93.4 (87.5) |
| 3840x2160 | 940.3 (933.4) | 557.9 (553.7) | 1498.2 (1487.1) | 768.8 (724.2) | 506.2 (478.1) | 286.9 (284.8) | 202.5 (196.9) |

Table 2: DLSS (NGX) internal memory. The library does not report it (section 5). The figures are
NVIDIA's from the SDK guides in `build/_deps/ngx-src/doc/`, in "MB" as printed (the guides do not
say if MB means 10^6 or 2^20 bytes), for RTX 50 / 40 series. Add them to the SR and RR columns
of table 1.

| Output | DLSS SR, presets J/K (K = NGX default for Quality) | DLSS RR, presets D/E (D = default seen in the NGX log, `docs/DLSS.md`) | DLSS RR, presets "Default"/F |
|---|---|---|---|
| 1920x1080 | 85.77 MB | 121.09 MB | 154.08 MB |
| 2560x1440 | 143.54 MB | 212.03 MB | 268.37 MB |
| 3840x2160 | 307.37 MB | 467.97 MB | 578.84 MB |

Notes on the totals:

- "Native path" = path trace mode with NRD and no upscaler. Without denoising (`denoise = 0`) or
  in the reference mode there is no NRD instance: only the internal targets column applies. The
  primary mode allocates no internal target.
- The Unity columns include the optional voxel id target (`voxelIdTarget`, on by default) and the
  background input. The Unity usage flags are an assumption (section 2).
- The Rust host allocates much less (section 2): 2 swapchain images + the optional voxel id
  target = 33.8 / 60.0 / 127.5 MiB [drv] at the three output sizes.
- At a size change or an upscaler change, the old internal targets, NRD instance and upscaler
  stay alive until their last frame is safe (`Renderer::FreeRetired`). For a few frames the VRAM
  is the old set plus the new set.

## 2. Host targets (`CaelixRenderTargets`)

The HOST allocates every image of `CaelixRenderTargets` (`include/caelix/caelix.h`); the library
only creates and caches image views (16 views per renderer, `kMaxCachedViews`). Accepted formats
are `kTargetSpecs` in `lib/src/render/renderer_internal.h`. Every target needs STORAGE usage; the
background needs SAMPLED usage.

| Target | Binding (trace set) | Role | Accepted formats | Size without upscaler | Size with upscaler |
|---|---|---|---|---|---|
| `color` | 1 | primary mode: debug view; lighting modes: premultiplied colour, a = coverage | any storage format | common size | OUTPUT size |
| `albedo` | 2 | material albedo, a = coverage | R8G8B8A8_UNORM, R16G16B16A16_SFLOAT | common size | render size |
| `normal` | 3 | world face normal, w = coverage | R16G16B16A16_SFLOAT, R8G8B8A8_UNORM | common size | render size |
| `depth` | 4 | device depth of the host projection | R32_SFLOAT | common size | render size OR output size (`docs/VOXEL_ID.md` "Output-size depth") |
| `motion` | 5 | previous pixel - current pixel; RGBA: w = checkerboard flag | R16G16_SFLOAT, R32G32_SFLOAT, R16G16B16A16_SFLOAT, R32G32B32A32_SFLOAT | common size | render size |
| `view_z` | 6 | linear view depth (NRD IN_VIEWZ) | R32_SFLOAT | common size | render size |
| `voxel_id` | 7 | hash of the first-hit voxel (`docs/VOXEL_ID.md`) | R32_UINT, R32G32_UINT | common size | OUTPUT size |
| `background` | composite 12 (input, read only) | host image at the misses (`docs/BACKGROUND.md`) | any sampled colour format | common size | render size |

"Upscaler" means DLSS SR or RR in the path trace mode with a colour target (`docs/DLSS.md`). The
primary mode, the reference mode and a render without a colour target ignore the upscaler: then
every target has one size.

What each host allocates:

| Host | Target | Format | Size | Bytes/px | 1920x1080 | 2560x1440 | 3840x2160 |
|---|---|---|---|---|---|---|---|
| Rust (`host-rs`) | `color` = swapchain image (2 images [log]) | B8G8R8A8_UNORM | window | 4 | 2 x 8.4 | 2 x 15.0 | 2 x 31.9 |
| Rust | `voxel_id` (with `--voxel-id`) | R32G32_UINT | window | 8 | 16.9 | 30.0 | 63.8 |
| Unity | colour | R16G16B16A16_SFLOAT | camera | 8 | 16.9 | 30.0 | 63.8 |
| Unity | depth | R32_SFLOAT | camera (also while upscaling) | 4 | 8.4 | 15.0 | 31.9 |
| Unity | motion | R16G16B16A16_SFLOAT | geometry | 8 | 16.9 / 7.5 | 30.0 / 13.4 | 63.8 / 30.0 |
| Unity | albedo | R8G8B8A8_UNORM | geometry | 4 | 8.4 / 3.8 | 15.0 / 6.7 | 31.9 / 15.0 |
| Unity | voxel id (optional, default on) | R32_UINT | camera | 4 | 8.4 | 15.0 | 31.9 |
| Unity | background (no random write) | R16G16B16A16_SFLOAT | geometry | 8 | 16.9 / 7.5 | 30.0 / 13.4 | 63.8 / 30.0 |

MiB [drv]; "a / b" = native / DLSS Quality ("geometry" = camera size, or the render size while
the view upscales).

The Rust host (`host-rs/caelix-app/src/main.rs`, the `CaelixRenderTargets` literal) passes only the
acquired swapchain image as `color` (`caelix_vk::swapchain`: B8G8R8A8_UNORM, usage TRANSFER_SRC |
TRANSFER_DST | STORAGE | COLOR_ATTACHMENT) and, with `--voxel-id`, an R32G32_UINT storage image
at the window size (`caelix_vk::Context::create_storage_image`). Albedo, normal, depth, motion,
view_z and background are absent. The presentation engine owns the swapchain memory, so its [drv]
figure comes from an equivalent ordinary image. The Unity plugin
(`Caelix/Runtime/Rendering/Native/CaelixNativeViewTargets.cs`, `docs/UNITY_PLUGIN.md` "Packet v8")
allocates one set per camera view (up to 8 views): colour, depth, motion, albedo, the optional
voxel id and the background as RenderTextures without depth buffer; normal and view_z are absent.
Unity chooses the Vulkan usage flags of a RenderTexture; the [drv] figures assume STORAGE | SAMPLED |
COLOR_ATTACHMENT | TRANSFER (no STORAGE for the background). For these formats and sizes the
query gave the same size as STORAGE | SAMPLED. Unity's sky capture (128 px RGBA16F cube with mips)
adds about 1.0 MiB [calc].

## 3. Library internal targets

`Renderer::Internal` (`lib/src/render/renderer.h`) and `kInternalSpecs`
(`lib/src/render/renderer_internal.h`). `Renderer::EnsureInternalTargets`
(`renderer_targets.cpp`) creates ALL 18 images together at the first lighting render (path trace
or reference mode) and recreates them when the size changes. They are not created per mode: an
image that a mode does not use is still allocated. Usage STORAGE | SAMPLED, VMA
`AUTO_PREFER_DEVICE`, kept in GENERAL. Size = the render size (with an upscaler) or the common size
of the host targets.

| # | Image | Format | B/px | Written by | Read by | Used in |
|---|---|---|---|---|---|---|
| 0 | `view_z` | R32_SFLOAT | 4 | trace (binding 8) | NRD IN_VIEWZ | NRD |
| 1 | `normal_roughness` | A2B10G10R10_UNORM_PACK32 | 4 | trace (9) | NRD IN_NORMAL_ROUGHNESS | NRD |
| 2 | `motion` | R16G16B16A16_SFLOAT | 8 | trace (10) | NRD IN_MV, composite (7, w flag), cross resolve, DLSS motion | all lighting |
| 3 | `albedo` | R8G8B8A8_UNORM | 4 | trace (11) | composite (1), encode (2) | all lighting |
| 4 | `deterministic` | R16G16B16A16_SFLOAT | 8 | trace (12) | composite (2) | all lighting |
| 5 | `diff_in` | R16G16B16A16_SFLOAT | 8 | trace (13) | NRD IN_DIFF_RADIANCE_HITDIST, or composite (3) without NRD | all lighting |
| 6 | `spec_in` | R16G16B16A16_SFLOAT | 8 | trace (14) | NRD IN_SPEC_RADIANCE_HITDIST, or composite (4) without NRD | all lighting |
| 7 | `depth` | R32_SFLOAT | 4 | trace (15) | DLSS depth | upscaler |
| 8 | `rr_diffuse_albedo` | R8G8B8A8_UNORM | 4 | trace (16), RR frames only | DLSS RR | RR |
| 9 | `rr_specular_albedo` | R8G8B8A8_UNORM | 4 | trace (17), RR frames only | DLSS RR | RR |
| 10 | `rr_normal_roughness` | R16G16B16A16_SFLOAT | 8 | trace (18), RR frames only | DLSS RR | RR |
| 11 | `rr_hit_dist` | R16_SFLOAT | 2 | trace (19), RR frames only | DLSS RR | RR |
| 12 | `accum` | R32G32B32A32_SFLOAT | 16 | composite (5) | composite (5) | reference mode |
| 13 | `diff_out` | R16G16B16A16_SFLOAT | 8 | NRD OUT_DIFF_RADIANCE_HITDIST (also an NRD temporary) | composite (3) | NRD |
| 14 | `spec_out` | R16G16B16A16_SFLOAT | 8 | NRD OUT_SPEC_RADIANCE_HITDIST (also an NRD temporary) | composite (4) | NRD |
| 15 | `validation` | R8G8B8A8_UNORM | 4 | NRD OUT_VALIDATION (debug view `nrd-validation` only) | composite (6) | debug view |
| 16 | `resolve` | R16G16B16A16_SFLOAT | 8 | composite (8) | cross resolve | delta checkerboard (default on; RR forces it off) |
| 17 | `upscale_in` | R16G16B16A16_SFLOAT | 8 | composite or cross resolve (9) | DLSS colour, encode on NGX failure | upscaler |
| | **Sum** | | **118** | | | |

The trace writes images 0..7 in every lighting frame (images 8..11 only when the upscaler is RR).
Numbers in brackets are descriptor bindings (section 7).

Table 3a: internal target VRAM per size, MiB [drv] ([calc]).

| Size | Sum of 18 | `accum` | each RGBA16F | each R32F / RGBA8 / RGB10A2 | `rr_hit_dist` |
|---|---|---|---|---|---|
| 1280x720 | 110.6 (103.7) | 15.0 | 7.5 | 3.8 | 1.9 |
| 1707x960 | 197.2 (184.4) | 26.7 | 13.4 | 6.7 | 3.4 |
| 1920x1080 | 248.9 (233.3) | 33.8 | 16.9 | 8.4 | 4.2 |
| 2560x1440 | 442.5 (414.8) | 60.0 | 30.0 | 15.0 | 7.5 |
| 3840x2160 | 940.3 (933.4) | 127.5 | 63.8 | 31.9 | 15.9 |

Table 3b: internal targets that the path allocates but never reads, from the "Used in" column, MiB
[drv] (in brackets: [calc]).

| Path | Images not read | 1920x1080 | 2560x1440 | 3840x2160 |
|---|---|---|---|---|
| Native, NRD (output size) | depth, 4 RR guides, accum, validation, upscale_in | 105.5 (98.9) | 187.5 (175.8) | 398.4 (395.5) |
| DLSS SR (render size) | 4 RR guides, accum, validation | 35.6 (33.4) | 63.5 (59.4) | 142.5 (133.6) |
| DLSS RR (render size) | view_z, normal_roughness, accum, diff_out, spec_out, validation, resolve | 48.8 (45.7) | 86.9 (81.3) | 195.0 (182.8) |

(view_z is still written in RR frames, but nothing reads it.)

Other per-renderer and per-device GPU memory (independent of the resolution except `upscale_out_`):

| Object | Owner | Format / size | VRAM |
|---|---|---|---|
| `upscale_out_` | Renderer, upscaled frames only | R16G16B16A16_SFLOAT, OUTPUT size, STORAGE / SAMPLED / TRANSFER_DST | 16.9 / 30.0 / 63.8 MiB [drv] at 1080p / 1440p / 2160p |
| `sky_sh_buffer_` | Renderer | 9 float4 | 144 B [calc] |
| `FrameSlot::constants` | Renderer, one per frame slot (3 at start, at most 8) | `FrameConstants`, 576 B at `b34452d`, host-visible, mapped | 576 B per slot [calc] |
| NRD constant ring | NrdExecutor | 960 B stride [log] x `setsMaxNum` x 8 slots | not logged (`setsMaxNum` is not printed) |
| fallback sky cube | RenderKernels (per device) | 1 x 1 x 6 R8G8B8A8_UNORM | 24 B [calc] |

## 4. NRD (REBLUR_DIFFUSE_SPECULAR, NRD 4.18.0)

Table 4: the NRD log line "N permanent textures (X MiB), M transient textures (Y MiB)"
(`NrdExecutor::NrdExecutor`, `lib/src/render/nrd_executor.cpp`); the sizes are the sum of the VMA
allocation sizes. In brackets: [calc].

| NRD size | Run | Permanent (13) | Transient (8) | Sum |
|---|---|---|---|---|
| 1280x720 | 1920x1080, DLSS SR [log] | 39.4 (36.9) | 26.3 (24.6) | 65.7 |
| 1707x960 | 2560x1440, DLSS SR [log] | 70.5 (65.6) | 46.9 (43.8) | 117.4 |
| 1920x1080 | 1920x1080, native [log] | 88.6 (83.1) | 59.1 (55.4) | 147.7 |
| 2560x1440 | 2560x1440, native [log]; 3840x2160, DLSS SR [log] | 157.5 (147.7) | 105.0 (98.5) | 262.5 |
| 3840x2160 | 3840x2160, native [log] | 334.7 (332.2) | 223.2 (221.5) | 557.9 |

NRD runs only in the path trace mode with `denoise` on and not under DLSS RR (RR retires the
instance). Its size is the trace size (render size under DLSS SR). Every instance logged
14 pipelines and an NRD constant block of 960 bytes [log].

Pool textures (`build/_deps/nrd-src/Source/Denoisers/Reblur_DiffuseSpecular.hpp`, formats from
`Source/Reblur.cpp`; `NRD_NORMAL_ENCODING` = 2, so PREV_NORMAL_ROUGHNESS is R10G10B10A2; all
textures have downsample factor 1 except TILES):

| Pool | Texture | Format | B/px |
|---|---|---|---|
| permanent | PREV_VIEWZ | R32_SFLOAT | 4 |
| permanent | PREV_NORMAL_ROUGHNESS | R10_G10_B10_A2_UNORM | 4 |
| permanent | PREV_INTERNAL_DATA | R16_UINT | 2 |
| permanent | DIFF_HISTORY, SPEC_HISTORY | RGBA16_SFLOAT | 8 each |
| permanent | DIFF_FAST_HISTORY, SPEC_FAST_HISTORY | R16_SFLOAT | 2 each |
| permanent | DIFF/SPEC_HISTORY_STABILIZED_PING/PONG (4) | R16_SFLOAT | 2 each |
| permanent | SPEC_HITDIST_FOR_TRACKING_PING/PONG (2) | R16_SFLOAT | 2 each |
| | permanent sum | | 42 |
| transient | DATA1 | RG8_UNORM | 2 |
| transient | DATA2 | R32_UINT | 4 |
| transient | SPEC_HITDIST_FOR_TRACKING | R16_SFLOAT | 2 |
| transient | DIFF_TMP2, SPEC_TMP2 | RGBA16_SFLOAT | 8 each |
| transient | DIFF_FAST_HISTORY, SPEC_FAST_HISTORY | R16_SFLOAT | 2 each |
| transient | TILES | R8_UNORM, downsample 16 | 1 per 16 x 16 px |
| | transient sum | | 28 + tiles |

The executor creates the transient pool as ordinary images (no aliasing), so the transient pool
costs VRAM all the time, like the permanent pool.

## 5. DLSS SR and RR

The library does not query NGX memory: `lib/src/render/dlss_upscaler.cpp` does not call
`NGX_DLSS_GET_STATS` / `NVSDK_NGX_Parameter_DLSSGetStatsCallback` (SR) or the
`DLSSDGetStatsCallback` (RR). The SDK has these calls (`build/_deps/ngx-src/include/nvsdk_ngx_helpers.h`,
`nvsdk_ngx_helpers_dlssd_cuda.h`; they report `NVSDK_NGX_Parameter_SizeInBytes`). Table 2 has the
public figures from `DLSS_Programming_Guide_Release.pdf` (revision 310.6.0, section "Execution
Times and GPU RAM Usage", note 1) and `DLSS-RR Integration Guide.pdf` (section 2.2, tables 1 and 2).
The build uses SDK tag v310.9.16 (`docs/DLSS.md`), so the guides are slightly older than the DLLs.
The SR guide calls its figures "a ballpark number" and says an RGBA16 output can lower them
(`upscale_out_` is RGBA16F). The figure is a process-wide total: it includes memory that NGX
caches after `ReleaseFeature` unless `NVSDK_NGX_Parameter_FreeMemOnReleaseFeature` is set.

The library's own DLSS memory: `upscale_out_` (RGBA16F at the output size, section 3) plus the
internal targets at the render size. DLSS reads `upscale_in`, `depth` and `motion` (SR and RR) and
the four RR guides (RR) from the internal targets; it needs no extra library image.

## 6. Per-path image list

| Path | Trace size | Library images | Host images (minimum) |
|---|---|---|---|
| Primary mode | common | none (no internal targets, no NRD) | any present target |
| Path trace, no denoise | common | 18 internal | colour |
| Path trace, NRD | common | 18 internal + NRD pools (21) | colour |
| Path trace, DLSS SR | render | 18 internal + NRD pools at render size + `upscale_out_` + NGX | colour at output size (+ voxel id at output size, + depth at either size) |
| Path trace, DLSS RR | render | 18 internal + `upscale_out_` + NGX (no NRD) | colour at output size |
| Reference mode | common | 18 internal (uses `accum`), no NRD, no upscaler | colour |

## 7. Descriptor layout

Every set is created in `RenderKernels` (`lib/src/render/render_kernels.h` / `.cpp`); every
binding is PARTIALLY_BOUND. Each kernel binds its own set as set 0 and the frame set as set 1
(binding 0 = the frame slot's `FrameConstants` as a uniform buffer,
`shaders/common/caelix_frame_ubo.hlsl`). No push constants.

| Set | Kernels | Bindings |
|---|---|---|
| trace (set 0) | `caelix_path_trace`, `caelix_path_trace_no_light_nee`, `caelix_path_trace_sss`, `caelix_path_trace_sss_no_light_nee`, `caelix_voxel_id` | 0 TLAS; 1..7 host targets colour, albedo, normal, depth, motion, view_z, voxel_id (`kTargetCount` = 7); 8..19 internal targets 0..11 (`kTraceInternalCount` = 12: view_z .. rr_hit_dist); 20 sky cubemap (sampled); 21 sky sampler. `caelix_voxel_id` uses only 0, 4 and 7. |
| composite (set 0) | `caelix_composite`, `caelix_cross_resolve` | 0 host colour; 1 albedo; 2 deterministic; 3 diff_out (NRD) or diff_in; 4 spec_out (NRD) or spec_in; 5 accum; 6 validation; 7 motion; 8 resolve; 9 upscale_in; 10 sky cubemap; 11 sky sampler; 12 host background (sampled, render size) |
| encode (set 0) | `caelix_upscale_encode` (upscaled frames) | 0 host colour; 1 `upscale_out_` (or `upscale_in` after an NGX failure); 2 internal albedo |
| sky SH (set 0) | `caelix_sky_sh` | 0 sky cubemap; 1 sampler; writes `sky_sh_buffer_` by device address |
| frame (set 1) | all six frame-constant kernels | 0 `FrameConstants` uniform buffer |
| NRD set 0 / set 1 | the 14 NRD pipelines (`NrdKernels`) | set 0: NRD textures (sampled + storage, from per-slot pools); set 1: dynamic uniform buffer (NRD constant ring) + the immutable samplers |

The trace shader declares the internal formats with `[[vk::image_format]]`
(`shaders/render/caelix_path_trace.hlsl`, `caelix_composite_set.hlsl`,
`caelix_upscale_encode.hlsl`); the host targets have no format qualifier (unknown format, written
with `shaderStorageImageWriteWithoutFormat`). Each frame slot owns one descriptor pool with the
five sets (trace, composite, sky, encode, frame).

## Raw data

Host runs (copied binaries, `caelix-app.exe --test-entity --mode pt --size WxH --voxel-id
--frames 40 --no-input [--upscaler dlss-sr|dlss-rr --upscale-quality quality]`,
`RUST_LOG=info,caelix_vk=debug`, the two Overwolf variables set; all exited with code 0, no warning
or error), in
`C:/Users/betairya/AppData/Local/Temp/claude/F--RP-Games-Caelix-family-Caelix-Renderer/ceb8da46-3eb5-411d-b334-6c5ff104e59d/scratchpad/vram-host/logs/`:

- `1920x1080_native.log`, `1920x1080_sr.log`, `1920x1080_rr.log`
- `2560x1440_native.log`, `2560x1440_sr.log`, `2560x1440_rr.log`
- `3840x2160_native.log`, `3840x2160_sr.log`, `3840x2160_rr.log`
- `probe.log` (a first 1920x1080 native run, info level)

Tools and results in
`C:/Users/betairya/AppData/Local/Temp/claude/F--RP-Games-Caelix-family-Caelix-Renderer/ceb8da46-3eb5-411d-b334-6c5ff104e59d/scratchpad/tools/`:
`image_reqs.py` (ctypes Vulkan query of image memory requirements, no build), `vram_tables.py`
(the image lists of this document, [calc] and [drv] per image, writes `vram_tables.json`),
`totals.py` (the sums of tables 1 and 3b). The scratch folder is temporary; copy the logs to
`Artifacts/` to keep them.
