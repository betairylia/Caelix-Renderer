# Step 6: path-traced lighting and NRD

Status: design, 2026-10-14. Owner decisions already taken: one lighting path (the path tracer),
the budget renderer and the NAADF resampler are not ported now, NRD is the denoiser (license
accepted 2026-10-14, obligations in `nrd-reference` memory and below).

## Goal

A lit image of the 8K citadel from the native renderer at the two milestone-4 poses, at 1 sample
per pixel, denoised by NRD `REBLUR_DIFFUSE_SPECULAR`, with the same lighting model as the Unity
path tracer (`CaelixPathTraceCommon.hlsl`). Three deliverables:

1. A reference image per pose: the same kernel accumulated over 1024 frames with a static camera.
   This is the ground truth the denoised image is judged against.
2. The denoised 1 spp image per pose, and the GPU time of trace, NRD and composite.
3. A side-by-side with the Unity captures at the same poses. Lighting will differ in the sky
   model; geometry and shading response must match.

## What is kept from Unity, exactly

The shading rules of `CaelixShadeVoxelHit` and the bounce loop (lines 269-350 and 565-714 of
`CaelixPathTraceCommon.hlsl`):

- Primary emission goes to a deterministic target. Nothing else is deterministic in this step.
- Opaque hit: albedo and emission from the packed-scene decode; specular chance
  `lerp(metallic, 1, fresnel * smoothness)` (zero for packed materials, which have smoothness 0,
  so packed worlds are diffuse only); cosine-weighted diffuse direction `normalize(N + unit)`.
- Bounce loop: up to `bounce_count` stochastic segments, throughput times albedo, Russian
  roulette on the max throughput channel, sky radiance on a miss, emission added per hit.
- The first stochastic segment's hit distance is recorded; `K_T_MAX` when the first bounce
  reaches the sky.
- The incident radiance of the path is written to the diffuse OR the specular target according to
  the lobe chosen at the primary hit; the other target gets zero radiance and zero hit distance.
  This is NRD's required split and hit-distance rule.
- Demodulation: the targets hold incident radiance without the primary albedo. The composite
  multiplies by albedo after denoising.

Not ported in this step: the transparent delta chain (`AdvanceDeltaRay`, primary surface
replacement, checkerboard parity), medium extinction, the STBN blue-noise sampler, the Halton
jitter and the colour resolve. The citadel has no glass, so the milestone does not need them.
Glass is step 6d.

## Sky and sun

Unity samples a PBR sky cubemap with a sun disk in it. The native host has no such sky. The
library gets a procedural sky, parameters in the render options:

- `sun_dir`, `sun_radiance` (RGB, linear), `sun_angular_radius` (default 0.5 degrees).
- `sky_zenith`, `sky_horizon` (RGB), blended by the direction's elevation; `sky_ground` below
  the horizon.
- Sampling: a ray that leaves the scene returns `sky(dir) + sun disk`. No next-event estimation
  in this step; the sun is found by chance like in Unity (the sun disk is small, so the diffuse
  signal is noisy; this is the signal NRD is built for and it is what Unity does).

The Unity bridge will later feed the real cubemap through an optional sky image in the options.

## Kernels and targets

All internal targets are owned by the renderer and sized to the host's colour target. They are
recreated on a size change, which also recreates the NRD instance (NRD has no resize).

| Target | Format | Written by | Read by |
|---|---|---|---|
| `view_z` | R32F | trace | NRD IN_VIEWZ, composite |
| `normal_roughness` | R10G10B10A2 | trace (`NRD_FrontEnd_PackNormalAndRoughness`) | NRD |
| `motion` | RGBA16F, 2.5D: uv delta xy, viewZ delta z | trace | NRD IN_MV |
| `albedo` | RGBA8 | trace | composite |
| `deterministic` | RGBA16F | trace | composite |
| `diff_in`, `spec_in` | RGBA16F (`REBLUR_FrontEnd_PackRadianceAndNormHitDist`) | trace | NRD |
| `diff_out`, `spec_out` | RGBA16F | NRD | composite |
| `accum` | RGBA32F | composite in reference mode | composite |
| NRD permanent + transient pools | per `InstanceDesc` | NRD | NRD |

The host-visible optional targets from the previous step (albedo, normal, depth, motion, view_z)
are written by the same trace kernel when present. The renderer no longer keeps separate code
paths for "primary only" and "path traced": one kernel, `bounce_count = 0` gives the primary
image.

Passes per frame, all inside `caelix_renderer_render`:

1. `caelix_path_trace.hlsl`: primary ray, G-buffer writes, shading, bounce loop, NRD packing.
2. NRD dispatches from `GetComputeDispatches` (skipped in reference mode and when the denoiser
   is off).
3. `caelix_composite.hlsl`: `color = deterministic + albedo * (unpack(diff_out) +
   unpack(spec_out))`; in reference mode `accum = lerp(accum, raw, 1/(n+1))` and the output is
   `accum`. Exposure multiply, then sRGB encode when the host target is a UNORM format.

Timestamps around each pass; `CaelixGpuTimings` grows: `trace_ms`, `denoise_ms`,
`composite_ms`.

## NRD integration: `NrdExecutor` (lib/src/nrd_executor.*)

NRD v4.18.0 by FetchContent, static library, `NRD_EMBEDS_SPIRV_SHADERS=ON`,
`NRD_NORMAL_ENCODING=2`, `NRD_ROUGHNESS_ENCODING=1`. The build needs CMake 3.30 or newer and
DXC from the Vulkan SDK (ShaderMake fetches itself and MathLib).

Device: `VK_KHR_compute_shader_derivatives` becomes a requested extension (feature bit 10,
optional: when absent the library builds NRD with `NRD_SUPPORTS_QUAD_INTRINSICS=OFF`; this is a
build-time choice, so the default build assumes the extension and the CMake option
`CAELIX_NRD_QUAD_INTRINSICS` turns it off). `shaderStorageImageReadWithoutFormat` is added to
the requirements next to the write feature.

Executor contract, from `GetLibraryDesc`/`GetInstanceDesc`:

- Pipeline layout per NRD pipeline: set 0 = textures from binding 20 and storage textures from
  binding 3 (both partially bound, counts from `descriptorPoolDesc`); set 1 = two immutable
  samplers at bindings 0 and 1 (nearest clamp, linear clamp) and the constant buffer at binding
  2. Offsets are read from `LibraryDesc::spirvBindingOffsets`, never hard-coded.
- Constant buffer: a host-visible ring sized `align(constantBufferMaxDataSize) * setsMaxNum *
  frames_in_flight`, written per dispatch unless `constantBufferDataMatchesPreviousDispatch`.
- Pools: one image per `TextureDesc` with `SAMPLED | STORAGE`, size `resourceSize /
  downsampleFactor`, mip 1. Permanent pool images survive frames; transient images are reused.
- Per dispatch: write one descriptor set, record a barrier per resource to the state the range
  wants (sampled read or storage), dispatch `(gridWidth, gridHeight, 1)`.
- `CommonSettings` per frame: `viewToClipMatrix` and `worldToViewMatrix` for this frame and the
  previous one, column-major, column vectors, non-jittered, built from `CaelixCamera` (basis +
  projection); `motionVectorScale = (1/w, 1/h, 1)` because the trace writes pixels in xy and
  viewZ delta in z; `cameraJitter = 0` in this step; `frameIndex` increments; `resourceSize =
  rectSize = target size`; `denoisingRange` from the camera far plane; `accumulationMode =
  RESTART` on the first frame after a reset.
- `ReblurSettings`: defaults, `hitDistanceParameters` = the same A, B, C the trace used for
  `REBLUR_FrontEnd_GetNormHitDist` (3, 0.1, 20), `enableAntiFirefly = true`,
  `hitDistanceReconstructionMode = AREA_3X3` (recommended for probabilistic lobe selection).

License obligations the code and docs carry: the NRD notice file in `third_party/NOTICES.md`,
the credits text "uses NVIDIA Real-Time Denoisers" plus the NVIDIA mark requirement documented in
the README for the game side. NRD source is not modified; only the compiled library and its
embedded shaders ship.

## ABI changes (version 2, still in development)

`CaelixRenderOptions` gains:

```
uint32_t mode;              /* CAELIX_MODE_PRIMARY (debug views as today), CAELIX_MODE_PATH_TRACE, CAELIX_MODE_REFERENCE (accumulate) */
uint32_t bounce_count;      /* stochastic bounces, 0..14; Unity default 4 */
uint32_t denoise;           /* 0 = raw 1 spp, 1 = NRD */
uint32_t reset_accumulation;/* 1 = restart the reference accumulation and NRD history this frame */
float    exposure;          /* composite multiply before sRGB; 1.0 = none */
float    sun_dir[3], sun_radiance[3], sun_angular_radius;
float    sky_zenith[3], sky_horizon[3], sky_ground[3];
uint32_t frame_seed;        /* RNG seed, host increments per frame */
```

`CaelixGpuTimings` gains `denoise_ms`, `composite_ms`. `caelix_renderer_get_nrd_info` returns
the NRD version and pool sizes for the host's log.

## Host (Rust app)

Flags: `--mode primary|pt|ref`, `--bounces N`, `--no-denoise`, `--exposure`, `--sun x,y,z`,
`--sun-radiance r,g,b`, `--sky-zenith`, `--sky-horizon`, `--accumulate N` (reference mode: take
the screenshot after N frames with a static camera; any camera input resets the accumulation).
The title shows trace, denoise, composite and total ms.

## Sub-steps and their checks

- **6a. Lighting kernel + composite + reference mode.** No NRD. Checks: trace test with a white
  emissive box lighting a grey floor (energy within tolerance against an analytic estimate for
  one bounce); reference images of the citadel at both poses (1024 frames); timing of the trace
  at 4 bounces.
- **6b. NrdExecutor.** Checks: NRD validation layer output clean (`enableValidation` view
  renders, inspected once); denoised citadel images at both poses versus the 6a references
  (PSNR logged; expected >= 30 dB on static views); denoise_ms at 1280x720 and 2560x1440.
- **6c. Unity comparison.** Side-by-side at both poses, whole-frame GPU time native versus Unity
  6.5 ms, in `Artifacts/native-renderer-step6-<date>/`.
- **6d. Glass.** Port the delta chain and medium extinction, test on the fixture world.

## 6a as built (2026-10-14)

Differences from the plan above, for 6b to pick up:

- `normal_roughness` is RGBA16F (world normal, roughness 1), not R10G10B10A2 with NRD packing;
  `diff_in` / `spec_in` hold plain radiance and the hit distance in world units (no
  `REBLUR_FrontEnd_PackRadianceAndNormHitDist` yet). 6b switches both to NRD packing.
- Coverage is `albedo.a`; there is no separate coverage target.
- `K_T_MAX` = 1e4 (Unity's value): the length of bounce rays and the "no hit" hit distance.
  The internal `view_z` of a miss is 1e30.
- The primary lobe decides the target (`specular_lobe` of the primary shade). Unity's current
  code routes by the delta-chain flag (`PathBounceIndexIsSpecular`), so an opaque specular sample
  goes to the diffuse target there; packed materials never sample the specular lobe, so the two
  agree on every packed world.
- `sun_dir` is the existing field of `CaelixRenderOptions` (shared with the primary mode's
  Lambert term). `CaelixGpuTimings` has `composite_ms` and a pad (6b can use it for `denoise_ms`).
- Sun: with the proposed radiance (20, 19, 18) and a 0.5-degree disk the sun adds about 0.003
  to the irradiance of a lit face, against about 1.5 from the proposed sky, so the sun is
  invisible. A sun that lights the scene needs a radiance of about 10^4 at 0.5 degrees, which
  without next-event estimation is a firefly source that 1024 frames do not converge. Decide
  for 6c: next-event estimation to the sun, or a larger disk.

## Open questions for the owner

1. Sun and sky parameters for the comparison. Proposal: sun at Unity's PBR sky default direction
   if it is recorded in the scene; else elevation 45 degrees, azimuth 135, radiance (20, 19, 18),
   zenith (0.2, 0.4, 0.9), horizon (0.7, 0.8, 0.95).
2. Jitter and a colour resolve are deferred to a later upscaling step. Agree?
3. Reference accumulation uses the raw 1 spp signal with the same RNG stream. 1024 frames at
   0.3 ms each is about half a second of GPU time per pose; fine.

## Sun NEE as built (2026-10-14)

The 6a open point "next-event estimation or a larger disk" is decided: next-event estimation.

- Flag `CAELIX_RENDER_FLAG_SUN_NEE` (bit 2; the host sets it by default, `--no-sun-nee` clears
  it). The library skips NEE when `sun_radiance` is 0, so a sun-off frame costs nothing extra.
  Frame constants: `sun_cos_radius` (416) and the new `sun_solid_angle` (428) =
  `2 pi (1 - cos r)`, computed in double as `4 pi sin^2(r / 2)` (the float `1 - cos r` of a
  0.5-degree disk keeps only 3 significant digits). The shader takes `1 - cos r` back from it.
- Where: every vertex that sampled the diffuse lobe AND has a next segment: the primary hit
  when `bounce_count >= 1`, a bounce hit while `bounce_index <= bounce_count`. NEE stands in
  for exactly the sun light that the next BSDF segment would have found, so the expected value
  equals the NEE-off value at the same `bounce_count`, and `bounce_count = 0` stays "primary
  emission only".
- Estimator: `L` uniform in the sun cone (pdf `1 / omega`), shadow ray from the pushed-off
  origin, `TMax = 1e30`, `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH` (the first candidate whose
  DDA hits in range is committed and the loop breaks). Unoccluded:
  `sun_radiance * omega * dot(N, L) / pi`, times the throughput that already holds the
  vertex's albedo (1 / k at the primary: demodulated), added before Russian roulette.
- No double counting: the segment that leaves an NEE vertex uses `SkyRadianceNoSun`; a segment
  that leaves a specular-lobe vertex (no NEE there) uses `SkyRadiance` with the disk. This is
  more general than "skip NEE for a specular primary": it holds at every vertex. Packed worlds
  never sample the specular lobe. A primary miss (composite, sky on miss) always shows the disk.
- Random numbers: NEE has its own stream `InitRandom(pixel, frame_seed ^ 0x68E31DA4)`. The BSDF
  path consumes the same numbers with NEE on or off, so NEE on/off frames are correlated (useful
  for comparisons) and a sun-off frame is bit-identical to 6a. Checked on the citadel overview:
  `--no-sun-nee --sun-radiance 20,19,18` (the 6a settings) reproduces 6a's
  `citadel-overview-ref.png` byte for byte; the new default (sun off, NEE on) is byte-identical
  to sun off with NEE off and differs from 6a's image only by 6a's chance-found radiance-20 sun
  (99.1 % of the pixels equal, at most 13 levels, mean -0.02 levels).
- Hit distance for NRD: unchanged (the first BSDF segment); the shadow ray is not a segment.
- Host defaults: sun radiance (0, 0, 0) = sun off (owner decision: the Unity comparison is
  sky light only). Suggested sun: `--sun-radiance 2e4,1.9e4,1.8e4` at 0.5 degrees: irradiance
  (4.79, 4.55, 4.31) on a surface facing the sun, luminance 4.58, about 3.1x the default sky's
  irradiance on an up-facing surface (`pi (0.2 horizon + 0.8 zenith)` = (0.94, 1.51, 2.86),
  luminance 1.48). The default sun direction is 44.6 degrees above the horizon, so flat ground
  gets 0.70 of it. With exposure 1 bright sand then reaches about 1.0 and a few pixels clip;
  exposure about 0.7 keeps the highlights.

Tests (`caelix_trace_test`, "lighting: sun next-event estimation ..."; grey floor, albedo
123/255 after the RGBA8 albedo target, black sky, sun straight up, R = 1000, r = 2 degrees,
omega = 3.8276e-3 sr; expected `albedo R omega (1 + cos r) / 2 / pi` = 0.58749):

| Check | Measured | Expected | Ratio |
|---|---|---|---|
| NEE, 4 bounces, reference 256 frames, centre 64x64 mean | 0.58729 | 0.58749 | 0.99965 (largest pixel error 0.043 %) |
| NEE off, 1 bounce, reference 4096 frames | 0.58294 | 0.58749 | 0.99226 (0.99260 to NEE; limit 10 %) |
| NEE, 1 bounce (double-count guard) | 0.58729 | test 1 | 1.00000 |
| NEE, 0 bounces | 0 | 0 | |
| floor under a 16^3 block 8 above it, 1 bounce | 0 (max 0) | 0 | |
| floor far from the block | 0.58729 | test 1 | 1.00000 |

The 0.035 % deficit of the NEE value is the FP16 diffuse target (spacing 2^-10 near 1.2).

Timings (`pt`, 4 bounces, 1280x720, RTX 5080, release, FIFO 144 Hz, median of 13 one-second
log samples, suggested sun on): overview 0.53 ms without NEE, 0.69 ms with NEE (+30 %); detail
0.63 ms without, 0.86 ms with (+37 %). Less than the expected 2x: shadow rays stop at the first
hit, and back-facing samples trace nothing. The 6a run of the same settings measured 0.49 and
0.57 ms (another session; FIFO leaves the GPU partly idle).

Images: `Artifacts/native-renderer-step6a-20261014/*-nee.png`.
