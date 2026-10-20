# Per-pixel ReSTIR DI on the light list (step 2)

Branch `gi-experiments`, 2026-10-21. Builds on `docs/LIGHT_LIST.md` (step 1: light table, light
cells, NEE at every diffuse vertex behind `CAELIX_GI_FLAG_LIGHT_NEE`). Design by Fable; the
"As built" section at the end is the implementer's.

## What it does

With `CAELIX_GI_FLAG_RESTIR_DI` (needs `LIGHT_NEE`), the direct light of the list lights at the
primary surface is resampled per pixel (Bitterli et al. 2020): the trace kernel draws a few
candidates from the pixel's light cell and keeps one in a per-pixel reservoir; a temporal pass
merges the reprojected previous reservoir; spatial passes merge neighbours; a shade pass traces
one visibility ray to the selected light point and adds the contribution to the diffuse radiance
target. The bounce vertices keep step 1's one-sample NEE. The primary vertex's one-sample NEE of
step 1 is replaced by the reservoir (it is not added twice).

A light sample is a fixed point on a voxel face, so there is no reconnection Jacobian: the
phase 1 fireflies cannot occur here. The reuse knobs and flags of `CaelixGiOptions` apply as
for ReSTIR GI: `TEMPORAL`, `SPATIAL`, `VISIBILITY` (a shadow ray per reused neighbour sample),
`UNBIASED` (Z-count weights with the rays), `temporal_m_cap`, `spatial_samples`,
`spatial_passes`, `spatial_radius`, `depth_threshold`. `CaelixGiOptions::light_reserved` becomes
`di_candidates` (0 = 4; 1 .. 32): initial candidates per pixel.

## Reservoir (32 B, four buffers per pixel like ReSTIR GI: initial, prev, temp, out)

```
DiReservoir
  uint light;             //  0  light table entry index; 0xFFFFFFFF = no sample
  uint voxel_face;        //  4  bits 0..8 voxel index v = x | y<<3 | z<<6, bits 9..11 face (+X0 -X1 +Y2 -Y3 +Z4 -Z5), bit 12 valid
  uint uv;                //  8  2 x f16: point on the face in (-0.5, 0.5)^2 (e1, e2 of GiFace basis)
  float W;                // 12  unbiased contribution weight (area measure at the light point)
  float3 visible_pos;     // 16  x_v (the pushed-off primary point)
  uint visible_flags_m;   // 28  bits 0..2 face of n_v, bits 8..31 M  (as GiReservoir)
```

The light point `y`, its normal `n_y` and radiance `L_e` are recomputed from (light, voxel,
face, uv) through the light table entry and the entity's CURRENT transform every time they are
needed (a light on a moving entity moves with it; a stale entry index after a table rebuild is
accepted: the table is rebuilt only in flushes that finalized bricks, and the reservoir history
is dropped when `CaelixLightStats::rebuilds` changed since the previous render of this renderer
(the renderer keeps the last value it saw); document this in As built).

Target at pixel r: `p_hat_r(y) = Luminance(L_e) * max(0, dot(n_v, w)) / pi * max(0, dot(n_y, -w)) / d^2`,
`w = normalize(y - x_v)`, `d = |y - x_v|`, no visibility. Contribution:
`L_e * max(0, dot(n_v, w)) / pi * max(0, dot(n_y, -w)) / d^2 * V * W` (albedo demodulated as the
diffuse target expects).

## Passes

1. Trace kernel (primary vertex, diffuse lobe, `bounce_count >= 1`): `di_candidates` candidates.
   Each: one of the pixel's cell reservoirs (fallback: the global CDF) -> entry, `W_brick`;
   `SampleLightEntry` -> point with `pdf_area`; candidate weight `w = p_hat_r * W_brick /
   pdf_area`. Streaming RIS; the result `W = w_sum / (M0 * p_hat_sel)` (0 when w_sum = 0), M = 1
   (the candidates are one initial sample; M counts reservoir merges as in the GI code: the
   initial reservoir has M = 1). Written to `di_initial`. No shadow ray here. The primary
   vertex's one-sample NEE of step 1 is skipped in this mode.
2. `caelix_restir_di_temporal`: as the GI temporal pass (reprojection by the internal motion,
   `GiSimilar`-style test on `visible_pos` and face, M clamped to the cap), merge weight
   `p_hat_r(y_prev) * W_prev * M_prev` with `p_hat_r` recomputed at this pixel; no Jacobian.
3. `caelix_restir_di_spatial` x `spatial_passes`: neighbours in a disk as the GI pass; the same
   `VISIBILITY` / `UNBIASED` semantics (Z = sum of M over the pixel and the accepted neighbours
   at which the selected sample has a nonzero target and (UNBIASED) is visible).
4. `caelix_restir_di_shade`: one shadow ray from `x_v` to `y` (TMax = d - 2 push-offs); adds the
   contribution to `g_int_diff` (unpack the REBLUR-packed value, add, repack with the existing
   hit distance; when the stored hit distance is 0 / no GI, use d). In ReSTIR GI mode the GI
   shade runs first and the DI shade adds after it (order: GI passes, then DI passes).

Buffer rotation, retirement, size changes, history validity and the `reset` option follow the GI
buffers (`EnsureGiBuffers` pattern: `EnsureDiBuffers`, `di_index_`, `di_history_valid_`). The GI
timer brackets the DI passes too (`gi_ms` = all reuse passes); `CaelixGiStats::reservoir_bytes`
includes the DI buffers.

## ABI

`CAELIX_GI_FLAG_RESTIR_DI (1u << 6)`; `light_reserved` -> `di_candidates`; validation:
`RESTIR_DI` without `LIGHT_NEE` -> INVALID_ARGUMENT; `di_candidates <= 32`. Host: flag
`--gi-restir-di`, `--gi-di-candidates <n>`, UI checkbox + field. Bench runs: `di` = `--mode pt
--no-denoise --warmup 64 --gi pt --gi-light-nee --gi-restir-di`, `di_unbiased` (+
`--gi-unbiased`), `di_conv` = `--mode ref --accumulate 512 --gi pt --gi-light-nee --gi-restir-di`.

## Tests (`tests/trace_test.cpp`)

1. "ReSTIR DI unbiased": the step 1 box with one strength-3 voxel; 1024 accumulated frames of
   `LIGHT_NEE` vs 1024 of `LIGHT_NEE | RESTIR_DI | TEMPORAL | SPATIAL | UNBIASED`: mean ratio
   0.97 .. 1.03, PSNR > 30 dB.
2. "ReSTIR DI lowers the error": 64 frames warm-up, the single frame's RMSE against the 1024-frame
   reference is lower with RESTIR_DI than with LIGHT_NEE alone (same seed sequence).
3. "RESTIR_DI needs LIGHT_NEE": the option call returns INVALID_ARGUMENT.
4. The step 1 tests and the 7 existing cases still pass.

## As built

Built on 2026-10-21 (branch `gi-experiments`, on top of 50c8076). Files:
`shaders/render/caelix_restir_di.hlsl` (reservoir, light point, target, merge, initial
candidates), `caelix_restir_di_temporal.hlsl`, `caelix_restir_di_spatial.hlsl`,
`caelix_restir_di_shade.hlsl`, `shaders/render/caelix_light_sample.hlsl` (`SampleLightEntry` split
into shared helpers and `LightPointAt`), `shaders/render/caelix_path_trace.hlsl` (initial reservoir,
primary NEE skipped), `shaders/common/caelix_frame.hlsl` + `lib/src/render/frame_constants.h`,
`lib/src/render/renderer*.{h,cpp}`, `render_kernels.*`, `lib/CMakeLists.txt`,
`include/caelix/caelix.h`, `tests/trace_test.cpp` (three cases), host flags and UI, README,
`tools/gi_bench.py`.

### Sizes

| item | size |
| --- | --- |
| `DiReservoir` | 32 B, as designed; bit 3 of `visible_flags_m` = diffuse lobe (difference 2) |
| DI reservoir buffers | 4 x 32 B per pixel: 2 MiB at 128 x 128, 112.5 MiB at 1280 x 720, 253 MiB at 1920 x 1080 |
| `FrameConstants` | 672 -> 704 B: `di_initial` 648, `di_prev` 656, `di_temp` 664, `di_out` 672, `di_candidates` 680, `di_history_valid` 684, pad to 704 |
| `CaelixGiOptions` | unchanged size (48 B); `light_reserved` -> `di_candidates` |
| `CaelixGiStats` | unchanged; `reservoir_bytes` = GI + DI buffers, `gi_ms` = every reuse pass |

### Pass order

```
[light cell fill] -> [sky SH] -> trace (initial GI and DI reservoirs)
-> barrier -> gi_timer begin
   -> [GI temporal -> GI spatial x passes -> GI shade]          (method RESTIR)
   -> [DI temporal -> DI spatial x passes -> DI shade]          (RESTIR_DI)
-> gi_timer end -> barrier -> NRD -> composite
```

A compute -> compute barrier sits between every two passes. The DI passes run when the flag is
set, light NEE runs, the scene has a light table (`light_table_address() != 0`) and
`bounce_count >= 1`; otherwise the frame is a plain light NEE frame and the DI history is dropped.
In the path trace method the renderer now passes `RESTIR_DI` and the four reuse flags in
`gi_flags` when DI runs (before, only `DIFFUSE_ONLY | LIGHT_NEE` passed there). The buffers rotate
as the GI buffers (`EnsureDiBuffers`, `di_index_`, `di_history_valid_`, `retired_di_`).
`di_history_valid` = previous frame ran DI on the same buffers, no restart, no `reset` / method /
flag change, and `CaelixLightStats::rebuilds` equal to the value of the previous render
(`di_light_rebuilds_`, read with `Scene::GetLightStats`; no scene change).

The path tracer without `RESTIR_DI` is bit-identical to 50c8076: the colour hash of 4 frames of the
light box was compared before and after the change for plain path tracing, path tracing with
`LIGHT_NEE` and ReSTIR GI with `LIGHT_NEE` (a temporary test, removed).

### Differences from the design, and why

1. **uv is 2 x unorm16, not 2 x f16.** The f16 grid is not uniform over (-0.5, 0.5) (dense near
   0), so a rounded point is not uniformly distributed. The trace kernel's candidates quantize the
   uv first (`SampleLightEntryWith(..., quantize_uv = true)`: the point is the centre of its
   1/65536 cell) and evaluate the target at exactly the point that the reservoir names. The
   one-sample NEE (`SampleLightEntry`) is unchanged (no quantization).
2. **The initial reservoir is written at every opaque primary surface, not only after the diffuse
   lobe**, with a "diffuse lobe" bit (bit 3 of `visible_flags_m`, copied from the pixel's own
   reservoir by every merge). The shade pass adds the contribution only when the bit is set (that
   is where step 1's NEE ran, so the expected value is the same as step 1). A pixel that took the
   specular lobe this frame keeps its DI history for the next frame (in the design it would have
   stored an empty reservoir and broken the temporal chain on glossy surfaces). The first segment
   after a diffuse primary vertex still skips list-light emission (`segment_light_nee`), so the
   light is not counted twice. A transparent primary surface and `bounce_count 0` get an empty
   reservoir (M 0).
3. **"A surface" is M > 0, "a sample" is the valid bit.** A pixel whose candidates all had a zero
   target stores M = 1 without a sample; `DiSimilar` accepts reservoirs with M > 0 (the GI test
   asks for a valid sample), so such a reservoir still counts in M, as RIS requires.
4. **`UNBIASED`: Z traces no visibility ray** (the design: "at which the selected sample has a
   nonzero target and is visible"). The DI target has no visibility and the initial candidates are
   not shadow-tested, so the support of a neighbour's reservoir does not depend on visibility; Z
   must count every reservoir whose target is nonzero at y. With a visibility test, a y visible
   from the pixel but occluded from a neighbour would drop that neighbour's M from Z and the
   estimate would be too bright at shadow boundaries. The merge itself keeps the visibility rays
   (`VISIBILITY` or `UNBIASED`): they only zero candidates whose contribution at the pixel is zero
   anyway, so they cost no bias. Measured in the test box, the design's variant gives the same mean
   (ratio 0.9979 vs 0.9991, 1024 frames, one bounce): in that box a light point visible from a pixel
   is visible from all its neighbours, so the test cannot show the difference. The design's variant
   also has a side effect: a sample occluded from the pixel itself gets Z = 0 and W = 0, an implicit
   visibility reuse (single frame RMSE 0.0468 vs 0.0555).
5. **The temporal pass uses Z too with `UNBIASED`** (no ray: the pixel's M when p_hat_r(y) > 0, plus
   the capped M of the previous reservoir when the target at its own visible point and face is
   nonzero). The GI temporal pass keeps 1/M.
6. **Visibility reuse in the shade pass, without `UNBIASED`** (Bitterli et al. 2020, section 4.4;
   not in the design): when the shade's shadow ray is occluded, the final reservoir (the next
   frame's `di_prev`) is stored again with W = 0 (M kept). Reason: with a light voxel set into a
   wall or a ceiling, two of its three candidate faces are covered; the unshadowed target samples
   them as often as the open face, so most reservoirs hold an occluded sample, and without this the
   default `TEMPORAL | SPATIAL` mode gained almost nothing over one-sample NEE in the test box
   (single frame, one bounce, RMSE against a 2048-frame reference):

   | flags | without reuse | with reuse |
   | --- | --- | --- |
   | light NEE | 0.1307 | - |
   | DI, no reuse flags | 0.1304 | 0.1304 |
   | DI `T` | 0.1314 | 0.0950 |
   | DI `TS` (host default) | 0.1270 | 0.0505 |
   | DI `TSV` | 0.0553 | 0.0468 |
   | DI `TSU` | 0.0555 | 0.0555 (not applied) |

   Cost: a bias (the next frame treats the occluded sample as worthless although its target is
   not zero): the 1024-frame mean ratio of `TS` against light NEE went from 0.9994 to 0.9933 (box,
   one bounce). With `UNBIASED` the reservoir is left as it is, because the Z counts assume supports
   that do not depend on visibility. To remove it, delete the `StoreDiReservoir` in the occluded
   branch of `caelix_restir_di_shade.hlsl`.
7. **Entry index guard.** `DiResolve` treats an entry index >= the table's current count as no
   sample (the history is dropped on every rebuild anyway; this is the guard). The light cell is
   looked up once per pixel for all candidates.
8. **Bench runs** are in `RUNS` of `tools/gi_bench.py` (`di`, `di_unbiased`, `di_conv`); the json
   holds only the scenes.

### Measurements (RTX 5080, driver 610.88)

Tests (`tests/trace_test.cpp`, light box 128 x 128, 4 bounces):

- "ReSTIR DI unbiased": 1024 frames `LIGHT_NEE | RESTIR_DI | TEMPORAL | SPATIAL | UNBIASED` vs 1024
  frames `LIGHT_NEE`: mean 0.13916 vs 0.13959, ratio 0.9969, pixel PSNR 42.78 dB (light NEE against
  itself with other seeds: ratio 1.0002, 43.23 dB), 16 x 16 block PSNR 60.2 dB; reservoir bytes
  2097152.
- "ReSTIR DI lowers the error": single frame after 64 warm-up frames, RMSE against a 1024-frame
  light NEE reference: light NEE 0.1568, DI `TS` 0.0999 (0.64x), DI `TSU` 0.1025 (0.65x). The
  indirect light (bounce NEE) is in all three.
- "RESTIR_DI needs LIGHT_NEE": `RESTIR_DI` alone (both methods) and `di_candidates` 33 ->
  INVALID_ARGUMENT; 32 is accepted. The GI validation case now uses `1u << 7` as its unknown flag.
- All 7 ctest cases pass in `build-gi`; 0 validation errors.

Cornell (`cornell.cxw`, pose 24, 20, 1.5, fov 60, black sky, `--gi pt --gi-light-nee`, 64 warm-up
frames, `--no-denoise`; metrics of the 8-bit PNGs against a 2048-frame light NEE reference):

| run | trace ms | gi ms | PSNR | mean ratio |
| --- | --- | --- | --- | --- |
| 1280 x 720, light NEE | 1.48 | - | 14.37 dB | 0.826 |
| 1280 x 720, DI (`TS`) | 1.51 | 0.28 | 15.51 dB | 0.839 |
| 1280 x 720, DI unbiased | 1.50 | 0.49 | 15.54 dB | 0.841 |
| 1280 x 720, DI temporal only | 1.50 | 0.14 | 15.58 dB | 0.842 |
| 1280 x 720, DI no reuse | 1.50 | 0.10 | 15.26 dB | 0.839 |
| 1280 x 720, DI 2 spatial passes | 1.50 | 0.43 | 15.46 dB | 0.836 |
| 1920 x 1080, light NEE | 3.26 | - | | |
| 1920 x 1080, DI (`TS`) | 3.32 | 0.91 | | |
| 1920 x 1080, DI unbiased | 3.32 | 1.36 | | |

The single-frame mean ratios are below 1 for every run because the 8-bit PNG clips the noisy
bright pixels; DI is not darker than light NEE. Converged (512 frames): light NEE 0.9855, DI
0.9822, DI unbiased 0.9841 (same clipping effect). The full image is dominated by the indirect
noise of the bounces; the direct light alone (`--bounces 1`, reference 2048 frames): light NEE
18.56 dB (mean 0.917), DI `TS` 37.92 dB (0.988), DI unbiased 38.09 dB (0.994), DI `TSV` 38.30 dB
(0.989), DI temporal only 40.05 dB (0.999). The DI image is visibly less noisy on the directly lit
walls and not darker or brighter. With ReSTIR GI (`--gi restir --gi-light-nee`): 19.83 dB ->
21.81 dB with DI, gi_ms 0.31 -> 0.63. With NRD (64 frames): 24.21 dB vs 24.11 dB, denoise 0.50 vs
0.65 ms.

### Open points

- **History lost on every light table rebuild.** Every flush that finalizes bricks rebuilds the
  table (and the compact order is not stable), so a scene with edits or moving entities in every
  frame (`--test-edits`, the Unity FractalCave) runs DI without temporal reuse. A stable index (a
  reservoir keyed by group + dense brick id, remapped through a small table) would keep it.
- **Covered faces** still waste candidates (step 1's open point); visibility reuse hides most of
  it in the temporal history, but the initial candidates and the spatial neighbours still pick
  covered faces. Per-face open bits per light voxel would remove it at the source.
- **1/M darkening** of the spatial pass: about 1 % on the Cornell direct light (`TS` 0.988 vs
  temporal only 0.999), at face borders and shadow edges; `UNBIASED` removes it at 0.2 ms more at
  720p.
- `n_v` is the axis of the face index (exact for unrotated entities only), as in the GI kernels.
- Visibility reuse (difference 6) is a design change; the owner may prefer it behind a flag.
- Not measured: the bench matrix (`di`, `di_unbiased`, `di_conv` on the four scenes), moving
  cameras and moving light entities.
