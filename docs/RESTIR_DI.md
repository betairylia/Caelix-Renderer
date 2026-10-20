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

(Implementer.)
