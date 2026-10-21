# MIS between the light NEE and the BSDF path (step 3)

Design (Fable, 2026-10-21). Follows `LIGHT_LIST.md` (step 1) and `RESTIR_DI.md` (step 2). The
"As built" section at the end is the truth once the step is implemented.

## Why

Step 1 made the list lights NEE-only: a segment that leaves a vertex with light NEE skips the
emission of a list light it hits, except in the near field (`nee_min_distance`, 8 units), which
is the BSDF path's. Two problems remain:

- Bright emitters just outside the near field keep a high variance: the NEE reaches them through a
  brick choice that ignores their size on the receiver's hemisphere, while a cosine-sampled bounce
  hits a large close emitter often and with a bounded weight. Speckles in the 256-frame light NEE
  accumulation of the Cornell box; the near-field split only moved the problem to 8 units.
- A latent bias: a list-light brick that the table dropped (beyond `max_bricks`), or a voxel whose
  entry mask bit is clear (materials changed after the light word was written), is never sampled by
  the NEE, but the BSDF path still skips its emission. That light is lost.

Multiple importance sampling (Veach 1997, balance heuristic) combines the two strategies: at every
diffuse vertex the NEE contribution is weighted by `w_nee(y) = p_nee(y) / (p_nee(y) + p_bsdf(y))`
and a list-light hit of the next segment by `w_bsdf(y) = p_bsdf(y) / (p_nee(y) + p_bsdf(y))`. The
weights sum to 1 at every light point, so the sum of the two estimators is unbiased for any
densities as long as (a) both sides evaluate the SAME functions `p_nee` and `p_bsdf` of the vertex
and the point, and (b) `w_nee(y) = 0` wherever the NEE of this frame cannot produce `y`. Neither
density has to be the true density of the strategy: a proxy keeps the unbiasedness and only
changes the variance (Veach 1997, section 9.2.2).

## 1. The densities

`p_bsdf` is the solid-angle density of the cosine-weighted diffuse direction at the vertex,
`cos_r / pi` with `cos_r = dot(n_r, w)`. It is the exact density of `ShadeVoxelHit`'s diffuse
lobe (`normalize(N + unit vector)`). The specular lobe is not combined: a segment that leaves a
specular-lobe vertex counts every emission in full, as before (no NEE ran there).

`p_nee` is a PROXY for the area density of the light NEE at the vertex, in the area measure at the
light point, converted to solid angle by `d^2 / cos_l`. The true density of the NEE is not
evaluable (the brick comes from one of 32 ReGIR reservoirs whose contents are random, the cell is
jittered, empty reservoirs fall back to the global CDF). The proxy mirrors the cell fill's target
instead:

```
p_area(y) = p_brick(b) * (region_weight[z] / total_weight_b) * (1 / n_z) * (1 / n_faces)
p_brick(b) = p_hat_c(b) / C_c           with a cell c (section 3), or
           = total_weight_b / total     without one (the global CDF)
p_hat_c(b) = total_weight_b * orient / max(|centre_b - centre_c|^2, near_d2(level))
```

`region_weight[z]`, `total_weight_b`, `n_z` (the set bits of the slice mask), `centre_b` (brick
origin + 4, world) and `orient` (the mean-normal term of `caelix_light_cells.hlsl`, 1 without a
mean normal) come from the light table entry of the brick; `n_faces` is the number of axis faces
of the voxel whose outward normal points to the receiver's side of the voxel centre (1..3, exactly
as `SampleLightEntryWith` counts them); `C_c` is the cell's normalisation of section 3. The entry's
`total_weight_b` cancels inside the cell term:

```
p_area(y) = orient / max(d2_bc, near_d2) * region_weight[z] / (C_c * n_z * n_faces)   (cell)
p_area(y) = region_weight[z] / (total * n_z * n_faces)                                (global)
```

**Since step 4** (`docs/LIGHT_FACES.md`, per-face masks) the in-brick density comes from the face
weights: `region_weight[z] / (n_z n_faces)` above is replaced by `total_weight_b * face_weight[f] /
(W_cand n_f)`, with `f` the point's face, `n_f` the set bits of that face's mask and `W_cand` the sum
of the weights of the brick's candidate faces seen from the receiver (`LightCandidateWeight`, the
sampler's rule). The description carries the brick's world origin and axes, the six face weights,
the face and `1 / n_f` instead of the voxel frame and `pdf_base`, and the mean normal is derived
from the face weights (`LightBrickNormalLocal`).

Both sides evaluate `p_area` through ONE function, `LightNeeProxyPdf`, from the same inputs: a
per-vertex context (`LightNeeProxyContext`: whether a cell was found, its world centre, `near_d2`,
`C_c`, and the table's `total`) built by `LightNeeProxyAt(fc, x_r, group, object_point)` from the
UNJITTERED cell of the vertex (the jitter of section 2.5 of `LIGHT_LIST.md` only moves the cell
the NEE samples from; the proxy is a deterministic function of the vertex), and a per-light
description (`LightMisDesc`: the voxel frame = world centre + the three entity axes, `pdf_base =
region_weight[z] / n_z`, `brick_centre`, `brick_normal` (world, length 0 without a mean normal),
`total_weight_b`).

`w_nee = 0` where the NEE cannot sample (condition (b)): points closer than `nee_min_distance` (the
NEE returns 0 there; the BSDF side keeps `w_bsdf = 1`, as today), bricks without a table entry
(section 2), voxels whose mask bit is clear, and materials below `min_emission` (not list lights:
their emission is counted in full, as today). Where the NEE CAN sample but this frame's cell
happens not to hold the brick, `w_nee > 0` is still correct: every reservoir's `W` is an unbiased
weight of every brick, so the NEE estimator of `f * w_nee` is unbiased in expectation over the
fill (the same argument as step 1).

The balance heuristic: `w_nee = p_sa / (p_sa + p_bsdf)` with `p_sa = p_area * d^2 / cos_l`. No
power heuristic (the densities are proxies; the balance heuristic is the one that is provably never
much worse than either strategy alone).

### Vertex proxy (`CAELIX_GI_FLAG_LIGHT_PROXY_VERTEX`, 2026-10-22)

With the flag (and `LIGHT_MIS`), a vertex whose context has a cell (`ctx.cell`, `C_c > 0`) uses
the VERTEX form of the area density instead of the cell form above:

```
p_area(y) = L_f * cos_r * cos_l / max(d^2, 1) / C_c       (0 when W_cand = 0)
L_f       = face_weight[f] / n_f                          (the voxel-face weight of the point's face)
cos_r     = max(0, dot(n_r, w)),  cos_l = the light face's cosine toward the receiver
```

so `p_sa = L_f * cos_r * d^2 / max(d^2, 1) / C_c`. It follows the receiver's own geometry term
(its distance and both cosines) where the cell form only knows the distance from the cell centre
to the brick centre, which is what the per-vertex resampling (`LIGHT_LIST.md` section 2.6)
samples by. A vertex without a cell keeps the global form. `LightNeeProxyContext::vertex` is set
from the flag wherever a context is built or rebuilt (`LightNeeProxyOfSlot`, `DiReceiverOf`), so
the NEE side (`LightNextEvent`, `LightNextEventResampled`), the BSDF side (`TracePath`) and ReSTIR
DI's target and shade evaluate the same function, `LightNeeMisWeight`.

It is a PROXY like the cell form: any function that both strategies evaluate identically gives
weights that sum to 1 for every point, so the combination stays unbiased; only the variance
depends on how well it follows the true NEE density. Its scale is only approximate: `C_c` is the
cell fill's normalisation of the BRICK-level target `total_b * G` (section 3), not of the
voxel-face target `L_f * G_vertex`, so the vertex form over- or under-states the NEE density by a
factor that depends on the brick's size in voxels and on the cell's light distribution.

## 2. Brick -> entry index map (scene, `light_indices_`)

The BSDF side needs the light table entry of the brick a segment hit. A new growable buffer of
`u32` per AABB row, next to `light_words_` and managed exactly like it (reserved by
`ReserveLightWords`, moved by the same `light_moves_` copies, zero-filled for new rows and freed
ids by the same fills and pending copies), holds `entry index + 1` (0 = no entry). The compact
kernel writes it for EVERY dense id below the group's high water (0 for a light word of 0, for a
dropped entry and for a zero total), so a rebuild leaves no stale value behind; rows above the
high water are never hit. `LightPush` grows by one address (80 bytes). `CaelixFrameConstants`
carries the buffer address at offset 688 (`light_indices`, 0 without a table), and
`CaelixScene` exposes it like `light_table_address()`.

Section 2's map also fixes the latent bias in the non-MIS path: a list-light hit is skipped only
when the map has an entry for the brick AND the voxel's mask bit is set (the NEE can sample it).

## 3. Cell normalisation `C_c` (renderer, light cells)

The cell fill's RIS gives every reservoir `W = (w_sum / K) / p_hat(selected)`, and `w_sum / K`
is an unbiased estimate of `C_c = sum over the table of p_hat_c(b)`. The fill kernel averages
`w_sum / K` over its 32 threads (groupshared, thread 0 sums) and stores it in the cell header. The
trace of the same frame must not read the value computed from the reservoirs it samples (a ratio
estimator bias), so the header keeps two values: `C_prev` (the fill before the latest one; the
trace reads this one) and `C_cur` (written by the latest fill). The fill moves `C_cur` into
`C_prev` before writing. A cell inserted by a lookup starts with both at 0; an evicted cell gets
both 0; `C_prev = 0` selects the global proxy (the first frame of a cell).

Cell layout (832 bytes since the push-back of shadow ray results; 304 since the temporal cell
reuse; 296 in step 3, 288 before): `0 key_lo, 4 key_hi, 8 touched_frame, 12 entity, 16 coord (3),
28 level, 32 C_prev, 36 C_cur, 40 M, 44 C_n, 48 push_count, 64 reservoirs (32 x 8), 320 push ring
(32 x 16)`. `kLightCellBytes`, `kLightCellReservoirOffset` and the renderer's `kLightCellBytes`
(`renderer.h`) change together; the ABI comments say "832 bytes per cell".

With `CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL` (docs/LIGHT_LIST.md section 2.2) the normalisation comes
from dedicated draws: each fill thread draws one more entry from the CDF only for the estimate,
and C_cur is the running mean of those draws over at most `light_temporal_frames` x 32 draws
(count in C_n). The reason: with temporal reuse the previous fill's candidates are part of this
frame's reservoirs, so `w_sum / K` (even one frame delayed through C_prev) would be correlated
with the reservoirs that the trace samples. The trace still reads C_prev.

With `CAELIX_GI_FLAG_LIGHT_CELL_PUSH` (docs/LIGHT_LIST.md section 2.2, 2026-10-21) the cell
reservoirs also hold voxel-face samples with a voxel-face target, but the proxy stays brick-level
(`C_c` from the brick-level fresh or dedicated draws, `p_hat_c(b)` times the in-brick density of
the sampler at x_r) for both kinds of sample. That is valid: the balance heuristic only needs a
weighting function that both strategies evaluate the same way for the same point and that is 0
only where the NEE cannot produce the point; it does not need the NEE's true density. A proxy
further from the true density costs variance, not bias. The voxel-face samples still come from
reservoirs whose fresh candidates cover every brick, so the NEE support is unchanged.

## 4. Path tracer (`TracePath`, flag `CAELIX_GI_FLAG_LIGHT_MIS`)

At a vertex with light NEE (diffuse lobe, next segment exists), `LightNextEvent` takes `mis` and
returns its contribution times `w_nee` (the proxy context is built from the unjittered cell; with
the jitter on, the sampling cell is a second lookup). The vertex remembers `x_r`, `n_r` and its
proxy context for the segment that leaves it.

When that segment hits a list light (`emission_luminance >= light_min_emission`) at `T >=
nee_min_distance`:

```
index = light_indices[aabb_row_base(group) + id]        (0 -> w_bsdf = 1)
entry = LoadLightEntry(index - 1)
voxel = floor(object_hit - 0.25 * object normal) - entry.origin   (outside 0..7 or mask bit clear -> w_bsdf = 1)
face  = the hit face (from the object normal); n_faces counted from x_r as SampleLightEntryWith does
p_sa  = LightNeeProxyPdf(context, desc, voxel slice, n_faces) * T^2 / cos_l
w_bsdf = p_bsdf / (p_bsdf + p_sa),  p_bsdf = dot(n_r, direction) / pi
radiance += emission * throughput * w_bsdf
```

Without the flag (or without MIS at this vertex): `w_bsdf = 0` when the NEE can sample the point
(entry found and mask bit set), else 1. That is step 1's rule with the bias fix of section 2.

Vertices that keep step 1's exclusive rule even with the flag (their BSDF sample is reused by
other pixels, so a pixel-dependent weight would bias the reuse): the primary vertex in ReSTIR GI
mode (`restir`). In ReSTIR DI mode (`restir_di`) the primary vertex takes part since section 5
(phase B).

## 5. ReSTIR DI (second commit)

The primary vertex's NEE is the DI reservoir. Its sample is reused by other pixels, so the
receiver's proxy context must travel with the reservoir: `DiReservoir` grows to 48 bytes
(`float3 cell_centre`, `float cell_C` (0 = global proxy or no cell), level in flag bit 4), filled
by `DiInitialReservoir` from the primary vertex's cell lookup (no jitter there, as today) and kept
as "own" fields by every merge (like `visible_pos`). The target function becomes `p_hat(y) =
luminance(L_e) * G * w_nee(y; receiver)` (the integrand of the NEE strategy; the Z counts of the
spatial pass evaluate it at the neighbours' receivers from their stored context), the shade pass
contributes `L_e * G * V * W * w_nee`, and the first segment of a diffuse primary vertex in DI mode
uses `w_bsdf` from the same context as section 4. `DiLight` carries the `LightMisDesc` of its entry
(resolved by `DiResolve` together with the point). ReSTIR GI keeps the exclusive rule at the
primary vertex (its sample radiance `L_o(x_s)` is pixel-independent by construction).

## 6. ABI

Flags only: `CAELIX_GI_FLAG_LIGHT_MIS (1u << 8)` ("with LIGHT_NEE: MIS between the light NEE and
the diffuse bounce; without it a list-light hit after a light NEE vertex is skipped"). The renderer
keeps it in the path-trace method like `LIGHT_CELL_JITTER`. `CaelixGiOptions` is unchanged. Host:
`--light-no-mis` (the flag is on by default with `--gi-light-nee`); bench runs `pt_nee_nomis`.

## 7. Tests (`tests/trace_test.cpp`)

- "light NEE with MIS is unbiased": the light box of step 1 with `LIGHT_NEE | LIGHT_MIS`, 1024
  reference frames against the 4096-frame path tracer: ratio in 0.97..1.03, block PSNR > 30 dB.
- "MIS on a large emitter": the same box with an 8 x 8 emissive panel (one brick, strength 3) in
  the ceiling at distance > 8 from the floor, so that both strategies carry weight: with and
  without the flag against the path tracer, ratio in 0.97..1.03 both; the per-pixel RMSE of a
  64-frame accumulation is printed for both (expected: lower with MIS).
- "dropped bricks keep their emission": `max_bricks 1` with two light bricks: without MIS the
  dropped brick's emission is still counted (ratio against the path tracer in 0.97..1.03).
- The step 1 cases keep passing (the cell layout changed: "cells fill and sample" reads the cells).

## 8. Bench

`tools/gi_bench.py`: `pt_nee` gains the flag by default (host default), `pt_nee_nomis` is the old
behaviour; the four scenes at 1 spp against their references.

## As built (phase A: sections 1-4, 6-8; 2026-10-21)

Step 4 (`docs/LIGHT_FACES.md`, "As built") changed the proxy's in-brick density to the face weights
(`LightNeeProxyPdfArea(ctx, d, x_r)`, `LightCandidateWeight` instead of `LightCandidateFaces`, a new
`LightMisDesc`); the description below is the phase A state, and the measurements of both phases
were taken before step 4.

Section 5 (ReSTIR DI with MIS) was built in phase B (see "As built (phase B)" below). In phase A
the DI target, reservoir and shade pass were unchanged, and the primary vertex kept the exclusive
rule in ReSTIR DI mode (with the bias fix of section 2).

### Sizes

| item | size |
| --- | --- |
| `light_indices_` (scene) | 4 B per AABB row, next to `light_words_` (256 KiB at the initial 65536 rows); counted in `CaelixLightStats::table_bytes` |
| `LightPush` | 72 -> 80 B (`light_indices` after `light_words`; C++ and both shader copies) |
| light cell | 288 -> 296 B (`C_prev` at 32, `C_cur` at 36, reservoirs at 40) |
| `FrameConstants` | 704 B unchanged: `light_indices` (u64) at 688 replaces `pad_688`/`pad_692` (720 B since the per-vertex resampling, `LIGHT_LIST.md` section 2.6) |
| public ABI | `CAELIX_GI_FLAG_LIGHT_MIS (1u << 8)` only; no struct changed |

### Kernels and code touched

- `caelix_light_compact.hlsl`: `CompactBrick` returns entry index + 1 (0 = no entry); `main` writes
  the map for every dense id below the high water.
- `caelix_light_scan.hlsl`: the push struct only.
- `caelix_light_cells.hlsl`: `gs_c[32]` + one barrier after the reservoir store; thread 0 moves
  `C_cur` to `C_prev` and stores the mean of `w_sum / K`. Eviction and the "no usable table" case
  zero both values. Every early return stays workgroup-uniform; the normal path has none before the
  barrier.
- `caelix_light_sample.hlsl`: `RowsTransform`/`RowsRotate` (moved from the fill kernel),
  `LightMisDesc` (+ `LightMisDescWith`, `LightMisDescOf`), `LightCandidateFaces`,
  `LightNeeProxyContext` (+ `LightNeeProxyOfSlot`, `LightNeeProxyAt`), `LightNeeProxyPdfArea`,
  `LightNeeMisWeight`, `LightListCanSample`; `LightSample` carries `mis`; `LightNextEvent` takes
  `mis` and returns the context through `out LightNeeProxyContext ctx`.
- `caelix_path_trace.hlsl` (`TracePath`): `light_mis`, the per-segment state (`segment_mis`,
  `segment_x_r`, `segment_n_r`, `segment_ctx`) and the hit rule of section 4.
- Scene: the map is reserved, grown, moved (`light_moves_`), zero-filled (new rows) and zeroed (freed
  ids) exactly like `light_words_`. Renderer: `constants.light_indices`, `kCellBytes` 296, the flag in
  `kKnownFlags` and in the path-trace flag mask.
- Host: `GiSettings::light_mis` (default on, only with `light_nee`), `--light-no-mis`, a "MIS light
  NEE / diffuse bounce" checkbox in the GI section. `tools/gi_bench.py`: `pt_nee_nomis` and
  `pt_nee_nomis_conv`.

### Differences from the design, and why

1. `LightNextEvent` builds the proxy context (and does the extra unjittered lookup) only when `mis`
   is set; without it the context is the global one, and the lookups and random numbers are the
   same as before step 3 (without MIS nothing reads the context). With `mis` and the jitter on, a
   vertex does two lookups (unjittered for the proxy, jittered for sampling); with the jitter off one
   lookup serves both. The MIS draws no random number.
2. The function the design calls `LightNeeProxyPdf` is `LightNeeProxyPdfArea`; the shared cell part
   is `LightNeeProxyOfSlot`, so that `LightNextEvent` and `LightNeeProxyAt` build the context from
   the same code. `LightNeeProxyAt` was not called in phase A (phase B calls it in `TracePath`).
3. `LightMisDesc` computes the brick centre and the world mean normal from the voxel frame's axes
   (`centre_b = voxel centre + R (3.5 - voxel)`) instead of a second load of the entity rows. The NEE
   side (`SampleLightEntryWith`) and the BSDF side (`LightMisDescOf`) both go through
   `LightMisDescWith` with the same frame function, so both evaluate the same numbers.
4. Degenerate densities in `LightNeeMisWeight`: `cos_l <= 0` or a non-finite `p_sa` counts as
   `p_sa = +inf` (NEE 1, BSDF 0); `p_sa + p_bsdf = 0` gives NEE 0, BSDF 1. Every branch keeps
   `w_nee + w_bsdf = 1` for the same inputs.
5. The "dropped bricks keep their emission" test also runs with MIS (the dropped brick gets
   `w_bsdf = 1` there as well).
6. Test helpers: `LightBoxWalls()` builds the walls for the three box variants; the pixel metric
   helpers (`PixelPsnr`, `PixelRmse`, `ColorPixels`) moved up in `trace_test.cpp` so that the new
   tests can use them.
7. "ReSTIR GI: set_gi_options validation" used `1u << 7` as an unknown flag, which has been
   `LIGHT_CELL_JITTER` since the cell jitter commit (the test failed since then); it uses `1u << 31`
   now. All 27 trace test cases pass.

### Measurements (`tests/trace_test.cpp`, RTX 5080, 128 x 128, 4 bounces)

| test | result |
| --- | --- |
| light NEE with MIS is unbiased (1 voxel, 1024 frames vs 4096-frame PT) | ratio 1.0095, block PSNR 45.78 dB (light NEE without MIS: 1.0095, 45.78 dB; PT vs PT other seeds: 1.0060, 42.04 dB) |
| MIS on a large emitter (8 x 8 panel), light NEE | ratio 1.0035, 64-frame RMSE 1.4757 |
| MIS on a large emitter, light NEE + MIS | ratio 1.0032, 64-frame RMSE 1.4578 (0.988 x) |
| dropped bricks keep their emission (`max_bricks` 1, 1 of 2 bricks dropped), light NEE | ratio 1.0032, block PSNR 39.55 dB |
| dropped bricks, light NEE + MIS | ratio 1.0032, block PSNR 39.55 dB |

The single-voxel light: from the floor a voxel face subtends about 1 / d^2 = 1 / 600 sr, so
`p_sa` is about 3 orders of magnitude above `p_bsdf` and `w_nee` is about 0.999: the image is the
NEE image to 4 digits. The panel (0.1 sr from the floor) gives `w_nee` about 0.97 and a small gain
(1.2 %): the 64-frame error there is dominated by other terms (indirect bounces, NEE samples on the
covered side faces of the panel voxels). MIS helps where an emitter just beyond the near field
covers a large part of the hemisphere; the bench scenes (`pt_nee` vs `pt_nee_nomis`) are the real
measure and were not run in this step. The dropped-bricks case before this step lost the dropped
light (the BSDF path skipped it and the NEE could not sample it); its ratio is now 1.003.

## As built (phase B: section 5, ReSTIR DI with MIS; 2026-10-21)

### What changed

- `caelix_restir_di.hlsl`: `DiReservoir` is 48 bytes; the third word is the receiver's proxy
  context (`float3 cell_centre`, `float cell_C`; 0 = global proxy), the cell level is own flag bit 4
  (`kDiCellLevel`). `DiMisEnabled(fc)` = the MIS flag and not ReSTIR GI. `DiReceiver` +
  `DiReceiverOf(fc, r)` rebuild the receiver (x_v, the face's axis normal, the context with
  `near_d2` from the level and the table's `total`). `DiLight::mis` (`LightMisDescOf` in
  `DiResolve`). `DiGeometry` has an overload that also returns `w` and `cos_l`. `DiTargetPdf(l,
  receiver, mis)` multiplies by `LightNeeMisWeight(..., nee_side = true)`.
  `DiMergeFinishWithCount(s, own, count)` copies `visible_pos`, the own flags and the context from
  the receiver's own reservoir. `DiInitialReservoir` builds the context from its unjittered cell
  lookup (`LightNeeProxyOfSlot`) and weights each candidate's `p_hat` by `w_nee` (`s.mis`).
- Temporal / spatial passes: the receiver from `DiReceiverOf(own)`; the spatial pass keeps whole
  copies of the accepted neighbours and evaluates the Z counts at `DiReceiverOf(accepted[k])`.
- Shade pass: the contribution times `w_nee` at the receiver.
- `TracePath`: in ReSTIR DI mode with MIS (and without ReSTIR GI) the primary vertex sets
  `segment_mis`, `segment_ctx = LightNeeProxyAt(fc, bounce_origin, surface.group,
  LightCellPoint(surface))` and `segment_n_r = GiFaceNormal(GiFaceIndex(surface.normal))`.
- C++: DI reservoir buffers 48 bytes per pixel (`renderer_targets.cpp`), comments in `renderer.h`
  and `caelix.h`. No ABI struct changed.

### Differences from the design, and why

1. **Not with ReSTIR GI.** `DiMisEnabled` is false when `gi_method` is ReSTIR GI: the first segment
   of the primary vertex keeps the exclusive rule there (ReSTIR GI reuses it), so the DI must keep
   its full weight, or the two strategies would not sum to 1 and the direct light of the list
   lights would be too dark. With ReSTIR GI the DI behaves as in step 2 (zero context fields).
2. **The BSDF side uses the axis normal.** The DI's receiver normal is the axis normal of the
   visible face (`GiFaceNormal`, as in all DI passes); `TracePath` uses the same normal for
   `p_bsdf` at the primary vertex in DI mode (only in the MIS weight; the bounce direction and its
   cosine are unchanged), so that `w_nee + w_bsdf = 1` holds exactly also on rotated entities.
   Elsewhere `segment_n_r` stays `surface.normal`.
3. **The context and the MIS descriptions are built only with `DiMisEnabled`.** Without the flag
   the initial reservoir stores zeros in the third word and `DiResolve` does not load the voxel
   frame a second time; the images without the flag are the same as before (the DI test cases give
   identical numbers with the phase A shaders loaded through `CAELIX_SHADER_DIR`).
4. **Both lookups are the same lookup.** `LightNeeProxyAt` in `TracePath` runs before
   `DiInitialReservoir` in the same thread and inserts the cell when it is missing; the DI lookup
   then finds that slot. `C_prev` is written only by the fill kernel, so both read the same value.
5. **Tests**: the DI+MIS unbiasedness check compares against the 1024-frame light NEE reference,
   as the existing DI case does (light NEE itself matches the path tracer, "light NEE is unbiased");
   the comparison against the path tracer is in "MIS on a large emitter", which got two more
   modes (ReSTIR DI and ReSTIR DI + MIS, both `TEMPORAL | SPATIAL | UNBIASED`).

### Measurements (`tests/trace_test.cpp`, RTX 5080, 128 x 128, 4 bounces)

| test | result |
| --- | --- |
| ReSTIR DI unbiased vs light NEE (1024 frames each), without MIS | ratio 0.9943, block PSNR 54.88 dB, pixel PSNR 38.78 dB (unchanged from phase A) |
| ReSTIR DI unbiased + MIS vs light NEE | ratio 0.9998, block PSNR 55.55 dB, pixel PSNR 38.71 dB |
| single frame RMSE after 64 warm-up frames (vs 1024-frame light NEE) | light NEE 0.15678, DI `TS` 0.15367, DI `TSU` 0.15484, DI `TSU` + MIS 0.15430 |
| panel (8 x 8 emitter) vs 4096-frame path tracer, ReSTIR DI | ratio 1.0076, 64-frame RMSE 2.5797 |
| panel, ReSTIR DI + MIS | ratio 1.0074, 64-frame RMSE 2.3871 (0.925x) |

All 27 trace test cases pass. The single-voxel light box gives `w_nee` near 1 (see phase A), so
the DI with and without MIS are almost the same image there; the panel shows the gain. The 64-frame
RMSE of the DI runs is above the one-sample light NEE (1.46-1.48) because the temporal reuse
correlates the frames of a short accumulation; it is not a per-frame comparison. The bench scenes
were not run in this step.
