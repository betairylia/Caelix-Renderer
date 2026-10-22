# Light NEE: RIS + NEE on world-space light cells (as built)

Branch `light-nee-experiments`, 2026-10-22. This document describes the many-lights direct light
path as the code has it. It replaces `LIGHT_LIST.md`, `LIGHT_FACES.md`, `LIGHT_MIS.md` and
`RESTIR_DI.md`; those, with their design history and every measurement, are copied verbatim to
`F:/RP_Games/Caelix-family/Artifacts/gi-experiments-20261020/docs/`. Measurements and images are in
`F:/RP_Games/Caelix-family/Artifacts/gi-experiments-20261020/` (section 8). ReSTIR GI is described
in `docs/GI_EXPERIMENTS.md`.

Code: `shaders/common/caelix_light.hlsl` (list-light rule, table layout), `shaders/scene/
caelix_finalize.hlsl` (light words), `shaders/scene/caelix_light_compact.hlsl` and
`caelix_light_scan.hlsl` (table rebuild), `shaders/render/caelix_light_cells.hlsl` (cell fill),
`shaders/render/caelix_light_sample.hlsl` (cells, sampling, NEE, MIS proxy), `TracePath` in
`shaders/render/caelix_path_trace.hlsl`, `lib/src/scene/scene_lights.cpp`, `lib/src/render/
renderer_render.cpp` and `renderer_targets.cpp`. ABI: `CaelixLightOptions`, `CaelixLightStats`,
`caelix_scene_set_light_options`, `caelix_scene_get_light_stats`, the `light_*` fields of
`CaelixGiOptions` and `CaelixGiStats`, and the `CAELIX_GI_FLAG_LIGHT_*` flags (experimental, no ABI
bump).

## 1. Direction

The owner chose the many-lights direction on 2026-10-22: resampled importance sampling (RIS) and
next-event estimation (NEE) on a coarse world-space grid of light cells.

- Every emissive voxel bright enough is a "list light" in a per-brick light table (section 2).
- A hash table of world-space cells holds a few reservoirs of brick-level light samples per cell.
  A fill kernel refills them from the table before every trace (section 4).
- Every diffuse path vertex draws a few light candidates from its cell, keeps one by its own target,
  traces one shadow ray (section 5) and combines the result with the BSDF path by MIS (section 6).

The aim is a raw signal whose spatial and temporal correlation stays low, so that a denoiser or an
upscaler (NRD, DLSS RR) can carry the remaining noise. Paths that reuse whole shaded light samples
across pixels or frames were tried and removed (section 9): per-pixel ReSTIR DI and the push-back
of shadow ray results. No ReSTIR PT (owner decision).

## 2. Light list (scene)

### 2.1 Light words

A material is a list light when the luminance of its emission, `dot(emission, (0.2126, 0.7152,
0.0722))` before `CaelixRenderOptions::emission_scale`, is at least `min_emission`. The same rule
holds for both encodings (block-ID is the standard, packed colours are a test bed). The finalize
kernel writes one 4-byte light word per AABB row (`Scene::light_words_`, the same index space as the
AABB rows): `asuint` of the luminance sum of the brick's alive list-light voxels, 0 when there is
none. New and grown AABB ranges get zero words, freed ids get a zero word, and range moves copy the
words with the rows.

### 2.2 Light table

Every flush that finalized bricks, freed ids, moved AABB ranges, released a group or changed the
options rebuilds the table with two kernels after the finalize dispatch (inside the flush timer):

1. `caelix_light_compact` (64 threads per group index): for every dense id with a light word, it
   recomputes the entry from the record and stores it at `InterlockedAdd(count, 1)`; an index at or
   above the capacity counts as dropped. An entry whose recomputed total is 0 is not stored.
2. `caelix_light_scan` (one workgroup of 1024): an inclusive prefix sum over the entries' total
   weights (the CDF), and the header's capacity and total weight.

Layout (one device-local buffer, capacity N = `max_bricks`):

```
header    64 B      uint count; uint dropped; uint capacity; uint pad; float total_weight; pad
entries   N x 448 B
cdf       N x 4 B   inclusive prefix sum of entry.total_weight (cdf[count - 1] = total_weight)
```

The table is opt-in: the scene builds it only after `caelix_scene_set_light_options` was called.
A host that never calls it (the Unity plugin) pays only the light words and the entry map (4 + 4
bytes per AABB row). `Scene::light_table_address()` is 0 until a rebuild was recorded; there is no
light NEE before that.

### 2.3 Entry layout and per-face masks

```
0     uint2 face_mask[6][8]   64 B per face f (+X 0, -X 1, +Y 2, -Y 3, +Z 4, -Z 5): bit v of slice z
                              is set when voxel v = x | y << 3 | z << 6 is a list light and its
                              face f is open
384   float face_weight[6]    luminance sum over the set bits of face_mask[f]
408   float total_weight      luminance sum over the brick's list-light voxels (the CDF weight)
412   uint group; 416 uint id; 420 uint entity; 424 int3 origin (entity-local, brick key x 8)
436   uint face_extent[2]     8 bits per face: for a +a face the lowest row (0..7) along axis a
                              over the set bits of its mask, for a -a face the highest; 0xFF for an
                              empty mask
444   uint pad
```

A face is open when it lies on the brick border or when the neighbour voxel inside the brick is not
occupied. A neighbour brick is not consulted, so a border face covered by the next brick counts as
open: its samples are wasted (the shadow ray finds the cover), not biased. The render kernels load
bytes 384..447 as a `LightEntryHeader` and a face mask only when they need it.

The brick's mean emitting direction (`LightBrickNormalLocal`) is the normalised luminance-weighted
sum of the open-face directions, `normalize(w0 - w1, w2 - w3, w4 - w5)`; it is 0 when every axis
emits the same both ways. The fill target and the MIS proxy use it.

### 2.4 CDF

A brick is drawn from the global table with `LightCdfSearch`: a binary search for the smallest index
whose CDF value is above `u * total_weight`. The probability of entry i is `total_i / total`.

### 2.5 Options and stats

`CaelixLightOptions`: `min_emission` (0 = 20.0: packed strength 2 with an albedo luminance of at
least 0.3125, and every bright strength 3 voxel; the packed emitter is `albedo x 8^s`),
`max_bricks` (0 = 65536, 1 .. 2^20), `nee_min_distance` (0 = 8 world units, section 2.6).
`min_emission` applies to bricks finalized after the call (call it before the uploads); a new
`max_bricks` reallocates the table at the next flush. `CaelixLightStats`: entries and dropped bricks
of the newest safe rebuild (read back), the table bytes (`64 + 452 x max_bricks` plus the light
words and the entry map) and the rebuild count. 29.6 MB at 65536 entries, 474 MB at 2^20 (the table is bound only by
the compact and scan kernels; the render kernels read it through addresses).

### 2.6 Near-field split

Light points closer than `nee_min_distance` to the receiver belong to the BSDF path. The NEE returns
0 for such a point before the shadow ray, and `TracePath` keeps the full emission of a list light
that a segment hits closer than that (`hit.T`). Both sides measure from the same pushed-off origin,
so the split is unbiased. The reason: uniform area sampling with a `1/d^2` term and a brick choice
that ignores the distance give an extremely heavy-tailed estimate near a lamp (bistro lost 14 % of
its light after 512 frames before the split; `light-nee/diagnosis/`). The BSDF path finds near
emitters with a bounded weight.

### 2.7 Brick -> entry map

`Scene::light_indices_` holds `entry index + 1` per AABB row (0 = no entry), managed like the light
words. The compact kernel writes it for every dense id below the group's high water (0 for no
light, a dropped entry or a zero total), so a rebuild leaves no stale value. The trace kernel reads
it through `CaelixFrameConstants::light_indices` (`LightListCanSample`, section 6.3).

## 3. Light cells (renderer)

### 3.1 Key

A cell is keyed by (entity, level, cell coordinate, receiver face). The coordinate of an
entity-local point p at level L is `floor(p) >> (5 + 2 L)`. The point of a vertex is its hit point
moved a quarter voxel into the air along the face normal (`LightCellPoint`), so that the cell is the
one of the space the vertex sees. With `CAELIX_GI_FLAG_LIGHT_CELL_ORIENTED` the receiver face is the
axis of the vertex's world normal (`GiFaceIndex`, 0..5); without it the face is
`kLightCellFaceNone` (6) and the cell is unoriented. Every spatial cell then splits into up to 6
oriented cells. The fingerprint (`LightCellKey`) is three rounds of splitmix64 over (entity, level),
(x, y) and z, and a fourth over the face of an oriented cell; the low word is forced non-zero.

### 3.2 Levels and footprint

Level 0 cells are 32 voxels on an edge, level 1 cells 128 voxels (`LightCellSize`). A vertex uses
level 0 when its pixel footprint `2 d tan_half_fov_y / size.y` (world units per pixel at its camera
distance d) is below `light_level_footprint` (0 = 0.25), else level 1. Every vertex of a path uses
its own camera distance.

### 3.3 Hash table, probes, eviction

Slot = `(key_lo + probe) & (capacity - 1)`, linear probing up to 8 slots (`LightCellLookup`). An
empty slot is claimed with a 32-bit compare-and-swap on `key_lo` (64-bit buffer atomics are not a
device requirement); the inserting thread then stores `key_hi`, the touch frame, entity, coordinate,
level, zero normalisations and the face. A `key_lo` match counts when `key_hi` matches or is still 0
(an insert in progress), so a thread that lost the CAS does not insert the same key again further
on. A low-word collision during a pending insert merges two cells; that costs importance, not
correctness, because every reservoir carries its own W. Every lookup writes `touched_frame`. A
lookup that finds neither the key nor a free slot in 8 probes returns no cell: the vertex samples
the global CDF. The fill evicts a cell (key 0, reservoirs and normalisations cleared) when it was not
looked up for 64 frames (`kLightCellMaxAge`); `frame_counter` (the renderer's render count) is the
clock.

### 3.4 Layout

A cell is `64 + 8 R` bytes (`LightCellBytes`), R = `light_cell_reservoirs` (a power of two in
1 .. 256, library default 8: 128 B):

| offset | field | meaning |
| --- | --- | --- |
| 0 | `uint key_lo, key_hi` | fingerprint; `key_lo` 0 = empty slot |
| 8 | `uint touched_frame` | `frame_counter` of the newest lookup |
| 12 | `uint entity` | |
| 16 | `int3 coord` | cell coordinate at `level` |
| 28 | `uint level` | 0: 32-voxel cells, 1: 128-voxel cells |
| 32 | `float C_prev` | cell normalisation of the fill before the latest one (the MIS proxy reads it; 0 = none) |
| 36 | `float C_cur` | cell normalisation of the latest fill |
| 40 | `float M` | temporal reuse: confidence sum behind every reservoir of the cell; 0 = no history |
| 44 | `float C_n` | temporal reuse: draw count behind `C_cur` |
| 48 | `uint reserved` | written 0 |
| 52 | `uint face` | receiver face of the key (0..5, or 6 for an unoriented cell) |
| 56 | 8 bytes | padding |
| 64 | `R x { uint light; float W }` | reservoirs: `light` = the entry index of a brick (`kNoLight` = 0xFFFFFFFF = empty), `W` = its unbiased contribution weight (0 = empty) |

A reservoir holds a brick only; the vertex samples the point inside the brick (section 5.4). The
renderer creates the buffer at the first render with `CAELIX_GI_FLAG_LIGHT_NEE`, zero-fills it, and
recreates it (the old one retired until its last frame is safe) when the capacity or R changes.

### 3.5 Usage counters

After the last cell (`LightCellCounterBase` = capacity x cell bytes) the buffer holds 16 bytes:
`live` (cells with a key after this frame's eviction), `active` (of those, looked up by the previous
frame's trace), `inserted` and `failed` (lookups of this frame). The renderer clears them before the
fill, copies them after the trace into an 8-slot host-visible ring, and `caelix_renderer_get_gi_stats`
reports the slot of the frame of `CaelixGpuTimings` as `CaelixGiStats::light_cells_live`,
`light_cells_active`, `light_cells_inserted` and `light_cell_lookups_failed`. The fill's thread 0 of
a cell counts `live` and `active` once per cell; the lookup counts `inserted` and `failed`.

### 3.6 Capacity and the buffer guard

The capacity is a power of two in 1024 .. 2^20 (library default 65536, host default 262144). The
usage counters showed that 65536 cells saturate at 1080p: the shroom forest at 960 x 540 looks up
about 74000 cells per frame and keeps about 200000 live (64 frames of retention), with about one
million failed lookups per frame at 65536 cells. Oriented cells need about 3.5 x more live cells.

The cell table is one storage buffer descriptor (trace set binding 21) and the kernels address it
with 32-bit byte offsets. `caelix_renderer_set_gi_options` therefore rejects (logged,
`CAELIX_ERROR_INVALID_ARGUMENT`) options whose table, `capacity x (64 + 8 R) + 16` bytes with the
defaults for zero fields, is larger than the device's `maxStorageBufferRange`
(`Device::max_storage_buffer_range`, read at adoption). The largest table, 2^20 cells of 256
reservoirs, is 2.2 GB.

## 4. Fill (`caelix_light_cells`)

### 4.1 Candidates and target

The fill runs before the trace (and before the sky SH kernel) when light NEE is on and the scene has
a table; its time is `CaelixGiStats::light_fill_ms`. Workgroups of 256 threads hold 256 / R cells,
R threads per cell; thread `r = thread % R` owns reservoir r. The dispatch is
`min(1024, G) x max(1, G / 1024)` with `G = capacity x R / 256` workgroups (at least 4 and exact,
because both are powers of two), slot = `(y * 1024 + x) * (256 / R) + thread / R`. A skipped, evicted
or out-of-range slot only clears its threads' `active` flag, so every thread reaches the one
barrier. Without a usable table the cell's reservoirs and normalisations are cleared.

Each thread draws K = `light_candidates` (0 = 8, 1 .. 64) bricks from the CDF (proposal
`p_i = total_i / total`) and keeps one by streaming RIS with the brick-level target

```
p_hat_i  = total_i * G_i
G_i      = orient * orient_r / max(|centre_i - cell centre|^2, (cell half size + 4)^2)
orient   = max(0.1, dot(mean normal_i, direction from the brick centre to the cell centre)),
           1 without a mean normal
orient_r = the receiver orientation term (section 4.2), 1 for an unoriented cell
```

in world space (the entity transforms of the brick and of the cell). The resampling weight of a
fresh candidate is `w_i = p_hat_i / p_i = G_i * total`; the reservoir stores the selected entry index
and `W = (w_sum / M) / p_hat(selected)`, M = K without temporal reuse. W is an unbiased contribution
weight of the brick-level integral for every receiver of the cell, because `p_hat_i > 0` for every
entry with `total_i > 0`. The random stream is `InitRandom(uint2(slot, r), frame_seed ^ 0x3C6EF372)`:
a function of the slot, the reservoir and the frame only, so the output does not depend on the
workgroup size.

### 4.2 Receiver orientation (oriented cells)

With `CAELIX_GI_FLAG_LIGHT_CELL_ORIENTED` the receivers of a cell face one way, and the target gets

```
orient_r = max(0.1, dot(n_cell, normalize(brick centre - back)))
back     = cell centre - (cell half size) * n_cell
```

with `n_cell` the world axis of the cell's face (`LightCellFaceNormal`, `LightCellReceiverOrient`).
Every receiver of the cell lies in front of the back plane, so a lamp at the receivers' own height
keeps a positive cosine. The floor 0.1 must not be 0: a receiver can read a cell whose orientation
or position is not its own (the jittered lookup reads a neighbouring cell, a key collision merges
two cells), and every reservoir must stay an unbiased contribution weight for it, which needs
`p_hat > 0` for every entry. A light behind the cell's face then costs noise (W up to 10 x), not
bias. The MIS proxy applies the same factor (section 6.1).

### 4.3 Temporal reuse

With `CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL` (static scenes only: no scene edits, no entity motion)
each thread merges the reservoir it wrote in the previous fill as one more candidate, after its K
fresh ones:

```
M_old = min(cell M, light_temporal_frames x K)       (0 drops the history)
w_old = p_hat(y_old) * W_old * M_old                 (p_hat with this frame's target)
M_new = K + M_old;  W = (w_sum / M_new) / p_hat(selected);  header M = M_old + K
```

Why it is unbiased: streaming RIS gives `E[f(y) W] = (1 / M) sum_i c_i E[f(y_i) W_i]` for every
receiver f of the cell when the presence and the confidence `c_i` of every input are constants (they
do not depend on the inputs' values) and each `W_i` is a contribution weight over the cell's
receivers. The confidences K and `M_old` depend only on the frame count, the old reservoir is such a
contribution weight (`E[p_hat(y_old) W_old] = C`, so `w_old` has expectation `M_old C`), every entry
is reachable from the fresh draws, and the per-candidate target cancels in W. The history's random
number is drawn whenever the flag is on, so the stream never depends on the data. The renderer
passes `light_temporal_frames = 0` for one fill after a light table rebuild (entry indices may have
moved; `CaelixLightStats::rebuilds` changed) or after `CaelixGiOptions::reset`. A fill without the
flag stores M = 0, so a later switch to the flag starts clean.

### 4.4 Cell normalisation (C_prev, C_cur)

The MIS proxy (section 6.1) needs the cell normalisation `C_c = sum over the table of p_hat_i`.
Without temporal reuse `w_sum / K` of the fresh candidates of every reservoir is an unbiased
estimate of it: thread r = 0 moves `C_cur` to `C_prev` and stores the mean over the cell's R threads
as the new `C_cur`. The trace reads `C_prev` only, so it never divides by an estimate made from the
reservoirs it samples (a ratio of two estimates from the same reservoirs would be biased).

With temporal reuse that is not enough: the previous fill's candidates are part of this frame's
reservoirs, so an estimate made from the candidates, also one frame delayed through `C_prev`, is
correlated with the reservoirs that the trace samples. Each thread therefore makes one dedicated CDF
draw `w_c = G_j * total`, independent of every reservoir, and thread r = 0 folds the mean of the R
into `C_cur` as a running mean over at most `light_temporal_frames x R` draws (`C_n` counts them).
An evicted cell and a cell filled without a usable table get `C_prev = C_cur = M = C_n = 0`; a cell
with `C_prev = 0` uses the global proxy.

## 5. NEE at a vertex (`LightNextEvent`)

The light NEE runs at the same vertices as the sun NEE: every vertex that took the diffuse lobe and
has a next segment (the primary vertex when `bounce_count >= 1`, every later diffuse vertex while
`bounce_index <= bounce_count`). Its random stream is `InitRandom(pixel, frame_seed ^ 0x7F4A7C15)`,
separate from the path's, so a frame without the flag stays bit-identical. The result is the
demodulated incident radiance of the Lambert lobe times the path throughput. In ReSTIR GI mode the
primary vertex's NEE goes unpacked into the diffuse target and the shade pass adds the reservoir's
contribution to it.

### 5.1 Lookups and jitter

The vertex looks up its cell (it inserts a missing cell; a new cell has empty reservoirs until the
next fill). With `CAELIX_GI_FLAG_LIGHT_CELL_JITTER` the sampling cell is looked up at the cell point
moved by a uniform offset in (-0.5, 0.5) cells per axis (three random numbers), so the pixels of one
brick read up to 8 cells. The cells are refilled every frame from a few draws each, so without the
jitter all pixels of a cell share one random cell content: a per-cell flicker that a spatial filter
cannot average. The jitter turns it into per-pixel noise. It is unbiased, because a reservoir's W
does not depend on the receiver. With MIS the vertex also looks up its unjittered cell for the proxy
context (section 6.1); without the jitter one lookup serves both.

### 5.2 One sample (light_nee_candidates 1)

`LightPickBrick`: a uniform reservoir `min(R - 1, u * R)` of the sampling cell when it holds a valid
one (`W_brick` = its W), else the global CDF (`W_brick = total / total_b`). The uniform choice among
the R reservoirs needs no factor: each is an unbiased estimator of the same integral. Then a point in
the brick (section 5.4) and the shadow ray (section 5.5). Contribution:

```
L = L_e * (cos_r / pi) * (cos_l / d^2) * V / pdf_area * W_brick      (times w_nee with MIS)
```

### 5.3 Per-vertex resampling (light_nee_candidates K >= 2)

`LightNextEventResampled` draws K candidates exactly as the one-sample NEE draws its sample (a
brick from the sampling cell or the CDF, then a point), with the contribution weight
`W_i = W_brick / pdf_area`, and keeps one by the target at the vertex

```
p_hat(y) = luminance(L_e) * cos_r * cos_l / d^2          (times w_nee(y) with MIS)
```

which is 0 inside the near field and where `cos_r` or `cos_l` is not positive. Candidate i is kept
with probability `w_i / w_sum`, `w_i = p_hat(y_i) W_i`, and `W_out = w_sum / (K p_hat(y))`; one
shadow ray to y. This is RIS with the equal input weights 1/K over unbiased contribution weights, so
`(y, W_out)` is unbiased for every integrand whose support lies in the target's support; the NEE's
integrand is 0 wherever the target is 0 (the visibility is in the integrand only). The K candidates
read the same R reservoirs and are correlated; that does not matter, because the estimator is
linear in the inputs' contribution weights. The point: the cell's reservoirs are chosen by the cell
centre's target, so a lamp close to the vertex but weak from the cell centre is drawn rarely; the
second stage at the vertex prefers it. With K = 1 the path of section 5.2 runs (the same random
stream).

### 5.4 Face and voxel inside a brick (`SampleLightEntry`)

Four random numbers in every branch (face, voxel, two for the point):

1. Candidate faces seen from the receiver: per axis a, `t_a = dot(axis_a, x_r - brick origin)` in
   voxel units; `+a` is a candidate when `t_a > min_row(+a) + 1` (the lowest `+a` face plane of the
   face's mask), `-a` when `t_a < max_row(-a)` (the highest `-a` plane); an empty mask never is
   (`LightCandidateFaceMask`). Every face that a point of the brick can show to the receiver is a
   candidate. Between the two planes of an axis both faces are candidates, and a voxel whose chosen
   face points away gives `cos_l <= 0`: a wasted sample, not a bias.
2. A face f among the candidates with probability `face_weight[f] / W_cand`; `W_cand = 0` means
   that the NEE cannot reach the brick from there (no sample).
3. A voxel uniformly among the `n_f` set bits of `face_mask[f]`.
4. A point uniform on that face (area 1): `pdf_area = face_weight[f] / W_cand / n_f`. The brick's
   own probability is not in it (the caller's `W_brick`).

A face covered inside the brick is never sampled. The radiance is the voxel's emission times
`emission_scale`.

### 5.5 Shadow ray

From the pushed-off receiver toward the point with `TMax = d - 2 K_RAY_ORIGIN_PUSH_OFF` (the point
lies on a voxel face). Shadow rays ignore the fog.

## 6. MIS with the BSDF path

With `CAELIX_GI_FLAG_LIGHT_MIS` the NEE contribution is weighted by `w_nee(y) = p_sa / (p_sa +
p_bsdf)` and a list-light hit of the segment that leaves the vertex by `w_bsdf = p_bsdf / (p_sa +
p_bsdf)` (balance heuristic, `LightNeeMisWeight`). The two weights sum to 1 at every light point,
so the combination is unbiased for any densities, provided that (a) both sides evaluate the same
functions of the vertex and the point and (b) `w_nee = 0` wherever the NEE cannot produce the point.
Neither density has to be the true density of its strategy; a proxy changes only the variance.

### 6.1 The proxy

`p_bsdf = cos_r / pi`, the exact density of the cosine-weighted diffuse bounce. The true density of
the NEE cannot be evaluated (random reservoir contents, the jitter, the resampling). The proxy
`p_sa = p_area * d^2 / cos_l` mirrors the fill's target instead (`LightNeeProxyPdfArea`):

```
in_brick = face_weight[f] / (n_f * W_cand(x_r))                 (0 when W_cand = 0)
p_area   = G_c(b) * total_b * in_brick / C_prev      with a cell (C_prev > 0)
         = total_b * in_brick / total                without one (the global CDF)
G_c(b)   = orient * orient_r / max(|centre_b - cell centre|^2, near_d2)   (as in section 4.1)
```

The per-vertex part is a `LightNeeProxyContext` (`LightNeeProxyOfSlot`): whether a cell with
`C_prev > 0` was found, its world centre, the world normal of its face (oriented cells),
`near_d2`, `C_prev` and the table total. It comes from the vertex's UNJITTERED cell and the lookup's
own coordinate and face, so it is a deterministic function of the vertex. The per-light part is a
`LightMisDesc` (brick frame, face extents, face weights, the point's face, `1 / n_f`, brick total,
centre and mean normal), built by `LightMisDescWith` on both sides. The vertex keeps its context,
`x_r` and `n_r` for the segment that leaves it.

### 6.2 Balance heuristic

The NEE side multiplies its contribution (and, with resampling, its target) by `w_nee`. When the
segment hits a list light (`emission_luminance >= min_emission`) beyond the near field and the NEE
can sample the hit point (section 6.3), the hit counts its emission times `w_bsdf` of that point.
Degenerate cases keep the sum 1: `cos_l <= 0` or a non-finite `p_sa` gives NEE 1, BSDF 0; both
densities 0 give NEE 0, BSDF 1. The MIS draws no random number. Without the flag a hit that the NEE
can sample counts 0 (the exclusive rule). The primary vertex in ReSTIR GI mode keeps the exclusive
rule with the flag too: its BSDF sample is reused by other pixels, so a pixel-dependent weight would
bias the reuse.

### 6.3 Bias rules that still apply

- `LightListCanSample`: the hit brick has an entry in the brick -> entry map (section 2.7) and the
  hit voxel's bit is set in the mask of the HIT face. Otherwise the BSDF path keeps the full
  emission: a brick dropped beyond `max_bricks`, a voxel whose material is no longer a list light,
  a face that the entry counts as covered. No NEE samples those points, so they must stay with the
  BSDF path (before this rule a dropped brick's light was lost).
- The near field (section 2.6) is the BSDF path's on both sides, with and without MIS.
- Only a segment that leaves a vertex that did light NEE skips or weights list-light emission. A
  segment after a specular or transparent vertex counts every emission.
- A cell that does not hold the brick in this frame still gives `w_nee > 0`: that is correct,
  because every reservoir's W is an unbiased weight of every brick.
- Material changes do not rewrite light words: a brick whose word is 0 stays out of the table until
  it is finalized again, and the map then has no entry for it (its emission stays with the BSDF
  path).

## 7. Options and defaults

The library's zero values give the defaults of the left column; the library is opt-in (no light NEE
without `caelix_renderer_set_gi_options` and `CAELIX_GI_FLAG_LIGHT_NEE`). The Rust host turns on every
kept option by default (owner decision, 2026-10-22).

| field / flag | library (0 =) | host default | host flag |
| --- | --- | --- | --- |
| `CAELIX_GI_FLAG_LIGHT_NEE` (bit 5) | off | on | `--gi-no-light-nee` |
| `CAELIX_GI_FLAG_LIGHT_CELL_JITTER` (bit 6) | off | on | `--light-no-cell-jitter` |
| `CAELIX_GI_FLAG_LIGHT_MIS` (bit 7) | off | on | `--light-no-mis` |
| `CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL` (bit 8) | off | on | `--light-no-cell-temporal` |
| `CAELIX_GI_FLAG_LIGHT_CELL_ORIENTED` (bit 9) | off | on | `--light-no-cell-oriented` |
| `light_cell_capacity` | 65536 (1024 .. 2^20, power of two) | 262144 | `--light-cells` |
| `light_cell_reservoirs` | 8 (1 .. 256, power of two) | 64 | `--light-cell-reservoirs` |
| `light_candidates` | 8 (1 .. 64) | 16 | `--light-candidates` |
| `light_level_footprint` | 0.25 | 0.25 | `--light-footprint` |
| `light_temporal_frames` | 20 (1 .. 1024) | 128 | `--light-temporal-frames` |
| `light_nee_candidates` | 1 (1 .. 32) | 4 | `--light-nee-candidates` |
| `CaelixLightOptions::min_emission` | 20.0 | 0 (library) | `--light-min-emission` |
| `CaelixLightOptions::max_bricks` | 65536 | 65536 | `--light-max-bricks` |
| `CaelixLightOptions::nee_min_distance` | 8.0 | 0 (library) | `--light-nee-distance` |

The host's light NEE applies to both GI methods (`--gi pt` and `--gi restir`). At the host defaults
a cell is 576 B and the table 151 MB. Smoke run on bistro (1920 x 1080, dark sky, `--mode pt`, one
frame's GPU timestamps): fill 3.3 ms, trace 22 ms, 53349 live and 28042 active cells, 1 failed
lookup. The defaults as a set are not benchmarked yet (`tools/gi_bench.py` runs `pt_nee*`).

## 8. Measured results

Summary of the bench and test results that led to the kept path (RTX 5080, 1920 x 1080, folders
under `F:/RP_Games/Caelix-family/Artifacts/gi-experiments-20261020/`; the old docs in `docs/` there
have the full tables):

- `light-nee/` (light list + cells + NEE): converges to the path tracer reference in all four bench
  scenes after the near-field split (mean ratio 1.000; bistro 0.876 -> 0.999).
- `light-mis/` (MIS): unbiased in all four scenes, removes the speckles next to lamps at convergence
  (bistro relative RMSE 0.91 -> 0.74), no change at 1 spp, +0.3 .. 0.5 ms.
- `light-faces/`, `light-faces-extent/` (per-face masks and face extents): unbiased; bistro lit
  pixels at 1 spp 7.5 -> 13.4 %; the extents recovered the Cornell loss of the first face rule
  (16.12 dB against 16.03 dB before the masks).
- `cell-jitter/`: unbiased on bistro_dark; halves the frame-to-frame correlation of block means.
- `cell-temporal/`: unbiased at the mean (ratio 0.9994 .. 1.0009); 1 spp block PSNR +0.5 dB on
  bistro, +0.2 dB on shroom; fill +0.1 ms; the accumulated image converges slower per frame on
  bistro (block PSNR -1.3 dB at 512 frames, -0.5 dB at 2048: frame-to-frame correlation).
- `cell-push/` (also the usage counters): the 65536-cell default saturates at 1080p (section 3.6);
  the bench uses 262144 cells since then.
- `nee-resample/` (per-vertex resampling): unbiased; K 4 = +0.3 .. 0.5 dB at 1 spp for +20 % trace
  time, K 16 = +0.4 .. 0.8 dB for +80 %.
- `oriented-cells/`: unbiased; +0.25 dB at 1 spp, +0.1 .. 0.5 dB at 512 frames, fill x 2, live
  cells x 3.5. 8 reservoirs instead of 32 cost 0.07 .. 0.2 dB at 512 frames.
- `firefly-split/` (rootfall): the remaining salt comes from rare large samples of far small lamps
  whose share in a cell is tiny (bounce hits under MIS and the NEE); resampling, oriented cells and
  the variants tested change it by less than 10 %.
- The 32 / 128-voxel cells and the tunable reservoir count (up to 256) are not measured on the bench
  scenes yet. `tests/trace_test.cpp` checks every light NEE mode against a 4096-frame path tracer
  reference in a closed box (mean ratio within 3 %, block PSNR above 30 dB), the last case with every
  option at once and 256 reservoirs per cell.

## 9. Removed paths

Removed on 2026-10-22 (branch `light-nee-experiments`); the code is at tag `light-nee-cleanup-base`
(and `archive/gi-experiments-facebit`, `archive/gi-experiments` for the earlier ones).

- Per-pixel ReSTIR DI at the primary surface (`CAELIX_GI_FLAG_RESTIR_DI`, 4 reservoir buffers,
  temporal and spatial reuse, one shadow ray per pixel). Verdict: unbiased with the Z weights
  (1/M lost 1 .. 4 %), but only +0.2 .. 0.5 dB at 1 spp over the one-sample NEE before the face
  masks and +1.6 dB block PSNR after them, for four more passes and 4 x 48 B per pixel; reuse across
  pixels adds correlation that the denoiser must remove. Archived: `RESTIR_DI.md`, `LIGHT_MIS.md`
  section 5.
- Push-back of shadow ray results into the cells (`CAELIX_GI_FLAG_LIGHT_CELL_PUSH`, voxel-face
  reservoir words, `light_push_confidence`). Verdict: unbiased after four bias fixes, but a 1 spp
  gain only on bistro (+0.17 dB), losses on shroom and Cornell, about 3 x the low-frequency variance
  at convergence, and 63 .. 99 % of the pushes dropped. Archived: `LIGHT_LIST.md` section 2.2.
- Vertex form of the MIS proxy (`CAELIX_GI_FLAG_LIGHT_PROXY_VERTEX`). Verdict: unbiased, no
  measurable change (`nee-resample/`). Archived: `LIGHT_MIS.md` "Vertex proxy".
- Ball and cube light sampling (branch `gi-experiments`, removed before this branch). Verdict:
  unbiased, noisier than the per-face masks at 1 spp in bistro and shroom. Archived:
  `LIGHT_BALLS.md`.
