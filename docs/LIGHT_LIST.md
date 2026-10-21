# Emitter light list, light cells and NEE on emissive voxels

Branch `gi-experiments`, 2026-10-21. Design by Fable, approved by the owner. The sections "As built"
at the end are written by the implementer and are the truth once they exist.

## Why

`docs/GI_EXPERIMENTS.md` phase 1 found emitters by BSDF paths and kept them by reuse. The result:
correct images after a long warm-up, blotches after every disocclusion, and fireflies on concave
edges (`Artifacts/gi-experiments-20261020/fireflies/`). The owner decided to build a light list
after all. The finalize kernel already visits every dirty brick, and the record already holds per
voxel "alive" (at least one face not covered by an opaque neighbour), so the list costs little.

Owner decisions (2026-10-21):

- Block-ID worlds are the standard; packed colours are a test bed. The rule "is a list light" is a
  luminance threshold on the material's emission, the same for both encodings.
- Only bright emitters go into the list (`min_emission`, default 2.0 in linear luminance: packed
  strength 2 (6.4 x albedo) and up; strength 0 (0.1 x) and 1 (0.8 x) keep today's behaviour,
  emission counted when a path hits them). Shroom forest: 2.9 M emissive voxels in 206 k bricks,
  70 k of them at strength >= 2.
- No paging of the light pool: a fixed cap, dropped bricks are counted, no exception.
- The cells are the world-space reuse for secondary vertices. No ReSTIR PT. Step 1 = list + cells
  + NEE at every diffuse vertex of the path tracer; step 2 = per-pixel ReSTIR DI at the primary
  surface. The ReSTIR GI kernels of phase 1 stay optional on top.
- Emission of list lights is NOT counted when a BSDF path hits them (NEE only). This is unbiased
  because every alive emissive face is reachable by NEE. Large, bright, close emitters get more
  variance until MIS is added (later).
- Visible emitters (the primary surface itself) keep their emission in the deterministic target.

## 1. Data

### 1.1 Per-brick light word (scene, `light_words_`)

A `GrowableBuffer` of `uint32_t`, one per AABB row (index = `aabb_row_base + id`, the same index
space as the AABB rows; it is reserved to `aabbs_` capacity rows x 4 B whenever `aabbs_` grows).
Value = `asuint(float total_weight)` of the brick's list lights, 0 when the brick holds none. The
finalize kernel writes it for every brick it finalizes; the CPU records a 4 B zero fill for every
freed id where it records the NaN AABB row (`nan_ids`), and a 4 B-scaled copy for every AABB range
move (`aabb_moves_`) and growth copy. Nothing else writes it.

`total_weight` = sum over alive voxels v of the brick with `IsListLight(material(v))` of
`Luminance(material(v).emission)`. `Luminance(c) = dot(c, (0.2126, 0.7152, 0.0722))`.
`IsListLight(m) = Luminance(m.emission) >= min_emission` (the material's emission BEFORE
`emission_scale`; the trace kernel applies the same rule with the same threshold from the frame
constants). "Alive" is the finalize kernel's `alive` (opaque_count < 6) AND `result != 0`, i.e.
the voxel's micro occupancy bit.

The finalize kernel gets the material table and the threshold: `FinalizePush` grows by
`uint64_t materials` and `float min_emission` (+ pad; 88 -> 104 B, still <= 128). It includes
`common/caelix_material.hlsl` and sets `g_materialTable` / `g_blockEncoding` (the
`CaelixMaterialCode` of an opaque voxel is the raw value; the kernel's `v` after re-encoding is
the material code). Thread t accumulates the luminance of its row's list lights in a float,
reduced with `InterlockedAdd` on a groupshared uint of fixed point (1/1024 units) or a two-step
groupshared float reduction; thread 0 stores the word. A brick without list lights stores 0.

### 1.2 Light table (scene, `light_table_`)

**Step 4 replaced the entry layout** (`docs/LIGHT_FACES.md`, 2026-10-21): the entry is 448 B with
six per-face voxel masks and six face weights instead of the one voxel mask, the slice weights and
`mean_normal` below, and the sampler of section 2.1 picks a face direction by weight before the
voxel. The text below is the step 1 design and stays as history; the header, the CDF, the kernels'
dispatch and the entry map are unchanged.

One device-local buffer, rebuilt by two kernels inside `Scene::Flush` after the finalize dispatch
whenever `lights_dirty_` (set by every flush that dispatched the finalize kernel or recorded a
zero fill / range move of light words; cleared after the rebuild). Layout, `max_bricks` = N
(`CaelixLightOptions::max_bricks`, 0 = 65536, 1 .. 2^20):

```
header    64 B   uint count; uint dropped; uint pad[2]; float total_weight; float pad[11]
entries   N x 128 B   LightEntry
cdf       N x 4 B     float, inclusive prefix sum of entry.total_weight (cdf[count-1] = total)
```

`LightEntry` (128 B, `shaders/common/caelix_light.hlsl` + `lib/src/scene/gpu_structs.h`):

```
uint64_t mask[8];        //   0  bit v of word (v >> 6): voxel v = x | y << 3 | z << 6 is alive and a list light
float region_weight[8];  //  64  luminance sum of the set bits of word w
float total_weight;      //  96
uint group;              // 100  scene-wide group index
uint id;                 // 104  dense brick id
uint entity;             // 108
int3 origin;             // 112  entity-local voxel origin of the brick (brick key * 8)
uint mean_normal;        // 124  octahedral 2 x 16 bit snorm of the mean open-face direction (0 = none)
```

Voxel index convention: `v = x | (y << 3) | (z << 6)` (x fastest), so word w = z-slice w. This is
NOT the micro occupancy order of the record; the build kernel converts.

`mean_normal`: sum over the brick's list-light voxels of the unit directions of their open faces
(a face is open when the neighbour voxel inside the brick is not occupied in the micro occupancy;
a face on the brick border counts as open), normalised; 0 when the sum is 0. It is an importance
hint only (section 2.2).

Kernels (scene-owned `ComputePipeline`s next to `finalize_`, push constants `LightPush`, see
`gpu_structs.h`; the shader files are `shaders/scene/caelix_light_compact.hlsl` and
`shaders/scene/caelix_light_scan.hlsl`, added to the kernel list of `lib/CMakeLists.txt` and
`render_kernels.cpp`'s LoadShader is NOT used: the scene loads them as it loads the finalize
kernel):

1. `caelix_light_compact`: zero the header (thread 0 of workgroup 0 cannot order against the
   others, so the CPU records a `vkCmdFillBuffer` of the header's first 16 B to 0 before the
   dispatch instead). One workgroup of 64 threads per alive group index in [0, group_count)
   (the CPU passes `group_count` = `groups_.size()`; a freed group has `high_water` rows that are
   NaN / 0 so it contributes nothing, but the kernel also skips a group whose `GroupInfo.entity`
   is `0xFFFFFFFF` if the CPU marks freed groups so; otherwise read the light words, they are 0).
   Thread t walks ids t, t + 64, ... < `high_water`: light word != 0 -> `index =
   InterlockedAdd(header.count, 1)`; index >= N -> `InterlockedAdd(header.dropped, 1)` and skip;
   else build the entry from the record: slot row -> record address -> the 16 micro occupancy
   words and the 256 payload words; for each voxel: occupied bit && IsListLight(material(code))
   -> set the mask bit, add the luminance to its region weight and the total, add the open-face
   directions to the normal sum. Store the entry (and the entry's total into cdf[index] for the
   scan). Dispatch count can exceed 65535 groups: split like the finalize dispatch.
2. `caelix_light_scan`: one workgroup of 1024 threads, an inclusive prefix sum over
   cdf[0 .. count) in place (loop over chunks of 1024 with a carried offset), writes
   `header.total_weight`. count = 0 -> total 0.

Barriers: finalize writes -> compact reads (compute -> compute), compact -> scan, scan -> the
"everything visible" barrier that already ends the flush. The flush timer brackets them.
`CaelixLightStats` (section 4) reports the count, the dropped count, the bytes and the last
rebuild's presence in the flush time (no separate timer).

### 1.3 Light cells (renderer, ReGIR)

A hash table of cells owned by the renderer (like the GI reservoir buffers: created at the first
render that needs it, recreated when the capacity option changes, retired safely). Capacity C =
`CaelixGiOptions::light_cell_capacity` (0 = 65536, power of two, 2^10 .. 2^20). R = 32 reservoirs
per cell (`kLightCellReservoirs`, a shader constant).

```
Cell (304 B since the temporal cell reuse, section 2.2; 296 B since step 3, docs/LIGHT_MIS.md;
288 B before, without C_prev and C_cur)
  uint64_t key;          //   0  0 = empty; else the fingerprint of (entity, level, cell coordinate)
  uint touched_frame;    //   8  fc.frame_counter (section 3) of the last lookup
  uint entity;           //  12
  int3 coord;            //  16  cell coordinate at `level`
  uint level;            //  28
  float C_prev;          //  32  cell normalisation of the fill before the latest one (MIS proxy)
  float C_cur;           //  36  cell normalisation of the latest fill (mean of w_sum / K; with
                         //      LIGHT_CELL_TEMPORAL the running mean of dedicated draws)
  float M;               //  40  LIGHT_CELL_TEMPORAL: the candidate count behind every reservoir of
                         //      the cell (uniform over the 32: all threads merge the same way);
                         //      0 = no history
  float C_n;             //  44  LIGHT_CELL_TEMPORAL: the draw count behind C_cur; 0 = none
  LightCellReservoir r[32];  // 48 .. 304
LightCellReservoir (8 B)
  uint light;            // entry index in the light table, 0xFFFFFFFF = empty
  float W;               // unbiased contribution weight of the brick-level target (0 = empty)
```
An unfilled cell has empty reservoirs (all `W` 0); the sampling then falls back to the global
table (section 2.3).

Cell coordinate of an entity-local point p (voxel units) at level L: `floor(p) >> (3 + 2 L)`
(level 0 = one brick = 8 voxels, level 1 = 32 voxels). Level by footprint: with `d` = the world
distance from the camera position to the vertex, `footprint = 2 d tan_half_fov_y / size.y`
(world units per pixel); level 0 when `footprint < light_level_footprint` (option, 0 = 0.25),
else level 1. Every vertex of a path uses its own camera distance.

Key fingerprint: `Hash64(entity, level, cx, cy, cz)` = a 64-bit mix (e.g. two rounds of
splitmix64 over the packed 32-bit words), forced non-zero. Slot = `key & (C - 1)`, linear
probing up to 8 slots. A fingerprint collision merges two cells: harmless for correctness
(every reservoir carries its own W), only the importance suffers.

Lookup from the trace kernel (`LightCellLookup`): probe; a matching key -> write
`touched_frame = fc.frame_counter` (plain store) and return the cell; an empty slot ->
`InterlockedCompareExchange(key, 0, fingerprint)`; success or another thread's success with the
same key -> that cell (touched, reservoirs empty until the next fill); 8 probes without a
match -> no cell (the vertex samples the global table, section 2.3). The inserting thread also
writes entity, coord and level and clears the 32 reservoirs (plain stores after the CAS; a
concurrent reader of the same new cell may see partly cleared reservoirs and reads either an
empty one or a stale `W` 0 slot, both harmless).

Eviction: the fill pass clears a cell (key = 0, reservoirs empty) when `touched_frame + 64 <
fc.frame_counter`. Hash-table removal under linear probing leaves tombstone-free holes, which
is fine here: a later lookup of a displaced key simply inserts again in an earlier slot and the
displaced old cell ages out.

## 2. Sampling

### 2.1 Within a brick (`SampleLightEntry`)

Given an entry and three random numbers:

1. Region: pick word w with probability `region_weight[w] / total_weight` (a running sum over 8).
2. Voxel: `n = countbits(mask[w])` (two 32-bit `countbits`), `k = min(n - 1, floor(u * n))`, the
   k-th set bit by popcount halving (32, 16, 8, 4, 2, 1). pdf within the region `1 / n`.
3. Face: the voxel centre `c = origin + (x, y, z) + 0.5` in entity-local units, world `cw =
   EntityCurWorld(group, c)`; the receiver `x_r` (world). Candidate faces = the axis faces whose
   outward normal has `dot(n_f, x_r - cw) > 0` (1 .. 3 of them; the entity rotation is applied
   to the normals through the EntityInfo rows, rotation part only). Pick one with equal
   probability `1 / n_faces`. Whether the chosen face is covered by a neighbour voxel is NOT
   tested (the entry has no per-face data). A covered face gives an occluded shadow ray: a wasted
   sample, not a bias.
4. Point: uniform on the unit face: `y = cw + n_f 0.5 + t1 e1 + t2 e2`, t uniform in (-0.5,
   0.5) (rotated with the entity). Area pdf 1.

Result: `LightSample { float3 position; float3 normal; float3 radiance (material.emission *
emission_scale of the voxel's code, read from the record payload word: 2 loads); float pdf_area =
(region_weight[w] / total) * (1 / n) * (1 / n_faces) }`. The brick-level probability is NOT in
`pdf_area`; the caller multiplies by the brick's 1/p (the cell reservoir's W, or the CDF
probability of the global table).

Contribution at a diffuse receiver with normal n_r and demodulated BRDF 1/pi:
`L = radiance * (max(0, dot(n_r, w)) / pi) * (max(0, dot(n_f, -w)) / d^2) * V / pdf_area * W_brick`,
`w = normalize(y - x_r)`, `d = |y - x_r|`, V from a shadow ray from the pushed-off receiver
toward y with `TMax = d - 2 K_RAY_ORIGIN_PUSH_OFF` (the sample lies on a voxel face, as in
`GiVisible`). The albedo of the receiver is applied by the caller like the sun NEE (demodulated
at the primary vertex, in the throughput after it).

### 2.2 Cell fill (`caelix_light_cells`, renderer, trace layout + TracePush, before the trace)

One workgroup of 32 threads per cell slot (C workgroups, split at 65535 like the others). Thread
r owns reservoir r. Skip the cell when key == 0; evict when aged (section 1.3). Skip the fill
(keep the reservoirs) when `light_table.count == 0` (write empty reservoirs instead). Otherwise
each thread draws K = `light_candidates` (0 = 8, 1 .. 64) candidates from the global CDF:
`u * total_weight` -> binary search over cdf[0 .. count) -> entry i, proposal pdf
`p_i = entry.total_weight / total_weight`; target `p_hat_i = entry.total_weight * G_i`, where

```
centre   = EntityCurWorld(entry.group, entry.origin + 4)             // brick centre, world
cell_c   = EntityCurWorld(cell entity, (cell coordinate + 0.5) * cell size)  // the cell's centre
d2       = max(dot(centre - cell_c, centre - cell_c), (cell half size + 4)^2)   // clamp near
orient   = mean_normal == 0 ? 1 : max(0.1, dot(decode(mean_normal rotated), normalize(cell_c - centre)))
G_i      = orient / d2
```

The cell's entity, coordinate and level come from the cell header. RNG per (cell slot,
reservoir, frame): `InitRandom(uint2(slot, r), fc.frame_seed ^ 0x3C6EF372u)`.

Streaming RIS over the K candidates with weight `p_hat_i / p_i`; the result reservoir
`light = selected i, W = (w_sum / K) / p_hat_selected` (0 when w_sum = 0). Unbiased for the
brick-level target. No temporal reuse of cell reservoirs in step 1.

Temporal reuse (`CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL`, 2026-10-21). Test feature: it assumes a
static scene (no scene edits, no entity motion), because the merge evaluates the old sample with
this frame's target. After the K fresh candidates (same draws in the same order as without the
flag) each thread merges the reservoir it wrote in the previous fill as one more candidate:

```
m_cap  = light_temporal_frames * K          // FrameConstants::light_temporal_frames, 0 = drop
M_old  = min(cell.M, m_cap)
old    = this thread's reservoir (light, W_old)
u_old  = RandomFloat01(rng)                 // always drawn with the flag: the stream does not
                                            // depend on the data
if M_old > 0 and old.light valid (< count) and W_old > 0 and p_hat_old = total_old * G_old > 0:
    w_old = p_hat_old * W_old * M_old;  w_sum += w_old;  select old with probability w_old / w_sum
M_new  = M_old + K
W      = (w_sum / M_new) / p_hat_selected
```

Thread 0 stores `M = M_new` in the header (uniform for the cell: every thread merges the same
way). `CaelixGiOptions::light_temporal_frames` (0 = 20, 1 .. 1024; host `--light-temporal-frames`)
sets the cap. The renderer passes `light_temporal_frames = 0` (the fill drops the history for one
frame) when the scene rebuilt its light table since the previous fill (`CaelixLightStats::rebuilds`
changed: entry indices may have moved) or after `CaelixGiOptions::reset`. A new cell (insert,
eviction, a fill without a usable table) has M = 0; a fill without the flag stores M = 0, so a
later switch to the flag starts clean.

Unbiased: `E[p_hat(y_old) W_old] = C` (the cell normalisation), so `w_old` has expectation
`M_old C` and `w_sum / M_new` estimates C; the merge weights (`M_old / M_new` for the history,
`1 / M_new` per fresh candidate) sum to 1, every entry with total > 0 is reachable from the fresh
draws, and the target is the same every frame (static scene), so W stays an unbiased contribution
weight.

Cell normalisation with the flag: the previous fill's candidates are part of this frame's
reservoirs, so a C estimate made from the candidates (even one frame delayed through `C_prev`) is
correlated with the reservoirs the trace samples, and the MIS proxy (docs/LIGHT_MIS.md section 3)
must stay independent of them. Each thread therefore makes one dedicated draw from the CDF
(`w_c = G_j * total`, an unbiased estimate of C), and thread 0 folds the mean of the 32 into
`C_cur` as a running mean: `n_old = min(C_n, light_temporal_frames * 32)`,
`C_cur = (C_cur_old * n_old + sum of w_c) / (n_old + 32)`, `C_n = n_old + 32`; `C_prev = C_cur_old`
as before. Cost: one more CDF search, entry load and G per thread (the shroom forest at 960 x 540:
fill 0.6 -> 0.75 ms in one frame's timings, not a careful measurement).

Host: `--light-cell-temporal`, `--light-temporal-frames N` (default 20), UI checkbox "temporal
reuse in light cell fill" and slider "light temporal frames". Bench runs `pt_nee_temporal`,
`pt_nee_temporal_conv`, `di_temporal`. Test: "light NEE with temporal cell reuse is unbiased" (four
single-voxel lights in four ceiling bricks: with one brick the merge cannot change W).

### 2.3 At a path vertex (`SampleLightAtVertex`)

```
cell = LightCellLookup(entity, level, coordinate)      // touches / inserts
if (cell && reservoir r = cell.r[floor(u * 32)] has W > 0)  -> entry = light table[r.light], W_brick = r.W
else                                                    -> CDF sample from the global table (binary
                                                           search), W_brick = total_weight / entry.total_weight
sample = SampleLightEntry(entry, x_r, rng)
contribution as in 2.1, times W_brick
```

The uniform choice of one of the 32 reservoirs needs no factor: each reservoir is an unbiased
estimator of the same integral.

### 2.4 In the path tracer (`TracePath`, flag `CAELIX_GI_FLAG_LIGHT_NEE`)

Exactly where `SunNextEvent` is called (the primary vertex when it took the diffuse lobe and
`bounce_count >= 1`; every later diffuse vertex while `bounce_index <= bounce_count`), add
`LightNextEvent(fc, origin, normal, group, id, object position, rng) * throughput`. The RNG: a
separate stream like `nee_rng` (`InitRandom(pixel, fc.frame_seed ^ 0x7F4A7C15u)`), so the path
without the flag stays bit-identical. The vertex's cell: entity = `GroupEntity(group)`,
coordinate from `object_hit` (entity-local voxel units; the primary vertex has
`surface.object_hit`, a bounce hit `hit.object_hit`).

Hit emission: `radiance += shade.emission * throughput` becomes
`radiance += (light_nee && IsListLight(material)) ? 0 : shade.emission * throughput` at every
bounce hit. The primary surface's emission (deterministic target) is unchanged. `ShadeVoxelHit`
needs no change: `IsListLight` reads `GET_MATERIAL(block).emission` again (or
`CaelixShadeResult` gains `bool list_light`; the implementer chooses, the second avoids a
second table read).

ReSTIR GI mode (`gi_method = RESTIR`) with the flag: the GI path applies the same NEE at x_s and
after (inside `L_o`), and the primary vertex's NEE result is written by the trace kernel into
`g_int_diff.rgb` (unpacked, hit distance 0) in ReSTIR mode; `caelix_restir_gi_shade.hlsl` reads
that value first and ADDS the reservoir's contribution before packing. Without the flag the
trace kernel writes 0 there as today.

Sun NEE and fog are untouched. Specular vertices get no light NEE (as the sun).

Step 3 (docs/LIGHT_MIS.md) replaces this exclusive rule: a list-light hit is skipped only when the
NEE can sample the point (the brick has a table entry and the voxel's mask bit is set), and with
`CAELIX_GI_FLAG_LIGHT_MIS` the NEE and the hit are weighted by the balance heuristic instead.

### 2.5 Jittered cell lookup (`CAELIX_GI_FLAG_LIGHT_CELL_JITTER`, 2026-10-21)

Observed (owner, 1 spp, light NEE alone, a scene with many lights): the image flickers in a
blocky per-brick pattern. Cause: every pixel whose vertex lies in one cell reads the same 32
reservoirs of that frame, and the fill rebuilds them every frame from 32 x K draws of the global
CDF (at most 256 of N light bricks, chosen by emission, not by distance). With N >> 256 the
frames in which the near strong bricks are among the draws give the whole cell a large W, the
other frames leave it dark. The pixels of a cell are fully correlated, neighbouring cells are
independent: a per-cell flicker that no spatial filter can average. The estimate stays unbiased
(the bench converged, section 5); only the variance is concentrated in the cell choice. The RNG
(one stream per pixel and frame in the trace, one per slot, reservoir and frame in the fill) is
not the cause.

Fix 1 (built): with the flag, `LightNextEvent` looks the cell up at `object_point + (u - 0.5) *
cell_size` per axis (`u` three fresh numbers of the light stream), so the pixels of one brick read
up to 8 different cells and the block pattern becomes per-pixel noise. Unbiased: a cell's
reservoir is an unbiased estimator of the brick-level integral for every receiver (its W does
not depend on the receiver). Costs: up to 8 x more live cells near the camera (more fill work, the
capacity stays 65536) and one more frame with empty reservoirs for newly touched cells. The jitter
is per vertex, every bounce, path trace and ReSTIR GI methods; ReSTIR DI's initial candidates
(`DiInitialReservoir`) are NOT jittered (the DI reuse already decorrelates the cells). Without the
flag the random stream and the lookup are unchanged. Host: on by default with `light NEE`,
`--light-no-cell-jitter` / the "jitter light cell lookup" checkbox turn it off.

Built since (variance, not pattern): temporal reuse of the cell reservoirs in the fill (thread r
merges its old reservoir, M-capped, as one more candidate; `CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL`,
section 2.2). Not built: several reservoirs per vertex resampled with the exact point target.

## 3. Frame constants and push constants

`FrameConstants` grows from 608 to 672 B (CPU + HLSL + static_asserts):

```
uint64_t light_table;        // 608  scene light table (section 1.2), 0 = none
uint64_t light_cells;        // 616  renderer cell table, 0 = none
uint32_t light_cell_capacity;// 624  C (power of two)
uint32_t light_candidates;   // 628  K
float light_level_footprint; // 632
float light_min_emission;    // 636  the scene's threshold (IsListLight in the kernels)
uint32_t frame_counter;      // 640  renderer render count (1, 2, ...), for touched_frame
uint32_t light_entity_mask;  // 644  reserved 0
uint32_t pad_648[6];         // 648 .. 671
```

(Later steps filled the padding and grew the struct to 704 B; the temporal cell reuse uses offset
696: `uint32_t light_temporal_frames` = the frames of history the fill keeps, 0 = none this frame;
`pad_700` follows.)

`TracePush` is unchanged. Scene kernels use `LightPush` (section 1.2, `gpu_structs.h`):
`{ uint64_t group_infos, slot_tables, page_table, light_words, light_table, materials,
entity_infos; uint group_count, max_bricks, records_per_page_shift, record_bytes, group_shift,
encoding; float min_emission; uint pad }` (96 B).

## 4. ABI (functions and new structs only; no existing struct changes except the two branch-only
experimental ones noted)

```c
typedef struct CaelixLightOptions
{
    float min_emission;    /* 0 = 2.0: list light when luminance(material.emission) >= this */
    uint32_t max_bricks;   /* 0 = 65536; 1 .. 1048576; the light table's capacity */
    uint32_t reserved[2];
} CaelixLightOptions;
/* Scene-wide. Applies to bricks finalized after the call; the bricks already finalized keep their
   light word until they are finalized again (a change of min_emission does not reclassify them:
   call before the uploads). max_bricks reallocates the table at the next flush. */
CAELIX_API CaelixResult caelix_scene_set_light_options(CaelixScene* scene, const CaelixLightOptions* options);
typedef struct CaelixLightStats
{
    uint32_t bricks;          /* entries in the light table after the last flush that rebuilt it (CPU readback? NO: see below) */
    uint32_t bricks_dropped;
    uint64_t table_bytes;
    uint64_t rebuilds;        /* flushes that rebuilt the table */
} CaelixLightStats;
CAELIX_API void caelix_scene_get_light_stats(const CaelixScene* scene, CaelixLightStats* out);
```

`bricks` / `bricks_dropped` need a readback of the header: the scene copies the header's first
16 B into a host-visible ring slot at the end of the rebuild (one 16 B copy per rebuild; the slot
of the safe frame is read in `GetLightStats`, like the GPU timers resolve). If the ring is more
than ~30 lines of code, report `bricks = 0xFFFFFFFF` ("unknown") and leave a TODO; the test then
checks the image instead.

`CaelixGiOptions` (branch-only, experimental, not mirrored in C#) gains after `reset`:
`uint32_t light_cell_capacity; uint32_t light_candidates; float light_level_footprint;
uint32_t light_reserved;`. `CaelixGiStats` gains `uint64_t light_cell_bytes; float
light_fill_ms;` (the fill dispatch gets its own `GpuTimer`, `light_timer_`). Flag
`CAELIX_GI_FLAG_LIGHT_NEE (1u << 5)`. Update the comments of both structs. The
`caelix_renderer_set_gi_options` validation: capacity 0 or a power of two in [2^10, 2^20],
candidates <= 64, footprint finite and >= 0.

`host-rs`: `caelix-sys` is bindgen (regenerates). `caelix-vk`/`caelix-app`: flags
`--gi-light-nee`, `--light-min-emission <f>`, `--light-max-bricks <n>`, `--light-cells <n>`,
`--light-candidates <n>`, `--light-footprint <f>`; `caelix_scene_set_light_options` is called
once after the scene is created (before the world loads); the GI panel gets a "light NEE"
checkbox, the three cell fields and stats rows (light bricks, dropped, cell MB, fill ms); the
screenshot `timings:` line gains `light_ms=`.

## 5. Render order (renderer_render.cpp)

```
sky SH (if sky) -> [light cells fill, if LIGHT_NEE and light_table != 0] -> barrier -> trace
-> ReSTIR GI passes -> NRD -> composite
```

The fill runs before the trace with a compute -> compute barrier after it (the trace reads the
cells and inserts new ones; the fill of the NEXT frame fills them). The fill reads the light
table that the scene's flush of THIS frame rebuilt (the flush's final barrier covers it).

`frame_counter`: a renderer member incremented per render call, written into the constants.

## 6. Tests (`tests/trace_test.cpp`, Catch2, build-gi)

1. "light table counts one emissive brick": a 2-brick scene (one brick with 3 list-light voxels
   of strength 3 white, one plain brick), flush, `caelix_scene_get_light_stats`: bricks 1 (or the
   unknown marker with the TODO), dropped 0. With `min_emission` above every emission: 0.
2. "light NEE is unbiased": a closed box (6 walls, 24^3 interior, grey) with one strength-3 white
   voxel in the ceiling, camera inside; reference mode: 1024 frames with the plain path tracer
   vs 1024 frames with LIGHT_NEE, 4 bounces, 128x128. Mean linear brightness ratio in 0.97 ..
   1.03 and PSNR > 30 dB between the two. (The plain path tracer finds a 1-voxel light at 1 spp
   rarely: 1024 frames over 16 k pixels is enough for the mean; raise to 4096 if the ratio
   is noisy, and say so in the report.)
3. "path tracer unchanged without the flag": with `caelix_scene_set_light_options` called and
   the table built, one frame without LIGHT_NEE is bit-identical to the same frame on a scene
   without light options (the light words and table must not influence the trace).
4. "cells fill and sample": LIGHT_NEE with `light_cell_capacity` 1024: after 3 frames the GI
   stats report `light_cell_bytes` > 0 and `light_fill_ms` resolves (>= 0) for a safe frame;
   the image is not black.

All 7 existing ctest cases must still pass in `build-gi`.

## 7. Bench

`tools/gi_bench.json`: add run `pt_nee` = `--mode pt --no-denoise --warmup 64 --gi pt
--gi-light-nee`, `pt_nee_conv` = `--mode ref --accumulate 512 --gi pt --gi-light-nee`, and the
scene `shroom` = `F:/RP_Games/Caelix-family/ScenesPlayground/Artifacts/ancient-shroom-forest-20261015/out/ancient-shroom-forest.cxw`
with the file's stored pose (`--pose` omitted; add a bench option `"pose": null` = do not pass
`--pose`) and `--sky-zenith 0,0,0 --sky-horizon 0,0,0 --sky-ground 0,0,0`. The validator runs
cornell, ashwick, bistro_dark, shroom for `reference`, `pt`, `pt_nee`, `pt_nee_conv`.

## As built

### Near field (Fable, 2026-10-21, after the bench)

The bench found the converged light-NEE image 14 % too dark on bistro_dark, uniformly, at the
bounce vertices (`Artifacts/gi-experiments-20261020/light-nee/README.md`). The diagnosis
(`.../light-nee/diagnosis/README.md`, 20 shader variants on a clean copy of 50c8076) found no code
bug: light points closer than about 8 voxels to the receiver make the NEE estimate extremely
heavy-tailed (uniform area sampling with 1/d^2, a brick choice by power only), so 512 frames
deliver only a fraction of their mean, and the step 1 rule had moved exactly those points from the
BSDF path (bounded weight) into NEE. Fix, validated at 2 and 4 bounces (bistro 0.876 -> 0.9988,
Cornell 0.9954 -> 0.9997, converged relative RMSE 1.135 -> 0.870): a distance split.
`CaelixLightOptions::nee_min_distance` (0 = 8 world units; host `--light-nee-distance`) goes to
`CaelixFrameConstants::light_nee_min_distance` (offset 644); `LightNextEvent` returns 0 for a
point closer than that (before the shadow ray), `TracePath` keeps the emission of a list light hit
closer than that (`hit.T`, same origin), and `DiGeometry` (ReSTIR DI target and shade) returns 0
for such points. Both sides measure from the same pushed-off origin, so the split is unbiased.
Consequence: light within 8 voxels of a lamp is found by the BSDF path only (1 spp noise there).
The long-term alternative is MIS between NEE and BSDF hits, which needs a true NEE pdf at a hit
point (the cell RIS weight is not one).

Review change (Fable, after the implementation): the light table is opt-in. `Scene::EnsureLightTable`
builds it only once `caelix_scene_set_light_options` was called (`lights_enabled_`); the Unity plugin
never calls it and pays only the 4 B light words. The host calls it at scene creation; the tests
call `EnableLights` (zero options = defaults).

Step 1 as built on 2026-10-21 (branch `gi-experiments`, on top of 03b021c). Files: `shaders/common/
caelix_light.hlsl` (list-light rule, table layout, entry load, octahedral code),
`shaders/scene/caelix_finalize.hlsl` (light word), `shaders/scene/caelix_light_compact.hlsl` and
`caelix_light_scan.hlsl` (table rebuild), `shaders/render/caelix_light_sample.hlsl` (cells,
CDF search, `SampleLightEntry`, `LightNextEvent`), `shaders/render/caelix_light_cells.hlsl`
(cell fill), `lib/src/scene/scene_lights.cpp` (options, stats, table allocation, rebuild
recording), the `Scene::Flush` and `Renderer::Render` hooks, `tests/trace_test.cpp` (four cases),
host flags and UI, `tools/gi_bench.*`.

### Sizes

| item | size |
| --- | --- |
| light word | 4 B per AABB row (`Scene::light_words_`, grows with the AABB pool; 256 KiB at the initial 65536 rows) |
| light table | 64 B header + 132 B per `max_bricks` entry (128 B entry + 4 B CDF): 8.3 MiB at 65536; since step 4 (`LIGHT_FACES.md`) 452 B per entry (448 B entry + 4 B CDF): 28.3 MiB at 65536, 452 MiB at 2^20 |
| light cells | 288 B per cell: 18 MiB at 65536, 288 KiB at 1024 (296 B since step 3, docs/LIGHT_MIS.md; 304 B since the temporal cell reuse: 19 MiB at 65536) |
| `FinalizePush` | 88 -> 104 B (`light_words`, `materials`, `min_emission` in place of `pad`) |
| `LightPush` | 72 B (see differences); 80 B since step 3 (`light_indices`, docs/LIGHT_MIS.md) |
| `FrameConstants` | 608 -> 672 B, fields as in section 3 (`frame_counter` at 640) |
| `CaelixGiOptions` | 32 -> 48 B (52 B since the temporal cell reuse: `light_temporal_frames`); `CaelixGiStats` 16 -> 32 B |
| `CaelixLightOptions` | 16 B; `CaelixLightStats` 24 B |

Header word 2 (the design's `pad[0]`) is now `capacity` (= max_bricks, written by the scan): the
render kernels need it to find the CDF behind the entries.

### Kernels and passes

- `caelix_finalize`: per thread the luminance sum of the row's occupied list-light voxels (one
  material read per distinct code), 64 floats in groupshared memory, thread 0 sums them and
  stores `asuint(sum)` (0 when none) at `light_words + (aabb_row_base + id) * 4`.
- `caelix_light_compact` (scene, 64 threads per group index, split at 65535 groups) and
  `caelix_light_scan` (scene, one workgroup of 1024, Hillis-Steele in groupshared memory per
  chunk of 1024 with a carried sum; no subgroup operations). Recorded by
  `Scene::RecordLightRebuild` after the finalize barrier: fill header[0..16) = 0, barrier,
  compact, barrier, scan, barrier, copy header[0..16) to a host readback ring (8 slots), barrier
  to HOST, then a barrier to ALL_COMMANDS reads. Inside the flush timer.
- `caelix_light_cells` (renderer, trace set + `TracePush`, 32 threads per cell, dispatch
  1024 x capacity / 1024). Before the sky SH kernel and the trace, with its own `GpuTimer`
  (`light_timer_` -> `light_fill_ms`) and a compute -> compute barrier after it. A new cell
  buffer is zero-filled (`vkCmdFillBuffer`) before its first fill.
- Light NEE in `TracePath` at the sun NEE's vertices, own stream
  `InitRandom(pixel, frame_seed ^ 0x7F4A7C15)`. ReSTIR mode: the primary NEE goes unpacked into
  `g_int_diff`, `caelix_restir_gi_shade` adds the reservoir's contribution and packs the sum.

### Differences from the design, and why

1. **`min_emission` default is 20, not 2.0.** The library decodes a packed emitter as
   `albedo * 8^s` (`material_table.cpp`, `exp2(3 s)`): there is no `* 0.1` (the 0.1 in this
   document, in `GI_EXPERIMENTS.md` and in the ScenesPlayground notes is not in the native
   renderer's table). With 2.0, every strength-1 voxel with an albedo luminance >= 0.25 would be
   a list light, against the decision "strength 0 and 1 keep today's behaviour". 20 = the
   decision's 2.0 on the library's scale (strength 2 from albedo luminance 0.3125, every bright
   strength 3). Measured on the shroom forest: 2.0 -> 19769 light bricks, 8 -> 2710, 20 -> 2706
   (none dropped at 65536). `--light-min-emission` / `CaelixLightOptions::min_emission` change it.
2. **Atomics go through descriptors.** DXC 1.8 of the Vulkan SDK 1.4.304.1 has no
   `vk::BufferPointer`, and `vk::RawBufferLoad/Store` have no atomic form. The light table is
   set 0 binding 0 of the two scene kernels (a `RWByteAddressBuffer`; the scene owns the set
   layout and one descriptor pool + set per table allocation, retired with the buffer), and the
   light cells are trace set binding 21 (storage buffer, PARTIALLY_BOUND, written when the flag
   is on). The render kernels read the table through `fc.light_table` (an address);
   `fc.light_cells` is the cells' address but serves only as "present" (all cell accesses use the
   binding). The 64-bit cell key is claimed with a 32-bit compare-and-swap on its low word (64-bit
   buffer atomics are not a device requirement; the low word is forced non-zero); the high word
   is stored after the claim. A lookup that sees the low word match before the high word is
   written probes on and may insert the same key twice (harmless, as for fingerprint collisions).
3. **Released groups.** The design assumed a freed group's rows read as NaN / 0. They do not:
   the freed AABB and light-word ranges go back to the row pool and belong to other groups later,
   while the GPU GroupInfo row of the freed index still points at them, so the compact kernel
   would create entries with the wrong group. `ReleaseGroup` now marks the group's info dirty and
   the flush restages its `high_water` as 0 (a 4-byte copy from a zero word in the ring;
   `Scene::StageReleasedGroup`). Only `high_water` changes, so a stale TLAS instance of the freed
   group (before the TLAS rebuild) reads the same slot rows as before.
4. **Light words of new and moved ranges.** A new or grown AABB range gets its light words
   zero-filled (`[kept, rows)`, next to the NaN fill), so an id that waits for finalization
   reads 0, not old memory. Range moves copy the light words with the AABB rows
   (`light_moves_`). Freed ids get their zero word as a 4-byte copy from the ring (a zero word
   next to the NaN row, `EnsureNanRow`), not a `vkCmdFillBuffer`: copies run after the range
   moves (as the NaN rows), fills before them, and a fill could race a move into the same row.
5. **Compact computes before it allocates.** An entry whose recomputed total is 0 (its
   materials changed since the light word was written, or `min_emission` was raised) is not
   stored. The scan clamps `count` to the capacity (the compact kernel counts dropped bricks in it
   too) and writes `capacity` and `total_weight`.
6. **Hit emission rule.** A list light's emission is skipped only on a segment that leaves a
   vertex that did light NEE (`segment_light_nee`, like the sun's `segment_sees_sun`), not at
   every bounce hit: a segment after a specular or transparent vertex (no NEE there) still counts
   it. The design's rule would lose that light (bias with smooth or glass materials). The
   primary surface's emission stays deterministic. `CaelixShadeResult` gained
   `emission_luminance` (the second option of the design: no second table read).
7. **Cell point.** The cell coordinate uses `object_hit + 0.25 * object face normal` (the air
   side of the face): the hit point lies on the face plane, and `floor` of it would pick the
   solid voxel's cell half of the time.
8. **`LightPush`** is `{group_infos, slot_tables, page_table, light_words, materials,
   group_base, group_count, max_bricks, records_per_page_shift, record_bytes, group_shift,
   encoding, min_emission}` (72 B): no `entity_infos` (the compact kernel needs none),
   `group_base` for the dispatch split, the table is the binding (difference 2).
9. **Cell insertion does not clear the reservoirs**: a slot whose key is 0 has empty reservoirs
   (zero at creation, cleared at eviction by the fill pass).
10. **Table lifetime.** The table is created by the first flush that can rebuild it (dirty
    bricks, dirty groups or a dirty table) and is rebuilt by every flush that finalized bricks,
    freed ids, moved AABB ranges, released a group or changed the options, when the ring was not
    full. `Scene::light_table_address()` is 0 until a rebuild was recorded (and again after a
    reallocation until the next rebuild): no light NEE then. The table is built whether or not a
    renderer uses light NEE (the Unity plugin pays 8.3 MiB and the rebuild work per changing
    flush).
11. **Stats readback** is built (8 host-visible 16-byte slots; `Recycle` reads the newest slot
    whose frame is safe), so `CaelixLightStats::bricks` is never the "unknown" marker.
12. **Tests.** Test 2: the plain path tracer needs 4096 reference frames (light NEE 1024),
    and the PSNR is taken on 16 x 16 block means: per pixel, the path tracer finds a 1-voxel
    light only a few times in 4096 samples, so its own per-pixel noise keeps any per-pixel PSNR
    against it far below 30 dB. The test also
    prints the path tracer's own noise floor. Test 3 compares the ReSTIR GI room (a strength-1
    emitter) on a scene with `min_emission` 1 (its table has 4 entries) and a scene with the
    defaults (no entry). Test 1 also checks the threshold set before the upload and a capacity of
    one (one entry, one dropped). The existing validation case now uses `1u << 6` as its unknown
    flag.

### Measurements (RTX 5080, driver 610.88)

- Tests: light table 1 brick, 0 dropped; light NEE vs path tracer mean ratio 1.0095 (path tracer
  vs path tracer, other seeds: 1.0060), 16 x 16 block PSNR 45.8 dB (noise floor 42.0 dB); path
  tracer frame bit-identical; cells 294912 bytes, fill 0.0045 ms at 1024 cells, 60 % of the
  pixels lit at 1 spp (the light is a ceiling voxel: 2 of its 3 candidate faces are covered).
- Cornell (`--gi pt --no-denoise`, warm-up 32/64): 640 x 360: trace 0.21 ms without the flag,
  0.35 ms with it, light fill 0.035 ms (65536 cells); 1920 x 1080: trace 1.86 ms -> 3.19 ms, fill
  0.035 ms; with 1024 cells fill 0.006 ms, trace 3.42 ms (more global-CDF fallbacks and probes).
  The room is lit at 1 spp with the flag and black without it.
- Shroom forest (3.24 M bricks, 8457 groups, 1280 x 720): trace 2.98 ms without the flag, 5.9 ms
  with it (2706 light bricks), fill 0.33 ms; flush mean 0.44 ms with `--test-edits` every frame
  (finalize + light table rebuild + BLAS builds).

### Open points

- Covered faces: done in step 4 (`docs/LIGHT_FACES.md`): the entry has a voxel mask and a weight
  per face direction, and a face covered inside the brick is never sampled (the light box's floor
  and walls go from about 60 % to 100 % lit pixels at 1 spp). A face on the brick border that a
  neighbour brick covers can still be sampled.
- MIS with the BSDF path and the bias fix for dropped bricks: see docs/LIGHT_MIS.md (step 3).
- Material changes do not rewrite light words: a brick whose word is 0 under the old table stays
  out until it is finalized again (an entry whose lights vanished is dropped by the compact
  kernel, difference 5).
- Temporal reuse of the cell reservoirs is built behind `CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL`
  (section 2.2) for static scenes only; it has no motion or edit handling beyond dropping the
  history after a light table rebuild. The fill touches every live cell every frame.
- `maxStorageBufferRange`: the table binding at `max_bricks` 2^20 is 474 MB since step 4 (138 MB
  before), above the Vulkan minimum of 128 MB (NVIDIA allows 4 GiB). Only the compact and scan
  kernels bind it; the render kernels read it through addresses.
- The bench runs (`pt_nee`, `pt_nee_conv`, scene `shroom` with its stored pose) are configured in
  `tools/gi_bench.json`, not run in this step.
