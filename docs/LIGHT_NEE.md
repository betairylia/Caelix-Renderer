# Light NEE on `main` (spec, 2026-10-24)

Status: approved by the owner (2026-10-24) after two external reviews (GPT Astra, Fable); implemented
2026-10-24 (section 11 is the truth for the code where it differs). Base: the light tree (`docs/LIGHT_TREE.md`, commit `6c6cbbf`).
Reference implementation: branch `light-nee-experiments` (`docs/LIGHT_NEE.md` there; "old" below).
Owner goals (`docs/human_wrote/Caelix-Renderer Towards A Million Stars.md`): a raw signal with low
spatial and temporal correlation, the denoiser/upscaler carries the rest; no visibility reuse yet; the
trace kernel stays at or below 128 registers. Owner rule: visuals and stability come before strict
unbiasedness when the difference is within what the eye accepts.

## 1. Decisions

| topic | decision |
| --- | --- |
| scope | light cells + fill + NEE at the vertex + sun/lamp selection + NEE/BSDF MIS, in the library; host defaults in the Rust host; no Unity work |
| order of work | (1) hash table module + tests, (2) light cells + fill + a fill bench on real cell sets, (3) NEE + MIS in the trace kernel; M and the reservoir mode follow from the bench of (2) |
| candidate source | the light tree, drawn for the cell's receiver BOX and the cell's receiver normal (section 5) |
| fill scope | ACTIVE cells only (looked up by the previous frame's trace) |
| fill schedule | M fresh candidates per refreshed reservoir: M = 2 if the fill with M = 2 costs < 1 ms on rootfall, else M = 1 (then a reservoir is a plain tree draw, no RIS) |
| reservoirs | 64 per cell, 16 B each (experimental option: 128 or 256, section 11.6) |
| cell lifetime | evicted after 128 frames without a lookup |
| history | NO temporal reuse (owner decision 2026-10-24). Reservoir mode "off" (every reservoir is refreshed every fill) or "rotate" (a rotating quarter is refreshed per fill, each reservoir lives at most 4 fills); the default is chosen by the fill bench (section 3.2) |
| sun / lamp | one shadow ray per NEE vertex, chosen between the sun and the lamp candidate by their unoccluded contributions (one-sample RIS; old `nee_combine 1`), with the per-cell sun visibility hint |
| NEE / BSDF MIS | kept (balance heuristic with a proxy NEE density), only after diffuse bounces |
| no usable cell | no lamp NEE at that vertex; the next BSDF hit counts the full emission (no global fallback) |
| hash map | a reusable module: key table separate from the payload, ~20 % key fill (4 x key slots per payload), payload pool from a GPU free stack |
| cell size | footprint levels 32 / 128 voxels; a controller that may only coarsen (section 3.5) |

## 2. Reusable GPU hash table (`HashTable`, `shaders/common/caelix_hash_table.hlsl`)

A module with no knowledge of light cells; the light cells are its first user, a per-voxel colour
average is a planned second one. The user owns a payload pool; the table maps 64-bit keys to payload
indices.

- **Key slot** (16 B): `uint key_lo, key_hi` (64-bit fingerprint; `key_lo` 0 = empty, real keys have
  `key_lo >= 1` and `key_hi != 0`), `uint payload` (`kHashNone` = none), `uint pad`.
- **Payload header contract** (the user's payload starts with these words): `state` (free /
  uninitialised / ready), `touched` (frame of the newest lookup), and the user's key fields (so the key
  can be recomputed from the payload).
- **Capacity**: key slots = a power of two >= 4 x payload capacity (about 20 % fill when the payload is
  80 % full: 1.2 probes per hit, < 0.1 % failed inserts with 8 probes; table in section 9).
- **Lookup(key)**: probe `(hash + i) & mask`, i < 8. Both words equal = hit: the caller loads the payload
  index with ACQUIRE order and writes the payload's `touched`. `key_lo` equal and `key_hi` 0 = an insert
  in progress = pending (no entry this frame, no insert). Empty = miss (stop). Other keys are skipped.
- **LookupOrInsert(key)**: after a miss, CAS `key_lo` from 0 on the first empty slot of the window. The
  CAS returns the value it found:
  - 0 (won): pop a payload index; the user writes the payload header (key fields, `touched = frame`,
    `state = uninitialised`; never the rest of the payload); store `payload`; then a RELEASE store of
    `key_hi`. HLSL `Interlocked*` imply no barrier: the platform's acquire/release atomics
    (`CAELIX_ATOMIC_*`, inline SPIR-V) are used on both sides.
  - the same `key_lo`: another thread inserts or inserted this key: treat as pending; never insert further
    on. This keeps a key from being present twice.
  - another value: continue the probe.
  An empty free stack stores `payload = kHashNone`: the entry counts as absent until the next rebuild.
- **Free stack**: `int top` + `uint stack[P]`, initially `0 .. P-1`. Pop (inserts, during the trace):
  `i = atomicAdd(top, -1) - 1`, `i < 0` = empty. Push (evictions, in the maintenance pass):
  `stack[atomicAdd(top, 1)] = freed`. A one-thread fix-up clamps `top` to 0 before any push. Pops and
  pushes never run in the same dispatch.
- **Maintenance pass** (once per frame per table, before the user's own pass):
  1. over the payload pool: a ready or uninitialised payload with `frame - touched > max_age` is freed
     (`state = free`, push); the pass also writes the compact list of ACTIVE payloads (touched by the
     previous frame) and the indirect dispatch arguments of the user's pass;
  2. the key table is rebuilt from the live payloads: clear, then insert every live payload's key
     (recomputed from its header). No tombstones, no rehash threshold, no back reference; pending and
     payload-less entries of the previous frame disappear by construction. A key whose 8-probe window is
     full at the rebuild frees its payload. Estimated ~0.1 ms at 262144 payloads (a 16 MiB clear + the
     inserts); measured in the bench, and rebuilt less often only if that matters.
  Counters: live, active, inserted, pending, failed (no slot), failed (no payload), evicted, rebuild ms.
- **C++ side**: `HashTable` owns the key buffer, the free stack, the counters and their readback; the user
  owns the payload buffer. Every layout is defined once with names and mirrored in HLSL with
  static_asserts (owner rule: no magic offsets).
- **Fingerprint**: splitmix64 rounds over the user's key words (old `LightCellKey`), `key_lo >= 1`,
  `key_hi != 0` forced. A 64-bit collision merges two keys (~2^-64 per pair); documented, not handled.

## 3. Light cells (payload of the hash table)

### 3.1 Key, levels, footprint, jitter

- Key = (entity, entity life, level, cell coordinate, receiver face 0..5). Cells are entity-local (a
  moving entity carries its cells). Entity ids are reused after removal: the scene keeps a per-entity
  `life` counter (bumped when an id is handed out again, stored in the free `EntityInfo::pad` word), so
  the cells of a removed entity are never found again and age out.
- The receiver face is the ENTITY-LOCAL face of the hit voxel. Voxel surfaces are axis-aligned in their
  entity, so every receiver of a cell has exactly the normal `R e_face` in world space (R = the entity
  rotation): no normal cone is needed.
- Cell point = the hit point moved a quarter voxel into the air along the face normal. Coordinate at
  level L = `floor(p) >> (5 + 2 L)` (32 and 128 voxels).
- Level: 0 when the vertex's pixel footprint `2 d tan_half_fov_y / size.y` is below
  `footprint x controller scale`, else 1.
- Jitter: the sampling cell is looked up at the cell point moved by a uniform offset in (-0.5, 0.5)
  cells per axis (per-cell flicker becomes per-pixel noise).
- The MIS proxy uses the JITTERED sampling cell (owner decision). The segment state keeps that slot, so
  both sides read the same cell header and no second lookup is needed. The jitter is drawn before the
  BSDF direction and does not depend on it.

### 3.2 Reservoir lifetime (no temporal reuse)

Owner decision (2026-10-24): no temporal reuse in the cells. The old 128-frame history was the cause of
the slow patch flicker on crimson-crown (`Artifacts/light-flicker-crimson-20261023/`: lag-1
autocorrelation 0.97, and the NRD output was steadier without it), and the light tree's box draw is close
enough to the target that fresh draws give low-variance weights.

- **Mode "off"**: every fill refreshes all 64 reservoirs of every active cell.
- **Mode "rotate"**: every fill refreshes the reservoirs `r` with `(r + fill_index) % 4 == 0`, so each
  reservoir is at most 4 fills old; a reservoir that is stale (generation mismatch, see below) is refreshed
  at once whatever its turn. Each reservoir is still a valid contribution weight of the frame it was drawn
  in; between refreshes a moving light lags by at most 4 frames (accepted).
- The default mode is chosen by the fill bench: "off" if its fill time is acceptable, else "rotate".
  Chosen by the bench rule: "rotate" (section 11.3). The owner changed the default to "off" after her
  visual check (2026-10-24): white frame-to-frame noise for the denoiser (section 11.4), at the higher
  fill cost.
- **Identity**: a reservoir stores the group, its index entry and the group's 32-bit generation
  `(group life << 16) | group version` (life bumped when the group index is released, version at every
  rebuild within one life). A mismatch means the entry may point at another brick: the fill refreshes the
  reservoir; the trace (a cell that was inactive while its group was rebuilt) treats it as an empty pick
  (zero lamp contribution for that frame, still on the MIS path). No re-find by position is needed.
- Cells of moving entities and moving lights need nothing special: the next refresh uses the new
  positions.

### 3.3 Payload layout (one cell)

```
header 64 B: state, touched, entity, entity life, coord (int3), level, face,
             fills (since insert), C, C_n, sun_tests, sun_visible, pad
reservoirs 64 x 16 B: uint group; uint entry; float W; uint generation ((group life << 16) | version)
```

1088 B per cell (64 + 16 R with R reservoirs, section 11.6); at the default payload capacity 262144:
272 MiB + key table 16 MiB (per renderer;
section 7). A reservoir is one aligned 16-byte load. The fill initialises the reservoirs of an
uninitialised cell (empty, no history) and sets it ready; a trace that sees an uninitialised cell does
no lamp NEE, so it never reads the W, C or sun counters of a previous cell in a reused payload. The
per-slot visibility counters of a later visibility reuse need their own space (decided with that step).

### 3.4 Fill (indirect dispatch over the active list; 64 threads per cell, thread r owns reservoir r)

1. Receiver region: the cell box inflated by half a cell on every side (jittered lookups), as a world
   AABB of the entity-transformed box; receiver normal `R e_face` (section 3.1).
2. `M` fresh candidates per reservoir from `LightTreeSampleBrickBox(both trees, box, normal)`
   (section 5): the proposal pdf `p(y)` is exact, and the call also returns the leaf's box importance,
   which is the TARGET `p_hat(y)` (the importance of the chosen leaf's world summary: projected face
   sums and box, for the cell's box and normal). No record load in the fill. A leaf that is not in a BVH
   (dropped) can never be drawn and has no target, so the proxy (section 6) gives `w_nee = 0` for it.
3. Only the reservoirs due for a refresh (section 3.2) draw; the others are kept as they are.
4. `W = (w_sum / M) / p_hat(selected)` for M = 2 (streaming RIS); `W = 1 / p(y)` for M = 1 (a plain draw:
   the target is not needed then).
5. Cell normalisation for the MIS proxy: one dedicated draw per cell per fill (a rotating thread),
   independent of the reservoirs, folded into `C` as a running mean over the last 32 fills
   (`kNormalisationFills`). Because it is
   independent of the reservoirs, the trace reads the current `C` (the old one-frame delay is gone); a
   cell with `C_n = 0` gives `p_sa = 0` (the BSDF keeps the full emission).
6. Sun visibility hint: decay of `sun_tests` / `sun_visible` by 1/8 per fill (old section 10.2).
7. `fills += 1`.

Random stream: a function of (payload index, r, frame seed) only.

Cost model (measured first, section 8): a light tree draw costs ~1 ns of GPU throughput in the light tree
bench with random receivers, against ~0.03 ns for the old CDF draw on bistro (its CDF fits in L2). An
active cell costs 64 x M + 1 draws per fill in mode "off", 16 x M + 1 in mode "rotate". The 64 threads of a cell share one receiver box, so
their descents are coherent and should cost much less than the bench's; the fill bench measures it.

### 3.5 Cell-size controller (CPU)

The fill cost follows the ACTIVE cells, the memory the live ones. Every 16 frames, from the readback:
when the live count passes 0.85 P (P = payload capacity) or inserts fail for lack of payloads, the
footprint threshold shrinks by 10 % (fewer fine level-0 cells); when the live count is below 0.70 P it
grows by 10 %, but never above the configured footprint (the controller only coarsens). Clamp:
[0.25, 1] x the configured footprint. Reported in the stats. The 128-frame eviction lag makes the live
count react slowly; the rate is checked in the bench for oscillation.

## 4. NEE at a vertex

Runs at the vertices where the sun NEE runs today: every vertex that took the diffuse lobe and has a
next segment. Its own random stream with a fixed number of draws per vertex (frames without light NEE
stay bit-identical).

1. Lookup (jittered) of the sampling cell; insert when missing. "No lamp NEE" (the next BSDF hit counts
   the full emission) only on CELL-level conditions that do not depend on reservoir values: no slot,
   pending, no payload, uninitialised, or `C_n = 0`. A ready cell whose picked reservoir is empty or
   stale gives a zero lamp contribution and stays on the MIS path (deciding "no lamp NEE" from the
   picked reservoir would make the result too bright).
2. Lamp candidate: a uniform reservoir of the cell -> generation check -> `LightBrickSamplePoint` (point
   on the brick, in-brick pdf) -> `W_l = W_brick / pdf_in_brick`.
3. Sun candidate (sun NEE on): as today, `W_sun = 1 / pdf_sun`.
4. One-sample RIS over the present candidates with the vertex target
   `p_hat = luminance(L_e) cos_r cos_l / d^2` (times `w_nee` with MIS; 0 inside the near field), the sun
   term times the cell's sun visibility hint `max(0.05, visible / tests)` (1 when `tests = 0`). The
   domains are disjoint (sun, lamps), so `W_out = w_sum / p_hat(kept)`. The kept candidate is replayed
   from its saved random state (old section 5.3: no candidate values stay live).
5. One shadow ray (glass occludes, as today). The sun hint counters are updated when the sun was kept,
   wave-aggregated (old section 10.2; per-lane atomics cost 1.4 ms there).
6. Near field (8 world units): lamp points closer than that belong to the BSDF path (old 2.6).
7. One lamp candidate per vertex (owner decision: K = 1; per-vertex resampling, old 5.3, is reconsidered
   after visibility reuse).

## 5. Light tree API additions

- Box receiver importance: the light tree's importance (`LIGHT_TREE.md` section 5.1) with the receiver
  box R in place of the point: D = [R.lo - B.hi, R.hi - B.lo]; the emitter cosine bound over D as today;
  the receiver term `cr = min(1, m / dmin)` with `m = max over D of dot(n, -v)` and `dmin` the distance
  between the boxes (the same code as the point case, which is R.lo = R.hi); the distance term
  `d2 = max(|c_R - c_B|^2, max(1, (|half_R| + |half_B|)^2))` (reduces to the tree's formula for a point).
  One implementation for point and box; the point form must compile to the same code as today (check the
  SPIR-V of the light tree kernels).
- `LightTreeSampleBrickBox(t, trees, box, normal, rng, out ref, out pdf, out leaf, out importance)` and
  `LightTreeBrickPdfBox(...)`; the pdf of a draw over both trees includes the tree choice probability.
- A tree selector on the existing calls (static, dynamic, or both as today).
- Tests: rotated entities, the jitter range and numeric edges (receiver on a box face, zero-size boxes);
  the support check of section 8.

## 6. MIS with the BSDF path (old section 6, adapted)

- `w_nee = p_sa / (p_sa + p_bsdf)` on the NEE contribution; `w_bsdf = p_bsdf / (p_sa + p_bsdf)` on a lamp
  hit of the segment that leaves a vertex that did lamp NEE. `p_bsdf = cos_r / pi` (the diffuse bounce).
- Proxy: `p_area(y) = p_hat_cell(y) / C x in_brick(y)` from the vertex's jittered sampling cell;
  `p_hat_cell` = the box importance of y's LEAF summary (the fill's target, section 3.4); `in_brick` from
  the record (family weight at the receiver, count). No BVH leaf (dropped, not gathered) = `p_sa = 0` =
  the BSDF keeps the full emission. Both sides use the same functions.
- Rules kept from old 6.2 / 6.3: near field = BSDF; segments after specular or transparent vertices count
  every emission; a hit that the NEE cannot sample (not a light face, a dropped brick or leaf) keeps the
  full emission.
- State across the bounce trace: one packed word (jittered cell slot + flags: did lamp NEE, MIS on); the
  proxy context (cell box, normal, C) is rebuilt from the cell header after a lamp hit (old 6.1).
- BSDF side per lamp hit: index entry (binary search by brick position) -> `bvh_leaf` -> leaf summary,
  and the family search for `in_brick`. If the registers or the time show it, the group build writes a
  per-dense-id entry map (one load instead of the search).

## 7. Options, stats, ABI, binding

- `CaelixLightNeeOptions` + `caelix_renderer_set_light_nee_options` (per renderer): enabled, payload
  capacity (0 = 262144), footprint (0 = 0.25), reservoir mode (off / rotate; 0 = the bench's default),
  fill candidates M (0 = the bench's default, 1 or 2), max age (0 = 128), flags (jitter, MIS, sun hint,
  controller; each with a host opt-out).
  Library default: off. Rust host default: on.
- `CaelixLightNeeStats` (newest safe frame): live and active cells, inserted, pending, failed (slot),
  failed (payload), evicted, fresh draws, normalisation draws, stale reservoirs refreshed, controller
  scale, maintenance ms, fill ms.
- Per-frame order per renderer: maintenance (evict, active list, key rebuild) -> fill -> trace. Several
  renderers of one frame each own their table (288 MiB each at the defaults). Cells depend on the view
  only through the level choice, so one table per scene would also work; that is a later decision for
  multi-view (Unity).
- Binding: the payload, keys and free stack through buffer device addresses (288 MiB is above Vulkan's
  guaranteed 128 MiB `maxStorageBufferRange`; the old storage-buffer binding stays an option only if a
  register measurement favours it). `FrameConstants` additions (bases, mask, footprint scale, max age,
  flags) follow the uniform-buffer layout rule (`caelix_frame_layout_test`).

## 8. Validation and measurement

- Hash table: unit tests against a CPU model with heavy contention (many threads inserting the same and
  colliding keys, pending states, payload exhaustion and recovery, eviction, key rebuild, free stack).
- Fill bench (before the trace integration): real cell sets (rootfall, bistro 8192, crimson-crown)
  with fill ms, maintenance ms, active cells, fresh and normalisation draws per frame, for M = 1 and 2 and
  the modes off and rotate. Decides M (owner rule: M = 2 if the fill with M = 2 costs < 1 ms on rootfall,
  else M = 1) and the default mode.
- Temporal behaviour (modes off and rotate): brightness time series and lag-1 autocorrelation of fixed
  wall regions on crimson-crown (the method of `Artifacts/light-flicker-crimson-20261023/`), raw and after
  NRD, plus 1-spp PSNR / block PSNR vs 1024-frame references.
- Support check: random receiver points that actually read a cell (inside the inflated box, with the
  cell's normal); for every light face with a non-zero unoccluded contribution, the box sampler's pdf and
  the target must be > 0; rotated entities, jitter range, numeric edges.
- Unbiasedness (static scene): the closed light-box test of the old branch, ported:
  NEE + MIS vs the BSDF-only reference, 512 frames, linear ratio within run noise, with lamps, the sun and
  glass. MIS weights sum to 1 at sampled points (probe kernel). These cover only part of the design; the
  next two groups check the identity and lifecycle rules.
- Dynamic response: a lamp switched on, a lamp removed, a lamp moved into a room, a receiver entity
  rotated; response curves of a fixed region's NEE brightness over frames.
- Lifecycle stress: group and entity removal with immediate index/id reuse, a renderer paused while the
  scene changes many times, payload exhaustion then recovery.
- Registers: `CAELIX_PIPELINE_STATS` after every step; target <= 128 for the full kernel.
- Scenes (host, 1440p and 1080p, 100-frame means): bistro 8192, emberdeep, crimson-crown, shroom; trace
  ms, maintenance and fill ms, live/active cells; compared with the old branch's numbers.

## 9. Linear probing reference (simulation, 2^18 slots, cap 8)

| fill | probes per hit | probes per miss | new keys failing (8 probes) |
| --- | --- | --- | --- |
| 0.10 | 1.06 | 1.12 | 0.00 % |
| 0.25 | 1.17 | 1.38 | 0.06 % |
| 0.40 | 1.32 | 1.89 | 0.9 % |
| 0.50 | 1.45 | 2.44 | 3.5 % |
| 0.60 | 1.60 | 3.39 | 8.7 % |

## 10. Owner decisions (2026-10-24)

No temporal reuse in the cells (modes off and rotate); M by the rootfall rule; the history caps, the
re-find by brick position and the confidence bookkeeping of the reviews are therefore not built.

## 11. As built (2026-10-24)

Implemented on `main` (working tree on top of `4fe9f59`). This section is the truth for the code where
it differs from sections 1-10. Measurements, commands and images:
`F:/RP_Games/Caelix-family/Artifacts/light-nee-main-20261024/README.md`.

### 11.1 Files

- Hash table: `lib/src/rhi/hash_table.{h,cpp}`, `lib/src/rhi/hash_table_layout.h`,
  `shaders/common/caelix_hash_table.hlsl` (the device API), `shaders/hash/caelix_hash_{kernels,init,begin,
  maintain,finish}.hlsl` (the maintenance kernels).
- Light cells: `lib/src/render/light_cells.{h,cpp}`, `lib/src/render/light_cells_layout.h`,
  `shaders/common/caelix_light_cells_layout.hlsl`, `shaders/render/caelix_light_cells.hlsl` (cell rule,
  region, MIS weight, proxy), `shaders/render/caelix_light_cell_fill.hlsl`; renderer side
  `lib/src/render/renderer_light.cpp` (options, lifetime, stats) and `renderer_render.cpp`.
- Trace: `shaders/render/caelix_path_trace.hlsl` (LightCellAtVertex, LampCandidate, SunCandidate,
  NextEventAtVertex, CountSunRay, LampHitWeight) and the kernel variant
  `shaders/render/caelix_path_trace_no_light_nee.hlsl`.
- Light tree additions: `shaders/scene/caelix_light_tree.hlsl`, the generation split in
  `caelix_light_tree_build.hlsl` and the layout headers; entity life in `gpu_structs.h`,
  `caelix_record.hlsl`, `Scene::AddEntity`, `Scene::StageEntity`.
- ABI: `CaelixLightNeeOptions`, `caelix_renderer_set_light_nee_options`, `CaelixLightNeeStats`,
  `caelix_renderer_get_light_nee_stats` (no version bump). Rust host: `--no-light-nee`, `--light-*`,
  `--screenshot-frame`, the Stats window's "Light NEE" section, `light_nee:` on `timings_avg:`.
- Tests: `tests/hash_table_test.cpp` + `tests/shaders/caelix_hash_probe.hlsl`, `tests/light_nee_test.cpp`
  + `tests/shaders/caelix_light_nee_probe.hlsl`, the "light NEE" cases at the end of
  `tests/trace_test.cpp`, `tests/test_kernels.{h,cpp}` (shared probe helpers), `frame_layout_test.cpp`
  (the new push blocks and layout headers), `abi_c99.c`.

### 11.2 Differences from the spec

1. **Rotate mode packs 4 cells into a fill workgroup.** With 64 threads per cell (thread r owns
   reservoir r, section 3.4) the rotate mode drew a quarter of the reservoirs but cost as much as mode
   "off" (rootfall 1440p: 1.16 vs 1.25 ms at M = 1): a wave costs its instructions whatever the share
   of active lanes. Mode "rotate" now runs 4 cells per workgroup, 16 threads per cell, thread j owns the
   reservoirs 4j .. 4j + 3 of which exactly one is due by the rotation (stale ones are refreshed too).
   Mode "off" keeps one cell per workgroup, thread r = reservoir r. The hash table's maintenance takes
   `items_per_group` per call (1 or 4).
2. **An empty reservoir is not stale.** With M = 2 a refresh can end empty (both candidates have target
   0). Refreshing empty reservoirs out of turn in mode "rotate" would make them rarer than the
   resampling makes them: a bias upward. Stale = a generation mismatch (or a new cell).
3. **The lamp candidate stays live; only the sun candidate is replayed.** Replaying the lamp candidate
   (a second LightBrickSamplePoint and proxy) cost 1.5 ms at 2560x1440 on rootfall and more spill
   (48 vs 16 bytes); keeping its 7 values over the cheap sun draw does not.
4. **A trace kernel without light NEE.** With the light NEE code in the kernel (128 registers instead of
   96), a frame without light NEE traced 20 % slower (rootfall 1440p 9.9 vs 8.2 ms). The build compiles
   `caelix_path_trace_no_light_nee` (CAELIX_SPEC_NO_LIGHT_NEE: `light_nee_flags` 0 at compile time,
   96 registers, 48 bytes of spill as before) and the renderer binds it for every frame that runs no
   light NEE. A kernel variant source carries the marker CAELIX_KERNEL_VARIANT (cmake/Shaders.cmake).
5. **Receiver box margin.** The region is the cell inflated by half a cell plus 1 voxel
   (`kLightCellBoxMarginVoxels`): the receiver lies 0.248 voxels below the cell point, so half a cell
   alone does not hold every receiver of a jittered lookup.
6. **Options as opt-outs.** `CaelixLightNeeOptions::flags` holds opt-outs (`NO_JITTER`, `NO_MIS`,
   `NO_SUN_HINT`, `NO_CONTROLLER`), so 0 is the designed configuration, plus the measurement flag
   `LOOKUP_ONLY` (cells looked up, inserted and filled, no lamp NEE: the fill bench). The capacity
   field is `cell_capacity` (1024 .. 2^21).
7. **FrameConstants** grew to 608 bytes (light_nee_flags, light_frame, the cell, key, stack and counter
   addresses, the key mask, the footprint). The composite, cross resolve, sky SH and upscale encode
   kernels change only in the uniform block's struct type (member decorations and names): their
   function bodies are identical to before (checked by comparing the disassembly modulo id numbering).
8. **Light tree.** `LightTreeView` (LightTreeOpen without the check for lights) serves the trace's
   lookups; `LightTreeFindLightFace` serves the BSDF side and LightTreePointPdf; the tree selector is a
   field of `LightTree` (`LightTreeSelect`). The light tree probe's SPIR-V is not byte-identical to the
   one before the box form (id numbering, the order of constants and local variables, one swapped block
   pair; the same instructions); no library kernel includes caelix_light_tree.hlsl.
9. **Light NEE runs in both lighting modes** (path trace and reference) and restarts its cells when the
   scene's light tree is recreated (a new tree's generations start over).
10. **Counters.** The hash table counts `rebuild_dropped` (keys whose window was full at the rebuild)
    apart from the lookups' `failed_slot`; `stale_refreshed` counts every refresh out of turn (stale
    and new cells).

### 11.3 Fill bench and the defaults

Light NEE in lookup-only mode (`--light-lookup-only`: cells looked up, inserted and filled, no lamp
NEE), host defaults otherwise, 2560x1440, 100-frame means, dark sky for rootfall and bistro (folder
`fill-bench/` of the artifacts; the first, uncompacted rotate run in `fill-bench-v1-uncompacted/`):

| scene (active cells) | off, M 1 | off, M 2 | rotate, M 1 | rotate, M 2 | maintenance |
|---|---|---|---|---|---|
| rootfall (42 808) | 1.39 ms | 2.29 ms | 0.61 ms | 1.03 ms (1.04, 1.05 in two repeats) | 0.059 ms |
| bistro 8192 (32 112) | 1.20 ms | 2.21 ms | 0.52 ms | 0.86 ms | 0.059 ms |
| crimson-crown (14 257) | 0.52 ms | 0.77 ms | 0.24 ms | 0.39 ms | 0.055 ms |

Fresh draws per frame on rootfall: 2.74 M (off, M 1), 0.69 M (rotate, M 1); normalisation draws = the
active cells. Rootfall pose: the first keyframe of `rootfall.packed.street1.sequence.json`,
`-21.961,270.802,-933.939,-3.500,44.000`.

**Decision by the owner's rule:** fill(off, M 1) = 1.39 ms >= 1 ms on rootfall -> mode **rotate**;
fill(rotate, M 2) = 1.03 .. 1.05 ms >= 1 ms -> **M = 1**. The library's 0 values and the Rust host used
rotate and M = 1. (M = 2 adds little here anyway: the box draw's proposal already follows the leaf
importance, which is the RIS target.) The temporal cost of rotate is in 11.4.

**Owner change (2026-10-24, after her visual check):** the default mode is **off** (white
frame-to-frame noise, section 11.4) at 1.39 ms of fill on rootfall instead of 0.61 ms; M stays 1. The
library's 0 values and the Rust host use off and M = 1 (`d5670aa`).

### 11.4 Measurements

Rust host, RTX 5080, driver 610.88, 100-frame means of the trace dispatch, one GPU job at a time; the
light NEE at the defaults of the time (rotate, M 1, jitter, MIS, sun hint), the host's 4 bounces, no
sun (sky only; dark sky for rootfall, bistro and shroom). "off" = `--no-light-nee` (the trace kernel
variant without light NEE).

| scene | 2560x1440 on / off | 1920x1080 on / off | fill, maintenance (1440p) | live / active cells (1440p) |
|---|---|---|---|---|
| rootfall | 20.46 / 8.27 ms | 11.73 / 4.86 ms | 0.61, 0.060 ms | 61 531 / 42 951 |
| bistro 8192 | 21.55 / 8.09 ms | 11.84 / 4.75 ms | 0.55, 0.060 ms | 60 050 / 32 016 |
| crimson-crown | 26.95 / 10.88 ms | 15.31 / 6.60 ms | 0.24, 0.053 ms | 22 958 / 14 187 |
| emberdeep | 12.65 / 4.22 ms | 6.95 / 2.45 ms | 0.15, 0.048 ms | 6 946 / 5 243 |
| shroom | 19.29 / 6.79 ms | 10.80 / 4.16 ms | 0.83, 0.062 ms | 80 655 / 50 097 |

The "off" runs trace no shadow ray at all (no sun); with light NEE every diffuse vertex traces one. On
rootfall 1440p: lookups alone 10.9 ms, NEE without its shadow ray 12.8 ms, with it 20.3 ms (shadow rays
about 7.5 ms). In the configuration of the old branch's measurements (2 bounces, the sun 2e4 with sun
NEE, no denoiser, default sky): bistro 13.28 ms on / 8.20 off, emberdeep 10.14 / 5.83 (old branch after
its register work: 13.4 and 10.3 ms with light NEE).

Light NEE off against the binary before this work (rootfall 1440p, two runs each): 8.21 / 8.28 ms vs
8.26 / 8.18 ms. Registers (CAELIX_PIPELINE_STATS): trace with light NEE 128 + 16 bytes of spill (also
with the lookups alone), the variant without light NEE 96 + 48 bytes (as before), fill 80, hash kernels
16 .. 19.

Bias on real scenes: 1024-frame references with light NEE off vs on: mean ratio 1.005 rootfall, 0.999
bistro, 1.000 emberdeep (block PSNR 56 .. 62 dB).

Temporal behaviour on crimson-crown (256 consecutive frames, 1920x1080, exposure 4, the method and
regions of `Artifacts/light-flicker-crimson-20261023`, reference = 1024 frames with light NEE):

| variant | wall lag-1 autocorrelation (left, right, centre) | wall std of the 32-frame window mean / ref (white noise would give) |
|---|---|---|
| raw, rotate | 0.75, 0.69, 0.80 | 0.037, 0.035, 0.036 (0.016, 0.012, 0.016) |
| raw, off | -0.08, -0.04, -0.06 | 0.016, 0.012, 0.013 (0.016, 0.013, 0.015) |
| NRD, rotate | 0.996, 0.989, 0.994 | std(m_t) / ref 0.020, 0.009, 0.014 |
| NRD, off | 0.990, 0.993, 0.992 | std(m_t) / ref 0.009, 0.005, 0.012 |

The old branch (128-frame cell history): raw lag-1 0.94 .. 0.98, NRD std(m_t) / ref 0.03 .. 0.15. Mode
"off" gives white per-frame noise; "rotate" keeps a 4-frame memory (lag-1 about 0.75, lag-4 about 0.1)
and about twice the slow wall drift after NRD of "off".

### 11.5 Tests

All 11 ctests pass with 0 validation and 0 library errors: `caelix_hash_table_test` (5 cases),
`caelix_light_nee_test` (3 cases: cell regions, support, MIS), the two "light NEE" cases of
`caelix_trace_test` (unbiased against the path tracer: four option sets, lamps alone and with the sun
through an opening and glass, ratios 0.998 .. 0.999 against a path tracer floor of 0.998, block PSNR
39.3 .. 40.5 dB against the floor's 37.6 dB; lifecycle), `caelix_frame_layout_test` (the new push blocks,
the 608-byte uniform block of six kernels, 260+ named constants), `caelix_light_tree_test` and the rest
unchanged. Not built: the dynamic response curves of section 8 (a lamp switched on, removed, moved; a
receiver entity rotated). Since 11.6 the unbiasedness case runs six option sets (the two variants of
11.6 added) and the lifecycle case runs its step 4 (stale reservoirs refreshed out of turn) in mode
"rotate": with the default mode "off" every reservoir is due at every fill, so nothing is out of turn.

### 11.6 Experimental variants (2026-10-24, owner visual check)

Two variants for the owner's visual comparison: more reservoirs per cell, and no cells at all. The
defaults (mode "off", 64 reservoirs, M = 1) are unchanged: the full trace kernel differs only in reading
the reservoir count from the frame constants (128 registers and 16 bytes of spill as before), the
fill kernel only in reading it from its push constants, and every value at R = 64 is the one of before
(cell stride 1088, reservoir `min(63, u 64)`, random stream `FillHash(slot 64 + r)`).

**Reservoir count.** `CaelixLightNeeOptions::reservoirs` (the former `pad`; the struct stays 32 bytes):
0 = 64, else 64, 128 or 256 (a power of two in `kLightCellMinReservoirs` .. `kLightCellMaxReservoirs`).
A cell is `LightCellBytes(R) = 64 + 16 R` bytes (1088, 2112, 4160); the pool `cell_capacity x
LightCellBytes(R)` must stay within 4 GiB (`kMaxCellPoolBytes`; 262144 cells x 4160 bytes = 1.02 GiB),
otherwise the options are INVALID_ARGUMENT. A new R replaces the cells (as a new capacity does). The
trace reads R from `FrameConstants::light_cell_reservoirs` (offset 600, the former `pad_600`), the fill
from `LightCellFillPush::reservoirs`. Host: `--light-reservoir-count <0|64|128|256>`.

Fill mapping (`LightCellFillThreadsPerCell`, the same function in `light_cells_layout.h` and
`caelix_light_cells_layout.hlsl`): threads per cell T = min(64, R in mode "off", R / 4 in mode
"rotate"), 64 / T cells per workgroup (also the maintenance's `items_per_group`), thread j owns the R / T
reservoirs from j R / T.

| R | mode | threads per cell | cells per workgroup | reservoirs per thread | due per thread and fill |
|---|---|---|---|---|---|
| 64 | off | 64 | 1 | 1 | 1 |
| 64 | rotate | 16 | 4 | 4 | 1 |
| 128 | off | 64 | 1 | 2 | 2 |
| 128 | rotate | 32 | 2 | 4 | 1 |
| 256 | off | 64 | 1 | 4 | 4 |
| 256 | rotate | 64 | 1 | 4 | 1 |

R = 64 keeps the mappings of 11.2 item 1. "rotate" with R = 256 draws 64 reservoirs per cell and fill,
as many as "off" with R = 64 (2.75 M fresh draws per frame on rootfall in both), and every reservoir is
at most 4 fills old.

**No cells.** `CAELIX_LIGHT_NEE_FLAG_NO_CELLS` (host `--light-no-cells`): no hash table, fill,
maintenance or cell memory (cells of an earlier configuration are retired). Every NEE vertex draws its
lamp point with `LightTreeSample` for its own receiver x_r (the pushed-off origin) and normal n: the
point form of the importance at every tree level, then the in-brick draw for x_r, so p = P(brick | x_r,
n) P(face | brick, x_r) / face_count. Contribution `L_e (cos_r / pi) (cos_l / d^2) / p`, times the
balance heuristic weight with that p. The BSDF side of a lamp hit uses `LightTreeBrickPdf(x_r, n, brick)
x in_brick`, the same function of (x_r, n, point) as the draw: the segment's origin and `segment_normal`
are bit for bit the x_r and hit normal that the NEE got (checked in `TracePath`: both come from
`shade.bounce_origin` and the hit's normal), so the two MIS weights of a point sum to 1 up to the
rounding of the two pdf walks (top-down draw, bottom-up evaluation). No proxy. The near-field rule (8
world units on both sides) is kept, so only the source of the lamp candidate changes. The sun / lamp
one-sample RIS runs as before with hint 1 (no cell, no sun counters). The lamp stream draws one number
per vertex (the point seed). NO_JITTER, NO_SUN_HINT and NO_CONTROLLER have no effect; NO_CELLS with
LOOKUP_ONLY is INVALID_ARGUMENT. Stats: `enabled` and `running`, every cell field 0 (the Stats window
shows "no cells").

The trace runs a third kernel, `caelix_path_trace_light_tree_nee`
(`shaders/render/caelix_path_trace_light_tree_nee.hlsl`, `CAELIX_SPEC_LIGHT_NEE_DIRECT`; frame flags
`kLightNeeDirect | kLightNeeContribute`, plus `kLightNeeMis` without NO_MIS; the cell fields stay 0).
The direct path exists only in that variant (`#ifdef`); the full kernel keeps only the cell path, and
`caelix_path_trace_no_light_nee` is unchanged (its SPIR-V differs only in the member name of offset 600).

**Measurements** (`Artifacts/light-nee-variants-20261024/`, RTX 5080, driver 610.88, 2560x1440, the
settings of 11.4: 4 bounces, no sun, dark sky, NRD on, 100-frame means after 64 warm-up frames).

Registers (`CAELIX_PIPELINE_STATS`): `caelix_path_trace` 128 + 16 bytes of spill (as before; binary
210560 -> 211200 bytes), `caelix_path_trace_no_light_nee` 96 + 48 (unchanged),
`caelix_path_trace_light_tree_nee` 128 + 16, `caelix_light_cell_fill` 80 + 16 (driver shared memory
1280 -> 1536 bytes).

Trace times: on this machine on 2026-10-24 the frame times of the cell variants fell into two states,
about 16.5 ms and about 22.5 ms on both scenes, and the state changed from run to run whatever the
variant (the GPU clock stayed at about 2850 MHz; the desktop had other GPU clients). The table gives the
per-run medians of three interleaved runs; compare the variants within one state:

| scene | variant | trace, per-run medians | fill | maintenance | cell memory |
|---|---|---|---|---|---|
| rootfall | before this work (old kernels, `CAELIX_SHADER_DIR`) | 18.18, 18.02, 16.42 ms | 1.54 ms | 0.062 ms | 290 MiB |
| rootfall | default (off, 64) | 16.51, 16.31, 22.51 ms | 1.59 ms | 0.062 ms | 290 MiB |
| rootfall | rotate, 256 | 22.51, 22.60, 22.51 ms | 1.68 ms | 0.066 ms | 1.03 GiB |
| rootfall | no cells | 27.58, 27.55, 27.60 ms | - | - | 0 |
| bistro | before this work | 22.51, 16.89, 22.53 ms | 1.47 ms | 0.061 ms | 290 MiB |
| bistro | default (off, 64) | 22.70, 22.66, 22.72 ms | 1.52 ms | 0.061 ms | 290 MiB |
| bistro | rotate, 256 | 22.76, 16.90, 22.71 ms | 1.60 ms | 0.066 ms | 1.03 GiB |
| bistro | no cells | 28.07, 28.11, 28.12 ms | - | - | 0 |

Within one state the default with the new kernels equals the kernels of before (rootfall 16.31 / 16.51
vs 16.42 ms; bistro 22.66 .. 22.72 vs 22.51 / 22.53 ms), and rotate with 256 reservoirs equals the
default (bistro 16.90 vs 16.89 ms, 22.7 vs 22.7 ms); its fill costs 0.05 .. 0.1 ms more (the same
draws, a 4x larger pool). No cells never showed the faster state: 27.6 / 28.1 ms, about 5 ms (+22 %)
above the cell variants' slower state (a light tree descent with the point importance at every NEE
vertex instead of a reservoir load and an in-brick draw), without the fill's 1.5 ms.

Bias (bistro, reference mode, 512 frames, linear mean luminance ratio against the 1024-frame reference of
11.4 with light NEE in mode "rotate", `scripts/bias.py`): default 0.9994, rotate with 256 reservoirs
0.9998, no cells 0.9991 (quadrants 0.998 .. 1.001; block PSNR 56.7, 56.5, 56.9 dB). The test case
"light NEE: unbiased" (lamp box, against the 4096-frame path tracer; floor 0.9982 / 0.9986, 37.6 /
37.7 dB): rotate with 256 reservoirs 0.9983 / 0.9982 (sun 0 / 1000), 40.4 / 39.6 dB; no cells 0.9983 /
0.9984, 40.1 / 39.6 dB.

