# Light NEE on `main` (spec, 2026-10-24)

Status: approved by the owner (2026-10-24) after two external reviews (GPT Astra, Fable); being
implemented. Base: the light tree (`docs/LIGHT_TREE.md`, commit `6c6cbbf`).
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
| reservoirs | 64 per cell, 16 B each |
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

1088 B per cell; at the default payload capacity 262144: 272 MiB + key table 16 MiB (per renderer;
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
