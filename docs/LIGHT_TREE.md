# Light tree (as built, 2026-10-24)

Status: as built on `main` (2026-10-24). This file is the truth for the code.

Owner decisions:
- 2026-10-23: a light BVH at brick granularity, plus the per-cell sampling reuse. Reasons:
  volumetrics need a receiver-weighted light draw with an exact pdf at any world point, and many
  moving lights break a per-cell cut.
- 2026-10-24: strong emitters are rare in the game (meridian-deep is a stress case), so the hierarchy
  is two world-space 4-wide BVHs over light bricks (static and dynamic entities, chosen by the
  general entity flag `caelix_scene_set_entity_motion`, independent of physics; the host maps its
  own states), not an octree or pyramid over the group grid. Face lists are owned by the brick
  (self-contained brick light records). The double-buffered copy heap stays for now. Open faces are
  counted across brick and group borders of one entity. The light cells, the NEE and the MIS are
  specified in `docs/LIGHT_NEE.md` on top of this.

Background: `docs/human_wrote/Caelix-Renderer Towards A Million Stars.md`, the 2026-10-23
discussion summarised in the family `CLAUDE.md` ("Light sampling bench", "Lamp visibility",
"Crimson-crown"), `Artifacts/light-flicker-crimson-20261023/design-proposal-A.md` section 2.2 (the
bounds).

Streaming (planned, not built; to discuss with the owner): the render group is the streaming chunk;
the static world is one entity, so the group grid must become sparse too (the finalize neighbour
lookup uses the dense grid today); geometry LOD is decided on the GPU per view, but the light set is
decided on the CPU by range around the player, independent of view and of geometry residency; far
chunks keep their lights. The host will upload light records precomputed per chunk and per LOD
(a future `caelix_scene_upload_light_records`), and the library keeps them. Open: how the records are
produced (recommended: a GPU bake in the library with readback, stored by the host with the chunk,
so the open-face rule has one implementation); a versioned record format with a LOD field (scale
2^n, face weights x 4^n); the bake must see the neighbour chunks at the same LOD, and an edit at a
chunk border must mark the neighbour chunk's records stale; host-known record sizes allow CPU
in-place allocation, which then replaces the copy heap.

Scope of this step: the scene side (light bricks, the records, the BVHs, their updates) and the
device API (draw a light point, evaluate its pdf). No cells, no NEE, no MIS in the trace kernel.

## 1. Terms

- **Light voxel**: a voxel whose material emission luminance, `dot(emission, (0.2126, 0.7152,
  0.0722))` before `CaelixRenderOptions::emission_scale`, is at least `min_emission` (default 20).
  The tree keeps a 65536-bit **light code set** (8 KiB, GPU) indexed by the material code
  (`BlockMaterialCode`: transparent codes without their face bits), built on the CPU from the
  material table (`LightTree::RebuildCodes`) and uploaded with `vkCmdUpdateBuffer`.
- **Light face**: face `f` (+X 0, -X 1, +Y 2, -Y 3, +Z 4, -Z 5) of a light voxel whose neighbour
  voxel in direction `f` is not opaque. The neighbour can be in another brick or group of the same
  entity (resolved like the finalize kernel; an absent brick is air, so the face is open).
- **Light brick**: a brick with at least one light face.
- **Face weight**: the luminance of one light face (area 1: one voxel = one world unit, entities have
  no scale). A summary's `face[f]` is the sum of the face weights of family `f` below it.
- **Motion**: `CAELIX_ENTITY_STATIC` (default) or `CAELIX_ENTITY_DYNAMIC`, a per-entity flag of the
  scene (`Scene::Entity::motion`). It chooses the BVH that holds the entity's lights.

## 2. Structure

```
tree choice (2 root summaries, world space)
  -> BVH 0 (static entities) / BVH 1 (dynamic entities): 4-wide nodes, children inline
     -> leaf = one light brick (world box + projected face sums, its group and index entry)
        -> group block (heap): index entry -> brick light record (entity-local frame)
           -> face family -> face entry -> point on the unit face
```

Every layout below is defined once in `lib/src/scene/light_tree_layout.h` (structs, named offsets,
static_asserts) and mirrored by name in `shaders/common/caelix_light_tree_layout.hlsl`;
`caelix_frame_layout_test` compares the two files and the push constant blocks of the compiled
kernels with the C++ push structs. Heap offsets and sizes count 16-byte **units** (`kLtUnitBytes`).

Scene record word 1 (`kRecordBoundsWord`): bits 0..17 the tight bounds (readers mask them), bits
18..29 the light face count of the brick (`kLtRecordFaceCountShift` / `Mask`), bit 30 "light record
stale" (`kLtRecordStaleBit`). Word 1 is GPU-owned: an upload that replaces a resident brick
(`Scene::StageRecordCopy`) stages word 0 and words 2.. and leaves word 1 as it is, so the finalize
kernel sees the old light state; a new or reused record is copied whole (word 1 = 0).

### 2.1 Group block (in the light heap, section 4)

Every render group with light bricks owns one block. A group pass rebuilds it when any of its bricks
changed its light content (section 3); a rebuilt block recomputes only the stale bricks' records
and copies the others.

```
header 16 B         uint record_count; uint pad[3]   (LightBlockHeader; index entry 0 at
                    kLtBlockIndexOffset = 16)
index entries       [record_count] x 16 B, in Morton order of the brick position (LightIndexEntry):
                    uint pos (group-local brick position); uint dense_id;
                    uint record_offset (units from the block start);
                    uint bvh_leaf (tree << 31 | leaf index, written by the gather; 0xFFFFFFFF = none)
brick light records each starts at a unit, after the index entries
```

Brick light record (self-contained and position independent inside its group;
`BrickLightRecordHeader`):

```
0   float face[6]       face sums of the record's entries
24  uint box_lo         group-local voxels, x | y << 10 | z << 20, inclusive
28  uint box_hi         exclusive (tight box of the voxels with an entry; empty: lo = hi = brick corner)
32  uint pos            group-local brick position
36  uint entry_count
40  uint counts_lo      face counts of +X, -X, +Y (10 bits each)
44  uint counts_hi      face counts of -Y, +Z, -Z
48  uint entries[entry_count], zero-padded to a unit: the entries of face 0, then 1 .. 5, each family
    sorted by voxel index. Entry: voxel index in the brick (x | y << 3 | z << 6) in bits 0..8, the
    raw block value (as the finalized record holds it) in bits 16..31.
```

A group holds at most 4096 light bricks (`kLtMaxLightBricksPerGroup`); with `group_shift` 5 the
4096 lowest Morton positions are kept and the rest count as dropped
(`CaelixLightStats::bricks_dropped`).

`GroupLightRow` (32 B, by scene-wide group index, GPU buffer):

```
0   uint block_offset   units in the current heap buffer
4   uint block_units    0 = no block
8   uint generation     (life << 16) | version: life +1 when the group index is released, version +1
                        at every rebuild within one life (wraps at 2^16; LtNextLife, LtNextVersion)
12  uint gpu_flags      kLtGpuLightChanged: the finalize kernel saw a light change in the group
                        (atomic or)
16  uint cpu_flags      GPU-owned merge of the CPU's staged flags: kLtCpuRebuild, kLtCpuReleased,
                        kLtCpuFresh (a new life of the index: copy nothing from the old block)
20  uint bricks         light bricks (bits 0..15) | dropped light bricks << 16
24  uint light_faces
28  uint pad
```

The CPU never writes a row. Its flags go through a separate **staged flags** buffer (u32 per group,
0 = nothing new; one staged copy of the dense range of the listed groups per flush). The size kernel
merges them, `cpu_flags = (cpu_flags & fresh) | staged`, and clears the staged word, so a state
staged after a failed pass replaces rebuild / released but keeps a pending fresh. The build kernel
clears both flag words of the groups it processes. New rows and staged words (buffer growth) are
zero-filled.

### 2.2 The BVHs

Two world-space BVHs: tree 0 holds the light bricks of static entities, tree 1 those of dynamic
entities. Each has its own leaf array, node array, header and build state. They are rebuilt whole on
the GPU (section 3), so their depth follows the number of light bricks.

`LightBvhLeaf` (64 B): the summary `float face[6], lo[3], hi[3]` (world face sums, world AABB),
`uint group`, `uint entry` (index entry), `uint pad`, `uint parent` (node << 2 | slot). The world box
is the AABB of the rotated local record box under the entity's current transform; the world face
sums project the local ones, `W[+a] = sum_f F[f] * max(0, (R d_f)_a)` and `W[-a] = sum_f F[f] *
max(0, -(R d_f)_a)` (an upper bound of the cosine-weighted power for every direction; exact for
axis-aligned rotations).

`LightBvhNode` (224 B), children inline so that one node gives everything the child choice needs:

```
0    float child_face[4][6]
96   float child_lo[4][3]
144  float child_hi[4][3]
192  uint child[4]       bit 31: leaf index, else node index; 0xFFFFFFFF = empty slot
208  uint parent         0xFFFFFFFF = root
212  uint parent_slot
216  uint pad[2]
```

The slots are filled from 0. A node of a tree with two or more leaves has 2 .. 4 slots; the root of
a one-leaf tree holds the leaf in slot 0. The root is node 0.

Tree header (`LightTreeHeader`, 64 B, in the control buffer): the root summary `float face[6], lo[3],
hi[3]`, `uint root` (0xFFFFFFFF = empty tree), `uint leaf_count`, `uint node_count`, pad. Tree build
state (`LightTreeState`, 64 B): `building`, `leaves_found`, `leaves_dropped`, `builds`,
`node_overflow`, the bounds of the leaf box centres.

Capacities (CPU-managed, `LightTree::Prepare` / `Recycle`): leaves start at 1024 per tree and grow
to `max(1.5 x capacity, 1.5 x found)` (multiples of 64) when the light bricks found by a build exceed
75 % of the capacity or leaves were dropped; nodes are `leaves * 5 / 8 + 64` (a build over
Morton-ordered light bricks measured `node_count = leaves / 2`); a build that needs more nodes
leaves the tree empty, sets `node_overflow`, and the node capacity grows to the leaf capacity (the
bound) with a rebuild. Leaf and node buffers are copied on growth, so the old tree stays valid until
the rebuild. Memory per leaf of capacity: 64 B (leaf) + 140 B (nodes) + 88 B (build scratch, shared
by both trees).

Build scratch for a leaf capacity C: array X starts at byte `kLtScratchX * C` (keys and values of
the two sort buffers, the binary tree's left / right / parent, the parent of every sorted leaf, the
arrival counters, the even-depth flags, the binary node aggregates) and the radix sort histogram
follows at `kLtScratchHistogram * C` (`LightBvhScratchBytes`).

Limits: a group holds at most 4096 light bricks (above); a tree holds at most `kMaxDispatchGroups *
64` (4 194 240) leaves, one dispatch of the per-leaf kernels (`light_tree_record.cpp` `kMaxLeaves`;
the capacity never grows past it, the rest counts as dropped); the group kernels and the gathers run
in chunks of at most `kMaxDispatchGroups` workgroups, so the group count is the scene's (2^24, the
TLAS instance custom index); each heap buffer holds at most 2^32 units (u32 unit offsets,
`kLtMaxHeapUnits`; `CaelixLightOptions::heap_bytes` is checked against it, and growth stops there
with one error log).

### 2.3 Descriptor

The tree owns a 128-byte **descriptor** buffer (uploaded with `vkCmdUpdateBuffer` whenever a field
changes, for example when a buffer grows). Every kernel that uses the tree takes only its device
address: `FrameConstants::light_tree` (offset 544, 0 = no tree) for the render kernels,
`caelix_scene_get_light_tree` for host kernels and tests.

```
0   uint64 control        control buffer (section 2.4)
8   uint64 heap[2]        the two heap buffers
24  uint64 group_rows     GroupLightRow by group index
32  uint64 group_infos    GroupInfo (coord, entity)
40  uint64 entity_infos   EntityInfo (grid, transform)
48  uint64 entity_motion  u32 motion by entity id
56  uint64 grids          group grid cells
64  uint64 nodes[2]       LightBvhNode arrays
80  uint64 leaves[2]      LightBvhLeaf arrays
96  uint64 light_codes    65536-bit light code set
104 uint64 tree_headers   tree header 0 (tree 1 at + 64; inside the control buffer)
112 uint group_shift
116 uint encoding
120 uint pad
124 uint group_count      rows of the group passes (information for tests and hosts)
```

### 2.4 Control buffer (320 B)

`[0, 64)` the group pass (`LightControl`): `current`, `pass_result` (0 failed, 1 ok, 2 no change),
`needed_units`, pad, `light_groups`, `bricks_dropped`, `light_bricks` (u64), `light_faces` (u64),
`passes`, `failed_passes`, `pass_flags` (scratch of the size and scan kernels), `records_recomputed`,
`records_copied`, `trees_changed` (bit t: the last pass rebuilt or released a group of tree t).
`[64, 192)` the two tree headers, `[192, 320)` the two tree build states. The 64-bit counters stay
8-byte aligned. The whole buffer is copied to a host readback slot after every light update and
read in `Recycle` once its frame is safe.

## 3. Updates (inside `caelix_scene_flush`)

The light work runs after the finalize kernel and its barrier, inside the flush timer, with its own
GPU timer (`CaelixLightStats::gpu_ms`: group pass + gathers + builds). It is recorded only in a
flush whose scene tables are consistent (the scene's staging did not run out of ring space) and
whose light staging fit (else everything waits for the next flush).

1. **Finalize** (existing kernel; light work only with the tree enabled): per voxel, a light voxel
   counts its open faces (the same neighbour values the kernel already has). The brick's total goes
   into record word 1 bits 18..29. Thread 0 reads the old word 1 first (an upload that replaced the
   brick kept it); when the old or the new count is non-zero, it sets the stale bit (a set stale bit
   is kept) and `kLtGpuLightChanged` in the group's `gpu_flags`.
2. **Group pass**, when the flush finalized bricks, staged group flags (created, released, bricks
   removed) or a retry is pending. Four kernels over every group index `[0, group capacity)`:
   - `size`: merges the staged flags; released -> 0; flagged (cpu or gpu) -> the size of the rebuilt
     block (light brick scan + record plans, the same code as `build`); otherwise the current
     `block_units`. A released or flagged group sets `kLtPassFlagChanged` in `pass_flags` and the
     flag of the tree of its entity's motion.
   - `scan` (one workgroup): exclusive prefix sum of the sizes = the offsets in the target heap
     buffer (`1 - current`); the total; `pass_result` = no change (nothing flagged), ok (total <=
     capacity) or failed.
   - `build` (one workgroup of 256 per group; returns at once unless ok): released -> the row is
     cleared (next life); flagged -> the block is rebuilt (next version); otherwise the block
     is copied to its new offset. Every processed group writes its row and clears its flags.
   - `commit` (one workgroup): ok -> `current = 1 - current`, totals for the stats, passes + 1;
     failed -> passes + 1, failed_passes + 1.
   Rebuild of a flagged group:
   1. Scan the dense ids `[0, high_water)`: resident slot -> record word 1 -> light face count. A
      bit per light brick in a shared Morton bitmap; prefix popcounts give every light brick its
      Morton rank = its index entry (no sort).
   2. Plan per light brick: the record is **recomputed** when the stale bit is set, the group has no
      current block, the group is fresh, or no old index entry has its position (binary search over
      the old entries, which are in Morton order); otherwise it is **copied** from the old block. A
      recomputed record has room for the finalize kernel's count; a copied one keeps its size. The
      stale bit of every resident brick of the group is cleared (render kernels run after the flush
      and mask the bounds). On a failed pass nothing is cleared, so the retry recomputes the same
      bricks.
   3. One wave per light brick writes its index entry and its record. A recomputed record resolves
      the six neighbour records (`scene/caelix_scene_lookup.hlsl`, shared with the finalize kernel),
      finds the open faces of its light voxels in two passes over the voxels (counts, then entries;
      lane `l` owns voxels `[l k, (l + 1) k)`, so each family comes out sorted by voxel) and writes at
      most the finalize kernel's count of entries (they differ only while a neighbour's change waits
      for its finalization, which marks the record stale again).
   4. The row: offset, units, counts, the next version of the generation.
3. **Gather and build per tree**, recorded for a tree when the CPU asks for it (an entity of the
   tree moved, an entity changed its motion or was removed, a group of the tree's entities was
   released, a capacity grew) or when a group pass is recorded. The first kernel (`begin`) decides on
   the GPU: the tree is rebuilt when the CPU asked or when this flush's group pass committed with a
   rebuilt or released group of the tree (`trees_changed`). The other tree's leaves stay valid: they
   name (group, index entry) pairs, and the pass only copied those blocks (with their `bvh_leaf`).
   Every later kernel reads the decision and returns at once without it. Kernels:
   - `gather_count` (one thread per group, in dispatch chunks): the record count of the group's
     block when its entity has the tree's motion (and no release is pending), else 0; `bvh_scan`
     (one workgroup, generic exclusive scan) -> leaf offsets and the total; `gather_emit` (one thread
     per group, chunked): the leaves (dropped above the capacity) and `bvh_leaf` of the index entries
     of the CURRENT heap buffer.
   - `bvh_bounds` (one workgroup): the header's `leaf_count` = min(found, capacity) and the state's
     `leaves_dropped` (here, so that they are written also when the gather dispatched nothing), then
     the bounds of the leaf box centres; `bvh_morton`: 30-bit Morton codes of the centres, key
     (code, leaf index); the arrival counters are cleared.
   - Radix sort: stable LSD, 4 bits per pass, 8 passes: `sort_histogram` (digit counts per block of
     1024, digit-major), `bvh_scan`, `sort_scatter` (stable: per round of 256 elements the rank among
     equal digits comes from 16 wave prefix ballots plus the counts of the earlier waves and rounds).
   - `bvh_karras`: the binary radix tree (Karras 2012), one thread per internal node; equal codes are
     ordered by position (so every key is distinct).
   - `bvh_aggregate`: bottom-up, one thread per leaf; at every node the first arriving thread stops
     (acquire/release atomic counter), the second computes the node summary (sum of the face sums,
     union of the boxes) and goes on. Node summaries are written and read with coherent atomic word
     accesses.
   - Collapse to 4 wide: `bvh_depth` keeps the binary nodes at even depth (root depth 0), `bvh_scan`
     numbers them; `bvh_wide`: each kept node's slots are the children of its two children (a leaf
     child takes one slot, an internal child gives its two children), summaries inlined, parent
     links written for its slots. `bvh_finish`: the header (one-leaf and empty trees handled here),
     overflow check, builds + 1.
   The two trees build one after the other into the shared scratch.

The material code set changes when the materials or `min_emission` change; when it changes while
bricks are resident, every resident brick is queued for finalization again (cost of a reload; the
host should set materials and light options before uploads). Emission values that stay on the same
side of the threshold need nothing: the sample's radiance comes from the material table at draw
time, so stale weights only change the sampling quality.

Costs: moving a static entity rebuilds the whole static BVH (section 7: 4.7 ms at 1M light
bricks, 0.25 ms at 50k); an edit of a static entity's lights rebuilds the static BVH, of a dynamic
entity's lights the dynamic one; a streaming load rebuilds the static BVH at every flush that
commits a group pass.

## 4. Light heap

Two device-local buffers of the same capacity (default 16 MiB each, `CaelixLightOptions::heap_bytes`).
Every group pass with changes compacts all live blocks from the current buffer into the other one
(copy of the unchanged blocks, build of the flagged ones), so there is no allocator, no
fragmentation, and a pass either commits completely or changes nothing. A pass without a flagged or
released group (finalized bricks without light changes) copies nothing and keeps the buffer.

Growth: the CPU reads the control buffer back once the frame is safe. When `needed_units` exceeds 75
% of the capacity or the pass failed, it allocates two buffers of `max(2 x capacity, 1.5 x needed)`,
copies both old buffers into them (the CPU does not know which one is current), updates the
descriptor, and runs a group pass at the next flush. Every failed pass schedules that retry (its
flags are still set on the GPU), also one read back after the heap has grown since; only the growth
request of a result that belongs to a heap that has grown since is skipped (an epoch per slot).

Transient case after a failed pass: until the retry commits, the rows and blocks are the ones before
the failed pass. When a group index is released and starts a new life in that window (another entity
creates a group that reuses it), its old block stays in place and is gathered into the tree of the
new entity's motion with the new entity's transform; the retry then rebuilds the group (fresh: every
record recomputed) and the trees.

## 5. Device API (`shaders/scene/caelix_light_tree.hlsl`)

Includable by every kernel; it reads only the descriptor and the buffers it points to. All
positions and normals are world space. `n` is the receiver's unit normal, or 0 for a receiver
without a surface (a volume point).

```hlsl
struct LightTree { uint64_t descriptor; uint64_t heap; uint64_t headers; uint group_shift; };
bool LightTreeOpen(uint64_t descriptor, out LightTree tree);   // false: no tree, or no lights

struct LightBrickRef { uint group; uint entry; uint generation; };  // entry = index entry
struct LightPoint
{
    float3 position;     // on the light face (uniform on the unit square)
    float3 normal;       // the face's outward normal
    uint block;          // raw block value of the light voxel (material lookup by the caller)
    uint voxel;          // voxel index in the brick (x | y << 3 | z << 6)
    uint face;           // 0..5
    float pdf;           // area measure: P(brick | x, n) * P(face | brick, x, n) / face_count
    LightBrickRef brick;
};

bool LightTreeSample(LightTree t, float3 x, float3 n, inout uint rng, out LightPoint p);
bool LightTreeSampleBrick(LightTree t, float3 x, float3 n, inout uint rng,
                          out LightBrickRef brick, out float pdf_brick);
bool LightBrickSamplePoint(LightTree t, LightBrickRef brick, float3 x, float3 n, inout uint rng,
                           out LightPoint p);              // p.pdf = P(face | brick) / face_count
float LightTreeBrickPdf(LightTree t, float3 x, float3 n, LightBrickRef brick);
float LightTreePointPdf(LightTree t, float3 x, float3 n, uint group, uint3 voxel_in_group,
                        uint face, uint block);            // 0 when it is not a light face
bool LightBrickRefValid(LightTree t, LightBrickRef brick); // generation check
uint3 LightBrickVoxelOrigin(LightTree t, LightBrickRef brick);  // group-local voxel of the brick
                                                                // corner (with LightPoint::voxel:
                                                                // the light voxel, for NEE code)
float LightTreeImportance(float3 x, float3 n, float3 lo, float3 hi, float face[6]);  // section 5.1
uint LightTreeRandom(inout uint rng);       // PCG
float LightTreeRandomFloat(inout uint rng); // [0, 1)
```

`rng` is a PCG state owned by the caller; the API consumes a variable number of steps. A brick
reference stays valid while its group is not rebuilt (`LightBrickRefValid`).

Additions for the light NEE (docs/LIGHT_NEE.md section 5, 2026-10-24):

```hlsl
float LightTreeImportanceBox(float3 r_lo, float3 r_hi, float3 n, float3 lo, float3 hi, float face[6]);
bool LightTreeSampleBrickBox(LightTree t, float3 box_lo, float3 box_hi, float3 n, inout uint rng,
                             out LightBrickRef brick, out float pdf_brick, out uint leaf,
                             out float importance);  // leaf = tree << 31 | index; importance = the
                                                     // box importance of the leaf's summary
float LightTreeBrickPdfBox(LightTree t, float3 box_lo, float3 box_hi, float3 n, LightBrickRef brick);
float LightTreeLeafImportanceBox(LightTree t, LightBrickRef brick, float3 box_lo, float3 box_hi,
                                 float3 n);           // from the leaf array; 0 without a leaf
bool LightTreeFindLightFace(LightTree t, float3 x, uint group, uint3 voxel_in_group, uint face,
                            uint block_value, out LightBrickRef brick, out uint leaf,
                            out float in_brick);      // in_brick = P(face | brick, x) / face_count
void LightTreeSelect(inout LightTree t, uint trees);  // kLtSelectStatic / Dynamic / Both (default)
LightTree LightTreeView(uint64_t descriptor);         // LightTreeOpen without its check for lights
```

The box receiver uses the importance of section 5.1 with the receiver box R in place of the point:
the emitter cosines over D = [R.lo - hi, R.hi - lo]; `m = n . (c_B - c_R) + |n| . (h_B + h_R)` (the
maximum of n . (y - x) over both boxes) and `dmin` the distance between the boxes; the distance term
`max(|c_R - c_B|^2, max(1, (|h_R| + |h_B|)^2))`. Every term is a bound over the receiver points of the
box, so a light face that can reach one of them has a positive importance at every node above it.
One implementation (`LtImportance` with a constant `is_point`) serves both receivers, and the walk
functions take the receiver as `LtReceiver` (box) with the same constant; the point functions keep
their signatures. The light tree probe's SPIR-V is not byte-identical to the one before (ids, the
order of constants and local variables, one swapped block pair), with the same instructions; no
library kernel includes this file. LightTreePointPdf is LightTreeFindLightFace times the brick
probability. The selector makes an unselected tree count as empty (importance and power 0), so the
pdfs are those of draws over the selected trees.

Draw: the tree by the importance of the two root summaries, then down the 4-wide nodes by the
importance of the inline slots, to a leaf; the leaf gives the group and the index entry, the entry
the record. The face family, the voxel and the point use the record in the entity-local frame
(`x' = R^T (x - t)` from EntityInfo `cur_row*`, then group-local by subtracting the group origin).

Pdf: `LightTreePointPdf` finds the brick's index entry by binary search of the Morton code of its
position, then the face entry by binary search inside the face family. The brick probability walks
from the entry's `bvh_leaf` up the parent links: at every node the probability of the slot, times
the tree choice at the top. The leaf must point back at the same (group, entry), else the pdf is 0.

### 5.1 Importance of a summary

For a box `[lo, hi)` and face sums `F` in the receiver's frame:

```
D_lo = x - hi; D_hi = x - lo                      // box of the difference vectors x - y
cmax(+a) = D_hi[a] > 0 ? D_hi[a] / sqrt(D_hi[a]^2 + cb^2 + cc^2) : 0
cmax(-a) = D_lo[a] < 0 ? -D_lo[a] / sqrt(D_lo[a]^2 + cb^2 + cc^2) : 0
           with cb = clamp(0, D_lo[b], D_hi[b]) for the two other axes b, c
E        = sum_f F[f] * cmax(f)                    // exact maximum of the emitter cosine per family
m        = dot(n, centre - x) + dot(abs(n), half)  // maximum of n . (y - x) over the box
cr       = n == 0 ? 1 : (m <= 0 ? 0 : (dmin > 0 ? min(1, m / dmin) : 1))
d2       = max(|x - centre|^2, max(1, |half|^2))
I        = E * cr / d2
```

`I > 0` whenever any light face below the summary can reach the receiver (both cosines are bounds),
so the estimator stays unbiased. A child is drawn with probability `I_child / sum I` (one-pass
weighted reservoir, one random number per child, `LtPick*`). When every child's importance is 0 the
draw falls back to `power_child / sum power` (power = sum of the face sums; the same random numbers
feed both reservoirs): a draw then always reaches a light face, and the pdfs of all light faces sum
to 1 for every receiver (the normalisation test checks that). A node with one child is taken with
probability 1 and computes no importance. The pdf functions use the same functions on the same
data, so the pdf of a drawn sample equals the evaluated pdf up to rounding (the product order
differs).

Inside a record, the face family is drawn with `P(f) = F[f] * cmax(f) / sum` over the record box
(fallback `F[f] / sum F`), the voxel uniformly among the family's entries, the point uniformly on
the unit face.

## 6. Host API (C ABI additions, no version bump)

```c
typedef struct CaelixLightOptions
{
    uint32_t enabled;      /* 0 = no light tree (default), 1 = build it */
    float min_emission;    /* light voxel threshold, luminance of the emission before emission_scale.
                              0 = 20. Must be finite and >= 0 */
    uint64_t heap_bytes;   /* initial capacity of each heap buffer (rounded up to 16 B). 0 = 16 MiB.
                              Read when the tree is created; at most 2^32 x 16 bytes */
} CaelixLightOptions;      /* 16 bytes */
CaelixResult caelix_scene_set_light_options(CaelixScene*, const CaelixLightOptions*);

typedef struct CaelixLightStats   /* the newest light update read back; 104 bytes */
{
    uint32_t enabled, light_groups;
    uint64_t light_bricks, light_faces, bricks_dropped, heap_bytes_used, heap_bytes_capacity;
    uint32_t passes, failed_passes, records_recomputed, records_copied;
    float gpu_ms;
    uint32_t pad;
    uint32_t tree_leaves[2], tree_nodes[2], tree_builds[2], tree_leaves_dropped[2];
} CaelixLightStats;
void caelix_scene_get_light_stats(const CaelixScene*, CaelixLightStats*);
uint64_t caelix_scene_get_light_tree(const CaelixScene*);  /* descriptor address, 0 when disabled */

#define CAELIX_ENTITY_STATIC 0u   /* default */
#define CAELIX_ENTITY_DYNAMIC 1u
CaelixResult caelix_scene_set_entity_motion(CaelixScene*, CaelixEntityId, uint32_t motion);
```

Disabling releases the tree's buffers once the last flushed frame is safe; enabling again starts
from scratch (every resident brick is finalized again). EntityInfo word 7 (`kEntityInfoLife`, the
former pad) holds the entity's life: +1 every time `caelix_scene_add_entity` hands its id out again
(the light cells key on it). Entity motion is a general entity flag;
today only the light tree reads it. Renderer: `FrameConstants::light_tree` (offset 544; the struct is
560 bytes). No render kernel uses it in this step.

Rust host (`caelix-app`): `--light-tree`, `--light-min-emission <f>`; the Stats window (F1) shows a
"Light tree" section; `--timing-frames` prints the light stats on the `timings_avg:` line, plus
`flush_ms` and the load's `load_flush_ms` / `load_flush_max_ms`. The test entity is dynamic.

## 7. Validation and measurements

`caelix_light_tree_test` (Catch2, GPU, the public C ABI plus the probe kernel
`tests/shaders/caelix_light_tree_probe.hlsl`; the radix sort case runs the library's own sort
kernels from `build/shaders`; the test reads GPU memory through the library's layout headers):
- Structure against a CPU reference of every light face (exposure across bricks and groups, block-ID
  with host materials above and below the threshold and an emissive transparent id, and the packed
  encoding): rows, block headers, index entries (Morton order, positions, dense ids), records (face
  sums, boxes, counts, sorted entries, block values), `bvh_leaf` -> leaf -> entry, the group's entity
  (GroupInfo), every leaf of both BVHs reached exactly once, parent links, node slot sums within 1e-5
  and boxes, headers, stats.
- Normalisation (sum of `LightTreePointPdf` over all light faces = 1 within 1e-4, receivers with and
  without a normal, one next to a light face), sample/pdf consistency (1e-4), the drawn point on its
  face, a histogram of 2M draws against `N * pdf` (5 sigma per face), the bound property (4096
  random receiver/face pairs with both cosines above 0.05 have a positive pdf).
- Updates: light bricks added inside a group (only that group's generation changes) and across a
  group border, removed; `min_emission` up and down; an entity moved and rotated by 37 degrees; an
  entity removed; a record-copy case (one edited brick: exactly the edited brick and its re-finalized
  light neighbours are recomputed, the other records are byte-identical copies; the stats count
  them); re-uploaded bricks (an isolated light brick re-uploaded as stone and air leaves the tree,
  only through its own kept word 1; a light brick with a new light layout is recomputed, the other
  copied; in one window a new brick uploaded twice, a resident brick replaced then removed while a
  new brick takes its record, a brick replaced twice); enabling after the upload; static/dynamic: a
  dynamic entity moved for 5 frames (tree 1 rebuilt 5 times, tree 0 not), a static entity moved
  (tree 0 rebuilt once), motion toggles; heap growth from a 4 KiB heap (the first pass fails, later
  passes succeed); radix sort against `std::sort` (1, 2, 1000, 100k keys with duplicates); argument
  checks. Zero validation and library errors.

`caelix_frame_layout_test` checks the push constant blocks of every scene and light tree kernel
against the C++ push structs by member name, and every named constant of the HLSL layout headers
against the C++ constant of the same name.

Measured on the RTX 5080, driver 610.88 (2026-10-24).

`[.bench]` (synthetic: groups of 67 light bricks with 4 lamp voxels each):

| light bricks | groups | static BVH gather + build | pass, one group rebuilt + BVHs | pass, every group rebuilt + BVHs | LightTreeSample |
|---|---|---|---|---|---|
| 10 050 | 150 | 0.16 ms | 0.27 ms | 0.30 ms | 0.95 ns / draw |
| 53 600 | 800 | 0.25 ms | 0.35 ms | 0.70 ms | 1.48 ns / draw |
| 1 005 000 | 15 000 | 4.66 ms | 5.03 ms | 11.1 ms | 2.13 ns / draw |

Draws: 2^20 threads x 16 draws for 64 receivers. 1M light bricks: 24M light faces, heap 161 MB of
blocks, 502k BVH nodes.

Rust host smoke runs (1920x1080, `--timing-frames 100`, 100-frame means; `flush_ms` at rest is 0):

| scene | light bricks / faces / groups | heap used / capacity | trace without / with | load flush mean / max, without -> with | last light update |
|---|---|---|---|---|---|
| bistro 8192 packed (pose 3523.0,479.8,4912.0,-1.119,-127.260) | 2062 / 153 771 / 149 | 0.76 / 16.8 MB | 4.95 / 4.93 ms | 1.76 / 5.34 -> 1.90 / 5.54 ms | 0.28 ms |
| meridian-deep block-ID | 1 004 057 / 23 455 257 / 15 306 | 161 / 268 MB | 4.06 / 4.02 ms | 1.93 / 5.58 -> 3.56 / 11.0 ms | 5.4 ms |

The trace kernel does not use the tree yet: its time is unchanged (the differences are run noise,
about 0.15 ms between runs of one binary). On meridian-deep the load pays for a static BVH rebuild
at every flush that commits a group pass (128 builds during the load). The last light update is the
newest one of the load.

## 8. Implementation notes

- BVH atomics and BDA atomics: DXC 1.8 has no `vk::BufferPointer`; `caelix_platform.hlsl` declares
  OpConvertUToPtr + OpAtomic* through inline SPIR-V (`CAELIX_ATOMIC_*`).
- The workgroup prefix sums of the group pass and the BVH scan are one implementation
  (`shaders/common/caelix_workgroup_scan.hlsl`).
- The named offsets and bit fields that `shaders/common/caelix_record.hlsl` and
  `caelix_material.hlsl` added for the light tree are macros: a new static const in those headers,
  which every render kernel includes, reorders the SPIR-V constants of kernels that never use it
  (no code change, but the trace kernel's binary stays byte-identical with macros).

## 9. Files

- `lib/src/scene/light_tree.h`, `light_tree.cpp` (hooks, code set, Prepare, Stage),
  `light_tree_record.cpp` (transfers, group pass, gather and build recording, readback, stats),
  `light_tree_layout.h` (GPU layouts, named offsets, push structs), `scene_light.cpp` (the Scene
  side: options, stats, requeue), `staged_copy.h`.
- `shaders/common/caelix_light_tree_layout.hlsl` (layouts and their readers),
  `shaders/common/caelix_workgroup_scan.hlsl`, `shaders/scene/caelix_scene_lookup.hlsl` (record and
  neighbour lookups shared with the finalize kernel), `caelix_light_tree_group.hlsl` +
  `caelix_light_tree_{size,scan,build,commit}.hlsl` (group pass), `caelix_light_bvh_common.hlsl`,
  `caelix_light_bvh_sort.hlsl`, `caelix_light_bvh_{begin,scan,bounds,morton,sort_histogram,
  sort_scatter,karras,aggregate,depth,wide,finish}.hlsl`, `caelix_light_gather_{count,emit}.hlsl`
  (gather and build), `caelix_light_tree.hlsl` (device API).
- `tests/light_tree_test.cpp`, `tests/shaders/caelix_light_tree_probe.hlsl`,
  `tests/frame_layout_test.cpp` (layouts).
