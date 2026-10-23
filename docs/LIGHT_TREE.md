# Light tree (design, 2026-10-24)

Status: design for the implementation on `main`. After the implementation, this file becomes the
"as built" truth; the implementer updates it where the code differs.

Owner decision (2026-10-23): a light BVH at brick granularity, plus the per-cell sampling reuse
(the cells and the NEE are the owner's own work on top of this). Reasons: volumetrics need a
receiver-weighted light draw with an exact pdf at any world point, and many moving lights break a
per-cell cut. Background: `docs/human_wrote/Caelix-Renderer Towards A Million Stars.md`, the
2026-10-23 discussion summarised in the family `CLAUDE.md` ("Light sampling bench", "Lamp
visibility", "Crimson-crown"), `Artifacts/light-flicker-crimson-20261023/design-proposal-A.md`
section 2.2 (the bounds).

Scope of this step: the scene side (light bricks, the tree, its updates) and the device API (draw a
light point, evaluate its pdf). No cells, no NEE, no MIS in the trace kernel.

## 1. Terms

- **Light voxel**: a voxel whose material emission luminance, `dot(emission, (0.2126, 0.7152,
  0.0722))` before `CaelixRenderOptions::emission_scale`, is at least `min_emission` (default 20).
  The scene keeps a 65536-bit **light code set** (8 KiB, GPU) indexed by the material code
  (`CaelixMaterialCode`: transparent codes without their face bits).
- **Light face**: face `f` (+X 0, -X 1, +Y 2, -Y 3, +Z 4, -Z 5) of a light voxel whose neighbour
  voxel in direction `f` is not opaque. The neighbour can be in another brick or group of the same
  entity (resolved like the finalize kernel; an absent brick is air, so the face is open).
- **Light brick**: a brick with at least one light face.
- **Face weight**: the luminance of one light face (area 1: one voxel = one world unit, entities have
  no scale). A node's `face[f]` is the sum of the face weights of family `f` below it.

## 2. Structure

Three levels, the same split as the ray tracing BLAS/TLAS:

```
top tree (world space, CPU topology, GPU refit)        one leaf per entity with groups
  -> entity pyramid (entity-local, dense octree over    levels 1 .. L of the entity's group grid
     the group grid, GPU refit)
     -> group (GroupLightRow summary)                   level 0 = one render group
        -> group octree (sparse, built per group)       group_shift levels inside the group
           -> leaf = one light brick                    face lists: one entry per light face
```

### 2.1 Group level: the light block

Every render group with light bricks owns one **light block** in the light heap (section 4). A
group's block is rebuilt whole when any of its bricks changes its light content (section 3). The
block holds a sparse octree over the group's light bricks:

```
unit 0     header: uint node_count; uint leaf_first; uint face_area (byte offset from the block
           start); uint face_count
units 1..  LightNode[node_count], 48 B each (3 units)
           order: root (index 0), then the level group_shift-1 nodes, ..., the level 1 nodes, then the
           leaves (index leaf_first ..); every level in Morton order of its cell, so the children of a
           node are contiguous
then       uint face entries[face_count]: per leaf, the entries of face 0, then face 1, ... face 5
```

`LightNode` (48 B):

```
0   float face[6]   face sums
24  uint box_lo     group-local voxel coordinates, inclusive: x | y << 10 | z << 20
28  uint box_hi     exclusive, same packing (tight box of the light voxels below the node)
32  uint child      internal: first child index (bits 0..23) | (child_count - 1) << 24 (bits 24..26)
                    leaf: bit 31 set, dense brick id in bits 0..15
36  uint faces      leaf: index of the leaf's first face entry (from the face area)
40  uint counts_lo  leaf: face counts of +X, -X, +Y (10 bits each)
44  uint counts_hi  leaf: face counts of -Y, +Z, -Z
```

Face entry (4 B): voxel index `x | y << 3 | z << 6` in bits 0..8, the raw block value in bits
16..31 (the caller reads the material from it; no record lookup at sampling time).

The octree is not collapsed: a node with one child is kept (the traversal takes it with probability
1 and computes no importance). Internal levels: `group_shift` (4 for 16^3 groups: root covers 16
bricks, then 8, 4, 2; leaves are single bricks). A group holds at most `kMaxLightBricksPerGroup` =
4096 light bricks (every brick of a 16^3 group). With `group_shift` 5, the 4096 lowest Morton
positions are kept and the rest count as dropped (`CaelixLightStats::bricks_dropped`).

`GroupLightRow` (64 B, by scene-wide group index, GPU buffer):

```
0   float face[6]       the root node's face sums (0 = no lights)
24  uint box_lo, box_hi the root node's box (group-local voxels)
32  uint block_offset   in 16-B units, in the current heap buffer
36  uint block_units    0 = no block
40  uint generation     +1 at every rebuild or release of the group's block
44  uint gpu_flags      bit 0: the finalize kernel saw a light change in the group (InterlockedOr)
48  uint cpu_flags      staged by the CPU: bit 0 force a rebuild (bricks removed, new group),
                        bit 1 released
52  uint light_bricks
56  uint light_faces
60  uint pad
```

The CPU writes only `cpu_flags` (one 4-byte staged copy). The GPU clears both flag words when it
has processed them. New rows (buffer growth) are zero-filled.

### 2.2 Entity level: the pyramid

Every entity with a group grid owns a dense **pyramid** of `LightPyramidNode` (48 B: `float
face[6]; float lo[3]; float hi[3]`, entity-local voxel coordinates, empty = all faces 0) in a
`RowPool`. Level `L` (1 .. `levels`) has `max(1, grid_dims >> L)` nodes per axis; node `c` at level
`L` has the children `2c + (i & 1, i >> 1 & 1, i >> 2 & 1)` at level `L - 1` that lie inside the
level's dims. Level 0 is the group grid itself: the child is the group in that grid cell
(`kNoGroup` = empty), read through its `GroupLightRow` (box converted with the group origin
`coord * (8 << group_shift)`). `levels = log2(max(grid_dims))`; 0 means a one-cell grid whose
single group is the entity root. Storage order: level 1 first, root last; the offset of level
`L - 1` is the offset of level `L` minus the node count of level `L - 1`.

The pyramid has the same lifetime as the group grid: `RebuildGrid` (growth) reallocates it and
refits every node. Memory is about 1.7 times the grid's own memory (48 B per 7 cells against 4 B
per cell).

`EntityLightRow` (16 B, by entity id): `pyramid_base` (row), `levels`, `top_leaf` (index in the top
tree, `0xFFFFFFFF` = none), pad.

Refit: after the group pass (section 3), one dispatch per level `L = 1 .. max levels`, one thread
per listed node; the CPU lists the ancestors of every candidate group of the flush (groups with
finalize entries, removed bricks, creation or release) and every node of a new pyramid. A node is
the sum of its children's face sums and the union of their boxes.

### 2.3 Top tree

The CPU builds the topology over the alive entities that have a group grid: leaves sorted by the
Morton code of the entity's world position (quantised over the positions' bounds), parents =
consecutive runs of 8, up to one root. It rebuilds it when an entity is added or removed, gets its
first grid, or moves. `TopNode` (64 B):

```
0   float face[6]   world-space face sums
24  float lo[3]
36  float hi[3]     world AABB
48  uint child      internal: first child index; leaf: entity id
52  uint info       bit 31 leaf; bits 0..2 child_count - 1
56  uint parent     0xFFFFFFFF = root
60  uint pad
```

The CPU stages the nodes with zero aggregates; a refit kernel (one workgroup, levels bottom-up)
fills them. A leaf is the entity root (pyramid root, or the single group) moved to world space:
the box is the AABB of the rotated local box; the world face sums project the local ones,
`W[+a] = sum_f F[f] * max(0, (R d_f)_a)` and `W[-a] = sum_f F[f] * max(0, -(R d_f)_a)` (an upper
bound of the cosine-weighted power for every direction; exact for axis-aligned rotations). The
refit runs when the top topology was staged, an entity moved, or the group pass ran.

### 2.4 Descriptor

The scene owns a 128-byte **light tree descriptor** buffer (staged when a field changes). Every
kernel that uses the tree takes only its device address: `FrameConstants::light_tree` (offset 544,
0 = no tree) for the render kernels, `caelix_scene_get_light_tree` for host kernels and tests.

```
0   uint64 control       control buffer (section 4)
8   uint64 heap[2]       the two heap buffers
24  uint64 group_rows    GroupLightRow by group index
32  uint64 group_infos   GroupInfo (coord, entity)
40  uint64 entity_infos  EntityInfo (grid, transform)
48  uint64 entity_rows   EntityLightRow by entity id
56  uint64 grids         group grid cells
64  uint64 pyramids      pyramid row pool base
72  uint64 top_nodes     TopNode array
80  uint64 light_codes   65536-bit light code set
88  uint top_root        0xFFFFFFFF = no tree yet
92  uint group_shift
96  uint encoding
100 float min_emission
104 .. 127 pad
```

## 3. Updates (inside `caelix_scene_flush`)

1. **Finalize** (existing kernel, light tree enabled only): per voxel, a light voxel counts its open
   faces (the same neighbour values the kernel already has). The brick's total goes into record
   word 1 bits 18..29 (12 bits; the tight bounds use bits 0..17, readers mask them). Thread 0 reads
   the old count before it writes; when the old or the new count is non-zero, it sets bit 0 of the
   group's `gpu_flags`.
2. **Group pass**, when the flush finalized bricks, removed bricks, created or released groups, or
   a retry is pending (section 4). Four kernels over every group index `[0, group capacity)`:
   - `size`: released -> 0; flagged (cpu or gpu) -> the size of the rebuilt block (light brick scan
     + node counts, the same shared-memory code as `build`); otherwise the current `block_units`.
   - `scan` (one workgroup): exclusive prefix sum of the sizes = new offsets in the target heap
     buffer (`1 - current`); the total; `ok = total <= capacity`.
   - `build` (one workgroup per group; returns at once when `!ok`): released -> clear the row
     (generation + 1); flagged -> rebuild the block at its new offset (generation + 1); otherwise
     copy the old block to its new offset. Every group writes its row (offset, units, summary,
     counts) and clears its flags.
   - `commit` (one thread): when `ok`, `current = 1 - current`; totals for the stats; the control
     header goes to this frame's readback slot.
   When `!ok`, nothing changes on the GPU: the rows still point into the current buffer, the flags
   stay set, and the next pass retries them.
3. **Pyramid refit** (section 2.2).
4. **Top refit** (section 2.3).

The light kernels run after the finalize kernel and its barrier, inside the flush timer, with their
own GPU timer (`CaelixLightStats::gpu_ms`).

Rebuild of a flagged group (`build`): one workgroup (256 threads).
1. Scan the dense ids `[0, high_water)`: resident slot -> record word 1 -> light face count. Set a
   bit per light brick in a shared Morton bitmap; a prefix popcount gives every light brick its
   Morton rank (no sort).
2. Node counts per level from the sorted Morton codes; the block size.
3. Leaves: one wave per light brick resolves the six neighbour records (shared helper with the
   finalize kernel), finds the light faces of its 512 voxels, writes the face entries (wave prefix
   counts per family) and the leaf node (face sums, tight voxel box, counts, dense id).
4. Internal nodes bottom-up, one level at a time (device memory barrier between levels): sum and
   union of the children.
5. The row: root summary, offset, units, counts, generation + 1.

The material code set changes when the materials or `min_emission` change; when it changes while
bricks are resident, every resident brick is queued for finalization again (cost of a reload; the
host should set materials and light options before uploads). Emission values that stay on the same
side of the threshold need nothing: the sample's radiance comes from the material table at draw
time, so stale weights only change the sampling quality.

## 4. Light heap

Two device-local buffers of the same capacity (default 16 MiB each, `CaelixLightOptions::heap_bytes`).
Every group pass compacts all live blocks from the current buffer into the other one (copy of the
unchanged blocks, build of the flagged ones), so there is no allocator, no fragmentation, and a pass
either commits completely or changes nothing.

Control buffer (64 B): `current`, `ok`, `needed_units`, `capacity_units`, `light_groups`,
`light_bricks` (64-bit), `light_faces` (64-bit), `bricks_dropped`, `passes`, `failed_passes`.

Growth: the CPU reads the control header back once the frame is safe. When `needed_units` exceeds
75 % of the capacity or the pass failed, it allocates two buffers of `max(2 x capacity, 1.5 x
needed)`, copies both old buffers into them (the CPU does not know which one is current), updates
the descriptor, and runs a group pass at the next flush (retry) followed by a refit of every
pyramid node.

Cost: a pass copies every unchanged block. On a meridian-deep class scene (1M light bricks, about
100 MiB of blocks) that is about 0.3 to 0.5 ms per flush with light changes; small scenes pay
microseconds. If edits every frame on such scenes need it, a later step can add in-place rebuilds
with a bump region and compact only when the holes grow.

## 5. Device API (`shaders/scene/caelix_light_tree.hlsl`)

Includable by every kernel; it reads only the descriptor and the buffers it points to. All
positions and normals are world space. `n` is the receiver's unit normal, or 0 for a receiver
without a surface (a volume point).

```hlsl
struct LightTree;                          // the descriptor address and the fields used on the hot path
bool LightTreeOpen(uint64_t descriptor, out LightTree tree);   // false: no tree, or no lights

struct LightBrickRef { uint group; uint leaf; uint generation; };
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

// Draws a light brick for the receiver, then a point on it.
bool LightTreeSample(LightTree t, float3 x, float3 n, inout uint rng, out LightPoint p);
// The two halves, for reservoirs that store bricks.
bool LightTreeSampleBrick(LightTree t, float3 x, float3 n, inout uint rng,
                          out LightBrickRef brick, out float pdf_brick);
bool LightBrickSamplePoint(LightTree t, LightBrickRef brick, float3 x, float3 n, inout uint rng,
                           out LightPoint p);              // p.pdf = P(face | brick) / face_count
// pdfs for MIS and resampling.
float LightTreeBrickPdf(LightTree t, float3 x, float3 n, LightBrickRef brick);
float LightTreePointPdf(LightTree t, float3 x, float3 n, uint group, uint3 voxel_in_group,
                        uint face, uint block);            // 0 when it is not a light face
bool LightBrickRefValid(LightTree t, LightBrickRef brick); // generation check
```

`rng` is a PCG state owned by the caller; the API consumes a variable number of steps.

### 5.1 Importance of a node

For a node with box `[lo, hi)` and face sums `F` in the receiver's frame (world for the top tree,
entity-local below an entity leaf: `x' = R^T (x - t)`, `n' = R^T n`):

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

`I > 0` whenever any light face below the node can reach the receiver (both cosines are bounds),
so the estimator stays unbiased. The sampler picks a child with probability `I_child / sum I`
(one-pass weighted reservoir over the children, one random number per child; no arrays). The pdf
functions repeat the same importance computations along the path of the given brick: bottom-up
through the top tree from the entity leaf (`parent` links), top-down through the pyramid by group
coordinate, top-down through the group octree by containment of the leaf (or the voxel) in the
child boxes (siblings' boxes are disjoint). Both sides use the same function, so the pdf of a drawn
sample equals the evaluated pdf up to rounding.

Inside a leaf, the face family is drawn with `P(f) = F[f] * cmax(f) / sum` over the leaf box
(receiver-aware; the owner's sketch used `F[f] / total`), the voxel uniformly among the family's
entries, the point uniformly on the unit face.

## 6. Host API (C ABI additions, no version bump)

```c
typedef struct CaelixLightOptions
{
    uint32_t enabled;      /* 0 = no light tree (default: the scene pays nothing), 1 = build it */
    float min_emission;    /* light voxel threshold, luminance of the emission before emission_scale.
                              0 = 20 */
    uint64_t heap_bytes;   /* initial capacity of each of the two heap buffers. 0 = 16 MiB */
} CaelixLightOptions;
CaelixResult caelix_scene_set_light_options(CaelixScene*, const CaelixLightOptions*);

typedef struct CaelixLightStats { ... } CaelixLightStats;   /* newest completed pass */
void caelix_scene_get_light_stats(const CaelixScene*, CaelixLightStats*);
/* Device address of the descriptor (section 2.4), 0 when disabled. Valid for commands recorded
   after the flush that returned it. */
uint64_t caelix_scene_get_light_tree(const CaelixScene*);
```

Renderer: `FrameConstants::light_tree` (offset 544; the struct grows to 560). No render kernel uses
it in this step.

## 7. Validation

`caelix_light_tree_test` (Catch2, GPU) with a probe kernel compiled by the test build:
- CPU reference of every light face (exposure across bricks and groups) against the rows, leaves and
  face entries; face sums.
- Normalisation: for several receivers, the sum of `LightTreePointPdf` over every light face is 1.
- Consistency: for drawn samples, `p.pdf` equals `LightTreePointPdf` and `LightTreeBrickPdf` times
  the in-brick pdf.
- Histogram of drawn faces against `N * pdf`.
- Bound property: for random (receiver, light face) pairs with a positive contribution, the pdf is
  positive.
- Updates: add and remove light bricks (inside a group and across a group border), raise and lower
  `min_emission`, move and rotate an entity, remove an entity; the checks above after each step;
  generations change only for rebuilt groups.
- Heap growth: a tiny `heap_bytes` makes the first passes fail; after the readback the heap grows
  and the checks pass; the stats count the failed passes.
- Benchmark case (hidden tag `[.bench]`): draws per second on a synthetic scene of about 1M light
  bricks in 15k groups, and the pass cost.
