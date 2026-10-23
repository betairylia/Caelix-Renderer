#pragma once
#include <cstddef>
#include <cstdint>

namespace caelix {

// GPU layouts of the light tree (docs/LIGHT_TREE.md). The structs define the layouts; the kernels
// address the fields through the named offsets below, which static_asserts tie to the structs.
// shaders/common/caelix_light_tree_layout.hlsl mirrors every constant of this file by name:
// caelix_frame_layout_test compares the two files, and the push constant blocks of the compiled
// kernels with the push structs here.

constexpr uint32_t kLtNone = 0xFFFFFFFFu;
constexpr uint32_t kLtTrees = 2;  // 0 static entities, 1 dynamic entities (CAELIX_ENTITY_STATIC / DYNAMIC)
constexpr uint32_t kLtMaxLightBricksPerGroup = 4096;

// The luminance of a material emission (before CaelixRenderOptions::emission_scale) that the light
// voxel threshold compares; LtLuminance in the HLSL mirror.
constexpr float LightLuminance(const float emission[3])
{
    return 0.2126f * emission[0] + 0.7152f * emission[1] + 0.0722f * emission[2];
}

// ---------------------------------------------------------------------------
// Scene record word 1 (kRecordBoundsWord, record_layout.h): bits 0..17 hold the tight bounds; the finalize
// kernel adds the light state, which the group pass reads.
constexpr uint32_t kLtRecordFaceCountShift = 18;  // open faces of the brick's light voxels
constexpr uint32_t kLtRecordFaceCountMask = 0xFFFu;
constexpr uint32_t kLtRecordStaleBit = 1u << 30;  // the brick's light record must be recomputed

// ---------------------------------------------------------------------------
// The descriptor: every kernel that uses the tree receives only its address. 128 bytes.
struct LightTreeDescriptor
{
    uint64_t control;        // LightControl
    uint64_t heap[2];        // the two heap buffers
    uint64_t group_rows;     // GroupLightRow by group index
    uint64_t group_infos;    // GroupInfo by group index
    uint64_t entity_infos;   // EntityInfo by entity id
    uint64_t entity_motion;  // u32 CAELIX_ENTITY_STATIC / DYNAMIC by entity id
    uint64_t grids;          // group grid cells
    uint64_t nodes[2];       // LightBvhNode arrays
    uint64_t leaves[2];      // LightBvhLeaf arrays
    uint64_t light_codes;    // 65536-bit light code set
    uint64_t tree_headers;   // LightTreeHeader[2] (inside the control buffer)
    uint32_t group_shift;
    uint32_t encoding;
    uint32_t pad;
    uint32_t group_count;  // rows of the group passes; information for tests and hosts
};
constexpr uint32_t kLtDescControl = 0;
constexpr uint32_t kLtDescHeap0 = 8;
constexpr uint32_t kLtDescHeap1 = 16;
constexpr uint32_t kLtDescGroupRows = 24;
constexpr uint32_t kLtDescGroupInfos = 32;
constexpr uint32_t kLtDescEntityInfos = 40;
constexpr uint32_t kLtDescEntityMotion = 48;
constexpr uint32_t kLtDescGrids = 56;
constexpr uint32_t kLtDescNodes0 = 64;  // tree t at + t * kLtDescTreeStride
constexpr uint32_t kLtDescLeaves0 = 80;
constexpr uint32_t kLtDescTreeStride = 8;
constexpr uint32_t kLtDescLightCodes = 96;
constexpr uint32_t kLtDescTreeHeaders = 104;
constexpr uint32_t kLtDescGroupShift = 112;
constexpr uint32_t kLtDescEncoding = 116;
constexpr uint32_t kLtDescGroupCount = 124;
constexpr uint32_t kLtDescriptorBytes = 128;
static_assert(offsetof(LightTreeDescriptor, control) == kLtDescControl);
static_assert(offsetof(LightTreeDescriptor, heap) == kLtDescHeap0);
static_assert(offsetof(LightTreeDescriptor, heap) + sizeof(uint64_t) == kLtDescHeap1);
static_assert(offsetof(LightTreeDescriptor, group_rows) == kLtDescGroupRows);
static_assert(offsetof(LightTreeDescriptor, group_infos) == kLtDescGroupInfos);
static_assert(offsetof(LightTreeDescriptor, entity_infos) == kLtDescEntityInfos);
static_assert(offsetof(LightTreeDescriptor, entity_motion) == kLtDescEntityMotion);
static_assert(offsetof(LightTreeDescriptor, grids) == kLtDescGrids);
static_assert(offsetof(LightTreeDescriptor, nodes) == kLtDescNodes0);
static_assert(offsetof(LightTreeDescriptor, leaves) == kLtDescLeaves0);
static_assert(sizeof(uint64_t) == kLtDescTreeStride);
static_assert(offsetof(LightTreeDescriptor, light_codes) == kLtDescLightCodes);
static_assert(offsetof(LightTreeDescriptor, tree_headers) == kLtDescTreeHeaders);
static_assert(offsetof(LightTreeDescriptor, group_shift) == kLtDescGroupShift);
static_assert(offsetof(LightTreeDescriptor, encoding) == kLtDescEncoding);
static_assert(offsetof(LightTreeDescriptor, group_count) == kLtDescGroupCount);
static_assert(sizeof(LightTreeDescriptor) == kLtDescriptorBytes);

// ---------------------------------------------------------------------------
// A summary: face sums and a box. The BVH leaves, the tree headers and the build's binary node
// aggregates start with this layout.
struct LightSummary
{
    float face[6];
    float lo[3];
    float hi[3];
};
constexpr uint32_t kLtSummaryFace = 0;
constexpr uint32_t kLtSummaryLo = 24;
constexpr uint32_t kLtSummaryHi = 36;
constexpr uint32_t kLtSummaryBytes = 48;
static_assert(offsetof(LightSummary, face) == kLtSummaryFace);
static_assert(offsetof(LightSummary, lo) == kLtSummaryLo);
static_assert(offsetof(LightSummary, hi) == kLtSummaryHi);
static_assert(sizeof(LightSummary) == kLtSummaryBytes);

// The root summary of one BVH (world space). 64 bytes.
struct LightTreeHeader
{
    LightSummary summary;
    uint32_t root;  // node index; kLtNone = empty tree
    uint32_t leaf_count;
    uint32_t node_count;
    uint32_t pad;
};
constexpr uint32_t kLtHdrRoot = 48;
constexpr uint32_t kLtHdrLeafCount = 52;
constexpr uint32_t kLtHdrNodeCount = 56;
constexpr uint32_t kLtTreeHeaderBytes = 64;
static_assert(offsetof(LightTreeHeader, summary) == 0);
static_assert(offsetof(LightTreeHeader, root) == kLtHdrRoot);
static_assert(offsetof(LightTreeHeader, leaf_count) == kLtHdrLeafCount);
static_assert(offsetof(LightTreeHeader, node_count) == kLtHdrNodeCount);
static_assert(sizeof(LightTreeHeader) == kLtTreeHeaderBytes);

// The build state of one BVH. 64 bytes.
struct LightTreeState
{
    uint32_t building;        // 1 while this flush rebuilds the tree (gates every build kernel)
    uint32_t leaves_found;    // light bricks of the tree at the last build
    uint32_t leaves_dropped;  // of those, above the leaf capacity
    uint32_t builds;          // builds so far
    uint32_t node_overflow;   // 1 when the last build needed more nodes than the capacity
    uint32_t pad0[3];
    float bounds[6];  // lo[3], hi[3]: bounds of the leaf box centres
    uint32_t pad1[2];
};
constexpr uint32_t kLtStBuilding = 0;
constexpr uint32_t kLtStLeavesFound = 4;
constexpr uint32_t kLtStLeavesDropped = 8;
constexpr uint32_t kLtStBuilds = 12;
constexpr uint32_t kLtStNodeOverflow = 16;
constexpr uint32_t kLtStBounds = 32;
constexpr uint32_t kLtTreeStateBytes = 64;
static_assert(offsetof(LightTreeState, building) == kLtStBuilding);
static_assert(offsetof(LightTreeState, leaves_found) == kLtStLeavesFound);
static_assert(offsetof(LightTreeState, leaves_dropped) == kLtStLeavesDropped);
static_assert(offsetof(LightTreeState, builds) == kLtStBuilds);
static_assert(offsetof(LightTreeState, node_overflow) == kLtStNodeOverflow);
static_assert(offsetof(LightTreeState, bounds) == kLtStBounds);
static_assert(sizeof(LightTreeState) == kLtTreeStateBytes);

// The control buffer: the group pass state, then the two tree headers and states. 320 bytes. The
// library copies it whole to a host readback slot after every light update. The 64-bit counters
// stay 8-byte aligned.
struct LightControl
{
    uint32_t current;         // heap buffer the rows point into (0 or 1)
    uint32_t pass_result;     // last pass: kLtPassFailed / Ok / NoChange
    uint32_t needed_units;    // last pass with changes: kLtUnitBytes units of every live block
    uint32_t pad;
    uint32_t light_groups;    // after the last committed pass
    uint32_t bricks_dropped;  // after the last committed pass
    uint64_t light_bricks;
    uint64_t light_faces;
    uint32_t passes;              // passes that rebuilt or released at least one group
    uint32_t failed_passes;       // of those, the passes that did not fit the heap
    uint32_t pass_flags;          // kLtPassFlag*: set by the size kernel, cleared by the scan kernel
    uint32_t records_recomputed;  // last pass with changes: brick light records recomputed
    uint32_t records_copied;      // last pass with changes: records copied from the old block
    uint32_t trees_changed;       // last pass: bit t = a group of tree t was rebuilt or released
    LightTreeHeader headers[2];
    LightTreeState states[2];
};
constexpr uint32_t kLtCtlCurrent = 0;
constexpr uint32_t kLtCtlPassResult = 4;
constexpr uint32_t kLtCtlNeededUnits = 8;
constexpr uint32_t kLtCtlLightGroups = 16;
constexpr uint32_t kLtCtlBricksDropped = 20;
constexpr uint32_t kLtCtlLightBricks = 24;
constexpr uint32_t kLtCtlLightFaces = 32;
constexpr uint32_t kLtCtlPasses = 40;
constexpr uint32_t kLtCtlFailedPasses = 44;
constexpr uint32_t kLtCtlPassFlags = 48;
constexpr uint32_t kLtCtlRecordsRecomputed = 52;
constexpr uint32_t kLtCtlRecordsCopied = 56;
constexpr uint32_t kLtCtlTreesChanged = 60;
constexpr uint32_t kLtCtlTreeHeaders = 64;
constexpr uint32_t kLtCtlTreeStates = 192;
constexpr uint32_t kLtControlBytes = 320;
static_assert(offsetof(LightControl, current) == kLtCtlCurrent);
static_assert(offsetof(LightControl, pass_result) == kLtCtlPassResult);
static_assert(offsetof(LightControl, needed_units) == kLtCtlNeededUnits);
static_assert(offsetof(LightControl, light_groups) == kLtCtlLightGroups);
static_assert(offsetof(LightControl, bricks_dropped) == kLtCtlBricksDropped);
static_assert(offsetof(LightControl, light_bricks) == kLtCtlLightBricks);
static_assert(offsetof(LightControl, light_faces) == kLtCtlLightFaces);
static_assert(offsetof(LightControl, passes) == kLtCtlPasses);
static_assert(offsetof(LightControl, failed_passes) == kLtCtlFailedPasses);
static_assert(offsetof(LightControl, pass_flags) == kLtCtlPassFlags);
static_assert(offsetof(LightControl, records_recomputed) == kLtCtlRecordsRecomputed);
static_assert(offsetof(LightControl, records_copied) == kLtCtlRecordsCopied);
static_assert(offsetof(LightControl, trees_changed) == kLtCtlTreesChanged);
static_assert(offsetof(LightControl, headers) == kLtCtlTreeHeaders);
static_assert(offsetof(LightControl, states) == kLtCtlTreeStates);
static_assert(sizeof(LightControl) == kLtControlBytes);

// LightControl::pass_result.
constexpr uint32_t kLtPassFailed = 0;
constexpr uint32_t kLtPassOk = 1;
constexpr uint32_t kLtPassNoChange = 2;  // nothing flagged: no copy, no flip
// LightControl::pass_flags: a group was flagged or released; bit kLtPassFlagTreeShift + t: one of
// them belongs to tree t (its entity's motion).
constexpr uint32_t kLtPassFlagChanged = 1u;
constexpr uint32_t kLtPassFlagTreeShift = 1;

// ---------------------------------------------------------------------------
// GroupLightRow by group index. 32 bytes. The CPU never writes a row: its per-group flags go through
// the staged flags buffer (u32 per group index, kLtCpu*, 0 = nothing new), which the size kernel
// merges (cpu_flags = (cpu_flags & kLtCpuFresh) | staged) and clears.
struct GroupLightRow
{
    uint32_t block_offset;  // kLtUnitBytes units in the current heap buffer
    uint32_t block_units;   // 0 = no block
    uint32_t generation;    // +1 at every rebuild or release of the block
    uint32_t gpu_flags;     // kLtGpuLightChanged
    uint32_t cpu_flags;     // GPU-owned merge of the staged flags (kLtCpu*)
    uint32_t bricks;        // light bricks (kLtRowBricksMask) | dropped << kLtRowDroppedShift
    uint32_t light_faces;
    uint32_t pad;
};
constexpr uint32_t kLtRowBlockOffset = 0;
constexpr uint32_t kLtRowBlockUnits = 4;
constexpr uint32_t kLtRowGeneration = 8;
constexpr uint32_t kLtRowGpuFlags = 12;
constexpr uint32_t kLtRowCpuFlags = 16;
constexpr uint32_t kLtRowBricks = 20;
constexpr uint32_t kLtRowLightFaces = 24;
constexpr uint32_t kLtGroupRowBytes = 32;
static_assert(offsetof(GroupLightRow, block_offset) == kLtRowBlockOffset);
static_assert(offsetof(GroupLightRow, block_units) == kLtRowBlockUnits);
static_assert(offsetof(GroupLightRow, generation) == kLtRowGeneration);
static_assert(offsetof(GroupLightRow, gpu_flags) == kLtRowGpuFlags);
static_assert(offsetof(GroupLightRow, cpu_flags) == kLtRowCpuFlags);
static_assert(offsetof(GroupLightRow, bricks) == kLtRowBricks);
static_assert(offsetof(GroupLightRow, light_faces) == kLtRowLightFaces);
static_assert(sizeof(GroupLightRow) == kLtGroupRowBytes);
constexpr uint32_t kLtRowBricksMask = 0xFFFFu;
constexpr uint32_t kLtRowDroppedShift = 16;

constexpr uint32_t kLtGpuLightChanged = 1u;  // the finalize kernel saw a light change in the group
constexpr uint32_t kLtCpuRebuild = 1u;       // rebuild the block (bricks removed)
constexpr uint32_t kLtCpuReleased = 2u;      // the group index is free: clear the row
constexpr uint32_t kLtCpuFresh = 4u;         // a new life of the group index: copy nothing from the old block

// ---------------------------------------------------------------------------
// Group light block in a heap buffer: the header, the index entries (in Morton order of the brick
// position), the brick light records (each starts at a unit). Offsets and sizes in the heap count
// units of kLtUnitBytes.
constexpr uint32_t kLtUnitBytes = 16;
constexpr uint64_t kLtMaxHeapUnits = 0xFFFFFFFFull;  // per heap buffer: offsets are u32 units
struct LightBlockHeader
{
    uint32_t record_count;  // index entries = brick light records of the block
    uint32_t pad[3];
};
constexpr uint32_t kLtBlockIndexOffset = 16;  // bytes from the block start to index entry 0
static_assert(offsetof(LightBlockHeader, record_count) == 0);
static_assert(sizeof(LightBlockHeader) == kLtBlockIndexOffset);
static_assert(kLtBlockIndexOffset == kLtUnitBytes);

struct LightIndexEntry
{
    uint32_t pos;            // group-local brick position
    uint32_t dense_id;
    uint32_t record_offset;  // units from the block start
    uint32_t bvh_leaf;       // tree << kLtBvhLeafTreeShift | leaf index (the gather); kLtNone = none
};
constexpr uint32_t kLtIndexPos = 0;
constexpr uint32_t kLtIndexDenseId = 4;
constexpr uint32_t kLtIndexRecordOffset = 8;
constexpr uint32_t kLtIndexBvhLeaf = 12;
constexpr uint32_t kLtIndexEntryBytes = 16;
static_assert(offsetof(LightIndexEntry, pos) == kLtIndexPos);
static_assert(offsetof(LightIndexEntry, dense_id) == kLtIndexDenseId);
static_assert(offsetof(LightIndexEntry, record_offset) == kLtIndexRecordOffset);
static_assert(offsetof(LightIndexEntry, bvh_leaf) == kLtIndexBvhLeaf);
static_assert(sizeof(LightIndexEntry) == kLtIndexEntryBytes);
constexpr uint32_t kLtBvhLeafTreeShift = 31;
constexpr uint32_t kLtBvhLeafIndexMask = 0x7FFFFFFFu;

// Header of a brick light record; uint32_t entries[entry_count] follow, zero-padded to a unit: the
// entries of face 0, then 1 .. 5, each family sorted by voxel index.
struct BrickLightRecordHeader
{
    float face[6];        // face sums of the record's entries
    uint32_t box_lo;      // group-local voxels (LtPackVoxel), inclusive
    uint32_t box_hi;      // exclusive
    uint32_t pos;         // group-local brick position
    uint32_t entry_count;
    uint32_t counts_lo;   // face counts of +X, -X, +Y (kLtFamilyCountBits each)
    uint32_t counts_hi;   // face counts of -Y, +Z, -Z
};
constexpr uint32_t kLtRecordFace = 0;
constexpr uint32_t kLtRecordBoxLo = 24;
constexpr uint32_t kLtRecordBoxHi = 28;
constexpr uint32_t kLtRecordPos = 32;
constexpr uint32_t kLtRecordEntryCount = 36;
constexpr uint32_t kLtRecordCountsLo = 40;
constexpr uint32_t kLtRecordCountsHi = 44;
constexpr uint32_t kLtRecordHeaderBytes = 48;
static_assert(offsetof(BrickLightRecordHeader, face) == kLtRecordFace);
static_assert(offsetof(BrickLightRecordHeader, box_lo) == kLtRecordBoxLo);
static_assert(offsetof(BrickLightRecordHeader, box_hi) == kLtRecordBoxHi);
static_assert(offsetof(BrickLightRecordHeader, pos) == kLtRecordPos);
static_assert(offsetof(BrickLightRecordHeader, entry_count) == kLtRecordEntryCount);
static_assert(offsetof(BrickLightRecordHeader, counts_lo) == kLtRecordCountsLo);
static_assert(offsetof(BrickLightRecordHeader, counts_hi) == kLtRecordCountsHi);
static_assert(sizeof(BrickLightRecordHeader) == kLtRecordHeaderBytes);
static_assert(kLtRecordHeaderBytes % kLtUnitBytes == 0);
constexpr uint32_t kLtFamilyCountBits = 10;
constexpr uint32_t kLtFamilyCountMask = 0x3FFu;
constexpr uint32_t kLtFamiliesPerCountWord = 3;
// A face entry (u32): the voxel index in the brick (x | y << 3 | z << 6) and the raw block value.
constexpr uint32_t kLtFaceEntryBytes = 4;
constexpr uint32_t kLtFaceEntryVoxelMask = 0x1FFu;
constexpr uint32_t kLtFaceEntryBlockShift = 16;
// A packed group-local voxel (record boxes): x | y << kLtVoxelBits | z << 2 kLtVoxelBits.
constexpr uint32_t kLtVoxelBits = 10;
constexpr uint32_t kLtVoxelMask = 0x3FFu;

// ---------------------------------------------------------------------------
// BVH leaf: one light brick in world space. 64 bytes.
struct LightBvhLeaf
{
    LightSummary summary;  // world face sums (projected) and the world AABB
    uint32_t group;
    uint32_t entry;   // the brick's index entry in the group's block
    uint32_t pad;
    uint32_t parent;  // node << kLtParentSlotBits | slot
};
constexpr uint32_t kLtLeafGroup = 48;
constexpr uint32_t kLtLeafEntry = 52;
constexpr uint32_t kLtLeafParent = 60;
constexpr uint32_t kLtLeafBytes = 64;
static_assert(offsetof(LightBvhLeaf, summary) == 0);
static_assert(offsetof(LightBvhLeaf, group) == kLtLeafGroup);
static_assert(offsetof(LightBvhLeaf, entry) == kLtLeafEntry);
static_assert(offsetof(LightBvhLeaf, parent) == kLtLeafParent);
static_assert(sizeof(LightBvhLeaf) == kLtLeafBytes);
constexpr uint32_t kLtParentSlotBits = 2;
constexpr uint32_t kLtParentSlotMask = 3u;

// 4-wide BVH node, children inline. 224 bytes. The slots are filled from 0.
struct LightBvhNode
{
    float child_face[4][6];
    float child_lo[4][3];
    float child_hi[4][3];
    uint32_t child[4];  // kLtChildLeafBit: leaf index, else node index; kLtNone = empty slot
    uint32_t parent;    // kLtNone = root
    uint32_t parent_slot;
    uint32_t pad[2];
};
constexpr uint32_t kLtNodeChildFace = 0;
constexpr uint32_t kLtNodeChildLo = 96;
constexpr uint32_t kLtNodeChildHi = 144;
constexpr uint32_t kLtNodeChild = 192;
constexpr uint32_t kLtNodeParent = 208;
constexpr uint32_t kLtNodeParentSlot = 212;
constexpr uint32_t kLtNodeBytes = 224;
constexpr uint32_t kLtNodeSlots = 4;
constexpr uint32_t kLtNodeFaceBytes = 24;  // child_face[slot]
constexpr uint32_t kLtNodeBoxBytes = 12;   // child_lo[slot], child_hi[slot]
static_assert(offsetof(LightBvhNode, child_face) == kLtNodeChildFace);
static_assert(offsetof(LightBvhNode, child_lo) == kLtNodeChildLo);
static_assert(offsetof(LightBvhNode, child_hi) == kLtNodeChildHi);
static_assert(offsetof(LightBvhNode, child) == kLtNodeChild);
static_assert(offsetof(LightBvhNode, parent) == kLtNodeParent);
static_assert(offsetof(LightBvhNode, parent_slot) == kLtNodeParentSlot);
static_assert(sizeof(LightBvhNode) == kLtNodeBytes);
static_assert(sizeof(LightBvhNode::child_face[0]) == kLtNodeFaceBytes);
static_assert(sizeof(LightBvhNode::child_lo[0]) == kLtNodeBoxBytes);
static_assert(sizeof(LightBvhNode::child) / sizeof(uint32_t) == kLtNodeSlots);
constexpr uint32_t kLtChildLeafBit = 0x80000000u;

// ---------------------------------------------------------------------------
// Build scratch of the BVH kernels for a leaf capacity C (a multiple of 64). Array X has C elements
// and starts at byte kLtScratchX * C (u32 elements; the aggregates have kLtSummaryBytes each):
//   keys, values         the radix sort's (Morton code, leaf index) pairs; sort buffer b at
//                        + b * kLtScratchSortBufferStride * C; the sorted result is in buffer 0
//   left, right          binary node i's children (kLtChildLeafBit: the leaf at that sorted position)
//   parent               binary node i's parent (kLtNone = root)
//   sorted_leaf_parent   binary parent of the leaf at sorted position j
//   counter              aggregation arrival counters, then the node depths
//   wide                 1 for a binary node at even depth, then its 4-wide node index (scan)
//   aggregates           the summary of every binary node
//   histogram            the radix sort's digit counts: kLtSortDigits per sort block, digit-major
constexpr uint32_t kLtScratchKeys = 0;
constexpr uint32_t kLtScratchValues = 4;
constexpr uint32_t kLtScratchSortBufferStride = 8;
constexpr uint32_t kLtScratchLeft = 16;
constexpr uint32_t kLtScratchRight = 20;
constexpr uint32_t kLtScratchParent = 24;
constexpr uint32_t kLtScratchSortedLeafParent = 28;
constexpr uint32_t kLtScratchCounter = 32;
constexpr uint32_t kLtScratchWide = 36;
constexpr uint32_t kLtScratchAggregates = 40;
constexpr uint32_t kLtScratchHistogram = 88;
static_assert(kLtScratchValues == kLtScratchKeys + sizeof(uint32_t));
static_assert(kLtScratchSortBufferStride == kLtScratchValues + sizeof(uint32_t) - kLtScratchKeys);
static_assert(kLtScratchLeft == kLtScratchKeys + 2 * kLtScratchSortBufferStride);
static_assert(kLtScratchRight == kLtScratchLeft + sizeof(uint32_t));
static_assert(kLtScratchParent == kLtScratchRight + sizeof(uint32_t));
static_assert(kLtScratchSortedLeafParent == kLtScratchParent + sizeof(uint32_t));
static_assert(kLtScratchCounter == kLtScratchSortedLeafParent + sizeof(uint32_t));
static_assert(kLtScratchWide == kLtScratchCounter + sizeof(uint32_t));
static_assert(kLtScratchAggregates == kLtScratchWide + sizeof(uint32_t));
static_assert(kLtScratchHistogram == kLtScratchAggregates + kLtSummaryBytes);

// The radix sort: kLtSortBits per pass over 32-bit keys, blocks of kLtSortBlock elements.
constexpr uint32_t kLtSortBits = 4;
constexpr uint32_t kLtSortDigits = 16;
constexpr uint32_t kLtSortPasses = 8;
constexpr uint32_t kLtSortBlock = 1024;
static_assert(kLtSortDigits == 1u << kLtSortBits);
static_assert(kLtSortPasses * kLtSortBits == 32);

// Build scratch bytes for a leaf capacity.
constexpr uint64_t LightBvhScratchBytes(uint32_t leaf_capacity)
{
    const uint64_t blocks = (uint64_t{leaf_capacity} + kLtSortBlock - 1) / kLtSortBlock;
    const uint64_t histogram_bytes = uint64_t{kLtSortDigits} * sizeof(uint32_t) * blocks;
    return uint64_t{kLtScratchHistogram} * leaf_capacity + histogram_bytes;
}

// ---------------------------------------------------------------------------
// Push constants. The HLSL structs have the same member names (caelix_frame_layout_test checks the
// compiled blocks against these).

// The group pass kernels (shaders/scene/caelix_light_tree_group.hlsl). 80 bytes.
struct LightGroupPush
{
    uint64_t descriptor;
    uint64_t page_table;
    uint64_t slot_tables;
    uint64_t reverse_tables;
    uint64_t materials;
    uint64_t offsets;
    uint64_t staged_flags;
    uint32_t records_per_page_shift;
    uint32_t record_bytes;
    uint32_t group_base;
    uint32_t group_count;
    uint32_t capacity_units;
    uint32_t pad;
};
static_assert(offsetof(LightGroupPush, staged_flags) == 48);
static_assert(offsetof(LightGroupPush, records_per_page_shift) == 56);
static_assert(offsetof(LightGroupPush, capacity_units) == 72);
static_assert(sizeof(LightGroupPush) == 80);

// The gather and build kernels (shaders/scene/caelix_light_bvh_common.hlsl). 56 bytes.
struct LightBvhPush
{
    uint64_t descriptor;
    uint64_t scratch;
    uint64_t offsets;
    uint32_t tree;
    uint32_t leaf_capacity;
    uint32_t node_capacity;
    uint32_t group_count;
    uint32_t reason;
    uint32_t pass_recorded;
    uint32_t group_base;
    uint32_t pad;
};
static_assert(offsetof(LightBvhPush, tree) == 24);
static_assert(offsetof(LightBvhPush, group_base) == 48);
static_assert(sizeof(LightBvhPush) == 56);

// caelix_light_bvh_scan. 32 bytes.
struct LightScanPush
{
    uint64_t data;
    uint64_t total_address;
    uint64_t gate_address;
    uint32_t count;
    uint32_t pad;
};
static_assert(offsetof(LightScanPush, count) == 24);
static_assert(sizeof(LightScanPush) == 32);

// The radix sort kernels (shaders/scene/caelix_light_bvh_sort.hlsl). 64 bytes.
struct LightSortPush
{
    uint64_t keys_in;
    uint64_t values_in;
    uint64_t keys_out;
    uint64_t values_out;
    uint64_t histogram;
    uint64_t count_address;
    uint64_t gate_address;
    uint32_t shift;
    uint32_t block_count;
};
static_assert(offsetof(LightSortPush, shift) == 56);
static_assert(sizeof(LightSortPush) == 64);

}  // namespace caelix
