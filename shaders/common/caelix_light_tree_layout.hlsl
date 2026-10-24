// GPU layouts of the light tree (docs/LIGHT_TREE.md) and the helpers that read them. Mirrors
// lib/src/light/light_tree_layout.h, where the structs and their static_asserts are: every constant
// here has the same name and value there (caelix_frame_layout_test compares the two files). Every
// table is read and written through buffer device addresses.
#ifndef CAELIX_LIGHT_TREE_LAYOUT_HLSL
#define CAELIX_LIGHT_TREE_LAYOUT_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"

static const uint kLtNone = 0xFFFFFFFFu;
static const uint kLtTrees = 2;  // 0 static entities, 1 dynamic entities (CAELIX_ENTITY_STATIC / DYNAMIC)
static const uint kLtMaxLightBricksPerGroup = 4096;

// ---------------------------------------------------------------------------
// Scene record word 1 (kRecordBoundsWord): bits 0..17 hold the tight bounds; the finalize kernel adds
// the light state, which the group pass reads.
static const uint kLtRecordFaceCountShift = 18;  // open faces of the brick's light voxels
static const uint kLtRecordFaceCountMask = 0xFFFu;
static const uint kLtRecordStaleBit = 1u << 30;  // the brick's light record must be recomputed

// ---------------------------------------------------------------------------
// Descriptor (LightTreeDescriptor, 128 bytes). Every kernel that uses the tree receives only its
// address.
static const uint kLtDescControl = 0;        // uint64_t: control buffer
static const uint kLtDescHeap0 = 8;          // uint64_t: heap buffer 0
static const uint kLtDescHeap1 = 16;         // uint64_t: heap buffer 1
static const uint kLtDescGroupRows = 24;     // uint64_t: GroupLightRow by group index
static const uint kLtDescGroupInfos = 32;    // uint64_t: GroupInfo by group index
static const uint kLtDescEntityInfos = 40;   // uint64_t: EntityInfo by entity id
static const uint kLtDescEntityMotion = 48;  // uint64_t: u32 motion by entity id
static const uint kLtDescGrids = 56;         // uint64_t: group grid cells (u32)
static const uint kLtDescNodes0 = 64;        // uint64_t: LightBvhNode array of tree 0
static const uint kLtDescLeaves0 = 80;       // uint64_t: LightBvhLeaf array of tree 0
static const uint kLtDescTreeStride = 8;     // tree t's nodes / leaves at + t * kLtDescTreeStride
static const uint kLtDescLightCodes = 96;    // uint64_t: 65536-bit light code set
static const uint kLtDescTreeHeaders = 104;  // uint64_t: tree header 0 (inside the control buffer)
static const uint kLtDescGroupShift = 112;   // uint
static const uint kLtDescEncoding = 116;     // uint: kEncodingPacked or kEncodingBlockId
static const uint kLtDescGroupCount = 124;   // uint: rows of the group passes (tests and hosts)
static const uint kLtDescriptorBytes = 128;

uint64_t LtDescriptorU64(uint64_t descriptor, uint offset)
{
    return CAELIX_LOAD_U64(descriptor + offset);
}
uint LtDescriptorU32(uint64_t descriptor, uint offset)
{
    return CAELIX_LOAD_U32(descriptor + offset);
}

// ---------------------------------------------------------------------------
// Summary (LightSummary, 48 bytes): float face[6], lo[3], hi[3]. The BVH leaves, the tree headers
// and the build's binary node aggregates start with it.
static const uint kLtSummaryFace = 0;
static const uint kLtSummaryLo = 24;
static const uint kLtSummaryHi = 36;
static const uint kLtSummaryBytes = 48;

// Tree header (LightTreeHeader, 64 bytes, kLtCtlTreeHeaders + t * kLtTreeHeaderBytes in the control
// buffer): the root summary, the root node index (kLtNone = empty tree), the leaf and node counts.
static const uint kLtHdrRoot = 48;
static const uint kLtHdrLeafCount = 52;
static const uint kLtHdrNodeCount = 56;
static const uint kLtTreeHeaderBytes = 64;

// Tree build state (LightTreeState, 64 bytes, kLtCtlTreeStates + t * kLtTreeStateBytes).
static const uint kLtStBuilding = 0;       // 1 while this flush rebuilds the tree (gates every build kernel)
static const uint kLtStLeavesFound = 4;    // light bricks of the tree at the last build
static const uint kLtStLeavesDropped = 8;  // of those, above the leaf capacity
static const uint kLtStBuilds = 12;        // builds so far
static const uint kLtStNodeOverflow = 16;  // 1 when the last build needed more nodes than the capacity
static const uint kLtStBounds = 32;        // float lo[3], hi[3]: bounds of the leaf box centres
static const uint kLtTreeStateBytes = 64;

// ---------------------------------------------------------------------------
// Control buffer (LightControl, 320 bytes): the group pass state, the two tree headers, the two tree
// build states. The library reads it back whole after every light update.
static const uint kLtCtlCurrent = 0;             // heap buffer that the rows point into (0 or 1)
static const uint kLtCtlPassResult = 4;          // last pass: kLtPass*
static const uint kLtCtlNeededUnits = 8;         // last pass with changes: units of every live block
static const uint kLtCtlLightGroups = 16;        // after the last committed pass
static const uint kLtCtlBricksDropped = 20;      // after the last committed pass
static const uint kLtCtlLightBricks = 24;        // uint64_t
static const uint kLtCtlLightFaces = 32;         // uint64_t
static const uint kLtCtlPasses = 40;             // passes that rebuilt or released at least one group
static const uint kLtCtlFailedPasses = 44;       // of those, passes that did not fit the heap
static const uint kLtCtlPassFlags = 48;          // kLtPassFlag*: set by the size kernel, cleared by scan
static const uint kLtCtlRecordsRecomputed = 52;  // last pass with changes: brick light records recomputed
static const uint kLtCtlRecordsCopied = 56;      // last pass with changes: records copied
static const uint kLtCtlTreesChanged = 60;       // last pass: bit t = a group of tree t changed
static const uint kLtCtlTreeHeaders = 64;
static const uint kLtCtlTreeStates = 192;
static const uint kLtControlBytes = 320;

static const uint kLtPassFailed = 0;
static const uint kLtPassOk = 1;
static const uint kLtPassNoChange = 2;  // nothing flagged: no copy, no flip
static const uint kLtPassFlagChanged = 1u;   // a group was flagged or released
static const uint kLtPassFlagTreeShift = 1;  // bit kLtPassFlagTreeShift + t: one of tree t

// ---------------------------------------------------------------------------
// GroupLightRow (32 bytes) by scene-wide group index. The CPU never writes a row: its flags go
// through the staged flags buffer (u32 per group index, 0 = nothing new; LightGroupPush::
// staged_flags), which the size kernel merges into cpu_flags, cpu_flags = (cpu_flags & kLtCpuFresh)
// | staged, and clears.
static const uint kLtRowBlockOffset = 0;  // units in the current heap buffer
static const uint kLtRowBlockUnits = 4;   // 0 = no block
static const uint kLtRowGeneration = 8;   // life << kLtGenerationLifeShift | version (LtNext*)
static const uint kLtRowGpuFlags = 12;    // kLtGpuLightChanged
static const uint kLtRowCpuFlags = 16;    // kLtCpu*: the merge of the staged flags (cleared by the build)
static const uint kLtRowBricks = 20;      // light bricks | dropped light bricks << kLtRowDroppedShift
static const uint kLtRowLightFaces = 24;
static const uint kLtGroupRowBytes = 32;
static const uint kLtRowBricksMask = 0xFFFFu;
static const uint kLtRowDroppedShift = 16;

// The row's 32-bit generation: the group index's life (bumped when the index is released, so a
// reference into an earlier life never matches) in the high bits, the block version (bumped at every
// rebuild within one life, wrapping at 2^16) in the low bits. Macros: a new static const here would
// reorder the SPIR-V constants of every kernel that includes this header (docs/LIGHT_TREE.md section 8).
#define kLtGenerationLifeShift 16
#define kLtGenerationVersionMask 0xFFFFu

// The generation of a released group's row: the next life, version 0.
uint LtNextLife(uint generation)
{
    return ((generation >> kLtGenerationLifeShift) + 1u) << kLtGenerationLifeShift;
}

// The generation of a rebuilt group's row: the same life, the next version.
uint LtNextVersion(uint generation)
{
    return (generation & ~kLtGenerationVersionMask) | ((generation + 1u) & kLtGenerationVersionMask);
}

static const uint kLtGpuLightChanged = 1u;  // the finalize kernel saw a light change in the group
static const uint kLtCpuRebuild = 1u;       // rebuild the block (bricks removed)
static const uint kLtCpuReleased = 2u;      // the group index is free: clear the row
static const uint kLtCpuFresh = 4u;         // a new life of the group index: copy nothing from the old block

// ---------------------------------------------------------------------------
// Group light block in a heap buffer (offsets and sizes in units of kLtUnitBytes):
//   header: uint record_count; pad
//   index entries[record_count] from kLtBlockIndexOffset, in Morton order of the brick position
//   brick light records, each starting at a unit
static const uint kLtUnitBytes = 16;
static const uint kLtBlockIndexOffset = 16;
// Index entry (LightIndexEntry, 16 bytes).
static const uint kLtIndexPos = 0;           // group-local brick position
static const uint kLtIndexDenseId = 4;
static const uint kLtIndexRecordOffset = 8;  // units from the block start
static const uint kLtIndexBvhLeaf = 12;      // tree << kLtBvhLeafTreeShift | leaf index; kLtNone = none
static const uint kLtIndexEntryBytes = 16;
static const uint kLtBvhLeafTreeShift = 31;
static const uint kLtBvhLeafIndexMask = 0x7FFFFFFFu;
// Brick light record (BrickLightRecordHeader, then uint entries[entry_count] zero-padded to a unit:
// the entries of face 0, then 1 .. 5, each family sorted by voxel index). Self-contained and
// position independent inside its group, so a pass copies it when the brick did not change.
static const uint kLtRecordFace = 0;         // float face[6]: face sums of the entries
static const uint kLtRecordBoxLo = 24;       // group-local voxels (LtPackVoxel), inclusive
static const uint kLtRecordBoxHi = 28;       // exclusive
static const uint kLtRecordPos = 32;         // group-local brick position
static const uint kLtRecordEntryCount = 36;
static const uint kLtRecordCountsLo = 40;    // face counts of +X, -X, +Y
static const uint kLtRecordCountsHi = 44;    // face counts of -Y, +Z, -Z
static const uint kLtRecordHeaderBytes = 48;
static const uint kLtFamilyCountBits = 10;
static const uint kLtFamilyCountMask = 0x3FFu;
static const uint kLtFamiliesPerCountWord = 3;
// Face entry (u32): the voxel index in the brick (x | y << 3 | z << 6) and the raw block value.
static const uint kLtFaceEntryBytes = 4;
static const uint kLtFaceEntryVoxelMask = 0x1FFu;
static const uint kLtFaceEntryBlockShift = 16;
// Packed group-local voxel: x | y << kLtVoxelBits | z << 2 kLtVoxelBits.
static const uint kLtVoxelBits = 10;
static const uint kLtVoxelMask = 0x3FFu;

// ---------------------------------------------------------------------------
// BVH leaf (LightBvhLeaf, 64 bytes): the world summary, then the brick and the parent slot.
static const uint kLtLeafGroup = 48;
static const uint kLtLeafEntry = 52;   // index entry of the brick in the group's block
static const uint kLtLeafParent = 60;  // node << kLtParentSlotBits | slot
static const uint kLtLeafBytes = 64;
static const uint kLtParentSlotBits = 2;
static const uint kLtParentSlotMask = 3u;
// BVH node (LightBvhNode, 224 bytes), 4 slots filled from 0, summaries inline:
//   float child_face[4][6]; float child_lo[4][3]; float child_hi[4][3]; uint child[4] (kLtChildLeafBit:
//   leaf index, else node index; kLtNone = empty slot); uint parent (kLtNone = root); uint parent_slot
static const uint kLtNodeChildFace = 0;
static const uint kLtNodeChildLo = 96;
static const uint kLtNodeChildHi = 144;
static const uint kLtNodeChild = 192;
static const uint kLtNodeParent = 208;
static const uint kLtNodeParentSlot = 212;
static const uint kLtNodeBytes = 224;
static const uint kLtNodeSlots = 4;
static const uint kLtNodeFaceBytes = 24;
static const uint kLtNodeBoxBytes = 12;
static const uint kLtChildLeafBit = 0x80000000u;

// ---------------------------------------------------------------------------
// Build scratch of the BVH kernels for a leaf capacity C: array X starts at byte kLtScratchX * C
// (lib/src/light/light_tree_layout.h describes the arrays).
static const uint kLtScratchKeys = 0;
static const uint kLtScratchValues = 4;
static const uint kLtScratchSortBufferStride = 8;
static const uint kLtScratchLeft = 16;
static const uint kLtScratchRight = 20;
static const uint kLtScratchParent = 24;
static const uint kLtScratchSortedLeafParent = 28;
static const uint kLtScratchCounter = 32;
static const uint kLtScratchWide = 36;
static const uint kLtScratchAggregates = 40;
static const uint kLtScratchHistogram = 88;
// The radix sort.
static const uint kLtSortBits = 4;
static const uint kLtSortDigits = 16;
static const uint kLtSortPasses = 8;
static const uint kLtSortBlock = 1024;

// ---------------------------------------------------------------------------
// Helpers

uint LtPackVoxel(uint3 v)
{
    return v.x | (v.y << kLtVoxelBits) | (v.z << (2 * kLtVoxelBits));
}
uint3 LtUnpackVoxel(uint packed)
{
    return uint3(packed & kLtVoxelMask, (packed >> kLtVoxelBits) & kLtVoxelMask,
                 (packed >> (2 * kLtVoxelBits)) & kLtVoxelMask);
}

// The light face count of a scene record's word 1 (0: not a light brick).
uint LtLightFaceCount(uint word1)
{
    return (word1 >> kLtRecordFaceCountShift) & kLtRecordFaceCountMask;
}

// Luminance weights of the light voxel threshold (emission before emission_scale).
float LtLuminance(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

uint LtPart1By2(uint v)
{
    v &= 0x3FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

// 30-bit Morton code of a cell with 10-bit coordinates.
uint LtMorton(uint3 c)
{
    return LtPart1By2(c.x) | (LtPart1By2(c.y) << 1) | (LtPart1By2(c.z) << 2);
}

// A box with face sums: a BVH child, a leaf, a tree header.
struct LtSummary
{
    float face[6];
    float3 lo;
    float3 hi;
};

float LtPower(float face[6])
{
    return face[0] + face[1] + face[2] + face[3] + face[4] + face[5];
}

// The summary layout as three 16-byte vectors.
LtSummary LtLoadSummary(uint64_t address)
{
    const float4 a = asfloat(CAELIX_LOAD_U32x4(address));
    const float4 b = asfloat(CAELIX_LOAD_U32x4(address + 16));
    const float4 c = asfloat(CAELIX_LOAD_U32x4(address + 32));
    LtSummary s;
    s.face[0] = a.x;
    s.face[1] = a.y;
    s.face[2] = a.z;
    s.face[3] = a.w;
    s.face[4] = b.x;
    s.face[5] = b.y;
    s.lo = float3(b.zw, c.x);
    s.hi = c.yzw;
    return s;
}

void LtStoreSummary(uint64_t address, LtSummary s)
{
    CAELIX_STORE_U32x4(address, asuint(float4(s.face[0], s.face[1], s.face[2], s.face[3])));
    CAELIX_STORE_U32x4(address + 16, asuint(float4(s.face[4], s.face[5], s.lo.x, s.lo.y)));
    CAELIX_STORE_U32x4(address + 32, asuint(float4(s.lo.z, s.hi)));
}

LtSummary LtEmptySummary()
{
    LtSummary s;
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        s.face[f] = 0.0f;
    }
    s.lo = float3(0, 0, 0);
    s.hi = float3(0, 0, 0);
    return s;
}

// The union of two summaries (an empty one, power 0, is skipped).
LtSummary LtUnion(LtSummary a, LtSummary b)
{
    if (LtPower(a.face) <= 0.0f)
    {
        return b;
    }
    if (LtPower(b.face) <= 0.0f)
    {
        return a;
    }
    LtSummary s;
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        s.face[f] = a.face[f] + b.face[f];
    }
    s.lo = min(a.lo, b.lo);
    s.hi = max(a.hi, b.hi);
    return s;
}

// The index entry of the brick at Morton position `key` of a group's block (binary search over the
// entries, which are in Morton order of their position), or kLtNone.
uint LtFindIndexEntry(uint64_t block, uint record_count, uint key, uint group_shift)
{
    uint lo = 0;
    uint hi = record_count;
    while (lo < hi)
    {
        const uint mid = (lo + hi) >> 1;
        const uint pos = CAELIX_LOAD_U32(block + kLtBlockIndexOffset + uint64_t(mid) * kLtIndexEntryBytes);
        const uint mid_key = LtMorton(BrickOfPos(pos, group_shift));
        if (mid_key == key)
        {
            return mid;
        }
        if (mid_key < key)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return kLtNone;
}

// A brick light record's header and the address of its entries.
struct LtRecord
{
    float face[6];
    uint3 lo;  // group-local voxels
    uint3 hi;
    uint pos;
    uint entry_count;
    uint counts_lo;
    uint counts_hi;
    uint64_t entries;
};

// The brick light record of index entry `entry` of a block.
LtRecord LtLoadEntryRecord(uint64_t block, uint entry)
{
    const uint64_t index_entry = block + kLtBlockIndexOffset + uint64_t(entry) * kLtIndexEntryBytes;
    const uint record_units = CAELIX_LOAD_U32(index_entry + kLtIndexRecordOffset);
    const uint64_t address = block + uint64_t(record_units) * kLtUnitBytes;
    const float4 a = asfloat(CAELIX_LOAD_U32x4(address));
    const uint4 b = CAELIX_LOAD_U32x4(address + 16);
    const uint4 c = CAELIX_LOAD_U32x4(address + 32);
    LtRecord r;
    r.face[0] = a.x;
    r.face[1] = a.y;
    r.face[2] = a.z;
    r.face[3] = a.w;
    r.face[4] = asfloat(b.x);
    r.face[5] = asfloat(b.y);
    r.lo = LtUnpackVoxel(b.z);
    r.hi = LtUnpackVoxel(b.w);
    r.pos = c.x;
    r.entry_count = c.y;
    r.counts_lo = c.z;
    r.counts_hi = c.w;
    r.entries = address + kLtRecordHeaderBytes;
    return r;
}

uint LtFamilyCount(LtRecord r, uint f)
{
    const uint word = f < kLtFamiliesPerCountWord ? r.counts_lo : r.counts_hi;
    return (word >> (kLtFamilyCountBits * (f % kLtFamiliesPerCountWord))) & kLtFamilyCountMask;
}

// An entity's current frame: object-to-world rotation (rows) and translation.
struct LtFrame
{
    float3x3 rotation;
    float3 translation;
};

LtFrame LtEntityFrame(uint64_t entity_infos, uint entity)
{
    const uint64_t info = entity_infos + uint64_t(entity) * kEntityInfoBytes;
    const float4 r0 = asfloat(CAELIX_LOAD_U32x4(info + kEntityInfoCurRow0));
    const float4 r1 = asfloat(CAELIX_LOAD_U32x4(info + kEntityInfoCurRow1));
    const float4 r2 = asfloat(CAELIX_LOAD_U32x4(info + kEntityInfoCurRow2));
    LtFrame frame;
    frame.rotation = float3x3(r0.xyz, r1.xyz, r2.xyz);
    frame.translation = float3(r0.w, r1.w, r2.w);
    return frame;
}

// A group's entity-local voxel origin and its entity (GroupInfo coord and entity).
float3 LtGroupOrigin(uint64_t group_infos, uint group, uint group_shift, out uint entity)
{
    const uint4 w0 = CAELIX_LOAD_U32x4(group_infos + uint64_t(group) * kGroupInfoBytes + kGroupInfoCoord);
    entity = w0.w;  // kGroupInfoEntity
    return float3(int3(w0.xyz) * int(uint(kBrickEdge) << group_shift));
}

// The occupied slots of a node (its `child` words; the slots are filled from 0).
uint LtNodeChildCount(uint4 children)
{
    return (children.x != kLtNone ? 1u : 0u) + (children.y != kLtNone ? 1u : 0u) +
           (children.z != kLtNone ? 1u : 0u) + (children.w != kLtNone ? 1u : 0u);
}

#endif  // CAELIX_LIGHT_TREE_LAYOUT_HLSL
