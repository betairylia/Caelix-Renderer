// Shared code of the light BVH kernels (docs/LIGHT_TREE.md section 2.2): the gather of the light
// bricks into the leaves of one tree and its build (bounds, Morton codes, radix sort, Karras binary
// radix tree, bottom-up aggregation, collapse to a 4-wide BVH, header). Every kernel of a build
// returns at once unless the tree's build state says `building` (set by caelix_light_bvh_begin), so
// the CPU records the whole sequence and the GPU decides (after a group pass that may not commit).
// The build scratch layout (kLtScratch*) is in common/caelix_light_tree_layout.hlsl.
#ifndef CAELIX_LIGHT_BVH_COMMON_HLSL
#define CAELIX_LIGHT_BVH_COMMON_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_light_tree_layout.hlsl"

// Push constants of the gather and build kernels (LightBvhPush). 56 bytes.
struct LightBvhPush
{
    uint64_t descriptor;
    uint64_t scratch;  // build scratch
    uint64_t offsets;  // gather: u32 per group index (counts, then exclusive offsets)
    uint tree;         // 0 static, 1 dynamic
    uint leaf_capacity;
    uint node_capacity;
    uint group_count;
    uint reason;         // begin: the CPU asks for a rebuild of this tree
    uint pass_recorded;  // begin: this flush recorded a group pass (a committed pass rebuilds both trees)
    uint group_base;     // gather: first group index of this dispatch
    uint pad;
};

CAELIX_PUSH_CONSTANTS(LightBvhPush, g_push);

uint64_t LtControl()
{
    return LtDescriptorU64(g_push.descriptor, kLtDescControl);
}

uint64_t LtTreeHeader()
{
    return LtControl() + kLtCtlTreeHeaders + uint64_t(g_push.tree) * kLtTreeHeaderBytes;
}

uint64_t LtTreeState()
{
    return LtControl() + kLtCtlTreeStates + uint64_t(g_push.tree) * kLtTreeStateBytes;
}

bool LtBuilding()
{
    return CAELIX_LOAD_U32(LtTreeState() + kLtStBuilding) != 0;
}

uint LtLeafCount()
{
    return CAELIX_LOAD_U32(LtTreeHeader() + kLtHdrLeafCount);
}

uint64_t LtLeaves()
{
    return LtDescriptorU64(g_push.descriptor, kLtDescLeaves0 + kLtDescTreeStride * g_push.tree);
}

uint64_t LtNodes()
{
    return LtDescriptorU64(g_push.descriptor, kLtDescNodes0 + kLtDescTreeStride * g_push.tree);
}

// Array `offset` (kLtScratch*) of the build scratch.
uint64_t LtScratch(uint offset)
{
    return g_push.scratch + uint64_t(offset) * g_push.leaf_capacity;
}
// The radix sort's keys and values of sort buffer 0 or 1.
uint64_t LtKeys(uint buffer)
{
    return g_push.scratch + uint64_t(buffer) * kLtScratchSortBufferStride * g_push.leaf_capacity;
}
uint64_t LtValues(uint buffer)
{
    return LtKeys(buffer) + uint64_t(kLtScratchValues - kLtScratchKeys) * g_push.leaf_capacity;
}

uint LtLoadScratch(uint offset, uint i)
{
    return CAELIX_LOAD_U32(LtScratch(offset) + uint64_t(i) * 4);
}
void LtStoreScratch(uint offset, uint i, uint value)
{
    CAELIX_STORE_U32(LtScratch(offset) + uint64_t(i) * 4, value);
}

// The summary of a child of a binary node: a leaf (kLtChildLeafBit, its sorted position) or the
// aggregate of a binary node. `coherent` reads the aggregate with coherent atomic word loads (the
// aggregation, where another invocation of the same dispatch wrote it).
LtSummary LtBinaryChildSummary(uint child, bool coherent)
{
    if ((child & kLtChildLeafBit) != 0)
    {
        const uint leaf = CAELIX_LOAD_U32(LtValues(0) + uint64_t(child & ~kLtChildLeafBit) * 4);
        return LtLoadSummary(LtLeaves() + uint64_t(leaf) * kLtLeafBytes);
    }
    const uint64_t address = LtScratch(kLtScratchAggregates) + uint64_t(child) * kLtSummaryBytes;
    if (!coherent)
    {
        return LtLoadSummary(address);
    }
    uint words[12];
    CAELIX_UNROLL
    for (uint w = 0; w < 12; ++w)
    {
        words[w] = CAELIX_ATOMIC_LOAD_U32(address + w * 4);
    }
    LtSummary s;
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        s.face[f] = asfloat(words[f]);
    }
    s.lo = asfloat(uint3(words[6], words[7], words[8]));
    s.hi = asfloat(uint3(words[9], words[10], words[11]));
    return s;
}

#endif  // CAELIX_LIGHT_BVH_COMMON_HLSL
