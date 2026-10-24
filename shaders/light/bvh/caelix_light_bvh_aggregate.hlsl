// Light BVH build: bottom-up aggregation of the binary radix tree, one thread per leaf. A thread
// walks up from its leaf; at every node the first arriving thread stops and the second computes the
// node's summary (sum of the face sums, union of the boxes) from both children and goes on. The
// arrival counter is an acquire/release atomic, and the summaries of internal nodes are written and
// read with coherent atomic word accesses, so the second thread sees what the first one wrote.
#include "light/bvh/caelix_light_bvh_common.hlsl"

void LtStoreAggregateCoherent(uint node, LtSummary s)
{
    const uint64_t address = LtScratch(kLtScratchAggregates) + uint64_t(node) * kLtSummaryBytes;
    uint words[12];
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        words[f] = asuint(s.face[f]);
    }
    words[6] = asuint(s.lo.x);
    words[7] = asuint(s.lo.y);
    words[8] = asuint(s.lo.z);
    words[9] = asuint(s.hi.x);
    words[10] = asuint(s.hi.y);
    words[11] = asuint(s.hi.z);
    CAELIX_UNROLL
    for (uint w = 0; w < 12; ++w)
    {
        CAELIX_ATOMIC_STORE_U32(address + w * 4, words[w]);
    }
}

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    const uint j = dispatch_id.x;
    if (j >= g_push.leaf_capacity || !LtBuilding())
    {
        return;
    }
    const uint n = LtLeafCount();
    if (j >= n || n < 2)
    {
        return;
    }
    uint node = LtLoadScratch(kLtScratchSortedLeafParent, j);
    for (uint depth = 0; depth < 128 && node != kLtNone; ++depth)
    {
        if (CAELIX_ATOMIC_ADD_ACQ_REL_U32(LtScratch(kLtScratchCounter) + uint64_t(node) * 4, 1u) == 0)
        {
            return;  // the other child's thread completes this node
        }
        const LtSummary s = LtUnion(LtBinaryChildSummary(LtLoadScratch(kLtScratchLeft, node), true),
                                    LtBinaryChildSummary(LtLoadScratch(kLtScratchRight, node), true));
        LtStoreAggregateCoherent(node, s);
        node = LtLoadScratch(kLtScratchParent, node);
    }
}
