// Light BVH build, last kernel (one thread): the tree header. n = 0: an empty tree. n = 1: one
// 4-wide node with the single leaf in slot 0. n >= 2: the root summary is the binary root's, the
// root is node 0, node_count comes from the scan of the even-depth flags; more nodes than the
// capacity leave the tree empty and set the overflow flag (the CPU grows the node buffer and
// rebuilds). Counts the build.
#include "light/bvh/caelix_light_bvh_common.hlsl"

[numthreads(1, 1, 1)]
void main()
{
    if (!LtBuilding())
    {
        return;
    }
    const uint64_t header = LtTreeHeader();
    const uint64_t state = LtTreeState();
    const uint n = LtLeafCount();
    LtSummary root = LtEmptySummary();
    uint root_index = kLtNone;
    uint node_count = 0;
    uint overflow = 0;
    if (n == 1)
    {
        const uint64_t leaf = LtLeaves();
        root = LtLoadSummary(leaf);
        const uint64_t node = LtNodes();
        // Slot 0 holds the leaf; the summaries of slots 1 .. 3 are zero.
        for (uint q = 0; q < 6; ++q)
        {
            CAELIX_STORE_U32x4(node + kLtNodeChildFace + 16 * q, uint4(0, 0, 0, 0));
            CAELIX_STORE_U32x4(node + kLtNodeChildLo + 16 * q, uint4(0, 0, 0, 0));
        }
        CAELIX_STORE_U32x4(node + kLtNodeChildFace,
                           asuint(float4(root.face[0], root.face[1], root.face[2], root.face[3])));
        CAELIX_STORE_U32x2(node + kLtNodeChildFace + 16, asuint(float2(root.face[4], root.face[5])));
        CAELIX_STORE_U32x2(node + kLtNodeChildLo, asuint(root.lo.xy));
        CAELIX_STORE_U32(node + kLtNodeChildLo + 8, asuint(root.lo.z));
        CAELIX_STORE_U32x4(node + kLtNodeChildHi, asuint(float4(root.hi, 0.0f)));
        CAELIX_STORE_U32x4(node + kLtNodeChild, uint4(kLtChildLeafBit | 0u, kLtNone, kLtNone, kLtNone));
        CAELIX_STORE_U32x2(node + kLtNodeParent, uint2(kLtNone, 0u));  // parent, parent_slot
        CAELIX_STORE_U32(leaf + kLtLeafParent, 0u);                     // node 0, slot 0
        root_index = 0;
        node_count = 1;
    }
    else if (n >= 2)
    {
        node_count = CAELIX_LOAD_U32(header + kLtHdrNodeCount);  // the scan's total
        if (node_count > g_push.node_capacity)
        {
            overflow = 1;
        }
        else
        {
            root = LtLoadSummary(LtScratch(kLtScratchAggregates));  // binary node 0
            root_index = 0;
        }
    }
    LtStoreSummary(header, root);
    // root, leaf_count, node_count, pad
    CAELIX_STORE_U32x4(header + kLtHdrRoot, uint4(root_index, overflow != 0 ? 0u : n, node_count, 0u));
    CAELIX_STORE_U32(state + kLtStNodeOverflow, overflow);
    CAELIX_STORE_U32(state + kLtStBuilds, CAELIX_LOAD_U32(state + kLtStBuilds) + 1u);
}
