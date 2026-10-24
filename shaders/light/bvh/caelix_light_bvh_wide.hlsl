// Light BVH build: the collapse of the binary radix tree to 4-wide nodes, one thread per binary node
// at even depth (4-wide node index w = the scanned flag). Its slots are the children of its two
// children: a leaf child takes one slot, an internal child (odd depth) gives its own two children
// (leaves, or internal nodes at even depth = 4-wide nodes). So a 4-wide node has 2 .. 4 slots; the
// node's summaries are inlined per slot. The thread also writes the parent links of its slots
// (leaf.parent = w << 2 | slot; child node's parent / parent_slot). A node index at or above the
// node capacity is not written (caelix_light_bvh_finish then marks the tree empty and the overflow).
#include "light/bvh/caelix_light_bvh_common.hlsl"

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    const uint i = dispatch_id.x;
    if (i + 1 >= g_push.leaf_capacity || !LtBuilding())
    {
        return;
    }
    const uint n = LtLeafCount();
    if (i + 1 >= n || (LtLoadScratch(kLtScratchCounter, i) & 1u) != 0)
    {
        return;
    }
    const uint w = LtLoadScratch(kLtScratchWide, i);
    if (w >= g_push.node_capacity)
    {
        return;
    }
    // Slots: binary children (left, right) of node i, an internal child replaced by its children.
    uint slots[4] = {kLtNone, kLtNone, kLtNone, kLtNone};
    uint count = 0;
    const uint children[2] = {LtLoadScratch(kLtScratchLeft, i), LtLoadScratch(kLtScratchRight, i)};
    for (uint c = 0; c < 2; ++c)
    {
        const uint child = children[c];
        if ((child & kLtChildLeafBit) != 0)
        {
            slots[count++] = child;
        }
        else
        {
            slots[count++] = LtLoadScratch(kLtScratchLeft, child);
            slots[count++] = LtLoadScratch(kLtScratchRight, child);
        }
    }
    const uint64_t nodes = LtNodes();
    const uint64_t node = nodes + uint64_t(w) * kLtNodeBytes;
    const uint64_t leaves = LtLeaves();
    uint child_words[4] = {kLtNone, kLtNone, kLtNone, kLtNone};
    float faces[24];
    float bounds[24];  // lo[4][3], then hi[4][3]
    for (uint s = 0; s < 4; ++s)
    {
        LtSummary summary = LtEmptySummary();
        if (s < count)
        {
            const uint slot = slots[s];
            summary = LtBinaryChildSummary(slot, false);
            if ((slot & kLtChildLeafBit) != 0)
            {
                const uint leaf = CAELIX_LOAD_U32(LtValues(0) + uint64_t(slot & ~kLtChildLeafBit) * 4);
                child_words[s] = kLtChildLeafBit | leaf;
                CAELIX_STORE_U32(leaves + uint64_t(leaf) * kLtLeafBytes + kLtLeafParent,
                                 (w << kLtParentSlotBits) | s);
            }
            else
            {
                const uint child_node = LtLoadScratch(kLtScratchWide, slot);
                child_words[s] = child_node;
                if (child_node < g_push.node_capacity)
                {
                    const uint64_t child_address = nodes + uint64_t(child_node) * kLtNodeBytes;
                    CAELIX_STORE_U32x2(child_address + kLtNodeParent, uint2(w, s));  // parent, parent_slot
                }
            }
        }
        for (uint f = 0; f < 6; ++f)
        {
            faces[6 * s + f] = summary.face[f];
        }
        bounds[3 * s] = summary.lo.x;
        bounds[3 * s + 1] = summary.lo.y;
        bounds[3 * s + 2] = summary.lo.z;
        bounds[12 + 3 * s] = summary.hi.x;
        bounds[12 + 3 * s + 1] = summary.hi.y;
        bounds[12 + 3 * s + 2] = summary.hi.z;
    }
    for (uint q = 0; q < 6; ++q)
    {
        const float4 face4 = float4(faces[4 * q], faces[4 * q + 1], faces[4 * q + 2], faces[4 * q + 3]);
        const float4 bound4 = float4(bounds[4 * q], bounds[4 * q + 1], bounds[4 * q + 2], bounds[4 * q + 3]);
        CAELIX_STORE_U32x4(node + kLtNodeChildFace + 16 * q, asuint(face4));
        CAELIX_STORE_U32x4(node + kLtNodeChildLo + 16 * q, asuint(bound4));
    }
    CAELIX_STORE_U32x4(node + kLtNodeChild,
                       uint4(child_words[0], child_words[1], child_words[2], child_words[3]));
    if (i == 0)
    {
        CAELIX_STORE_U32x2(node + kLtNodeParent, uint2(kLtNone, 0u));  // the root: parent, parent_slot
    }
}
