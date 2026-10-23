// Light BVH build: the collapse to 4-wide nodes keeps the binary nodes at even depth (the root is at
// depth 0). One thread per binary node walks up to the root for its depth (stored in the counter
// array, free after the aggregation) and writes wide[i] = 1 for an even depth, else 0 (also for
// i >= n - 1). caelix_light_bvh_scan then turns the flags into 4-wide node indices.
#include "scene/caelix_light_bvh_common.hlsl"

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    const uint i = dispatch_id.x;
    if (i >= g_push.leaf_capacity || !LtBuilding())
    {
        return;
    }
    const uint n = LtLeafCount();
    uint flag = 0;
    if (i + 1 < n)
    {
        uint depth = 0;
        uint node = LtLoadScratch(kLtScratchParent, i);
        for (uint guard = 0; guard < 128 && node != kLtNone; ++guard)
        {
            ++depth;
            node = LtLoadScratch(kLtScratchParent, node);
        }
        LtStoreScratch(kLtScratchCounter, i, depth);
        flag = (depth & 1u) == 0 ? 1u : 0u;
    }
    LtStoreScratch(kLtScratchWide, i, flag);
}
