// Light BVH build, first kernel after the gather (one workgroup): the leaf count, min(leaves found,
// leaf capacity), and the dropped rest into the tree's header and state (also for a scene without
// groups, where the gather dispatches nothing), then the bounds of the leaf box centres for the
// Morton codes.
#include "scene/caelix_light_bvh_common.hlsl"

groupshared float3 gs_lo[256];
groupshared float3 gs_hi[256];

[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_GroupThreadID)
{
    if (!LtBuilding())
    {
        return;  // uniform
    }
    const uint t = thread_id.x;
    const uint found = CAELIX_LOAD_U32(LtTreeState() + kLtStLeavesFound);
    const uint n = min(found, g_push.leaf_capacity);
    if (t == 0)
    {
        CAELIX_STORE_U32(LtTreeHeader() + kLtHdrLeafCount, n);
        CAELIX_STORE_U32(LtTreeState() + kLtStLeavesDropped, found - n);
    }
    const uint64_t leaves = LtLeaves();
    float3 lo = float3(3.0e38f, 3.0e38f, 3.0e38f);
    float3 hi = -lo;
    for (uint i = t; i < n; i += 256)
    {
        const LtSummary leaf = LtLoadSummary(leaves + uint64_t(i) * kLtLeafBytes);
        const float3 centre = 0.5f * (leaf.lo + leaf.hi);
        lo = min(lo, centre);
        hi = max(hi, centre);
    }
    gs_lo[t] = lo;
    gs_hi[t] = hi;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (t < stride)
        {
            gs_lo[t] = min(gs_lo[t], gs_lo[t + stride]);
            gs_hi[t] = max(gs_hi[t], gs_hi[t + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0)
    {
        const uint64_t bounds = LtTreeState() + kLtStBounds;
        CAELIX_STORE_U32x4(bounds, asuint(float4(gs_lo[0], gs_hi[0].x)));
        CAELIX_STORE_U32x2(bounds + 16, asuint(gs_hi[0].yz));
    }
}
