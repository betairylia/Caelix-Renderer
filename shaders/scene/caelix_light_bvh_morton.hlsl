// Light BVH build: one thread per leaf writes the sort input (30-bit Morton code of the leaf box
// centre over the centre bounds, leaf index) and clears the aggregation counters.
#include "scene/caelix_light_bvh_common.hlsl"

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    const uint i = dispatch_id.x;
    if (i >= g_push.leaf_capacity || !LtBuilding())
    {
        return;
    }
    LtStoreScratch(kLtScratchCounter, i, 0u);
    if (i >= LtLeafCount())
    {
        return;
    }
    const uint64_t bounds = LtTreeState() + kLtStBounds;
    const float4 b0 = asfloat(CAELIX_LOAD_U32x4(bounds));
    const float2 b1 = asfloat(CAELIX_LOAD_U32x2(bounds + 16));
    const float3 lo = b0.xyz;
    const float3 extent = max(float3(b0.w, b1) - lo, float3(1e-6f, 1e-6f, 1e-6f));
    const LtSummary leaf = LtLoadSummary(LtLeaves() + uint64_t(i) * kLtLeafBytes);
    const float3 centre = 0.5f * (leaf.lo + leaf.hi);
    const uint3 cell = uint3(clamp((centre - lo) / extent * 1024.0f, 0.0f, 1023.0f));
    CAELIX_STORE_U32(LtKeys(0) + uint64_t(i) * 4, LtMorton(cell));
    CAELIX_STORE_U32(LtValues(0) + uint64_t(i) * 4, i);
}
