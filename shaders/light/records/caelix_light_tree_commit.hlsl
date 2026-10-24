// Light tree group pass, kernel 4 of 4 (docs/LIGHT_TREE.md section 3): one workgroup closes the
// pass. kLtPassOk: the target heap buffer becomes current, the totals over the rows are recounted
// for the stats, passes + 1. kLtPassFailed: passes + 1, failed_passes + 1 (the rows and flags stay;
// the next pass retries). kLtPassNoChange: nothing.
#include "light/records/caelix_light_tree_group.hlsl"

groupshared uint64_t gs_sums[3][kLtThreads];  // light bricks, light faces, dropped
groupshared uint gs_groups[kLtThreads];

[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_GroupThreadID)
{
    const uint t = thread_id.x;
    const uint64_t control = LtDescriptorU64(g_push.descriptor, kLtDescControl);
    const uint result = CAELIX_LOAD_U32(control + kLtCtlPassResult);
    if (result == kLtPassNoChange)
    {
        return;  // uniform
    }
    if (result == kLtPassFailed)
    {
        if (t == 0)
        {
            CAELIX_STORE_U32(control + kLtCtlPasses, CAELIX_LOAD_U32(control + kLtCtlPasses) + 1u);
            CAELIX_STORE_U32(control + kLtCtlFailedPasses,
                             CAELIX_LOAD_U32(control + kLtCtlFailedPasses) + 1u);
        }
        return;
    }
    const uint64_t rows = LtDescriptorU64(g_push.descriptor, kLtDescGroupRows);
    uint64_t bricks = 0;
    uint64_t faces = 0;
    uint64_t dropped = 0;
    uint groups = 0;
    for (uint g = t; g < g_push.group_count; g += kLtThreads)
    {
        // cpu_flags, bricks, light_faces, pad
        const uint4 counts = CAELIX_LOAD_U32x4(rows + uint64_t(g) * kLtGroupRowBytes + kLtRowCpuFlags);
        bricks += counts.y & kLtRowBricksMask;
        faces += counts.z;
        dropped += counts.y >> kLtRowDroppedShift;
        groups += (counts.y & kLtRowBricksMask) != 0 ? 1u : 0u;
    }
    gs_sums[0][t] = bricks;
    gs_sums[1][t] = faces;
    gs_sums[2][t] = dropped;
    gs_groups[t] = groups;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = kLtThreads / 2; stride > 0; stride >>= 1)
    {
        if (t < stride)
        {
            gs_sums[0][t] += gs_sums[0][t + stride];
            gs_sums[1][t] += gs_sums[1][t + stride];
            gs_sums[2][t] += gs_sums[2][t + stride];
            gs_groups[t] += gs_groups[t + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0)
    {
        CAELIX_STORE_U32(control + kLtCtlCurrent, 1u - (CAELIX_LOAD_U32(control + kLtCtlCurrent) & 1u));
        CAELIX_STORE_U32(control + kLtCtlLightGroups, gs_groups[0]);
        CAELIX_STORE_U32(control + kLtCtlBricksDropped,
                         uint(min(gs_sums[2][0], uint64_t(0xFFFFFFFFu))));
        CAELIX_STORE_U32x2(control + kLtCtlLightBricks,
                           uint2(uint(gs_sums[0][0]), uint(gs_sums[0][0] >> 32)));
        CAELIX_STORE_U32x2(control + kLtCtlLightFaces,
                           uint2(uint(gs_sums[1][0]), uint(gs_sums[1][0] >> 32)));
        CAELIX_STORE_U32(control + kLtCtlPasses, CAELIX_LOAD_U32(control + kLtCtlPasses) + 1u);
    }
}
