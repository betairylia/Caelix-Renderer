// Light BVH: exclusive prefix sum of `count` u32 values in place (one workgroup, chunks of 1024),
// used by the gather (leaf offsets), the radix sort (digit-major histograms) and the collapse to
// 4-wide nodes (node indices). Writes the total to total_address (0 = nowhere). Returns at once when
// gate_address is not 0 and the u32 there is 0 (the tree is not being built).
#include "common/caelix_platform.hlsl"
#include "common/caelix_workgroup_scan.hlsl"

// Push constants (LightScanPush). 32 bytes.
struct LightScanPush
{
    uint64_t data;
    uint64_t total_address;
    uint64_t gate_address;
    uint count;
    uint pad;
};

CAELIX_PUSH_CONSTANTS(LightScanPush, g_push);

static const uint kThreads = 256;

[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_GroupThreadID)
{
    if (g_push.gate_address != 0 && CAELIX_LOAD_U32(g_push.gate_address) == 0)
    {
        return;  // uniform
    }
    const uint t = thread_id.x;
    // Chunks of 4 * kThreads values, 4 per thread.
    uint carry = 0;
    for (uint base = 0; base < g_push.count; base += 4 * kThreads)
    {
        uint values[4];
        CAELIX_UNROLL
        for (uint i = 0; i < 4; ++i)
        {
            const uint index = base + 4 * t + i;
            values[i] = index < g_push.count ? CAELIX_LOAD_U32(g_push.data + uint64_t(index) * 4) : 0u;
        }
        uint total;
        WorkgroupExclusiveSum4(values, t, kThreads, total);
        CAELIX_UNROLL
        for (uint j = 0; j < 4; ++j)
        {
            const uint index = base + 4 * t + j;
            if (index < g_push.count)
            {
                CAELIX_STORE_U32(g_push.data + uint64_t(index) * 4, carry + values[j]);
            }
        }
        carry += total;
    }
    if (t == 0 && g_push.total_address != 0)
    {
        CAELIX_STORE_U32(g_push.total_address, carry);
    }
}
