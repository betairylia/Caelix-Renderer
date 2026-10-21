// Light table scan kernel (docs/LIGHT_LIST.md, section 1.2): after caelix_light_compact, turns
// cdf[0 .. count) of the scene's light table (common/caelix_light.hlsl) into the inclusive prefix
// sum of the entries' total weights, in place.
//
// One workgroup of 1024 threads: chunks of 1024 values, each scanned in groupshared memory
// (Hillis-Steele, 10 steps) and offset by the carried sum of the earlier chunks. Thread 0 then
// clamps the header's count to the capacity (the compact kernel counts dropped bricks in it too),
// and writes the capacity and total_weight = cdf[count - 1] (0 when count is 0).
//
// Inputs: LightPush (lib/src/scene/gpu_structs.h; max_bricks is the capacity); the light table is
// set 0 binding 0.
#include "common/caelix_platform.hlsl"
#include "common/caelix_light.hlsl"

struct LightPush
{
    uint64_t group_infos;
    uint64_t slot_tables;
    uint64_t page_table;
    uint64_t light_words;
    uint64_t light_indices;
    uint64_t materials;
    uint group_base;
    uint group_count;
    uint max_bricks;
    uint records_per_page_shift;
    uint record_bytes;
    uint group_shift;
    uint encoding;
    float min_emission;
};

CAELIX_PUSH_CONSTANTS(LightPush, g_push);
CAELIX_BINDING(0, 0) RWByteAddressBuffer g_table;

static const uint kScanThreads = 1024;

groupshared float gs_scan[2][kScanThreads];
groupshared float gs_carry;

[numthreads(1024, 1, 1)]
void main(uint3 thread_id : SV_GroupThreadID)
{
    const uint t = thread_id.x;
    const uint count = min(g_table.Load(kLightHeaderCount), g_push.max_bricks);
    const uint cdf = kLightHeaderBytes + g_push.max_bricks * kLightEntryBytes;
    if (t == 0)
    {
        gs_carry = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();
    CAELIX_LOOP
    for (uint base = 0; base < count; base += kScanThreads)
    {
        const uint i = base + t;
        uint source = 0;
        gs_scan[0][t] = i < count ? asfloat(g_table.Load(cdf + i * 4)) : 0.0f;
        GroupMemoryBarrierWithGroupSync();
        CAELIX_UNROLL
        for (uint step = 1; step < kScanThreads; step <<= 1)
        {
            const float value = gs_scan[source][t] + (t >= step ? gs_scan[source][t - step] : 0.0f);
            gs_scan[source ^ 1u][t] = value;
            source ^= 1u;
            GroupMemoryBarrierWithGroupSync();
        }
        const float carry = gs_carry;
        if (i < count)
        {
            g_table.Store(cdf + i * 4, asuint(carry + gs_scan[source][t]));
        }
        GroupMemoryBarrierWithGroupSync();
        if (t == 0)
        {
            gs_carry = carry + gs_scan[source][kScanThreads - 1];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0)
    {
        g_table.Store(kLightHeaderCount, count);
        g_table.Store(kLightHeaderCapacity, g_push.max_bricks);
        g_table.Store(kLightHeaderTotal, asuint(gs_carry));
    }
}
