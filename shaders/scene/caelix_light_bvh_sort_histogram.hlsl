// Light BVH radix sort pass, kernel 1 of 2 (scene/caelix_light_bvh_sort.hlsl): the digit counts of
// one block of 1024 elements, written to histogram[digit * block_count + block] (0 for a block past
// the element count).
#include "scene/caelix_light_bvh_sort.hlsl"

groupshared uint gs_counts[kLtSortDigits];

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID)
{
    if (LtSortSkipped())
    {
        return;  // uniform
    }
    const uint t = thread_id.x;
    const uint block = group_id.x;
    if (t < kLtSortDigits)
    {
        gs_counts[t] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint count = CAELIX_LOAD_U32(g_push.count_address);
    for (uint i = 0; i < kLtSortBlock / kLtSortThreads; ++i)
    {
        const uint element = block * kLtSortBlock + i * kLtSortThreads + t;
        if (element < count)
        {
            const uint key = CAELIX_LOAD_U32(g_push.keys_in + uint64_t(element) * 4);
            const uint digit = (key >> g_push.shift) & kLtSortDigitMask;
            InterlockedAdd(gs_counts[digit], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t < kLtSortDigits)
    {
        CAELIX_STORE_U32(g_push.histogram + (uint64_t(t) * g_push.block_count + block) * 4, gs_counts[t]);
    }
}
