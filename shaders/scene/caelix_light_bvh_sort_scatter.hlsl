// Light BVH radix sort pass, kernel 2 of 2 (scene/caelix_light_bvh_sort.hlsl): one block of 1024
// elements goes to its scanned digit offsets. Stable: the block's elements are taken in 4 rounds of
// 256 in input order; inside a round an element's rank among equal digits comes from the wave's
// prefix ballot counts plus the counts of the earlier waves and rounds.
#include "scene/caelix_light_bvh_sort.hlsl"

groupshared uint gs_running[kLtSortDigits];  // elements per digit written by the earlier rounds
groupshared uint gs_wave_counts[kLtSortMaxWaves][kLtSortDigits];  // elements per digit and wave, this round

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID)
{
    if (LtSortSkipped())
    {
        return;  // uniform
    }
    const uint t = thread_id.x;
    const uint block = group_id.x;
    const uint lanes = WaveGetLaneCount();
    const uint wave = t / lanes;
    const uint waves = kLtSortThreads / lanes;
    if (t < kLtSortDigits)
    {
        gs_running[t] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint count = CAELIX_LOAD_U32(g_push.count_address);
    for (uint r = 0; r < kLtSortBlock / kLtSortThreads; ++r)
    {
        const uint element = block * kLtSortBlock + r * kLtSortThreads + t;
        const bool valid = element < count;
        uint key = 0;
        uint value = 0;
        uint digit = kLtSortDigits;  // none
        if (valid)
        {
            key = CAELIX_LOAD_U32(g_push.keys_in + uint64_t(element) * 4);
            value = CAELIX_LOAD_U32(g_push.values_in + uint64_t(element) * 4);
            digit = (key >> g_push.shift) & kLtSortDigitMask;
        }
        uint rank = 0;
        for (uint d = 0; d < kLtSortDigits; ++d)
        {
            const bool match = digit == d;
            const uint before = WavePrefixCountBits(match);
            const uint total = WaveActiveCountBits(match);
            rank = match ? before : rank;
            if (WaveIsFirstLane())
            {
                gs_wave_counts[wave][d] = total;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (valid)
        {
            uint offset = gs_running[digit] + rank;
            for (uint w = 0; w < wave; ++w)
            {
                offset += gs_wave_counts[w][digit];
            }
            const uint64_t digit_base = g_push.histogram + (uint64_t(digit) * g_push.block_count + block) * 4;
            const uint destination = CAELIX_LOAD_U32(digit_base) + offset;
            CAELIX_STORE_U32(g_push.keys_out + uint64_t(destination) * 4, key);
            CAELIX_STORE_U32(g_push.values_out + uint64_t(destination) * 4, value);
        }
        GroupMemoryBarrierWithGroupSync();
        if (t < kLtSortDigits)
        {
            uint sum = 0;
            for (uint w2 = 0; w2 < waves; ++w2)
            {
                sum += gs_wave_counts[w2][t];
            }
            gs_running[t] += sum;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
