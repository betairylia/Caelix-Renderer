// Exclusive prefix sums over one workgroup: wave prefix sums plus the totals of the earlier waves
// (the waves of a workgroup are its consecutive thread ranges). Every thread of the workgroup calls
// these functions in uniform control flow; each call has two workgroup barriers.
#ifndef CAELIX_WORKGROUP_SCAN_HLSL
#define CAELIX_WORKGROUP_SCAN_HLSL

#include "common/caelix_platform.hlsl"

// Waves of the largest workgroup that uses the scan (256 threads at the smallest wave size of 4).
static const uint kWorkgroupScanMaxWaves = 64;

groupshared uint gs_wave_sums[kWorkgroupScanMaxWaves];

// The sum of `value` over the threads before thread t of a workgroup of `threads` threads; `total`
// is the sum over all threads.
uint WorkgroupExclusiveSum(uint value, uint t, uint threads, out uint total)
{
    const uint lanes = WaveGetLaneCount();
    const uint wave = t / lanes;
    const uint waves = threads / lanes;
    const uint prefix = WavePrefixSum(value);
    const uint wave_total = WaveActiveSum(value);
    if (WaveIsFirstLane())
    {
        gs_wave_sums[wave] = wave_total;
    }
    GroupMemoryBarrierWithGroupSync();
    uint base = 0;
    total = 0;
    for (uint w = 0; w < waves; ++w)
    {
        const uint sum = gs_wave_sums[w];
        base += w < wave ? sum : 0u;
        total += sum;
    }
    GroupMemoryBarrierWithGroupSync();
    return base + prefix;
}

// Four values per thread, thread t owning elements 4 t .. 4 t + 3: every value becomes the sum of the
// elements before it; `total` is the sum over the workgroup.
void WorkgroupExclusiveSum4(inout uint values[4], uint t, uint threads, out uint total)
{
    const uint sum = values[0] + values[1] + values[2] + values[3];
    uint running = WorkgroupExclusiveSum(sum, t, threads, total);
    CAELIX_UNROLL
    for (uint i = 0; i < 4; ++i)
    {
        const uint value = values[i];
        values[i] = running;
        running += value;
    }
}

#endif  // CAELIX_WORKGROUP_SCAN_HLSL
