// The light cell fill (docs/LIGHT_NEE.md section 3.4) over the ACTIVE cells (the hash table's active
// list: cells looked up by the previous frame's trace; an indirect dispatch written by the table's
// maintenance), before the trace, after the maintenance. Workgroups of 64 threads:
//   mode "off":    one cell per workgroup, thread r owns reservoir r;
//   mode "rotate": kLightCellRotation (4) cells per workgroup, 16 threads per cell, thread j owns the
//                  reservoirs 4 j .. 4 j + 3, exactly one of which is due by the rotation, so the waves
//                  carry only reservoirs that draw (with one cell per workgroup a quarter of the lanes
//                  would draw and the waves would cost as much as in mode "off").
// Per cell:
// 1. The receiver region (render/caelix_light_cells.hlsl): the inflated cell box as a world AABB and
//    the receivers' normal R e_face.
// 2. A reservoir is refreshed when it is due: every fill in mode "off"; in mode "rotate" when
//    (r + fills) % kLightCellRotation == 0; in both modes at once when it is stale (the generation of
//    its group no longer matches: the entry may name another brick), and every reservoir of a cell
//    that has just been inserted (uninitialised: its payload may hold an earlier cell). An EMPTY
//    reservoir (a resampled draw whose candidates all had target 0) is not stale: refreshing it out
//    of turn would make empty reservoirs rarer than the resampling makes them, which biases the cell
//    upward. Reservoirs that are not due keep their value: each is a valid contribution weight of the
//    fill it was drawn in, so between refreshes a moving light lags by at most kLightCellRotation fills.
// 3. A refreshed reservoir draws M = `candidates` bricks from LightTreeSampleBrickBox for the region
//    (proposal p(y) exact, target p_hat(y) = the box importance of the drawn leaf's summary):
//    M = 1: W = 1 / p(y) (a plain draw); M = 2: streaming RIS, kept with probability w_i / w_sum,
//    w_i = p_hat(y_i) / p(y_i), W = (w_sum / M) / p_hat(kept). W is an unbiased contribution weight of
//    the brick for every receiver of the region: p(y) > 0 and p_hat(y) > 0 for every light brick that
//    can reach any receiver of it (the box importance is a bound).
// 4. One normalisation draw per cell and fill (the cell's thread fills % threads per cell),
//    independent of every reservoir: w_c = p_hat(y) / p(y) (0 for a failed draw) estimates C = sum
//    over the bricks of p_hat; it is folded into the header's C as a running mean over at most
//    kLightCellNormalisationFills fills. The MIS proxy reads the current C (LightNeeProxyArea).
// 5. The cell's first thread, after a barrier (every thread has read the header): the sun hint counters
//    lose 1/8 (0 for a new cell), fills + 1, state ready.
// The random stream of a reservoir is a function of (cell slot, reservoir, frame seed) only. The counters
// of the fresh draws, the normalisation draws and the early (stale or new) refreshes go into the hash
// table's user counter words (one atomic per wave).
#include "render/caelix_light_cells.hlsl"

struct LightCellFillPush
{
    uint64_t payloads;
    uint64_t active;
    uint64_t dispatch;
    uint64_t counters;
    uint64_t light_tree;
    uint frame_seed;
    uint mode;
    uint candidates;
    uint pad;
};

CAELIX_PUSH_CONSTANTS(LightCellFillPush, g_push);

uint FillHash(uint x)
{
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

// One brick for the region: M candidates, W as in step 3.
uint4 FillReservoir(LightTree tree, LightCellRegion region, uint candidates, inout uint rng)
{
    float w_sum = 0.0f;
    float kept_target = 0.0f;
    float kept_pdf = 0.0f;
    LightBrickRef kept;
    kept.group = kLightCellNoGroup;
    kept.entry = 0;
    kept.generation = 0;
    for (uint m = 0; m < candidates; ++m)
    {
        LightBrickRef brick;
        float pdf;
        uint leaf;
        float target;
        const bool ok =
            LightTreeSampleBrickBox(tree, region.lo, region.hi, region.normal, rng, brick, pdf, leaf, target);
        const float u = LightTreeRandomFloat(rng);
        if (!ok || !(pdf > 0.0f))
        {
            continue;
        }
        if (candidates == 1)
        {
            kept = brick;
            kept_pdf = pdf;
            continue;
        }
        const float w = target > 0.0f ? target / pdf : 0.0f;
        if (!(w > 0.0f))
        {
            continue;
        }
        w_sum += w;
        if (u * w_sum < w)
        {
            kept = brick;
            kept_target = target;
        }
    }
    float weight = 0.0f;
    if (kept.group != kLightCellNoGroup)
    {
        weight = candidates == 1 ? 1.0f / kept_pdf : (w_sum / float(candidates)) / kept_target;
    }
    if (!(weight > 0.0f) || !(weight <= 3.0e38f))
    {
        return uint4(kLightCellNoGroup, 0u, 0u, 0u);
    }
    return uint4(kept.group, kept.entry, asuint(weight), kept.generation);
}

[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID)
{
    const bool rotate = g_push.mode == kLightCellModeRotate;
    const uint cells_per_group = rotate ? kLightCellRotation : 1u;
    const uint threads_per_cell = kLightCellFillThreads / cells_per_group;
    const uint reservoirs_per_thread = kLightCellReservoirCount / threads_per_cell;
    const uint j = thread_id.x % threads_per_cell;
    const uint group_index = group_id.y * kHashDispatchRowGroups + group_id.x;
    const uint index = group_index * cells_per_group + thread_id.x / threads_per_cell;
    // Every thread reaches the barrier below; a thread without a cell only skips the work.
    const bool active = index < CAELIX_LOAD_U32(g_push.dispatch + kHashDispatchCount);
    const uint slot = active ? CAELIX_LOAD_U32(g_push.active + uint64_t(index) * 4) : 0u;
    const uint64_t cell = LightCellAddress(g_push.payloads, slot);

    bool fresh = false;
    uint fills = 0;
    uint fresh_draws = 0;
    uint early = 0;
    bool normaliser = false;
    bool open = false;
    if (active)
    {
        // The header (before the barrier).
        fresh = CAELIX_LOAD_U32(cell + kLcState) == kHashStateUninitialised;
        fills = fresh ? 0u : CAELIX_LOAD_U32(cell + kLcFills);
        LightTree tree;
        open = LightTreeOpen(g_push.light_tree, tree);
        LightCellRegion region;
        region.lo = float3(0.0f, 0.0f, 0.0f);
        region.hi = float3(0.0f, 0.0f, 0.0f);
        region.normal = float3(0.0f, 0.0f, 0.0f);
        if (open)
        {
            region = LightCellRegionOfHeader(LtDescriptorU64(g_push.light_tree, kLtDescEntityInfos), cell);
        }

        // 2-3. This thread's reservoirs.
        for (uint k = 0; k < reservoirs_per_thread; ++k)
        {
            const uint r = j * reservoirs_per_thread + k;
            const uint64_t reservoir = cell + kLcReservoirs + uint64_t(r) * kLightCellReservoirBytes;
            const uint4 old = CAELIX_LOAD_U32x4(reservoir);
            const bool scheduled = !rotate || (r + fills) % kLightCellRotation == 0u;
            const bool mismatch =
                old.x != kLightCellNoGroup && !LightBrickRefValid(tree, LightCellReservoirRef(old));
            const bool stale = fresh || !open || mismatch;
            if (scheduled || stale)
            {
                uint rng = FillHash(slot * kLightCellReservoirCount + r) ^
                           FillHash(g_push.frame_seed ^ 0x3C6EF372u);
                const uint4 value = open ? FillReservoir(tree, region, g_push.candidates, rng)
                                         : uint4(kLightCellNoGroup, 0u, 0u, 0u);
                CAELIX_STORE_U32x4(reservoir, value);
                fresh_draws += open ? g_push.candidates : 0u;
                early += scheduled ? 0u : 1u;
            }
        }

        // 4. The normalisation draw.
        normaliser = j == fills % threads_per_cell;
        if (normaliser)
        {
            float w_c = 0.0f;
            if (open)
            {
                uint norm_rng = FillHash(slot ^ 0x5BD1E995u) ^ FillHash(g_push.frame_seed ^ 0x27D4EB2Fu);
                LightBrickRef brick;
                float pdf;
                uint leaf;
                float target;
                const bool drawn = LightTreeSampleBrickBox(tree, region.lo, region.hi, region.normal,
                                                           norm_rng, brick, pdf, leaf, target);
                if (drawn && pdf > 0.0f && target > 0.0f)
                {
                    w_c = target / pdf;
                }
            }
            const uint count = fresh ? 0u : CAELIX_LOAD_U32(cell + kLcNormCount);
            const float norm = fresh ? 0.0f : asfloat(CAELIX_LOAD_U32(cell + kLcNorm));
            const uint next_count = min(count + 1u, kLightCellNormalisationFills);
            const float next_norm = norm + (w_c - norm) / float(next_count);
            CAELIX_STORE_U32x2(cell + kLcNorm, uint2(asuint(next_norm), next_count));
        }
    }

    // The counters (one atomic per wave).
    const uint wave_fresh = WaveActiveSum(fresh_draws);
    const uint wave_norm = WaveActiveCountBits(normaliser && open);
    const uint wave_early = WaveActiveSum(early);
    if (WaveIsFirstLane())
    {
        if (wave_fresh != 0u)
        {
            CAELIX_ATOMIC_ADD_U32(g_push.counters + kLcCounterFreshDraws, wave_fresh);
        }
        if (wave_norm != 0u)
        {
            CAELIX_ATOMIC_ADD_U32(g_push.counters + kLcCounterNormDraws, wave_norm);
        }
        if (wave_early != 0u)
        {
            CAELIX_ATOMIC_ADD_U32(g_push.counters + kLcCounterStaleRefreshed, wave_early);
        }
    }

    // 5. The header words that every thread of the cell read above.
    GroupMemoryBarrierWithGroupSync();
    if (active && j == 0)
    {
        uint tests = 0;
        uint visible = 0;
        if (!fresh)
        {
            tests = CAELIX_LOAD_U32(cell + kLcSunTests);
            visible = CAELIX_LOAD_U32(cell + kLcSunVisible);
            tests -= tests >> kLightCellSunDecayShift;
            visible -= visible >> kLightCellSunDecayShift;
        }
        CAELIX_STORE_U32x2(cell + kLcSunTests, uint2(tests, visible));
        CAELIX_STORE_U32(cell + kLcFills, fills + 1u);
        CAELIX_STORE_U32(cell + kLcState, kHashStateReady);
    }
}
