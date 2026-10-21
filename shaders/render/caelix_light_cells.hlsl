// Light cell fill kernel (docs/LIGHT_LIST.md, section 2.2): refills the 32 reservoirs of every
// live light cell (render/caelix_light_sample.hlsl) from the scene's light table before the trace.
//
// One workgroup of 32 threads per cell slot (dispatch 1024 x capacity / 1024: slot = y * 1024 + x);
// thread r owns reservoir r. An empty slot (key_lo 0) is skipped. A cell not looked up for 64
// frames is evicted (key 0, reservoirs empty). Without a light table or with an empty one the
// reservoirs are emptied. Thread 0 of every cell that stays counts it in the usage counters after
// the last cell (live; active when the previous frame's trace looked it up). Otherwise each thread
// draws K = fc.light_candidates bricks from the table's CDF (proposal p_i = total_i / total) and
// keeps one by streaming RIS with the brick-level target p_hat_i = total_i * G_i (LightCellG):
//   G_i = orient / max(|centre_i - cell centre|^2, (cell half size + 4)^2),
//   orient = max(0.1, dot(mean normal_i, direction from the brick centre to the cell centre)),
//            1 without a mean normal (the normal comes from the entry's face weights,
//            LightBrickNormalLocal: the luminance-weighted open-face direction, docs/LIGHT_FACES.md),
// all in world space (entity transforms of the brick and of the cell). The reservoir stores the
// entry index and W = (w_sum / K) / p_hat(selected), w_i = p_hat_i / p_i: an unbiased contribution
// weight of the brick for every point of the cell, since p_hat_i > 0 for every entry. Random stream
// per (slot, reservoir, frame).
//
// Temporal reuse (kGiFlagLightCellTemporal, static scene only): after the K fresh candidates each
// thread merges the reservoir it wrote in the previous fill as one more candidate with the weight
// p_hat(y_old) * W_old * M_old, where M_old is the cell header's candidate count clamped to
// fc.light_temporal_frames x K (0 drops the history: the renderer sets 0 after a light table
// rebuild or a reset). The result is W = (w_sum / (M_old + K)) / p_hat(selected) and the header
// stores M = M_old + K (uniform over the 32 reservoirs: every thread merges the same way). Unbiased:
// E[p_hat(y_old) W_old] = C, so w_old has expectation M_old C and w_sum / (M_old + K) estimates C;
// the merge weights (M_old / M_new for the history, 1 / M_new per fresh candidate) sum to 1, every
// entry with total > 0 is reachable from the fresh draws, and the target is the same every frame
// (static scene), so W stays an unbiased contribution weight. The history candidate's random number
// is drawn whenever the flag is on (also with an empty history), so the stream never depends on the
// data. Without the flag the kernel, its random stream and its output are those of the fill without
// temporal reuse.
//
// The cell normalisation (docs/LIGHT_MIS.md section 3): without temporal reuse, w_sum / K of every
// reservoir is an unbiased estimate of C_c = sum over the table of p_hat_i; thread 0 moves the
// header's C_cur to C_prev and stores the mean over the 32 threads as the new C_cur. The trace reads
// C_prev only (the MIS proxy), so it never divides by an estimate made from the reservoirs it
// samples. With temporal reuse that is no longer enough: the previous fill's candidates ARE part of
// this frame's reservoirs, so an estimate made from the candidates (even one frame delayed through
// C_prev) is correlated with the reservoirs the trace samples. Each thread then makes one dedicated
// draw from the CDF (w_c = p_hat_j / p_j, independent of every reservoir) and thread 0 folds the
// mean of the 32 into C_cur as a running mean over at most fc.light_temporal_frames x 32 draws
// (draw count in the header's C_n). An evicted cell and a cell filled without a usable table get
// C_prev = C_cur = M = C_n = 0 (the global proxy, no history).
//
// Inputs: the trace set (binding 21 = the cells) and TracePush (the frame constants).
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_block.hlsl"
#include "render/caelix_scene_trace.hlsl"
#include "common/caelix_material.hlsl"
#include "common/caelix_light.hlsl"
#include "render/caelix_trace_set.hlsl"
#include "render/caelix_shading.hlsl"
#include "render/caelix_light_sample.hlsl"

// Slots per dispatch row (the dispatch is 1024 x capacity / 1024; the capacity is at least 1024).
static const uint kLightCellsPerRow = 1024;

// Per thread: w_sum / K of its reservoir, or its dedicated draw w_c with temporal reuse. Thread 0
// averages them into C_cur.
groupshared float gs_c[32];

// The geometry term G of the brick-level target p_hat = entry.total_weight * G for the cell centre
// (world space): orient / max(d^2, near_d2).
float LightCellG(CaelixFrameConstants fc, LightEntryHeader entry, float3 cell_centre, float near_d2)
{
    float4 e0;
    float4 e1;
    float4 e2;
    EntityCurRows(fc, entry.entity, e0, e1, e2);
    const float3 centre = RowsTransform(e0, e1, e2, float3(entry.origin) + 4.0f);
    const float3 to_cell = cell_centre - centre;
    const float distance2 = dot(to_cell, to_cell);
    float orient = 1.0f;
    const float3 n_local = LightBrickNormalLocal(entry);
    if (any(n_local != 0.0f) && distance2 > 1e-12f)
    {
        const float3 n = RowsRotate(e0, e1, e2, n_local);
        orient = max(0.1f, dot(n, to_cell * rsqrt(distance2)));
    }
    return orient / max(distance2, near_d2);
}

[numthreads(32, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID)
{
    const CaelixFrameConstants fc = CAELIX_LOAD_STRUCT(CaelixFrameConstants, g_push.frame_constants, 16);
    const uint slot = group_id.y * kLightCellsPerRow + group_id.x;
    if (slot >= fc.light_cell_capacity)
    {
        return;
    }
    const uint base = slot * kLightCellBytes;
    const uint reservoir = base + kLightCellReservoirOffset + thread_id.x * 8u;
    if (g_light_cells.Load(base) == 0u)
    {
        return;  // empty slot (uniform for the workgroup)
    }
    const uint touched = g_light_cells.Load(base + 8);
    if (touched + kLightCellMaxAge < fc.frame_counter)
    {
        // Evicted: the key, both normalisations, the history and every reservoir become empty.
        if (thread_id.x == 0u)
        {
            g_light_cells.Store2(base, uint2(0u, 0u));
            g_light_cells.Store4(base + kLightCellCPrevOffset, uint4(0u, 0u, 0u, 0u));
        }
        g_light_cells.Store2(reservoir, uint2(kNoLight, 0u));
        return;
    }
    // Usage counters: the cell stays live; it is active when the previous frame's trace looked it up.
    // Before the no-table return: a live cell is live without a table too.
    if (thread_id.x == 0u)
    {
        const uint counters = LightCellCounterBase(fc);
        g_light_cells.InterlockedAdd(counters + kLightCellCounterLive, 1u);
        if (touched == fc.frame_counter - 1u)
        {
            g_light_cells.InterlockedAdd(counters + kLightCellCounterActive, 1u);
        }
    }
    const uint64_t table = fc.light_table;
    const uint count = table != 0u ? CAELIX_LOAD_U32(table + kLightHeaderCount) : 0u;
    const float total = table != 0u ? asfloat(CAELIX_LOAD_U32(table + kLightHeaderTotal)) : 0.0f;
    if (count == 0u || !(total > 0.0f))
    {
        // No usable table (uniform for the workgroup): empty reservoirs, no normalisation, no history.
        if (thread_id.x == 0u)
        {
            g_light_cells.Store4(base + kLightCellCPrevOffset, uint4(0u, 0u, 0u, 0u));
        }
        g_light_cells.Store2(reservoir, uint2(kNoLight, 0u));
        return;
    }
    const uint capacity = CAELIX_LOAD_U32(table + kLightHeaderCapacity);
    const bool temporal = (fc.gi_flags & kGiFlagLightCellTemporal) != 0u;

    // The cell's centre in world space.
    const uint entity = g_light_cells.Load(base + 12);
    const int3 coord = int3(g_light_cells.Load3(base + 16));
    const uint level = g_light_cells.Load(base + 28);
    const float cell_size = float(8u << (2u * level));
    float4 c0;
    float4 c1;
    float4 c2;
    EntityCurRows(fc, entity, c0, c1, c2);
    const float3 cell_centre = RowsTransform(c0, c1, c2, (float3(coord) + 0.5f) * cell_size);
    const float near_d2 = (0.5f * cell_size + 4.0f) * (0.5f * cell_size + 4.0f);

    uint rng = InitRandom(uint2(slot, thread_id.x), fc.frame_seed ^ 0x3C6EF372u);
    const uint candidates = max(fc.light_candidates, 1u);
    uint selected = kNoLight;
    float p_hat_selected = 0.0f;
    float w_sum = 0.0f;
    CAELIX_LOOP
    for (uint k = 0; k < candidates; ++k)
    {
        const uint i = LightCdfSearch(table, capacity, count, RandomFloat01(rng) * total);
        const LightEntryHeader entry = LoadLightEntryHeader(LightEntryAddress(table, i));
        const float entry_total = entry.total_weight;
        const float u_select = RandomFloat01(rng);
        if (!(entry_total > 0.0f))
        {
            continue;
        }
        const float G = LightCellG(fc, entry, cell_centre, near_d2);
        const float p_hat = entry_total * G;
        // w = p_hat / p_i with p_i = entry_total / total.
        const float w = G * total;
        if (!(w > 0.0f))
        {
            continue;
        }
        w_sum += w;
        if (u_select * w_sum < w)
        {
            selected = i;
            p_hat_selected = p_hat;
        }
    }

    // The candidate count behind the new reservoir: K, plus the clamped history with temporal reuse.
    float m_new = float(candidates);
    float c_draw = 0.0f;
    if (temporal)
    {
        // The previous fill's reservoir of this thread as one more candidate. The header's M is read
        // by every thread before the barrier; thread 0 writes it after the barrier.
        const float m_cap = float(fc.light_temporal_frames) * float(candidates);
        const float m_old = min(asfloat(g_light_cells.Load(base + kLightCellMOffset)), m_cap);
        const uint2 old = g_light_cells.Load2(reservoir);
        const float w_old_reservoir = asfloat(old.y);
        const float u_old = RandomFloat01(rng);
        if (m_old > 0.0f && old.x != kNoLight && old.x < count && w_old_reservoir > 0.0f)
        {
            const LightEntryHeader entry_old = LoadLightEntryHeader(LightEntryAddress(table, old.x));
            if (entry_old.total_weight > 0.0f)
            {
                const float p_hat_old = entry_old.total_weight * LightCellG(fc, entry_old, cell_centre, near_d2);
                if (p_hat_old > 0.0f)
                {
                    const float w_old = p_hat_old * w_old_reservoir * m_old;
                    w_sum += w_old;
                    if (u_old * w_sum < w_old)
                    {
                        selected = old.x;
                        p_hat_selected = p_hat_old;
                    }
                }
            }
        }
        m_new = m_old + float(candidates);

        // One dedicated draw for the cell normalisation, independent of every reservoir.
        const float u_c = RandomFloat01(rng);
        const uint j = LightCdfSearch(table, capacity, count, u_c * total);
        const LightEntryHeader entry_c = LoadLightEntryHeader(LightEntryAddress(table, j));
        if (entry_c.total_weight > 0.0f)
        {
            const float w_c = LightCellG(fc, entry_c, cell_centre, near_d2) * total;
            c_draw = w_c > 0.0f ? w_c : 0.0f;
        }
    }
    const float W = (selected != kNoLight && p_hat_selected > 0.0f) ? (w_sum / m_new) / p_hat_selected : 0.0f;
    g_light_cells.Store2(reservoir, uint2(W > 0.0f ? selected : kNoLight, asuint(W)));

    // Every thread of the workgroup reaches this barrier: the returns above are uniform.
    gs_c[thread_id.x] = temporal ? c_draw : w_sum / float(candidates);
    GroupMemoryBarrierWithGroupSync();
    if (thread_id.x == 0u)
    {
        float c_sum = 0.0f;
        CAELIX_UNROLL
        for (uint t = 0; t < 32; ++t)
        {
            c_sum += gs_c[t];
        }
        const float c_prev = asfloat(g_light_cells.Load(base + kLightCellCCurOffset));
        if (temporal)
        {
            // Running mean of the dedicated draws over at most light_temporal_frames x 32 of them.
            const float n_old =
                min(asfloat(g_light_cells.Load(base + kLightCellCNOffset)), float(fc.light_temporal_frames) * 32.0f);
            const float c_cur = (c_prev * n_old + c_sum) / (n_old + 32.0f);
            g_light_cells.Store4(base + kLightCellCPrevOffset,
                                 uint4(asuint(c_prev), asuint(c_cur), asuint(m_new), asuint(n_old + 32.0f)));
        }
        else
        {
            // No history: M and C_n 0, so a later switch to temporal reuse starts clean.
            g_light_cells.Store4(base + kLightCellCPrevOffset,
                                 uint4(asuint(c_prev), asuint(c_sum * (1.0f / 32.0f)), 0u, 0u));
        }
    }
}
