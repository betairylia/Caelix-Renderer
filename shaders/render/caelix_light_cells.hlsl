// Light cell fill kernel (docs/LIGHT_LIST.md, sections 2.2 and 2.8): refills the R reservoirs
// (R = fc.light_cell_reservoirs, a power of two in 1 .. 64) of every live light cell
// (render/caelix_light_sample.hlsl) from the scene's light table before the trace.
//
// 64 / R cells per workgroup of 64 threads, R threads per cell (dispatch min(1024, G) x max(1, G / 1024)
// with G = capacity x R / 64 workgroups: slot = (y * 1024 + x) * (64 / R) + thread / R); thread
// r = thread % R owns reservoir r of its cell. An empty slot (key_lo 0) is skipped. A cell not looked
// up for 64 frames is evicted (key 0, reservoirs empty). Without a light table or with an empty one the
// reservoirs are emptied. The cells of a workgroup are independent: a skipped, evicted or
// emptied cell only clears its threads' `active`, so that every thread of the workgroup reaches both
// barriers (barriers in uniform control flow only). Thread r = 0 of every cell that stays counts it
// in the usage counters after the last cell (live; active when the previous frame's trace looked it
// up; the pushes of the previous frame's trace that got a ring entry and those beyond it). Otherwise
// each thread draws K = fc.light_candidates bricks from the table's CDF (proposal p_i = total_i /
// total) and keeps one by streaming RIS with the brick-level target p_hat_i = total_i * G_i
// (LightCellG):
//   G_i = orient * orient_r / max(|centre_i - cell centre|^2, (cell half size + 4)^2),
//   orient = max(0.1, dot(mean normal_i, direction from the brick centre to the cell centre)),
//            1 without a mean normal (the normal comes from the entry's face weights,
//            LightBrickNormalLocal: the luminance-weighted open-face direction, docs/LIGHT_FACES.md),
//   orient_r = the receiver orientation term of an oriented cell (below), 1 for an unoriented one,
// all in world space (entity transforms of the brick and of the cell). The reservoir stores the
// light sample word (a brick-level word for a fresh candidate) and W = (w_sum / M) / p_hat(selected),
// w_i = c_i p_hat_i W_i (a fresh candidate: c_i = 1, W_i = 1 / p_i), M = the sum of the confidences
// c_i: an unbiased contribution weight for every receiver of the cell, since p_hat_i > 0 for every
// entry. Random stream per (slot, reservoir, frame).
//
// The target of a voxel-face word (LightCellTargetOf; pushed samples and their descendants):
// p_hat = L_v * G_face, L_v = the luminance of the voxel's emission (the material table, before
// emission_scale), G_face = max(0.1, dot(face normal, direction from the face centre to the cell
// centre)) * orient_r / max(|face centre - cell centre|^2, (cell half size + 4)^2).
//
// Oriented cells (kGiFlagLightCellOriented, docs/LIGHT_LIST.md section 2.7): the cell key holds the
// receiver face (the axis of the receivers' world normal, header `face` 0..5; kLightCellFaceNone 6
// without the flag), so the receivers of one cell face one way. The target then has the receiver
// orientation term orient_r = max(kLightCellOrientFloor, dot(n_cell, direction from the cell's back
// plane point (the centre moved half a cell against n_cell: every receiver of the cell lies in front of
// it) to the light)), n_cell = the face's world axis (LightCellReceiverOrient), toward the brick centre for
// a brick-level target and toward the face centre for a voxel-face one; the dedicated C draw uses
// LightCellG and gets it too, and the MIS proxy's cell form (LightNeeProxyPdfArea) applies the same
// factor. The floor is never 0: a receiver may read a cell whose orientation or position is not its
// own (the jittered lookup, a slot merged by a key collision), and every reservoir must stay an
// unbiased contribution weight for every receiver, which needs p_hat > 0 for every entry. A light
// behind the cell's face then costs noise (W up to 10x), not bias.
//
// Temporal reuse (kGiFlagLightCellTemporal, static scene only): after the K fresh candidates each
// thread merges the reservoir it wrote in the previous fill as one more candidate with the weight
// p_hat(y_old) * W_old * M_old, where M_old is the cell header's confidence sum clamped to
// fc.light_temporal_frames x K (0 drops the history: the renderer sets 0 after a light table
// rebuild or a reset). The header stores the mean over the R threads of their M_new. E[p_hat(y_old)
// W_old] = C, so w_old has expectation M_old C; every entry with total > 0 is reachable from the
// fresh draws, so W stays an unbiased contribution weight. The history candidate's random number is
// drawn whenever the flag is on (also with an empty history), so the stream never depends on the
// data. Without the flag the kernel, its random stream and its output are those of the fill without
// temporal reuse.
//
// Push-back of shadow ray results (kGiFlagLightCellPush): the trace's vertices that sampled this
// cell pushed their light samples (LightNextEvent, LightCellClaimPush, LightCellStorePush): word, W
// and visible (0 = the shadow ray was traced and hit). Thread r merges ring entry r when
// r < min(push_count, R), with the constant confidence c_p = fc.light_push_confidence (M += c_p
// also for an empty entry) and the weight c_p * p_hat_p * W_p, p_hat_p = LightCellTargetOf(word)
// times kLightOccludedTarget when the entry is occluded. The old sample's target gets the same factor
// when an entry of this frame's ring holds its word occluded (several entries: occluded wins). The
// push's random number is drawn whenever the flag is on, after the history's and before the
// dedicated C draw.
// Why it is unbiased: streaming RIS gives E[f(y) W] = (1 / M) sum_i c_i E[f(y_i) W_i] for every
// receiver f of the cell, provided that the presence and the confidence c_i of every input are
// constants (do not depend on the inputs' values) and that each W_i is a contribution weight over
// the cell's receivers (E[f(y_i) W_i] = the integral for every receiver). The per-candidate target
// cancels (W divides by the selected candidate's own target), so it may be any positive function,
// visibility included. Hence the rules: (1) a ring entry's presence is its claim order (the first R
// claims), and a vertex claims before it draws its sample, so neither the presence nor the vertex
// that fills an entry depends on the sample; every vertex that used the cell pushes once, whatever
// its ray found (a claim after the shadow ray let the vertices without a ray win the entries); a
// variant that merges only visible pushes makes the presence depend on visibility and is biased
// bright (a lamp with one of two voxels hidden: the visible pushes always
// hold the visible voxel with W = 2, E[f W] = 2 I). (2) The confidences never depend on visibility;
// halving M_old when its sample was seen occluded correlates the history's weight with f and is
// biased bright too. (3) A vertex pushes a voxel-face word only when its candidate faces equal their
// union over the cell's receiver region (LightCellRegionFaceMask), else the brick-level word with
// W_brick: a point drawn from the vertex's own candidate faces does not cover a face that another
// receiver of the cell sees (a near brick whose face planes lie inside the region), which would bias
// dark. Visibility shapes only the targets: the floor kLightOccludedTarget keeps p_hat > 0, so a
// sample occluded here but visible elsewhere costs noise (W up to 10x), not bias.
//
// The cell normalisation (docs/LIGHT_MIS.md section 3): without temporal reuse, w_sum / K over the K
// fresh candidates of every reservoir is an unbiased estimate of C_c = sum over the table of p_hat_i;
// thread r = 0 moves the header's C_cur to C_prev and stores the mean over the R threads of the cell
// as the new C_cur. The trace reads C_prev only (the MIS proxy), so it never divides by an estimate
// made from the reservoirs it samples. With temporal reuse that is no longer enough: the previous
// fill's candidates ARE part of this frame's reservoirs, so an estimate made from the candidates
// (even one frame delayed through C_prev) is correlated with the reservoirs the trace samples. Each
// thread then makes one dedicated draw from the CDF (w_c = p_hat_j / p_j, independent of every
// reservoir) and thread r = 0 folds the mean of the R into C_cur as a running mean over at most
// fc.light_temporal_frames x R draws (draw count in the header's C_n). The pushes do not enter C_c
// (it stays the brick-level normalisation of the MIS proxy). An evicted cell and a cell filled
// without a usable table get C_prev = C_cur = M = C_n = push_count = 0 (the global proxy, no
// history).
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

// Workgroups per dispatch row (the dispatch is min(1024, G) x max(1, G / 1024), G = capacity x R / 64).
static const uint kLightCellGroupsPerRow = 1024;
// Threads per workgroup: 64 / R cells x R threads (R = fc.light_cell_reservoirs, a power of two in
// 1 .. 64, so the cells fill the workgroup exactly).
static const uint kLightCellGroupThreads = 64;

// Per thread: w_sum / K of its fresh candidates, or its dedicated draw w_c with temporal reuse; 0 for
// a thread whose cell is not filled. Thread r = 0 of each cell averages the cell's R into C_cur.
groupshared float gs_c[kLightCellGroupThreads];
// The push rings of this frame (kGiFlagLightCellPush), R entries per cell: entry r of the cell, or
// { kNoLight, 0, 1, 0 } for r >= min(push_count, R), without the flag and for a cell not filled.
groupshared uint4 gs_push[kLightCellGroupThreads];

// The geometry term G of the brick-level target p_hat = entry.total_weight * G for the cell centre
// (world space): orient * orient_r / max(d^2, near_d2); `n_cell` = the world normal of the cell's
// receiver face (zero for an unoriented cell: orient_r = 1).
float LightCellG(CaelixFrameConstants fc, LightEntryHeader entry, float3 cell_centre, float near_d2, float3 n_cell)
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
    // The receiver side: from the cell's back plane toward the brick centre.
    const float orient_r = LightCellReceiverOrient(n_cell, cell_centre, near_d2, centre);
    return orient * orient_r / max(distance2, near_d2);
}

// The target of the light sample word `word` for the cell centre (world space): a brick-level word
// total * LightCellG, a voxel-face word L_v * G_face (see the header). 0 for an invalid word, a
// voxel-face word whose voxel bit is clear (the scene changed) or a zero target.
float LightCellTargetOf(CaelixFrameConstants fc, uint64_t table, uint count, uint word, float3 cell_centre,
                        float near_d2, float3 n_cell)
{
    uint entry_index;
    uint voxel_index;
    uint face;
    if (!LightCellDecode(word, count, entry_index, voxel_index, face))
    {
        return 0.0f;
    }
    const uint64_t address = LightEntryAddress(table, entry_index);
    const LightEntryHeader entry = LoadLightEntryHeader(address);
    if (face == kLightCellFaceBrick)
    {
        return entry.total_weight > 0.0f ? entry.total_weight * LightCellG(fc, entry, cell_centre, near_d2, n_cell)
                                         : 0.0f;
    }
    const uint3 voxel = LightVoxel3(voxel_index);
    if (!LightFaceMaskBit(address, face, voxel))
    {
        return 0.0f;
    }
    const float luminance = MaterialEmissionLuminance(BrickVoxelValue(fc, entry.group, entry.id, voxel));
    if (!(luminance > 0.0f))
    {
        return 0.0f;
    }
    const LightVoxelFrame frame = LightVoxelFrameOf(fc, entry, voxel);
    float3 normal;
    float3 e1;
    float3 e2;
    LightFaceBasis(frame, face, normal, e1, e2);
    const float3 face_centre = frame.centre + 0.5f * normal;
    const float3 to_cell = cell_centre - face_centre;
    const float distance2 = dot(to_cell, to_cell);
    const float orient = distance2 > 1e-12f ? max(0.1f, dot(normal, to_cell) * rsqrt(distance2)) : 1.0f;
    // The receiver side: from the cell's back plane toward the face centre.
    const float orient_r = LightCellReceiverOrient(n_cell, cell_centre, near_d2, face_centre);
    return luminance * orient * orient_r / max(distance2, near_d2);
}

[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID)
{
    const CaelixFrameConstants fc = CAELIX_LOAD_STRUCT(CaelixFrameConstants, g_push.frame_constants, 16);
    g_materialTable = fc.materials;
    // This thread's cell of the workgroup, its reservoir r and the cell's first entry in groupshared.
    const uint R = fc.light_cell_reservoirs;
    const uint cells_per_group = kLightCellGroupThreads / R;
    const uint cell_in_group = thread_id.x / R;
    const uint r = thread_id.x % R;
    const uint gs_cell = cell_in_group * R;
    const uint slot = (group_id.y * kLightCellGroupsPerRow + group_id.x) * cells_per_group + cell_in_group;
    const uint base = slot * LightCellBytes(fc);
    const uint reservoir = base + kLightCellReservoirOffset + r * kLightCellReservoirBytes;

    // `active`: this thread's cell is filled below. Every thread reaches both barriers: a slot beyond
    // the capacity, an empty slot, an evicted cell and a fill without a usable table only clear it.
    bool active = slot < fc.light_cell_capacity;
    if (active)
    {
        active = g_light_cells.Load(base) != 0u;  // empty slot (uniform for the cell)
    }
    uint touched = 0u;
    if (active)
    {
        touched = g_light_cells.Load(base + 8);
        if (touched + kLightCellMaxAge < fc.frame_counter)
        {
            // Evicted: the key, both normalisations, the history, the push count and every reservoir
            // become empty.
            if (r == 0u)
            {
                g_light_cells.Store2(base, uint2(0u, 0u));
                g_light_cells.Store4(base + kLightCellCPrevOffset, uint4(0u, 0u, 0u, 0u));
                g_light_cells.Store(base + kLightCellPushCountOffset, 0u);
            }
            g_light_cells.Store2(reservoir, uint2(kNoLight, 0u));
            active = false;
        }
    }
    // Every thread of the cell reads the push count before the first barrier; thread r = 0 resets it
    // after it.
    uint push_count = 0u;
    if (active)
    {
        push_count = g_light_cells.Load(base + kLightCellPushCountOffset);
        // Usage counters: the cell stays live; it is active when the previous frame's trace looked it
        // up; the previous frame's pushes. Before the table check: a live cell is live without a table too.
        if (r == 0u)
        {
            const uint counters = LightCellCounterBase(fc);
            g_light_cells.InterlockedAdd(counters + kLightCellCounterLive, 1u);
            if (touched == fc.frame_counter - 1u)
            {
                g_light_cells.InterlockedAdd(counters + kLightCellCounterActive, 1u);
            }
            if (push_count != 0u)
            {
                const uint pushed = min(push_count, R);
                g_light_cells.InterlockedAdd(counters + kLightCellCounterPushed, pushed);
                if (push_count > pushed)
                {
                    g_light_cells.InterlockedAdd(counters + kLightCellCounterPushDropped, push_count - pushed);
                }
            }
        }
    }
    const uint64_t table = fc.light_table;
    const uint count = table != 0u ? CAELIX_LOAD_U32(table + kLightHeaderCount) : 0u;
    const float total = table != 0u ? asfloat(CAELIX_LOAD_U32(table + kLightHeaderTotal)) : 0.0f;
    if (active && (count == 0u || !(total > 0.0f)))
    {
        // No usable table: empty reservoirs, no normalisation, no history, no pushes.
        if (r == 0u)
        {
            g_light_cells.Store4(base + kLightCellCPrevOffset, uint4(0u, 0u, 0u, 0u));
            g_light_cells.Store(base + kLightCellPushCountOffset, 0u);
        }
        g_light_cells.Store2(reservoir, uint2(kNoLight, 0u));
        active = false;
    }
    const bool temporal = (fc.gi_flags & kGiFlagLightCellTemporal) != 0u;
    const bool push = (fc.gi_flags & kGiFlagLightCellPush) != 0u;

    // The push ring into groupshared memory, then the count is reset for this frame's trace.
    const uint push_entries = (active && push) ? min(push_count, R) : 0u;
    gs_push[thread_id.x] = r < push_entries
                               ? g_light_cells.Load4(base + LightCellPushOffset(fc) + r * kLightCellPushBytes)
                               : uint4(kNoLight, 0u, asuint(1.0f), 0u);
    GroupMemoryBarrierWithGroupSync();
    if (active && r == 0u)
    {
        g_light_cells.Store(base + kLightCellPushCountOffset, 0u);
    }

    const uint candidates = max(fc.light_candidates, 1u);
    // This thread's contribution to the cell normalisation (gs_c) and the history behind its reservoir.
    float c_value = 0.0f;
    float m_old = 0.0f;
    CAELIX_BRANCH
    if (active)
    {
        const uint capacity = CAELIX_LOAD_U32(table + kLightHeaderCapacity);
        // The cell's centre in world space, and the world normal of its receiver face (zero for an
        // unoriented cell).
        const uint entity = g_light_cells.Load(base + 12);
        const int3 coord = int3(g_light_cells.Load3(base + 16));
        const uint level = g_light_cells.Load(base + 28);
        const float3 n_cell = LightCellFaceNormal(g_light_cells.Load(base + kLightCellFaceOffset));
        const float cell_size = LightCellSize(level);
        float4 c0;
        float4 c1;
        float4 c2;
        EntityCurRows(fc, entity, c0, c1, c2);
        const float3 cell_centre = RowsTransform(c0, c1, c2, (float3(coord) + 0.5f) * cell_size);
        const float near_d2 = (0.5f * cell_size + 4.0f) * (0.5f * cell_size + 4.0f);

        uint rng = InitRandom(uint2(slot, r), fc.frame_seed ^ 0x3C6EF372u);
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
            const float G = LightCellG(fc, entry, cell_centre, near_d2, n_cell);
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
                selected = LightCellWord(i, 0u, kLightCellFaceBrick);
                p_hat_selected = p_hat;
            }
        }
        // The fresh candidates' w_sum / K estimates C_c (without temporal reuse).
        const float w_fresh = w_sum;

        // The confidence sum behind the new reservoir: K, plus the clamped history with temporal reuse,
        // plus c_p when this thread has a ring entry.
        float m_new = float(candidates);
        float c_draw = 0.0f;
        if (temporal)
        {
            // The previous fill's reservoir of this thread as one more candidate. The header's M is
            // read by every thread of the cell before the second barrier; thread r = 0 writes it after
            // that barrier.
            const float m_cap = float(fc.light_temporal_frames) * float(candidates);
            m_old = min(asfloat(g_light_cells.Load(base + kLightCellMOffset)), m_cap);
            const uint2 old = g_light_cells.Load2(reservoir);
            const float w_old_reservoir = asfloat(old.y);
            const float u_old = RandomFloat01(rng);
            if (m_old > 0.0f && old.x != kNoLight && w_old_reservoir > 0.0f)
            {
                float p_hat_old = LightCellTargetOf(fc, table, count, old.x, cell_centre, near_d2, n_cell);
                // A receiver of this cell saw the old sample occluded in the previous frame: its target
                // (not its confidence) takes the occluded factor.
                bool occluded = false;
                CAELIX_LOOP
                for (uint t = 0; t < R; ++t)
                {
                    const uint4 entry_t = gs_push[gs_cell + t];
                    occluded = occluded || (entry_t.x == old.x && asfloat(entry_t.z) == 0.0f);
                }
                p_hat_old *= occluded ? kLightOccludedTarget : 1.0f;
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
            m_new += m_old;
        }
        if (push)
        {
            // This thread's ring entry as one more candidate with the constant confidence c_p (also an
            // empty entry counts: its presence does not depend on any sample).
            const float u_push = RandomFloat01(rng);
            if (r < push_entries)
            {
                const float c_p = fc.light_push_confidence;
                const uint4 entry = gs_push[thread_id.x];
                const float W_p = asfloat(entry.y);
                m_new += c_p;
                if (W_p > 0.0f)
                {
                    const float p_hat_p =
                        LightCellTargetOf(fc, table, count, entry.x, cell_centre, near_d2, n_cell) *
                        (asfloat(entry.z) > 0.0f ? 1.0f : kLightOccludedTarget);
                    const float w_p = c_p * p_hat_p * W_p;
                    if (w_p > 0.0f)
                    {
                        w_sum += w_p;
                        if (u_push * w_sum < w_p)
                        {
                            selected = entry.x;
                            p_hat_selected = p_hat_p;
                        }
                    }
                }
            }
        }
        if (temporal)
        {
            // One dedicated draw for the cell normalisation, independent of every reservoir.
            const float u_c = RandomFloat01(rng);
            const uint j = LightCdfSearch(table, capacity, count, u_c * total);
            const LightEntryHeader entry_c = LoadLightEntryHeader(LightEntryAddress(table, j));
            if (entry_c.total_weight > 0.0f)
            {
                const float w_c = LightCellG(fc, entry_c, cell_centre, near_d2, n_cell) * total;
                c_draw = w_c > 0.0f ? w_c : 0.0f;
            }
        }
        const float W =
            (selected != kNoLight && p_hat_selected > 0.0f) ? (w_sum / m_new) / p_hat_selected : 0.0f;
        g_light_cells.Store2(reservoir, uint2(W > 0.0f ? selected : kNoLight, asuint(W)));
        c_value = temporal ? c_draw : w_fresh / float(candidates);
    }

    // Every thread of the workgroup reaches this barrier (no thread returned above).
    gs_c[thread_id.x] = c_value;
    GroupMemoryBarrierWithGroupSync();
    if (active && r == 0u)
    {
        float c_sum = 0.0f;
        CAELIX_LOOP
        for (uint t = 0; t < R; ++t)
        {
            c_sum += gs_c[gs_cell + t];
        }
        const float threads = float(R);
        const float c_prev = asfloat(g_light_cells.Load(base + kLightCellCCurOffset));
        if (temporal)
        {
            // Running mean of the dedicated draws over at most light_temporal_frames x R of them.
            // M: the mean of the R threads' confidence sums (they differ by c_p where the ring has
            // fewer than R entries); any constant is a valid confidence for the next merge.
            const float m_header = m_old + float(candidates) +
                                   (push ? fc.light_push_confidence * float(push_entries) / threads : 0.0f);
            const float n_old =
                min(asfloat(g_light_cells.Load(base + kLightCellCNOffset)), float(fc.light_temporal_frames) * threads);
            const float c_cur = (c_prev * n_old + c_sum) / (n_old + threads);
            g_light_cells.Store4(base + kLightCellCPrevOffset,
                                 uint4(asuint(c_prev), asuint(c_cur), asuint(m_header), asuint(n_old + threads)));
        }
        else
        {
            // No history: M and C_n 0, so a later switch to temporal reuse starts clean.
            g_light_cells.Store4(base + kLightCellCPrevOffset,
                                 uint4(asuint(c_prev), asuint(c_sum / threads), 0u, 0u));
        }
    }
}
