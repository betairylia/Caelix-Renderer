// Light cell fill kernel (docs/LIGHT_LIST.md, section 2.2): refills the 32 reservoirs of every
// live light cell (render/caelix_light_sample.hlsl) from the scene's light table before the trace.
//
// One workgroup of 32 threads per cell slot (dispatch 1024 x capacity / 1024: slot = y * 1024 + x);
// thread r owns reservoir r. An empty slot (key_lo 0) is skipped. A cell not looked up for 64
// frames is evicted (key 0, reservoirs empty). Without a light table or with an empty one the
// reservoirs are emptied. Otherwise each thread draws K = fc.light_candidates bricks from the
// table's CDF (proposal p_i = total_i / total) and keeps one by streaming RIS with the brick-level
// target p_hat_i = total_i * G_i:
//   G_i = orient / max(|centre_i - cell centre|^2, (cell half size + 4)^2),
//   orient = max(0.1, dot(mean normal_i, direction from the brick centre to the cell centre)),
//            1 without a mean normal,
// all in world space (entity transforms of the brick and of the cell). The reservoir stores the
// entry index and W = (w_sum / K) / p_hat(selected), w_i = p_hat_i / p_i: an unbiased contribution
// weight of the brick for every point of the cell, since p_hat_i > 0 for every entry. No temporal
// reuse. Random stream per (slot, reservoir, frame).
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

float3 RowsTransform(float4 r0, float4 r1, float4 r2, float3 p)
{
    const float4 h = float4(p, 1.0f);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

float3 RowsRotate(float4 r0, float4 r1, float4 r2, float3 v)
{
    return float3(dot(r0.xyz, v), dot(r1.xyz, v), dot(r2.xyz, v));
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
        // Evicted: the key and every reservoir become empty.
        if (thread_id.x == 0u)
        {
            g_light_cells.Store2(base, uint2(0u, 0u));
        }
        g_light_cells.Store2(reservoir, uint2(kNoLight, 0u));
        return;
    }
    const uint64_t table = fc.light_table;
    const uint count = table != 0u ? CAELIX_LOAD_U32(table + kLightHeaderCount) : 0u;
    const float total = table != 0u ? asfloat(CAELIX_LOAD_U32(table + kLightHeaderTotal)) : 0.0f;
    if (count == 0u || !(total > 0.0f))
    {
        g_light_cells.Store2(reservoir, uint2(kNoLight, 0u));
        return;
    }
    const uint capacity = CAELIX_LOAD_U32(table + kLightHeaderCapacity);

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
        const uint64_t entry = LightEntryAddress(table, i);
        const uint4 a = CAELIX_LOAD_U32x4(entry + 96);   // total, group, id, entity
        const uint4 b = CAELIX_LOAD_U32x4(entry + 112);  // origin, mean normal
        const float entry_total = asfloat(a.x);
        const float u_select = RandomFloat01(rng);
        if (!(entry_total > 0.0f))
        {
            continue;
        }
        float4 e0;
        float4 e1;
        float4 e2;
        EntityCurRows(fc, a.w, e0, e1, e2);
        const float3 centre = RowsTransform(e0, e1, e2, float3(int3(b.xyz)) + 4.0f);
        const float3 to_cell = cell_centre - centre;
        const float distance2 = dot(to_cell, to_cell);
        float orient = 1.0f;
        if (b.w != 0u && distance2 > 1e-12f)
        {
            const float3 n = RowsRotate(e0, e1, e2, DecodeOctahedral(b.w));
            orient = max(0.1f, dot(n, to_cell * rsqrt(distance2)));
        }
        const float G = orient / max(distance2, near_d2);
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
    const float W = (selected != kNoLight && p_hat_selected > 0.0f) ? (w_sum / float(candidates)) / p_hat_selected
                                                                     : 0.0f;
    g_light_cells.Store2(reservoir, uint2(W > 0.0f ? selected : kNoLight, asuint(W)));
}
