// ReSTIR DI temporal pass (render/caelix_restir_di.hlsl, docs/RESTIR_DI.md): merges the pixel's
// initial DI reservoir (fc.di_initial, written by the trace kernel) with the previous frame's final
// DI reservoir at the reprojected pixel (fc.di_prev) and writes fc.di_temp.
//
// As the GI temporal pass: the previous pixel is the sample position plus the internal motion; the
// previous reservoir is used when fc.di_history_valid = 1, kGiFlagTemporal is set, the pixel is
// inside the image and DiSimilar accepts it (a surface, the same visible face, visible points
// closer than gi_depth_threshold * view_z); its M is clamped to gi_temporal_m_cap. Merge weight
// p_hat_r(y_prev) * W_prev * min(M_prev, cap), with the light point rebuilt under the current
// transforms; no Jacobian, no visibility ray. kGiFlagUnbiased: the MIS denominator is Z = the M of
// the pixel and of the previous reservoir where the selected sample has a nonzero target function
// (at the previous reservoir's own visible point and face), instead of M. A pixel without a
// surface (M 0) stores an empty reservoir.
//
// Descriptors: render/caelix_trace_set.hlsl (internal motion and view z, light table reads).
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_block.hlsl"
#include "render/caelix_scene_trace.hlsl"
#include "common/caelix_material.hlsl"
#include "common/caelix_light.hlsl"
#include "render/caelix_trace_set.hlsl"
#include "render/caelix_shading.hlsl"
#include "render/caelix_restir_gi.hlsl"
#include "render/caelix_light_sample.hlsl"
#include "render/caelix_restir_di.hlsl"

[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    const CaelixFrameConstants fc = CAELIX_LOAD_STRUCT(CaelixFrameConstants, g_push.frame_constants, 16);
    if (any(thread_id.xy >= fc.size))
    {
        return;
    }
    g_blockEncoding = fc.encoding;
    g_materialTable = fc.materials;
    g_emissionScale = fc.emission_scale;

    const uint2 pixel = thread_id.xy;
    const uint64_t output = DiReservoirAddress(fc.di_temp, fc, pixel);
    const DiReservoir own = LoadDiReservoir(DiReservoirAddress(fc.di_initial, fc, pixel));
    if (DiM(own) == 0u)
    {
        StoreDiReservoir(output, DiEmptyReservoir());
        return;
    }
    const float3 x_v = own.visible_pos;
    const uint face = DiVisibleFace(own);
    const float3 n_v = GiFaceNormal(face);
    const float view_z = g_int_view_z[pixel].x;
    uint rng = InitRandom(pixel, fc.frame_seed ^ 0x6C8E9CF5u);

    const DiLight own_light = DiResolve(fc, own);
    DiMerge merge = DiMergeInit(own, own_light, DiTargetPdf(own_light, x_v, n_v));
    bool has_prev = false;
    DiReservoir prev = DiEmptyReservoir();
    uint m_prev = 0u;
    CAELIX_BRANCH
    if (fc.di_history_valid != 0u && (fc.gi_flags & kGiFlagTemporal) != 0u)
    {
        const float2 prev_position = CaelixSamplePosition(fc, pixel) + g_int_motion[pixel].xy;
        const int2 prev_pixel = int2(floor(prev_position));
        if (all(prev_pixel >= int2(0, 0)) && all(prev_pixel < int2(fc.size)))
        {
            prev = LoadDiReservoir(DiReservoirAddress(fc.di_prev, fc, uint2(prev_pixel)));
            if (DiSimilar(prev, face, x_v, view_z, fc.gi_depth_threshold))
            {
                has_prev = true;
                m_prev = min(DiM(prev), fc.gi_temporal_m_cap);
                const DiLight prev_light = DiResolve(fc, prev);
                DiMergeCandidate(merge, prev, prev_light, DiTargetPdf(prev_light, x_v, n_v), m_prev, 1.0f, rng);
            }
        }
    }
    uint count = merge.m;
    CAELIX_BRANCH
    if ((fc.gi_flags & kGiFlagUnbiased) != 0u)
    {
        // Z: the reservoirs whose own target function is nonzero at the selected sample (the
        // targets have no visibility, so no shadow ray).
        count = 0u;
        if (DiTargetPdf(merge.selected_light, x_v, n_v) > 0.0f)
        {
            count += DiM(own);
        }
        if (has_prev && DiTargetPdf(merge.selected_light, prev.visible_pos, GiFaceNormal(DiVisibleFace(prev))) > 0.0f)
        {
            count += m_prev;
        }
    }
    StoreDiReservoir(output, DiMergeFinishWithCount(merge, x_v, own.visible_flags_m, count));
}
