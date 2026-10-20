// ReSTIR DI shade pass (render/caelix_restir_di.hlsl, docs/RESTIR_DI.md): the pixel's final DI
// reservoir (fc.di_out after one spatial pass, else fc.di_temp) gives the demodulated direct light
// of the list lights at the primary surface,
//   L_e * max(0, dot(n_v, w)) / pi * max(0, dot(n_y, -w)) / d^2 * V * W,
// with one shadow ray from x_v toward the light point (TMax = d - 2 push-offs). It is added to the
// diffuse radiance target, which the trace kernel (path trace method) or the GI shade pass (ReSTIR
// GI, which runs first) has already packed (REBLUR_FrontEnd_PackRadianceAndNormHitDist): the value
// is unpacked, the contribution added, and the sum packed again with the stored normalized hit
// distance, or with d's when the stored radiance is 0 (no other sample). Only a pixel whose path
// took the diffuse lobe at the primary vertex (kDiDiffuseLobe) receives it: that is where step 1's
// one-sample NEE ran. Nothing is written to the diffuse target for a pixel without a contribution.
// Visibility reuse (without kGiFlagUnbiased): when the shadow ray is occluded, the final reservoir
// (the next frame's di_prev) is stored again with W = 0. This is a bias (the next frame's temporal
// merge treats the occluded sample as worthless although its target function is not 0) that
// removes most occluded samples from the history: a light voxel set into a wall, whose covered
// side faces are sampled as often as its open face, otherwise keeps occluded samples for many
// frames. With kGiFlagUnbiased the reservoir is left as it is (the Z counts of the reuse passes
// assume reservoirs whose support does not depend on visibility).
//
// Descriptors: render/caelix_trace_set.hlsl (TLAS, internal view z, diffuse radiance).
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_block.hlsl"
#include "render/caelix_scene_trace.hlsl"
#include "common/caelix_material.hlsl"
#include "common/caelix_light.hlsl"
#include "render/caelix_trace_set.hlsl"
#include "render/caelix_shading.hlsl"
#include "NRD.hlsli"
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
    g_lightNeeMinDistance = fc.light_nee_min_distance;

    const uint2 pixel = thread_id.xy;
    // 0 or 2 spatial passes end in di_temp, one in di_out.
    const bool spatial_out = fc.gi_spatial_passes == 1u && (fc.gi_flags & kGiFlagSpatial) != 0u;
    const uint64_t final_address = DiReservoirAddress(spatial_out ? fc.di_out : fc.di_temp, fc, pixel);
    DiReservoir r = LoadDiReservoir(final_address);
    if ((r.visible_flags_m & kDiDiffuseLobe) == 0u || !(r.W > 0.0f))
    {
        return;
    }
    const DiLight light = DiResolve(fc, r);
    if (!light.valid)
    {
        return;
    }
    const float3 x_v = r.visible_pos;
    const float3 n_v = GiFaceNormal(DiVisibleFace(r));
    float distance;
    const float geometry = DiGeometry(light.position, light.normal, x_v, n_v, distance);
    if (!(geometry > 0.0f) || !DiVisible(fc, x_v, light))
    {
        // Visibility reuse (Bitterli et al. 2020, section 4.4), not with kGiFlagUnbiased: an occluded
        // sample is dropped from the final reservoir (W = 0, M kept), so the next frame's temporal
        // pass does not carry it on.
        if ((fc.gi_flags & kGiFlagUnbiased) == 0u)
        {
            r.W = 0.0f;
            StoreDiReservoir(final_address, r);
        }
        return;
    }
    const float3 contribution = light.radiance * (geometry * r.W);
    const float4 stored = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(g_int_diff[pixel]);
    const float3 existing = max(stored.rgb, float3(0.0f, 0.0f, 0.0f));
    float norm_hit_dist = stored.a;
    if (!any(existing > 0.0f) || !(norm_hit_dist > 0.0f))
    {
        norm_hit_dist = REBLUR_FrontEnd_GetNormHitDist(distance, g_int_view_z[pixel].x, fc.hit_dist_params.xyz, 1.0f);
    }
    g_int_diff[pixel] =
        REBLUR_FrontEnd_PackRadianceAndNormHitDist(SanitizeRadiance(existing + contribution), norm_hit_dist, true);
}
