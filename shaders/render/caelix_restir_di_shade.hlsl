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
// No visibility reuse: an occluded final sample contributes 0 this frame and the reservoir is left
// as it is. Pruning it (W = 0 or an empty reservoir) makes the pixel's history a distribution
// proportional to target x visibility FROM THIS PIXEL; the spatial pass of the neighbours then
// counts this pixel's M (1/M or Z by target support) for samples it can no longer hold (occluded
// here, visible there), which lost 33 % of the direct light on bistro. Variance is the price: a
// light voxel set into a wall keeps occluded side-face samples in the history for some frames.
// With DiMisEnabled (kGiFlagLightMis, docs/LIGHT_MIS.md section 5) the contribution is multiplied by
// the NEE weight w_nee of the light point at this pixel's receiver (the proxy context stored in the
// reservoir): this pass contributes the NEE strategy's share of the list lights' direct light, the
// trace kernel's first segment after the diffuse primary vertex the BSDF's (w_bsdf, same context).
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
    const DiReceiver receiver = DiReceiverOf(fc, r);
    const float3 x_v = receiver.x_v;
    const float3 n_v = receiver.n_v;
    float distance;
    float3 w;
    float cos_l;
    const float geometry = DiGeometry(light.position, light.normal, x_v, n_v, distance, w, cos_l);
    if (!(geometry > 0.0f) || !DiVisible(fc, x_v, light))
    {
        return;  // occluded: nothing this frame, the history is not pruned (see the header)
    }
    float3 contribution = light.radiance * (geometry * r.W);
    if (DiMisEnabled(fc))
    {
        contribution *= LightNeeMisWeight(receiver.ctx, light.mis, x_v, n_v, w, distance, cos_l, true);
    }
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
