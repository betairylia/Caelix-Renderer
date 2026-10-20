// ReSTIR GI shade pass (render/caelix_restir_gi.hlsl, docs/GI_EXPERIMENTS.md): the pixel's final
// reservoir (fc.gi_out after one spatial pass, else fc.gi_temp) gives the demodulated diffuse
// radiance L_o * max(0, dot(n_v, w)) / pi * W, written to the diffuse radiance target in the
// trace kernel's packing (REBLUR_FrontEnd_PackRadianceAndNormHitDist, YCoCg, with the hit
// distance |x_s - x_v| normalized for roughness 1 at the pixel's view z; t_max for a sky sample),
// so NRD, the composite and the reference accumulation read it unchanged. An invalid reservoir
// or W = 0 writes 0 (NRD's "no sample").
//
// Descriptors: render/caelix_trace_set.hlsl (internal view z, diffuse radiance).
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_block.hlsl"
#include "render/caelix_scene_trace.hlsl"
#include "common/caelix_material.hlsl"
#include "render/caelix_trace_set.hlsl"
#include "render/caelix_shading.hlsl"
#include "NRD.hlsli"
#include "render/caelix_restir_gi.hlsl"

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
    g_forceDiffuse = (fc.gi_flags & kGiFlagDiffuseOnly) != 0u || fc.gi_method == kGiRestir;

    const uint2 pixel = thread_id.xy;
    // 0 or 2 spatial passes end in gi_temp, one in gi_out.
    const bool spatial_out = fc.gi_spatial_passes == 1u && (fc.gi_flags & kGiFlagSpatial) != 0u;
    const GiReservoir r = LoadGiReservoir(GiReservoirAddress(spatial_out ? fc.gi_out : fc.gi_temp, fc, pixel));
    float4 diffuse = float4(0.0f, 0.0f, 0.0f, 0.0f);
    CAELIX_BRANCH
    if (GiIsValid(r) && r.W > 0.0f)
    {
        const float3 n_v = GiFaceNormal(GiVisibleFace(r));
        float3 w = r.sample_pos;
        float hit_distance = fc.t_max;
        if (!GiIsSky(r))
        {
            const float3 d = r.sample_pos - r.visible_pos;
            hit_distance = length(d);
            w = hit_distance > 0.0f ? d / hit_distance : float3(0.0f, 0.0f, 0.0f);
        }
        const float3 radiance = r.radiance * (max(0.0f, dot(n_v, w)) * (1.0f / K_PI) * r.W);
        const float view_z = g_int_view_z[pixel].x;
        const float norm_hit_dist =
            REBLUR_FrontEnd_GetNormHitDist(hit_distance, view_z, fc.hit_dist_params.xyz, 1.0f);
        diffuse = REBLUR_FrontEnd_PackRadianceAndNormHitDist(SanitizeRadiance(radiance), norm_hit_dist, true);
    }
    g_int_diff[pixel] = diffuse;
}
