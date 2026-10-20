// ReSTIR DI spatial pass (render/caelix_restir_di.hlsl, docs/RESTIR_DI.md), run
// fc.gi_spatial_passes times when kGiFlagSpatial is set: pass g_push.gi_pass reads fc.di_temp
// (even pass) or fc.di_out (odd pass) and writes the other one.
//
// As the GI spatial pass: gi_spatial_samples neighbours, uniform in a disk of gi_spatial_radius
// pixels (the pixel itself and pixels outside the image are skipped), each accepted by DiSimilar
// and merged with its full M; weight p_hat_r(y) * W * M with the light point rebuilt under the
// current transforms (no Jacobian). kGiFlagVisibility or kGiFlagUnbiased: a shadow ray from this
// pixel's visible point to each neighbour's light point (occluded: weight 0, still counted in M).
// kGiFlagUnbiased: the MIS denominator is Z = the sum of M over this pixel and the accepted
// neighbours at which the selected sample has a nonzero target function, instead of M. Unlike the
// GI pass, Z traces no visibility ray: the DI targets have no visibility (the initial candidates
// are not shadow-tested), so a reservoir can hold an occluded sample and its support does not
// depend on visibility.
//
// Descriptors: render/caelix_trace_set.hlsl (TLAS for the shadow rays, internal view z).
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
    const bool even = (g_push.gi_pass & 1u) == 0u;
    const uint64_t input = even ? fc.di_temp : fc.di_out;
    const uint64_t output = DiReservoirAddress(even ? fc.di_out : fc.di_temp, fc, pixel);
    const DiReservoir own = LoadDiReservoir(DiReservoirAddress(input, fc, pixel));
    if (DiM(own) == 0u)
    {
        StoreDiReservoir(output, DiEmptyReservoir());
        return;
    }
    const float3 x_v = own.visible_pos;
    const uint face = DiVisibleFace(own);
    const float3 n_v = GiFaceNormal(face);
    const float view_z = g_int_view_z[pixel].x;
    uint rng = InitRandom(pixel, fc.frame_seed ^ (0x3B5A9D27u + g_push.gi_pass * 0x9E3779B9u));

    const bool unbiased = (fc.gi_flags & kGiFlagUnbiased) != 0u;
    const bool trace_visibility = (fc.gi_flags & (kGiFlagVisibility | kGiFlagUnbiased)) != 0u;
    // The accepted neighbours (counted in M): their visible points and face + M words, for Z.
    float3 accepted_pos[kGiMaxSpatialSamples];
    uint accepted_face_m[kGiMaxSpatialSamples];
    uint accepted_count = 0u;

    const DiLight own_light = DiResolve(fc, own);
    DiMerge merge = DiMergeInit(own, own_light, DiTargetPdf(own_light, x_v, n_v));
    const uint samples = min(fc.gi_spatial_samples, kGiMaxSpatialSamples);
    CAELIX_LOOP
    for (uint i = 0u; i < samples; ++i)
    {
        const float radius = fc.gi_spatial_radius * sqrt(RandomFloat01(rng));
        const float angle = K_TWO_PI * RandomFloat01(rng);
        const int2 q = int2(pixel) + int2(round(float2(cos(angle), sin(angle)) * radius));
        if (any(q < int2(0, 0)) || any(q >= int2(fc.size)) || all(q == int2(pixel)))
        {
            continue;
        }
        const DiReservoir c = LoadDiReservoir(DiReservoirAddress(input, fc, uint2(q)));
        if (!DiSimilar(c, face, x_v, view_z, fc.gi_depth_threshold))
        {
            continue;
        }
        const DiLight light = DiResolve(fc, c);
        const float p_hat = DiTargetPdf(light, x_v, n_v);
        float visibility = 1.0f;
        CAELIX_BRANCH
        if (trace_visibility && p_hat > 0.0f && c.W > 0.0f)
        {
            visibility = DiVisible(fc, x_v, light) ? 1.0f : 0.0f;
        }
        DiMergeCandidate(merge, c, light, p_hat, DiM(c), visibility, rng);
        accepted_pos[accepted_count] = c.visible_pos;
        accepted_face_m[accepted_count] = c.visible_flags_m;
        ++accepted_count;
    }

    uint count = merge.m;
    CAELIX_BRANCH
    if (unbiased)
    {
        const DiLight y = merge.selected_light;
        count = 0u;
        if (DiTargetPdf(y, x_v, n_v) > 0.0f)
        {
            count += DiM(own);
        }
        CAELIX_LOOP
        for (uint k = 0u; k < accepted_count; ++k)
        {
            const float3 n_k = GiFaceNormal(accepted_face_m[k] & kGiVisibleFaceMask);
            if (DiTargetPdf(y, accepted_pos[k], n_k) > 0.0f)
            {
                count += accepted_face_m[k] >> kDiMShift;
            }
        }
    }
    StoreDiReservoir(output, DiMergeFinishWithCount(merge, x_v, own.visible_flags_m, count));
}
