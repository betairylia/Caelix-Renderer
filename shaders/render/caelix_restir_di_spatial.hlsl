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
// depend on visibility. With DiMisEnabled the target function includes the NEE weight w_nee: the
// merge weights evaluate it at this pixel's receiver, the Z counts at each accepted neighbour's own
// receiver (DiReceiverOf of its reservoir: the support of the NEE strategy at that neighbour).
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
    g_lightNeeMinDistance = fc.light_nee_min_distance;

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
    const float view_z = g_int_view_z[pixel].x;
    uint rng = InitRandom(pixel, fc.frame_seed ^ (0x3B5A9D27u + g_push.gi_pass * 0x9E3779B9u));

    const bool unbiased = (fc.gi_flags & kGiFlagUnbiased) != 0u;
    // No visibility rays in the DI reuse (kGiFlagVisibility is ignored here): the DI targets have no
    // visibility, so Z by target support alone is unbiased (bistro 0.994 against 0.896 with 1/M)
    // and the shade pass's final shadow ray handles the visibility. Rays in the merge weights plus
    // visibility from the counted neighbours in Z (Bitterli et al. 2020, section 4.3) measured 1.27
    // (too bright: the pixel's own weight has no visibility test, so the techniques are not
    // consistent) at 11 ms; not worth fixing while the shade ray gives an unbiased result.
    const bool trace_visibility = false;
    const bool mis = DiMisEnabled(fc);
    const DiReceiver receiver = DiReceiverOf(fc, own);
    int selected = -1;  // the accepted neighbour whose sample was selected; -1 = the pixel's own
    // The accepted neighbours (counted in M), whole: their receivers (visible point, face, proxy
    // context) and M, for Z.
    DiReservoir accepted[kGiMaxSpatialSamples];
    uint accepted_count = 0u;

    const DiLight own_light = DiResolve(fc, own);
    DiMerge merge = DiMergeInit(own, own_light, DiTargetPdf(own_light, receiver, mis));
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
        const float p_hat = DiTargetPdf(light, receiver, mis);
        float visibility = 1.0f;
        CAELIX_BRANCH
        if (trace_visibility && p_hat > 0.0f && c.W > 0.0f)
        {
            visibility = DiVisible(fc, x_v, light) ? 1.0f : 0.0f;
        }
        if (DiMergeCandidate(merge, c, light, p_hat, DiM(c), visibility, rng))
        {
            selected = int(accepted_count);
        }
        accepted[accepted_count] = c;
        ++accepted_count;
    }

    uint count = merge.m;
    CAELIX_BRANCH
    if (unbiased)
    {
        const DiLight y = merge.selected_light;
        count = 0u;
        if (DiTargetPdf(y, receiver, mis) > 0.0f)
        {
            count += DiM(own);
        }
        CAELIX_LOOP
        for (uint k = 0u; k < accepted_count; ++k)
        {
            const DiReceiver receiver_k = DiReceiverOf(fc, accepted[k]);
            if (!(DiTargetPdf(y, receiver_k, mis) > 0.0f))
            {
                continue;
            }
            // The neighbour that owned Y sees it by construction (its weight passed the ray).
            if (!trace_visibility || int(k) == selected || DiVisible(fc, receiver_k.x_v, y))
            {
                count += DiM(accepted[k]);
            }
        }
    }
    StoreDiReservoir(output, DiMergeFinishWithCount(merge, own, count));
}
