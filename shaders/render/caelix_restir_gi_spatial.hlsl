// ReSTIR GI spatial pass (render/caelix_restir_gi.hlsl, docs/GI_EXPERIMENTS.md), run
// fc.gi_spatial_passes times when kGiFlagSpatial is set: pass g_push.gi_pass reads fc.gi_temp
// (even pass) or fc.gi_out (odd pass) and writes the other one.
//
// gi_spatial_samples neighbours, uniform in a disk of gi_spatial_radius pixels around the pixel
// (the pixel itself and pixels outside the image are skipped), each accepted by GiSimilar, shifted
// to this pixel's visible point with the reconnection Jacobian and merged with its full M.
// kGiFlagVisibility or kGiFlagUnbiased: a shadow ray from this pixel's visible point to each
// neighbour's sample (occluded: weight 0, still counted in M). kGiFlagUnbiased: after the
// selection of Y, the MIS denominator is Z = the sum of M over this pixel and the accepted
// neighbours at which Y has a nonzero target function and is visible, instead of M (1/M weights).
//
// Descriptors: render/caelix_trace_set.hlsl (TLAS for the shadow rays, internal view z).
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_block.hlsl"
#include "render/caelix_scene_trace.hlsl"
#include "common/caelix_material.hlsl"
#include "render/caelix_trace_set.hlsl"
#include "render/caelix_shading.hlsl"
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
    const bool even = (g_push.gi_pass & 1u) == 0u;
    const uint64_t input = even ? fc.gi_temp : fc.gi_out;
    const uint64_t output = GiReservoirAddress(even ? fc.gi_out : fc.gi_temp, fc, pixel);
    // The temporal pass stored this pixel's visible point and face (or an empty reservoir).
    const GiReservoir own = LoadGiReservoir(GiReservoirAddress(input, fc, pixel));
    if (!GiIsValid(own))
    {
        StoreGiReservoir(output, GiEmptyReservoir());
        return;
    }
    const float3 x_v = own.visible_pos;
    const uint face = GiVisibleFace(own);
    const float3 n_v = GiFaceNormal(face);
    const float view_z = g_int_view_z[pixel].x;
    uint rng = InitRandom(pixel, fc.frame_seed ^ (0x2C9B7A15u + g_push.gi_pass * 0x9E3779B9u));

    const bool unbiased = (fc.gi_flags & kGiFlagUnbiased) != 0u;
    const bool trace_visibility = (fc.gi_flags & (kGiFlagVisibility | kGiFlagUnbiased)) != 0u;
    // The accepted neighbours (counted in M): their visible points and face + M words, for Z.
    float3 accepted_pos[kGiMaxSpatialSamples];
    uint accepted_face_m[kGiMaxSpatialSamples];
    uint accepted_count = 0u;
    int selected = -1;  // the accepted neighbour whose sample was selected; -1 = the pixel's own

    GiMerge merge = GiMergeInit(own, GiTargetPdf(own, x_v, n_v));
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
        const GiReservoir c = LoadGiReservoir(GiReservoirAddress(input, fc, uint2(q)));
        if (!GiSimilar(c, face, x_v, view_z, fc.gi_depth_threshold))
        {
            continue;
        }
        const float p_hat = GiTargetPdf(c, x_v, n_v);
        const float jacobian = GiJacobian(c, c.visible_pos, x_v);
        float visibility = 1.0f;
        CAELIX_BRANCH
        if (trace_visibility && p_hat > 0.0f && jacobian > 0.0f)
        {
            visibility = GiVisible(fc, x_v, c) ? 1.0f : 0.0f;
        }
        if (GiMergeCandidate(merge, c, p_hat, jacobian, GiM(c), visibility, rng))
        {
            selected = int(accepted_count);
        }
        accepted_pos[accepted_count] = c.visible_pos;
        accepted_face_m[accepted_count] = c.visible_flags_m;
        ++accepted_count;
    }

    GiReservoir result;
    CAELIX_BRANCH
    if (unbiased)
    {
        const GiReservoir y = merge.selected;
        uint z = 0u;
        // This pixel: its own sample was traced from here; a neighbour's sample could only be
        // selected with a nonzero weight, which needed the visibility ray from here (UNBIASED
        // traces it for every neighbour with a nonzero target function) to pass.
        if (GiTargetPdf(y, x_v, n_v) > 0.0f)
        {
            z += GiM(own);
        }
        CAELIX_LOOP
        for (uint k = 0u; k < accepted_count; ++k)
        {
            const float3 n_k = GiFaceNormal(accepted_face_m[k] & kGiVisibleFaceMask);
            if (!(GiTargetPdf(y, accepted_pos[k], n_k) > 0.0f))
            {
                continue;
            }
            // The neighbour that owned Y sees it by construction.
            if (int(k) == selected || GiVisible(fc, accepted_pos[k], y))
            {
                z += accepted_face_m[k] >> kGiMShift;
            }
        }
        result = GiMergeFinishWithCount(merge, x_v, face, z);
    }
    else
    {
        result = GiMergeFinish(merge, x_v, face);
    }
    StoreGiReservoir(output, result);
}
