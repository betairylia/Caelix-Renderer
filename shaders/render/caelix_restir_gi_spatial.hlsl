// ReSTIR GI spatial pass (render/caelix_restir_gi.hlsl, docs/GI_EXPERIMENTS.md), run
// fc.gi_spatial_passes times when kGiFlagSpatial is set: pass g_push.gi_pass reads fc.gi_temp
// (even pass) or fc.gi_out (odd pass) and writes the other one.
//
// gi_spatial_samples neighbours, uniform in a disk of gi_spatial_radius pixels around the pixel
// (the pixel itself and pixels outside the image are skipped), each accepted by GiSimilar, shifted
// to this pixel's visible point with the reconnection Jacobian. The accepted neighbours are
// gathered first, then merged with one of three weighting schemes:
// - 1/M (default): each candidate is weighted with its full M; W = w_sum / (M * p_hat(Y)).
//   kGiFlagVisibility: a shadow ray from this pixel's visible point to each neighbour's sample
//   (occluded: weight 0, still counted in M).
// - Z-count (kGiFlagUnbiased): as 1/M with the visibility rays, but after the selection of Y the
//   MIS denominator is Z = the sum of M over this pixel and the accepted neighbours at which Y has
//   a nonzero target function and is visible.
// - Pairwise MIS (kGiFlagPairwiseMis; overrides the two above): the balance heuristic between each
//   neighbour i and the pixel's own technique c over confidence-weighted densities in this pixel's
//   domain, p_t(s) = M_t * p_hat_t(s) * J(s; r -> t), with visibility in p_c(X_i) and p_i(Y_c) (two
//   shadow rays per accepted neighbour). m_i = p_i(X_i) / (k p_i(X_i) + p_c(X_i)),
//   m_c = (1/k) sum_i p_c(Y_c) / (k p_i(Y_c) + p_c(Y_c)); w = m * p_hat_r * W * J (no factor M);
//   W = w_sum / p_hat_r(Y). The weights are bounded, so the reconnection Jacobian's outliers are
//   damped (docs/GI_EXPERIMENTS.md "Pairwise MIS").
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

    // Gather the accepted neighbours (each is counted in M whatever its weight).
    GiReservoir cand[kGiMaxSpatialSamples];
    uint count = 0u;
    const uint samples = min(fc.gi_spatial_samples, kGiMaxSpatialSamples);
    CAELIX_LOOP
    for (uint s = 0u; s < samples; ++s)
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
        cand[count] = c;
        ++count;
    }

    GiReservoir result;
    CAELIX_BRANCH
    if ((fc.gi_flags & kGiFlagPairwiseMis) != 0u)
    {
        // Pairwise MIS: m_i = p_i(X_i) / (k p_i(X_i) + p_c(X_i)) for a neighbour's sample,
        // m_c = (1/k) sum_i p_c(Y_c) / (k p_i(Y_c) + p_c(Y_c)) for the pixel's own, with visibility in
        // p_c(X_i) (ray from x_v to X_i) and p_i(Y_c) (ray from the neighbour's visible point to Y_c).
        const float k = float(count);
        const uint m_c = GiM(own);
        const float p_hat_own = GiTargetPdf(own, x_v, n_v);
        const float p_c_own = float(m_c) * p_hat_own;
        float mis_own = 1.0f;
        CAELIX_BRANCH
        if (count > 0u && p_c_own > 0.0f && own.W > 0.0f)
        {
            mis_own = 0.0f;
            CAELIX_LOOP
            for (uint i = 0u; i < count; ++i)
            {
                float p_i = GiTechniqueDensity(own, x_v, cand[i].visible_pos, n_v, GiM(cand[i]));
                if (p_i > 0.0f && !GiVisible(fc, cand[i].visible_pos, own))
                {
                    p_i = 0.0f;
                }
                mis_own += p_c_own / (k * p_i + p_c_own);
            }
            mis_own /= k;
        }
        GiMerge merge = GiMergeInitWeighted(own, p_hat_own);
        GiMergeWeighted(merge, own, p_hat_own, (p_c_own > 0.0f) ? mis_own * p_hat_own * own.W : 0.0f, m_c, rng);
        CAELIX_LOOP
        for (uint i = 0u; i < count; ++i)
        {
            const GiReservoir c = cand[i];
            const float p_hat_r = GiTargetPdf(c, x_v, n_v);
            const float jacobian = GiJacobian(c, c.visible_pos, x_v);
            float w = 0.0f;
            CAELIX_BRANCH
            if (p_hat_r > 0.0f && jacobian > 0.0f && c.W > 0.0f && GiVisible(fc, x_v, c))
            {
                const float p_i = GiTechniqueDensity(c, x_v, c.visible_pos, n_v, GiM(c));
                const float p_c = float(m_c) * p_hat_r;
                w = (p_i / (k * p_i + p_c)) * p_hat_r * c.W * jacobian;
            }
            GiMergeWeighted(merge, c, p_hat_r, w, GiM(c), rng);
        }
        result = GiMergeFinishMis(merge, x_v, face);
    }
    else
    {
        const bool unbiased = (fc.gi_flags & kGiFlagUnbiased) != 0u;
        const bool trace_visibility = (fc.gi_flags & (kGiFlagVisibility | kGiFlagUnbiased)) != 0u;
        int selected = -1;  // the accepted neighbour whose sample was selected; -1 = the pixel's own
        GiMerge merge = GiMergeInit(own, GiTargetPdf(own, x_v, n_v));
        CAELIX_LOOP
        for (uint i = 0u; i < count; ++i)
        {
            const GiReservoir c = cand[i];
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
                selected = int(i);
            }
        }

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
            for (uint j = 0u; j < count; ++j)
            {
                const float3 n_j = GiFaceNormal(GiVisibleFace(cand[j]));
                if (!(GiTargetPdf(y, cand[j].visible_pos, n_j) > 0.0f))
                {
                    continue;
                }
                // The neighbour that owned Y sees it by construction.
                if (int(j) == selected || GiVisible(fc, cand[j].visible_pos, y))
                {
                    z += GiM(cand[j]);
                }
            }
            result = GiMergeFinishWithCount(merge, x_v, face, z);
        }
        else
        {
            result = GiMergeFinish(merge, x_v, face);
        }
    }
    StoreGiReservoir(output, result);
}
