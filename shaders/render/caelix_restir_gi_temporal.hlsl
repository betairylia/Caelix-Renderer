// ReSTIR GI temporal pass (render/caelix_restir_gi.hlsl, docs/GI_EXPERIMENTS.md): merges the
// pixel's initial reservoir (fc.gi_initial, written by the trace kernel) with the previous
// frame's final reservoir at the reprojected pixel (fc.gi_prev) and writes fc.gi_temp.
//
// The previous pixel is the sample position plus the internal motion (pixels). The previous
// reservoir is used when fc.gi_history_valid = 1, kGiFlagTemporal is set, the pixel is inside
// the image and GiSimilar accepts it (valid, same visible face, visible points closer than
// gi_depth_threshold * view_z); its M is clamped to gi_temporal_m_cap and the reconnection
// Jacobian moves it from its own visible point to this one. No visibility ray: a previous sample
// that is occluded from the new visible point is mostly caught by the geometric test, and the
// spatial pass traces visibility (kGiFlagVisibility applies to spatial neighbours only).
// Weights: 1/M by default; with kGiFlagPairwiseMis the balance heuristic between the two techniques
// (the pixel's own with confidence M_c, the previous one with m = its capped M and visible point x_p)
// over the confidence-weighted densities p_c(s) = M_c * p_hat_r(s) and
// p_p(s) = m * p_hat_p(s) * J(s; r -> p): w_prev = p_p(X_p) / (p_p(X_p) + p_c(X_p)) * p_hat_r(X_p) * W_p
// * J(X_p; p -> r), w_own = p_c(Y_c) / (p_c(Y_c) + p_p(Y_c)) * p_hat_r(Y_c) * W_c, W = w_sum /
// p_hat_r(Y), M = M_c + m (docs/GI_EXPERIMENTS.md "Pairwise MIS"; a static camera gives the 1/M
// weights). Without the previous reservoir the output is the initial reservoir (W unchanged up to
// rounding). A pixel without a valid initial sample stores an empty reservoir.
//
// Descriptors: render/caelix_trace_set.hlsl (internal motion and view z).
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
    const uint64_t output = GiReservoirAddress(fc.gi_temp, fc, pixel);
    const GiReservoir own = LoadGiReservoir(GiReservoirAddress(fc.gi_initial, fc, pixel));
    if (!GiIsValid(own))
    {
        StoreGiReservoir(output, GiEmptyReservoir());
        return;
    }
    const float3 x_v = own.visible_pos;
    const uint face = GiVisibleFace(own);
    const float3 n_v = GiFaceNormal(face);
    const float view_z = g_int_view_z[pixel].x;
    uint rng = InitRandom(pixel, fc.frame_seed ^ 0x51A7C3E9u);

    const float p_hat_own = GiTargetPdf(own, x_v, n_v);
    GiMerge merge = GiMergeInit(own, p_hat_own);
    bool pairwise_done = false;
    GiReservoir result;
    CAELIX_BRANCH
    if (fc.gi_history_valid != 0u && (fc.gi_flags & kGiFlagTemporal) != 0u)
    {
        const float2 prev_position = CaelixSamplePosition(fc, pixel) + g_int_motion[pixel].xy;
        const int2 prev_pixel = int2(floor(prev_position));
        if (all(prev_pixel >= int2(0, 0)) && all(prev_pixel < int2(fc.size)))
        {
            const GiReservoir prev = LoadGiReservoir(GiReservoirAddress(fc.gi_prev, fc, uint2(prev_pixel)));
            if (GiSimilar(prev, face, x_v, view_z, fc.gi_depth_threshold))
            {
                const uint m = min(GiM(prev), fc.gi_temporal_m_cap);
                const float jacobian = GiJacobian(prev, prev.visible_pos, x_v);
                const float p_hat = GiTargetPdf(prev, x_v, n_v);
                CAELIX_BRANCH
                if ((fc.gi_flags & kGiFlagPairwiseMis) != 0u)
                {
                    // Balance heuristic over the two techniques (k = 1, no rays).
                    const float3 x_p = prev.visible_pos;
                    const uint m_c = GiM(own);
                    float w_prev = 0.0f;
                    const float p_p_prev = GiTechniqueDensity(prev, x_v, x_p, n_v, m);
                    const float p_c_prev = float(m_c) * p_hat;
                    if (p_hat > 0.0f && jacobian > 0.0f && prev.W > 0.0f && p_p_prev > 0.0f)
                    {
                        w_prev = (p_p_prev / (p_p_prev + p_c_prev)) * p_hat * prev.W * jacobian;
                    }
                    float w_own = 0.0f;
                    const float p_c_own = float(m_c) * p_hat_own;
                    if (p_c_own > 0.0f && own.W > 0.0f)
                    {
                        const float p_p_own = GiTechniqueDensity(own, x_v, x_p, n_v, m);
                        w_own = (p_c_own / (p_c_own + p_p_own)) * p_hat_own * own.W;
                    }
                    GiMerge mis = GiMergeInitWeighted(own, p_hat_own);
                    GiMergeWeighted(mis, own, p_hat_own, w_own, m_c, rng);
                    GiMergeWeighted(mis, prev, p_hat, w_prev, m, rng);
                    result = GiMergeFinishMis(mis, x_v, face);
                    pairwise_done = true;
                }
                else
                {
                    GiMergeCandidate(merge, prev, p_hat, jacobian, m, 1.0f, rng);
                }
            }
        }
    }
    if (!pairwise_done)
    {
        result = GiMergeFinish(merge, x_v, face);
    }
    StoreGiReservoir(output, result);
}
