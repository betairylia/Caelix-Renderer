// The parts of the light NEE (docs/LIGHT_NEE.md) that both sides of its MIS share: the near field and
// the balance heuristic weight. The trace kernel (render/caelix_path_trace.hlsl: LampCandidate on the
// NEE side, LampHitWeight on the BSDF side) and the test probe tests/shaders/caelix_light_nee_probe.hlsl
// include it. It reads no table and no frame constant.
#ifndef CAELIX_LIGHT_NEE_HLSL
#define CAELIX_LIGHT_NEE_HLSL

// Lamp points closer than this (world units) to the receiver belong to the BSDF path, on both sides
// of the MIS (docs/LIGHT_NEE.md section 5): area sampling with a 1 / d^2 term is extremely
// heavy-tailed near a lamp, the BSDF path finds near emitters with a bounded weight.
static const float kLightNeeNearField = 8.0f;

// The balance heuristic between the light NEE and the cosine-weighted diffuse bounce at the receiver
// with world normal n_r, for a lamp point with area density p_area in the unit direction w at
// distance d; cos_l = the lamp face's cosine toward the receiver. `nee_side`: the NEE's weight
// p_sa / (p_sa + p_bsdf), else the BSDF's p_bsdf / (p_sa + p_bsdf), with p_sa = p_area d^2 / cos_l and
// p_bsdf = cos_r / pi. Both sides call it with the density of the same light tree draw, so the two
// weights of a point sum to 1 in every branch (up to the rounding of the two pdf walks): a non-finite
// or degenerate p_sa (cos_l <= 0) counts as +inf (NEE 1, BSDF 0), both densities 0 as NEE 0, BSDF 1.
float LightNeeMisWeight(float p_area, float3 n_r, float3 w, float d, float cos_l, bool nee_side)
{
    const float p_sa = cos_l > 0.0f ? p_area * (d * d) / cos_l : 0.0f;
    const float p_bsdf = max(0.0f, dot(n_r, w)) * (1.0f / 3.1415926535f);
    if (!(cos_l > 0.0f) || !(p_sa <= 3.0e38f))
    {
        return nee_side ? 1.0f : 0.0f;
    }
    const float sum = p_sa + p_bsdf;
    if (!(sum > 0.0f))
    {
        return nee_side ? 0.0f : 1.0f;
    }
    return (nee_side ? p_sa : p_bsdf) / sum;
}

#endif  // CAELIX_LIGHT_NEE_HLSL
