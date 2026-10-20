// Screen-space ReSTIR GI (Ouyang et al. 2021), diffuse only: the reservoir, its storage in the
// reservoir buffers of CaelixFrameConstants (gi_initial, gi_prev, gi_temp, gi_out), the target
// function, the Jacobian of the reconnection shift, the geometric similarity test, the merge
// (resampled importance sampling with 1/M or Z-count weights) and the visibility ray.
// docs/GI_EXPERIMENTS.md is the design.
//
// One reservoir per pixel holds one sample: the first bounce of a path from the visible point
// x_v (the pushed-off primary surface point) to the sample point x_s, and the outgoing radiance
// L_o(x_s -> x_v) of the path from there on. A first segment that misses is a sky sample (the
// direction is stored, Jacobian 1). Target function at a visible point r with normal n_v:
//   p_hat_r(S) = luminance(L_o) * max(0, dot(n_v, w)) / pi,  w = normalize(x_s - x_v(r))
// (albedo demodulated: the composite multiplies by it). The contribution at r is
//   L_o * max(0, dot(n_v, w)) / pi * W.
// Normals are axis-aligned voxel faces, stored as a face index (+X 0, -X 1, +Y 2, -Y 3, +Z 4,
// -Z 5). LIMITATION: n_v is the axis of the face index, exact for unrotated entities only.
//
// Include after common/caelix_frame.hlsl, render/caelix_scene_trace.hlsl and
// render/caelix_shading.hlsl (K_PI, RandomFloat01, K_RAY_ORIGIN_PUSH_OFF, TraceShadow).
#ifndef CAELIX_RESTIR_GI_HLSL
#define CAELIX_RESTIR_GI_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"
#include "render/caelix_scene_trace.hlsl"
#include "render/caelix_shading.hlsl"

// 48 bytes in the reservoir buffers: three 16-byte words (LoadGiReservoir, StoreGiReservoir).
struct GiReservoir
{
    float3 sample_pos;     // x_s world position; for a sky sample the unit direction
    uint sample_flags;     // bits 0..2 face index of n_s (0..5), bit 3 sky sample, bit 4 valid
    float3 radiance;       // L_o(x_s -> x_v), linear RGB (emission + continuation bounces)
    float W;               // unbiased contribution weight (0 when invalid)
    float3 visible_pos;    // x_v world position (the pushed-off bounce origin of the primary surface)
    uint visible_flags_m;  // bits 0..2 face index of n_v, bits 8..31 M
};

static const uint kGiReservoirBytes = 48;
static const uint kGiSampleFaceMask = 7u;
static const uint kGiSampleSky = 1u << 3;
static const uint kGiSampleValid = 1u << 4;
static const uint kGiVisibleFaceMask = 7u;
static const uint kGiMShift = 8u;
// The most neighbours of one spatial pass (CaelixGiOptions::spatial_samples).
static const uint kGiMaxSpatialSamples = 8u;
// Upper bound of the reconnection Jacobian: a sample moved to a visible point much closer to it
// than its own (or seen at a much flatter angle there) would get a huge weight and show as a
// firefly. RTXDI clamps the Jacobian of its GI reuse for the same reason; the clamp is a bias.
static const float kGiMaxJacobian = 1e4f;

// The reservoir of `pixel` in the buffer at `base` (row-major, fc.size.x reservoirs per row).
uint64_t GiReservoirAddress(uint64_t base, CaelixFrameConstants fc, uint2 pixel)
{
    return base + (uint64_t(pixel.y) * fc.size.x + pixel.x) * kGiReservoirBytes;
}

// Three 16-byte loads (the struct itself has no guaranteed buffer layout).
GiReservoir LoadGiReservoir(uint64_t addr)
{
    const uint4 a = CAELIX_LOAD_U32x4(addr);
    const uint4 b = CAELIX_LOAD_U32x4(addr + 16);
    const uint4 c = CAELIX_LOAD_U32x4(addr + 32);
    GiReservoir r;
    r.sample_pos = asfloat(a.xyz);
    r.sample_flags = a.w;
    r.radiance = asfloat(b.xyz);
    r.W = asfloat(b.w);
    r.visible_pos = asfloat(c.xyz);
    r.visible_flags_m = c.w;
    return r;
}

void StoreGiReservoir(uint64_t addr, GiReservoir r)
{
    CAELIX_STORE_U32x4(addr, uint4(asuint(r.sample_pos), r.sample_flags));
    CAELIX_STORE_U32x4(addr + 16, uint4(asuint(r.radiance), asuint(r.W)));
    CAELIX_STORE_U32x4(addr + 32, uint4(asuint(r.visible_pos), r.visible_flags_m));
}

// All zero: invalid, W 0, M 0.
GiReservoir GiEmptyReservoir()
{
    GiReservoir r;
    r.sample_pos = float3(0.0f, 0.0f, 0.0f);
    r.sample_flags = 0u;
    r.radiance = float3(0.0f, 0.0f, 0.0f);
    r.W = 0.0f;
    r.visible_pos = float3(0.0f, 0.0f, 0.0f);
    r.visible_flags_m = 0u;
    return r;
}

bool GiIsValid(GiReservoir r)
{
    return (r.sample_flags & kGiSampleValid) != 0u;
}

bool GiIsSky(GiReservoir r)
{
    return (r.sample_flags & kGiSampleSky) != 0u;
}

uint GiM(GiReservoir r)
{
    return r.visible_flags_m >> kGiMShift;
}

uint GiVisibleFace(GiReservoir r)
{
    return r.visible_flags_m & kGiVisibleFaceMask;
}

// The face index of a normal: its largest axis and that axis' sign.
uint GiFaceIndex(float3 n)
{
    const float3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z)
    {
        return n.x >= 0.0f ? 0u : 1u;
    }
    if (a.y >= a.z)
    {
        return n.y >= 0.0f ? 2u : 3u;
    }
    return n.z >= 0.0f ? 4u : 5u;
}

// The axis normal of a face index.
float3 GiFaceNormal(uint index)
{
    const float sign = (index & 1u) != 0u ? -1.0f : 1.0f;
    const uint axis = min(index >> 1, 2u);
    return float3(axis == 0u ? sign : 0.0f, axis == 1u ? sign : 0.0f, axis == 2u ? sign : 0.0f);
}

float GiLuminance(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

// The target function of sample `r` at the visible point x_v with normal n_v.
float GiTargetPdf(GiReservoir r, float3 x_v, float3 n_v)
{
    const float luminance = GiLuminance(r.radiance);
    if (GiIsSky(r))
    {
        return luminance * max(0.0f, dot(n_v, r.sample_pos)) * (1.0f / K_PI);
    }
    const float3 w = r.sample_pos - x_v;
    const float d2 = dot(w, w);
    if (!(d2 > 0.0f))
    {
        return 0.0f;
    }
    return luminance * max(0.0f, dot(n_v, w * rsqrt(d2))) * (1.0f / K_PI);
}

// The Jacobian of moving sample `r` from the visible point x_v_from to x_v_to (reconnection
// shift, solid angle at the visible point): (cos_phi_to / cos_phi_from) * (d_from^2 / d_to^2),
// cos_phi = the cosine at the sample point between n_s and the direction to the visible point.
// 0 when the new visible point is behind the sample's face; 1 for a sky sample. Clamped to
// kGiMaxJacobian.
float GiJacobian(GiReservoir r, float3 x_v_from, float3 x_v_to)
{
    if (GiIsSky(r))
    {
        return 1.0f;
    }
    const float3 n_s = GiFaceNormal(r.sample_flags & kGiSampleFaceMask);
    const float3 from = x_v_from - r.sample_pos;
    const float3 to = x_v_to - r.sample_pos;
    const float d2_from = dot(from, from);
    const float d2_to = dot(to, to);
    if (!(d2_from > 0.0f) || !(d2_to > 0.0f))
    {
        return 0.0f;
    }
    const float cos_phi_from = max(0.0f, dot(n_s, from * rsqrt(d2_from)));
    const float cos_phi_to = max(0.0f, dot(n_s, to * rsqrt(d2_to)));
    if (cos_phi_to <= 0.0f || cos_phi_from <= 0.0f)
    {
        return 0.0f;
    }
    return min((cos_phi_to / cos_phi_from) * (d2_from / d2_to), kGiMaxJacobian);
}

// Geometric similarity of a candidate reservoir to the receiver r: a valid sample, the same
// visible face index, and visible points closer than threshold * view_z(r).
bool GiSimilar(GiReservoir candidate, uint face_r, float3 x_v_r, float view_z_r, float threshold)
{
    if (!GiIsValid(candidate) || GiVisibleFace(candidate) != face_r)
    {
        return false;
    }
    const float3 d = candidate.visible_pos - x_v_r;
    const float limit = threshold * view_z_r;
    return dot(d, d) <= limit * limit;
}

// Weighted reservoir selection over the receiver's own reservoir and shifted candidates.
struct GiMerge
{
    GiReservoir selected;
    float w_sum;
    uint m;
    float p_hat_selected;  // the target function of `selected` at the receiver
};

// Starts a merge with the receiver's own reservoir (no shift: Jacobian 1, visible by construction).
GiMerge GiMergeInit(GiReservoir own, float p_hat_own)
{
    GiMerge s;
    s.selected = own;
    s.w_sum = p_hat_own * own.W * float(GiM(own));
    s.m = GiM(own);
    s.p_hat_selected = p_hat_own;
    return s;
}

// Adds candidate `c`, shifted to the receiver: weight p_hat_at_r * W * m_capped * jacobian *
// visibility; it counts m_capped in M whatever its weight. Returns true when `c` was selected.
bool GiMergeCandidate(inout GiMerge s, GiReservoir c, float p_hat_at_r, float jacobian, uint m_capped,
                      float visibility, inout uint rng)
{
    const float w = p_hat_at_r * c.W * float(m_capped) * jacobian * visibility;
    s.m += m_capped;
    if (!(w > 0.0f))
    {
        return false;
    }
    s.w_sum += w;
    if (RandomFloat01(rng) * s.w_sum < w)
    {
        s.selected = c;
        s.p_hat_selected = p_hat_at_r;
        return true;
    }
    return false;
}

// The merged reservoir at the receiver (x_v_r, face_r) with the MIS denominator `count`
// (s.m: 1/M weights; Z: the unbiased combination): W = w_sum / (count * p_hat(selected)), M = s.m.
GiReservoir GiMergeFinishWithCount(GiMerge s, float3 x_v_r, uint face_r, uint count)
{
    GiReservoir r = s.selected;
    r.W = (s.p_hat_selected > 0.0f && s.w_sum > 0.0f && count > 0u)
              ? s.w_sum / (float(count) * s.p_hat_selected)
              : 0.0f;
    r.visible_pos = x_v_r;
    r.visible_flags_m = (face_r & kGiVisibleFaceMask) | (min(s.m, 0xFFFFFFu) << kGiMShift);
    return r;
}

GiReservoir GiMergeFinish(GiMerge s, float3 x_v_r, uint face_r)
{
    return GiMergeFinishWithCount(s, x_v_r, face_r, s.m);
}

// Whether sample `r` is visible from x_v (already pushed off its surface): a shadow ray toward
// the sample point that stops 2 push-offs short of it (the sample point lies on a voxel face), or
// toward a sky sample's direction up to fc.t_max (the length of the bounce rays).
bool GiVisible(CaelixFrameConstants fc, float3 x_v, GiReservoir r)
{
    RayDesc ray;
    ray.Origin = x_v;
    ray.TMin = 0.0f;
    if (GiIsSky(r))
    {
        ray.Direction = r.sample_pos;
        ray.TMax = fc.t_max;
    }
    else
    {
        const float3 d = r.sample_pos - x_v;
        const float distance = length(d);
        ray.TMax = distance - 2.0f * K_RAY_ORIGIN_PUSH_OFF;
        if (!(ray.TMax > 0.0f))
        {
            return true;
        }
        ray.Direction = d / distance;
    }
    return !TraceShadow(fc, ray);
}

#endif  // CAELIX_RESTIR_GI_HLSL
