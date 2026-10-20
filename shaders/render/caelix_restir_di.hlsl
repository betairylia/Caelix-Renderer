// Per-pixel ReSTIR DI on the light list (Bitterli et al. 2020; docs/RESTIR_DI.md): the reservoir,
// its storage in the DI reservoir buffers of CaelixFrameConstants (di_initial, di_prev, di_temp,
// di_out), the light point named by a reservoir, the target function, the merge and the initial
// candidates of the trace kernel.
//
// A sample is a fixed point y on a list-light voxel face: (light table entry, voxel, face, uv).
// y, its normal n_y and its radiance L_e are rebuilt from those indices through the light table
// and the entity's CURRENT transform (LightPointAt) every time a pass needs them, so there is no
// reconnection Jacobian. Target function at a visible point x_v with normal n_v:
//   p_hat(y) = luminance(L_e) * max(0, dot(n_v, w)) / pi * max(0, dot(n_y, -w)) / d^2,
//   w = normalize(y - x_v), d = |y - x_v|, no visibility.
// Contribution (demodulated, the composite multiplies by the albedo):
//   L_e * max(0, dot(n_v, w)) / pi * max(0, dot(n_y, -w)) / d^2 * V * W.
// n_v is the axis normal of the visible face index, as in render/caelix_restir_gi.hlsl (exact for
// unrotated entities only).
//
// Include after common/caelix_frame.hlsl, common/caelix_light.hlsl, render/caelix_trace_set.hlsl,
// render/caelix_scene_trace.hlsl, render/caelix_shading.hlsl, render/caelix_restir_gi.hlsl
// (GiFaceIndex, GiFaceNormal, GiLuminance) and render/caelix_light_sample.hlsl.
#ifndef CAELIX_RESTIR_DI_HLSL
#define CAELIX_RESTIR_DI_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_light.hlsl"
#include "render/caelix_scene_trace.hlsl"
#include "render/caelix_shading.hlsl"
#include "render/caelix_restir_gi.hlsl"
#include "render/caelix_light_sample.hlsl"

// 32 bytes in the reservoir buffers: two 16-byte words (LoadDiReservoir, StoreDiReservoir).
struct DiReservoir
{
    uint light;            // light table entry index; kNoLight = no sample
    uint voxel_face;       // bits 0..8 voxel v = x | y << 3 | z << 6, bits 9..11 face, bit 12 valid
    uint uv;               // 2 x unorm16: the point's cell on the face (LightUvQuantize)
    float W;               // unbiased contribution weight (area measure at the light point)
    float3 visible_pos;    // x_v world position (the pushed-off primary point)
    uint visible_flags_m;  // bits 0..2 face index of n_v, bit 3 diffuse lobe, bits 8..31 M
};

static const uint kDiReservoirBytes = 32;
static const uint kDiVoxelMask = 0x1FFu;
static const uint kDiFaceShift = 9u;
static const uint kDiFaceMask = 7u;
static const uint kDiSampleValid = 1u << 12;
// The pixel's path took the diffuse lobe at the primary vertex in this frame: the shade pass adds
// the contribution only then (step 1's NEE ran at that vertex only then).
static const uint kDiDiffuseLobe = 1u << 3;
// The low byte of visible_flags_m: the pixel's own flags (face and lobe), kept by every merge.
static const uint kDiVisibleFlagsMask = 0xFFu;
static const uint kDiMShift = 8u;
// Initial candidates per pixel: the maximum of CaelixGiOptions::di_candidates.
static const uint kDiMaxCandidates = 32u;

uint64_t DiReservoirAddress(uint64_t base, CaelixFrameConstants fc, uint2 pixel)
{
    return base + (uint64_t(pixel.y) * fc.size.x + pixel.x) * kDiReservoirBytes;
}

DiReservoir LoadDiReservoir(uint64_t addr)
{
    const uint4 a = CAELIX_LOAD_U32x4(addr);
    const uint4 b = CAELIX_LOAD_U32x4(addr + 16);
    DiReservoir r;
    r.light = a.x;
    r.voxel_face = a.y;
    r.uv = a.z;
    r.W = asfloat(a.w);
    r.visible_pos = asfloat(b.xyz);
    r.visible_flags_m = b.w;
    return r;
}

void StoreDiReservoir(uint64_t addr, DiReservoir r)
{
    CAELIX_STORE_U32x4(addr, uint4(r.light, r.voxel_face, r.uv, asuint(r.W)));
    CAELIX_STORE_U32x4(addr + 16, uint4(asuint(r.visible_pos), r.visible_flags_m));
}

// No sample, W 0, M 0: a pixel without a primary surface (or without DI).
DiReservoir DiEmptyReservoir()
{
    DiReservoir r;
    r.light = kNoLight;
    r.voxel_face = 0u;
    r.uv = 0u;
    r.W = 0.0f;
    r.visible_pos = float3(0.0f, 0.0f, 0.0f);
    r.visible_flags_m = 0u;
    return r;
}

bool DiHasSample(DiReservoir r)
{
    return (r.voxel_face & kDiSampleValid) != 0u && r.light != kNoLight;
}

uint DiM(DiReservoir r)
{
    return r.visible_flags_m >> kDiMShift;
}

uint DiVisibleFace(DiReservoir r)
{
    return r.visible_flags_m & kGiVisibleFaceMask;
}

// The light point of a reservoir's sample, rebuilt from its indices.
struct DiLight
{
    bool valid;
    float3 position;
    float3 normal;
    float3 radiance;
    float luminance;
};

DiLight DiInvalidLight()
{
    DiLight l;
    l.valid = false;
    l.position = float3(0.0f, 0.0f, 0.0f);
    l.normal = float3(0.0f, 0.0f, 0.0f);
    l.radiance = float3(0.0f, 0.0f, 0.0f);
    l.luminance = 0.0f;
    return l;
}

// Invalid without a sample or with an entry index beyond the table's count (the table changed:
// the renderer drops the history then, this is the guard).
DiLight DiResolve(CaelixFrameConstants fc, DiReservoir r)
{
    if (!DiHasSample(r) || fc.light_table == 0u)
    {
        return DiInvalidLight();
    }
    const uint count = CAELIX_LOAD_U32(fc.light_table + kLightHeaderCount);
    if (r.light >= count)
    {
        return DiInvalidLight();
    }
    const LightEntry entry = LoadLightEntry(LightEntryAddress(fc.light_table, r.light));
    const float2 uv = float2(LightUvDequantize(r.uv & 0xFFFFu), LightUvDequantize(r.uv >> 16));
    const LightPoint p =
        LightPointAt(fc, entry, r.voxel_face & kDiVoxelMask, (r.voxel_face >> kDiFaceShift) & kDiFaceMask, uv);
    DiLight l;
    l.valid = true;
    l.position = p.position;
    l.normal = p.normal;
    l.radiance = p.radiance;
    l.luminance = GiLuminance(p.radiance);
    return l;
}

// The unshadowed geometry term of the Lambert receiver (x_v, n_v) and the light face:
// max(0, dot(n_v, w)) / pi * max(0, dot(n_y, -w)) / d^2; `distance` receives d.
float DiGeometry(float3 position, float3 normal, float3 x_v, float3 n_v, out float distance)
{
    const float3 d = position - x_v;
    const float d2 = dot(d, d);
    distance = 0.0f;
    if (!(d2 > 0.0f))
    {
        return 0.0f;
    }
    distance = sqrt(d2);
    if (distance < g_lightNeeMinDistance)
    {
        return 0.0f;  // the near field is the BSDF path's (caelix_light_sample.hlsl)
    }
    const float3 w = d / distance;
    const float cos_v = dot(n_v, w);
    const float cos_y = -dot(normal, w);
    if (!(cos_v > 0.0f) || !(cos_y > 0.0f))
    {
        return 0.0f;
    }
    return (cos_v * (1.0f / K_PI)) * (cos_y / d2);
}

float DiTargetPdf(DiLight l, float3 x_v, float3 n_v)
{
    if (!l.valid)
    {
        return 0.0f;
    }
    float distance;
    return l.luminance * DiGeometry(l.position, l.normal, x_v, n_v, distance);
}

// Whether the light point is visible from x_v (already pushed off its surface): a shadow ray that
// stops 2 push-offs short of the point (it lies on a voxel face), as LightNextEvent.
bool DiVisible(CaelixFrameConstants fc, float3 x_v, DiLight l)
{
    const float3 d = l.position - x_v;
    const float distance = length(d);
    RayDesc ray;
    ray.Origin = x_v;
    ray.TMin = 0.0f;
    ray.TMax = distance - 2.0f * K_RAY_ORIGIN_PUSH_OFF;
    if (!(ray.TMax > 0.0f))
    {
        return true;
    }
    ray.Direction = d / distance;
    return !TraceShadow(fc, ray);
}

// Similarity of a candidate reservoir to the receiver: it belongs to a pixel with a surface (M > 0;
// a reservoir without a sample still counts in M), the same visible face index, and visible points
// closer than threshold * view_z.
bool DiSimilar(DiReservoir c, uint face_r, float3 x_v_r, float view_z_r, float threshold)
{
    if (DiM(c) == 0u || DiVisibleFace(c) != face_r)
    {
        return false;
    }
    const float3 d = c.visible_pos - x_v_r;
    const float limit = threshold * view_z_r;
    return dot(d, d) <= limit * limit;
}

// Weighted reservoir selection over the receiver's own reservoir and reused ones (no Jacobian).
struct DiMerge
{
    DiReservoir selected;
    DiLight selected_light;
    float w_sum;
    uint m;
    float p_hat_selected;  // the target function of `selected` at the receiver
};

DiMerge DiMergeInit(DiReservoir own, DiLight own_light, float p_hat_own)
{
    DiMerge s;
    s.selected = own;
    s.selected_light = own_light;
    s.w_sum = p_hat_own * own.W * float(DiM(own));
    s.m = DiM(own);
    s.p_hat_selected = p_hat_own;
    return s;
}

// Adds reservoir `c` (its light point `l`): weight p_hat_at_r * W * m * visibility; it counts m in M
// whatever its weight. Returns true when `c` was selected.
bool DiMergeCandidate(inout DiMerge s, DiReservoir c, DiLight l, float p_hat_at_r, uint m, float visibility,
                      inout uint rng)
{
    const float w = p_hat_at_r * c.W * float(m) * visibility;
    s.m += m;
    if (!(w > 0.0f))
    {
        return false;
    }
    s.w_sum += w;
    if (RandomFloat01(rng) * s.w_sum < w)
    {
        s.selected = c;
        s.selected_light = l;
        s.p_hat_selected = p_hat_at_r;
        return true;
    }
    return false;
}

// The merged reservoir at the receiver (x_v_r, own flags = face + lobe) with the MIS denominator
// `count` (s.m: 1/M weights; Z: the unbiased combination): W = w_sum / (count * p_hat(selected)).
DiReservoir DiMergeFinishWithCount(DiMerge s, float3 x_v_r, uint own_flags, uint count)
{
    DiReservoir r = s.selected;
    r.W = (s.p_hat_selected > 0.0f && s.w_sum > 0.0f && count > 0u)
              ? s.w_sum / (float(count) * s.p_hat_selected)
              : 0.0f;
    r.visible_pos = x_v_r;
    r.visible_flags_m = (own_flags & kDiVisibleFlagsMask) | (min(s.m, 0xFFFFFFu) << kDiMShift);
    return r;
}

// The initial reservoir of a pixel (trace kernel): `candidates` light points by streaming RIS at the
// visible point x_v (the pushed-off primary point) with the face index `face` of its normal. Each
// candidate: one of the 32 reservoirs of the pixel's light cell (looked up once; entry index and
// W_brick), or the global CDF when the cell or that reservoir is empty, then SampleLightEntryWith
// (uv quantized to the reservoir's 16 bits) -> point with pdf_area; candidate weight
// p_hat * W_brick / pdf_area. W = w_sum / (candidates * p_hat(selected)), M = 1. No shadow ray.
// `group` and `object_point` locate the cell as in LightNextEvent. Own random stream.
DiReservoir DiInitialReservoir(CaelixFrameConstants fc, uint2 pixel, float3 x_v, uint face, bool diffuse_lobe,
                               uint group, float3 object_point)
{
    DiReservoir r = DiEmptyReservoir();
    r.visible_pos = x_v;
    r.visible_flags_m = (face & kGiVisibleFaceMask) | (diffuse_lobe ? kDiDiffuseLobe : 0u) | (1u << kDiMShift);
    const uint64_t table = fc.light_table;
    const uint count = CAELIX_LOAD_U32(table + kLightHeaderCount);
    const float total = asfloat(CAELIX_LOAD_U32(table + kLightHeaderTotal));
    const uint candidates = min(fc.di_candidates, kDiMaxCandidates);
    if (count == 0u || !(total > 0.0f) || candidates == 0u)
    {
        return r;
    }
    const uint capacity = CAELIX_LOAD_U32(table + kLightHeaderCapacity);
    const float3 n_v = GiFaceNormal(face);
    uint rng = InitRandom(pixel, fc.frame_seed ^ 0x2D1B5A73u);

    int slot = -1;
    CAELIX_BRANCH
    if (fc.light_cells != 0u)
    {
        const uint level = LightCellLevel(fc, x_v);
        slot = LightCellLookup(fc, GroupEntity(fc, group), level, LightCellCoord(object_point, level));
    }

    float w_sum = 0.0f;
    float p_hat_selected = 0.0f;
    CAELIX_LOOP
    for (uint i = 0u; i < candidates; ++i)
    {
        const float u_reservoir = RandomFloat01(rng);
        const float u_global = RandomFloat01(rng);
        uint light = kNoLight;
        float W_brick = 0.0f;
        if (slot >= 0)
        {
            const uint cell_r = min(kLightCellReservoirs - 1u, uint(u_reservoir * float(kLightCellReservoirs)));
            const uint2 reservoir =
                g_light_cells.Load2(uint(slot) * kLightCellBytes + kLightCellReservoirOffset + cell_r * 8u);
            if (asfloat(reservoir.y) > 0.0f && reservoir.x < count)
            {
                light = reservoir.x;
                W_brick = asfloat(reservoir.y);
            }
        }
        LightEntry entry;
        if (light == kNoLight)
        {
            light = LightCdfSearch(table, capacity, count, u_global * total);
            entry = LoadLightEntry(LightEntryAddress(table, light));
            W_brick = entry.total_weight > 0.0f ? total / entry.total_weight : 0.0f;
        }
        else
        {
            entry = LoadLightEntry(LightEntryAddress(table, light));
        }
        const LightSample s = SampleLightEntryWith(fc, entry, x_v, true, rng);
        const float u_select = RandomFloat01(rng);
        if (!s.valid || !(W_brick > 0.0f))
        {
            continue;
        }
        float distance;
        const float p_hat = GiLuminance(s.radiance) * DiGeometry(s.position, s.normal, x_v, n_v, distance);
        const float w = p_hat * W_brick / s.pdf_area;
        if (!(w > 0.0f))
        {
            continue;
        }
        w_sum += w;
        if (u_select * w_sum < w)
        {
            r.light = light;
            r.voxel_face = (s.voxel & kDiVoxelMask) | ((s.face & kDiFaceMask) << kDiFaceShift) | kDiSampleValid;
            r.uv = LightUvQuantize(s.uv.x + 0.5f) | (LightUvQuantize(s.uv.y + 0.5f) << 16);
            p_hat_selected = p_hat;
        }
    }
    r.W = (w_sum > 0.0f && p_hat_selected > 0.0f) ? w_sum / (float(candidates) * p_hat_selected) : 0.0f;
    return r;
}

#endif  // CAELIX_RESTIR_DI_HLSL
