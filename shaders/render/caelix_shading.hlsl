// Shading helpers of the trace and composite kernels: block colours (debug views),
// the random number stream, the shading rule of Unity's path tracer
// (CaelixShadeVoxelHit, opaque and transparent branches), the interface decision of
// the delta chain, the medium extinction and the fog of the air (TraverseSegment),
// and the sky. Materials come from the scene's material table
// (common/caelix_material.hlsl, GET_MATERIAL); every emission is multiplied by
// g_emissionScale (CaelixRenderOptions::emission_scale).
//
// Sky: the host's cubemap when fc.sky_cubemap = 1 (mip 0 for primary misses and the
// delta chain, a roughness-chosen mip for specular bounces, the irradiance SH of
// caelix_sky_sh.hlsl for diffuse bounces), else the procedural sky. A kernel that
// samples the cubemap defines CAELIX_SKY_BINDING_IMAGE and CAELIX_SKY_BINDING_SAMPLER
// (set 0 bindings of g_sky and g_sky_sampler) before it includes this file; without
// them the sky functions are procedural only.
// Fog: the air (medium 0) is a homogeneous absorbing and emitting medium of density
// fc.fog_color.w and radiance fc.fog_color.rgb on every segment that ends on a surface.
//
// Ported from Unity (Caelix/Runtime/Rendering/Shaders):
//   Utils/Utils.hlsl                      K_T_MAX, K_RAY_ORIGIN_PUSH_OFF, FresnelReflectAmountOpaque,
//                                         FresnelReflectAmountTransparent
//   Utils/BlueNoise.hlsl                  RandomFloat01 / RandomUnitVector semantics (inout state)
//   PathTrace_full/CaelixPathTraceCommon.hlsl  CaelixShadeVoxelHit, AdvanceDeltaRay (interface),
//                                         CaelixMediumExtinction, CaelixSanitizeRadiance
// Differences, all marked NATIVE: the random stream is a hash + xorshift per pixel
// (Unity samples a spatio-temporal blue-noise texture); the sky is procedural or the
// host's cubemap, and diffuse bounces that miss see the cubemap's irradiance SH (Unity
// samples its cubemap directly); the fog of the air has no Unity counterpart.
#ifndef CAELIX_SHADING_HLSL
#define CAELIX_SHADING_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_block.hlsl"
#include "common/caelix_material.hlsl"

#ifdef CAELIX_SKY_BINDING_IMAGE
CAELIX_BINDING(CAELIX_SKY_BINDING_IMAGE, 0) TextureCube<float4> g_sky;
CAELIX_BINDING(CAELIX_SKY_BINDING_SAMPLER, 0) SamplerState g_sky_sampler;
#endif

// Set by every kernel from CaelixFrameConstants::emission_scale before the first shade.
static float g_emissionScale;
// Set by every kernel that shades before the first shade: true with kGiFlagDiffuseOnly or with
// ReSTIR GI (fc.gi_method = kGiRestir). ShadeVoxelHit then always takes the diffuse lobe at an
// opaque block (the lobe's random number is still drawn, so the stream does not change).
static bool g_forceDiffuse;

static const float K_PI = 3.1415926535f;
static const float K_TWO_PI = 6.283185307f;
// Unity Utils.hlsl: K_RAY_ORIGIN_PUSH_OFF 0.002 (world units along the normal).
static const float K_RAY_ORIGIN_PUSH_OFF = 0.002f;
// Largest finite FP16 value: radiance written to the RGBA16F targets is clamped to it.
static const float kHalfMax = 65504.0f;

static const float3 kAirColor = float3(1.0f, 0.0f, 1.0f);   // air hit: should not happen
static const float3 kGlassTint = float3(0.6f, 0.8f, 1.0f);  // debug views only

// ---------------------------------------------------------------------------
// Block colours

uint HashU32(uint value)
{
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    value ^= value >> 16;
    return value;
}

// A light, saturated-but-soft colour per hash value.
float3 Pastel(uint hash)
{
    const float3 rgb = float3(hash & 0xFFu, (hash >> 8) & 0xFFu, (hash >> 16) & 0xFFu) / 255.0f;
    return 0.35f + 0.6f * rgb;
}

// The albedo of a block value (face bits included) for the debug views and the
// G-buffer. Packed encoding: opaque rgb5 (bit 15 set), emissive rgb4 (bits 15..14
// = 01), glass with a fixed tint; block-ID encoding: a pastel hash of the id. Air
// (transparent id 0) is magenta, so a primary ray that reports an air boundary is
// visible.
float3 BlockAlbedo(uint v)
{
    if (g_blockEncoding == kEncodingPacked)
    {
        if ((v & 0xC000u) == 0u)
        {
            return CaelixTransparentId(v) == 0u ? kAirColor : kGlassTint;
        }
        if ((v & 0x8000u) != 0u)
        {
            return float3((v >> 10) & 31u, (v >> 5) & 31u, v & 31u) / 31.0f;
        }
        return float3((v >> 10) & 15u, (v >> 6) & 15u, (v >> 2) & 15u) / 15.0f;
    }
    if ((v & 0x8000u) == 0u)
    {
        const uint id = CaelixTransparentId(v);
        return id == 0u ? kAirColor : Pastel(HashU32(id | 0x10000u));
    }
    return Pastel(HashU32(v & 0x7FFFu));
}

// ---------------------------------------------------------------------------
// Random numbers. NATIVE: one 32-bit state per pixel, seeded by a hash of the
// pixel and the frame seed, advanced by xorshift32. Same helper semantics as
// Unity (an inout state, values in [0, 1)).

uint PcgHash(uint value)
{
    const uint state = value * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint InitRandom(uint2 pixel, uint frame_seed)
{
    const uint state = PcgHash(pixel.x ^ (pixel.y << 16) ^ (frame_seed * 0x9E3779B9u));
    return state != 0u ? state : 0x9E3779B9u;  // xorshift never leaves 0
}

// Uniform in [0, 1): the top 24 bits of the next xorshift32 state.
float RandomFloat01(inout uint state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(state >> 8) * (1.0f / 16777216.0f);
}

// Unity BlueNoise.hlsl RandomUnitVector(float2): a uniform direction on the sphere.
float3 RandomUnitVector(float2 sample)
{
    const float a = sample.x * K_TWO_PI;
    const float y = sample.y * 2.0f - 1.0f;
    const float sy = sqrt(max(1.0f - y * y, 0.0f));
    return float3(sin(a) * sy, y, cos(a) * sy);
}

float3 RandomUnitVector(inout uint state)
{
    const float u = RandomFloat01(state);
    const float v = RandomFloat01(state);
    return RandomUnitVector(float2(u, v));
}

// ---------------------------------------------------------------------------
// Shading

// URP Common.hlsl SafeNormalize: a zero vector stays zero.
float3 SafeNormalize(float3 v)
{
    const float dp3 = max(1.175494351e-38f, dot(v, v));
    return v * rsqrt(dp3);
}

// Unity Utils.hlsl: Schlick's approximation.
float FresnelReflectAmountOpaque(float n1, float n2, float3 incident, float3 normal)
{
    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    const float cos_x = -dot(normal, incident);
    const float x = 1.0f - cos_x;
    const float xx = x * x;
    return r0 + (1.0f - r0) * xx * xx * x;
}

// Unity Utils.hlsl: Schlick's approximation for an interface from a medium of ior n1
// into one of ior n2. When n1 > n2 the cosine of the transmitted direction is used,
// and past the critical angle it returns 1 (total internal reflection).
float FresnelReflectAmountTransparent(float n1, float n2, float3 incident, float3 normal)
{
    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    float cos_x = -dot(normal, incident);
    if (n1 > n2)
    {
        const float n = n1 / n2;
        const float sin_t2 = n * n * (1.0f - cos_x * cos_x);
        if (sin_t2 >= 1.0f)
        {
            return 1.0f;
        }
        cos_x = sqrt(1.0f - sin_t2);
    }
    const float x = 1.0f - cos_x;
    const float xx = x * x;
    return r0 + (1.0f - r0) * xx * xx * x;
}

// Unity CaelixMediumExtinction: the transmission of a straight segment of length t
// inside the transparent medium `medium` (a face-free transparent id; 0 = air: the
// fog, exp(-fog density * t), 1 without fog). A miss has t = 0, so the sky is never
// attenuated.
float3 MediumExtinction(CaelixFrameConstants fc, uint medium, float t)
{
    if (medium == 0u)
    {
        return exp(-fc.fog_color.w * t).xxx;
    }
    const CaelixMaterial m = GET_MATERIAL(medium);
    return exp(-(1.0f - m.albedo) * t * m.extinction);
}

// One segment of length t in `medium` that ENDS ON A SURFACE (t = 0 for a miss): the throughput
// loses the transmission; in the air with fog, the fog's emission along the segment,
// fog_color (1 - transmission), is added to `radiance` with the throughput before the segment.
void TraverseSegment(CaelixFrameConstants fc, uint medium, float t, inout float3 throughput,
                     inout float3 radiance)
{
    const float3 transmission = MediumExtinction(fc, medium, t);
    if (medium == 0u && fc.fog_color.w > 0.0f)
    {
        radiance += throughput * fc.fog_color.rgb * (1.0f - transmission);
    }
    throughput *= transmission;
}

// One reflect-or-refract decision at a transparent interface, shared by the delta
// chain (Unity AdvanceDeltaRay) and the transparent branch of ShadeVoxelHit: a ray
// in medium `source` (0 = air) meets a voxel of transparent id `destination` (ior
// `destination_ior`) through a face with world normal `normal`, which is flipped
// against the ray. Refraction is chosen when it exists and RandomFloat01 >= the
// Fresnel reflectance; the random number is drawn only when refraction exists, as
// in Unity (k = 1: the probability carries the Fresnel weight).
// `split`: kInterfaceRandom = the random choice above; kInterfaceSplitReflect /
// kInterfaceSplitRefract = the delta checkerboard (Unity's first-interface split): the
// branch is given, no random number is drawn, and `weight` = 2 F (reflection) or
// 2 (1 - F) (refraction) carries the Fresnel weight that the probability no longer does.
// Total internal reflection reflects with weight 1 and is not split (`split_applied` false).
static const int kInterfaceRandom = -1;
static const int kInterfaceSplitReflect = 0;
static const int kInterfaceSplitRefract = 1;

struct CaelixInterface
{
    float3 normal;     // flipped against the incoming ray
    float3 direction;  // the chosen direction, normalized
    bool refracted;
    bool can_refract;    // false: total internal reflection
    uint medium;         // the medium after the interface: destination when refracted, else source
    float weight;        // throughput factor: 1, or the checkerboard's 2 F / 2 (1 - F)
    bool split_applied;  // the checkerboard decided this interface
};

CaelixInterface CrossInterface(uint source, uint destination, float destination_ior,
                               float3 ray_direction, float3 normal, int split, inout uint rng)
{
    CaelixInterface it;
    it.normal = SafeNormalize(normal);
    if (dot(ray_direction, it.normal) > 0.0f)
    {
        it.normal = -it.normal;
    }
    const float source_ior = source != 0u ? GET_MATERIAL(source).ior : 1.0f;
    const float target_ior = destination != 0u ? destination_ior : 1.0f;
    const float eta = source_ior / target_ior;
    const float3 reflection = reflect(ray_direction, it.normal);
    const float3 refraction = refract(ray_direction, it.normal, eta);
    it.can_refract = dot(refraction, refraction) > 0.000001f;
    const float fresnel = it.can_refract
                              ? FresnelReflectAmountTransparent(source_ior, target_ior, ray_direction, it.normal)
                              : 1.0f;
    it.refracted = false;
    it.weight = 1.0f;
    it.split_applied = false;
    if (it.can_refract)
    {
        if (split == kInterfaceRandom)
        {
            it.refracted = RandomFloat01(rng) >= fresnel;
        }
        else
        {
            it.refracted = split == kInterfaceSplitRefract;
            it.weight = 2.0f * (it.refracted ? 1.0f - fresnel : fresnel);
            it.split_applied = true;
        }
    }
    it.direction = SafeNormalize(it.refracted ? refraction : reflection);
    it.medium = it.refracted ? destination : source;
    return it;
}

struct CaelixShadeResult
{
    float3 albedo;
    float3 emission;
    float emission_luminance;  // luminance of the material's emission before emission_scale (list-light test)
    float3 bounce_origin;
    float3 bounce_direction;
    bool specular_lobe;  // the bounce sampled the specular lobe (always at a transparent block)
    float k;             // lobe probability divisor (1, as in Unity)
    float roughness;     // NRD's linear roughness of the lobe: 1 - smoothness; 0 at a transparent block
};

// Unity CaelixShadeVoxelHit.
// Opaque branch: albedo and emission of the block, specular chance
// lerp(metallic, 1, fresnel * smoothness) (0 with g_forceDiffuse), cosine-weighted diffuse direction
// normalize(N + unit), the specular direction lerped by smoothness, the lobe chosen
// by one random number, k = 1, origin pushed off along N. The random numbers are
// drawn in Unity's order (lobe, then the unit vector).
// Transparent branch (the hit is an interface from the medium `medium` into the
// medium of the block's id): reflect or refract as in CrossInterface (whatever the
// smoothness, as in Unity), albedo 1 (the medium's colour is its extinction, a
// per-segment term), the block's emission, the specular lobe with roughness 0 (both
// directions are delta), origin pushed off to the side of the chosen direction;
// `medium` becomes the medium of the outgoing ray.
CaelixShadeResult ShadeVoxelHit(uint block, float3 hit_position, float3 ray_direction,
                                float3 normal, inout uint rng, inout uint medium)
{
    const CaelixMaterial material = GET_MATERIAL(block);
    CaelixShadeResult shade;
    shade.emission = material.emission * g_emissionScale;
    shade.emission_luminance = dot(material.emission, float3(0.2126f, 0.7152f, 0.0722f));
    shade.k = 1.0f;
    CAELIX_BRANCH
    if (!IsOpaque(int(block)))
    {
        const CaelixInterface it = CrossInterface(medium, CaelixTransparentId(block), material.ior,
                                                  ray_direction, normal, kInterfaceRandom, rng);
        shade.albedo = float3(1.0f, 1.0f, 1.0f);
        shade.bounce_origin =
            hit_position + (it.refracted ? -K_RAY_ORIGIN_PUSH_OFF : K_RAY_ORIGIN_PUSH_OFF) * it.normal;
        shade.bounce_direction = it.direction;
        shade.specular_lobe = true;
        shade.roughness = 0.0f;
        medium = it.medium;
        return shade;
    }
    shade.albedo = material.albedo;

    const float fresnel = FresnelReflectAmountOpaque(1.0f, material.ior, ray_direction, normal);
    const float specular_chance = g_forceDiffuse ? 0.0f : lerp(material.metallic, 1.0f, fresnel * material.smoothness);
    const float do_specular = (RandomFloat01(rng) < specular_chance) ? 1.0f : 0.0f;

    float3 diffuse_dir = SafeNormalize(normal + RandomUnitVector(rng));
    diffuse_dir = (dot(diffuse_dir, diffuse_dir) > 0.0f) ? diffuse_dir : normal;
    float3 specular_dir = reflect(ray_direction, normal);
    specular_dir = SafeNormalize(lerp(diffuse_dir, specular_dir, material.smoothness));

    shade.bounce_origin = hit_position + K_RAY_ORIGIN_PUSH_OFF * normal;
    shade.bounce_direction = lerp(diffuse_dir, specular_dir, do_specular);
    shade.specular_lobe = do_specular > 0.5f;
    shade.roughness = 1.0f - material.smoothness;
    return shade;
}

// The specular albedo guide of DLSS Ray Reconstruction (RR integration guide, appendix "Specular
// Albedo Calculation", from Ray Tracing Gems chapter 32): the average specular reflectance of a
// GGX lobe with reflectance `specular_color` (F0), alpha = roughness^2 and the view cosine `NoV`.
float3 EnvBRDFApprox2(float3 specular_color, float alpha, float NoV)
{
    NoV = abs(NoV);
    float4 X;
    X.x = 1.0f;
    X.y = NoV;
    X.z = NoV * NoV;
    X.w = NoV * X.z;
    float4 Y;
    Y.x = 1.0f;
    Y.y = alpha;
    Y.z = alpha * alpha;
    Y.w = alpha * Y.z;
    const float2x2 M1 = float2x2(0.99044f, -1.28514f, 1.29678f, -0.755907f);
    const float3x3 M2 = float3x3(1.0f, 2.92338f, 59.4188f, 20.3225f, -27.0302f, 222.592f, 121.563f, 626.13f,
                                 316.627f);
    const float2x2 M3 = float2x2(0.0365463f, 3.32707f, 9.0632f, -9.04756f);
    const float3x3 M4 = float3x3(1.0f, 3.59685f, -1.36772f, 9.04401f, -16.3174f, 9.22949f, 5.56589f, 19.7886f,
                                 -20.2123f);
    float bias = dot(mul(M1, X.xy), Y.xy) * rcp(dot(mul(M2, X.xyw), Y.xyw));
    const float scale = dot(mul(M3, X.xy), Y.xy) * rcp(dot(mul(M4, X.xzw), Y.xyw));
    // The guide's correction for a specular reflectance of 0.
    bias *= saturate(specular_color.g * 50.0f);
    return mad(specular_color, max(0.0f, scale), max(0.0f, bias));
}

// Unity CaelixSanitizeRadiance (any NaN or Inf channel drops the whole colour), then
// a clamp to the FP16 range of the radiance targets.
float3 SanitizeRadiance(float3 radiance)
{
    const float3 safe = any(isnan(radiance) | isinf(radiance)) ? float3(0.0f, 0.0f, 0.0f) : radiance;
    return min(safe, kHalfMax);
}

// ---------------------------------------------------------------------------
// Sky (NATIVE: the host's cubemap or a procedural sky; Unity samples a PBR sky cubemap)

// The 9 real spherical harmonics basis functions (bands 0..2, Ramamoorthi and Hanrahan
// 2001) at the unit direction d. Shared by the SH projection kernel (caelix_sky_sh.hlsl)
// and the irradiance lookup below, so both use the same order and constants.
void CaelixShBasis(float3 d, out float basis[9])
{
    basis[0] = 0.282095f;
    basis[1] = 0.488603f * d.y;
    basis[2] = 0.488603f * d.z;
    basis[3] = 0.488603f * d.x;
    basis[4] = 1.092548f * d.x * d.y;
    basis[5] = 1.092548f * d.y * d.z;
    basis[6] = 0.315392f * (3.0f * d.z * d.z - 1.0f);
    basis[7] = 1.092548f * d.x * d.z;
    basis[8] = 0.546274f * (d.x * d.x - d.y * d.y);
}

// Procedural sky: above the horizon lerp(horizon, zenith, sqrt(dir.y)), below it
// lerp(horizon, ground, sqrt(-dir.y)). No sun disk.
float3 ProceduralSky(CaelixFrameConstants fc, float3 direction)
{
    if (direction.y >= 0.0f)
    {
        return lerp(fc.sky_horizon.xyz, fc.sky_zenith.xyz, sqrt(saturate(direction.y)));
    }
    return lerp(fc.sky_horizon.xyz, fc.sky_ground.xyz, sqrt(saturate(-direction.y)));
}

// The sky without the sun disk: the cubemap at mip `lod` (clamped to the last mip) when
// the host gave one, else the procedural sky (lod ignored).
float3 SkyRadianceNoSunLod(CaelixFrameConstants fc, float3 direction, float lod)
{
#ifdef CAELIX_SKY_BINDING_IMAGE
    CAELIX_BRANCH
    if (fc.sky_cubemap != 0u)
    {
        return g_sky.SampleLevel(g_sky_sampler, direction, min(lod, fc.sky_max_lod)).rgb;
    }
#endif
    return ProceduralSky(fc, direction);
}

// The sky without the sun disk at full resolution (mip 0).
float3 SkyRadianceNoSun(CaelixFrameConstants fc, float3 direction)
{
    return SkyRadianceNoSunLod(fc, direction, 0.0f);
}

// The sun radiance inside the sun disk (dot(direction, sun_dir) > cos r), else 0.
float3 SunDisk(CaelixFrameConstants fc, float3 direction)
{
    return dot(direction, fc.sun_dir.xyz) > fc.sun_cos_radius ? fc.sun_radiance.xyz
                                                              : float3(0.0f, 0.0f, 0.0f);
}

// The sky plus the sun radiance inside the sun disk.
float3 SkyRadiance(CaelixFrameConstants fc, float3 direction)
{
    return SkyRadianceNoSun(fc, direction) + SunDisk(fc, direction);
}

// The irradiance of the cubemap sky at a surface with unit normal `normal`, divided by
// pi: the 9 coefficients at fc.sky_sh (already convolved with the clamped cosine) at the
// normal. Only valid when fc.sky_cubemap = 1.
float3 SkyIrradianceOverPi(CaelixFrameConstants fc, float3 normal)
{
    float basis[9];
    CaelixShBasis(normal, basis);
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    CAELIX_UNROLL
    for (uint i = 0; i < 9; ++i)
    {
        sum += asfloat(CAELIX_LOAD_U32x4(fc.sky_sh + uint64_t(i) * 16)).rgb * basis[i];
    }
    return max(sum, 0.0f) * (1.0f / K_PI);
}

// The radiance that a DIFFUSE bounce ray (cosine-weighted around `normal`, direction
// `direction`) receives when it misses. Cubemap: the SH irradiance / pi, the same value
// for every direction (a low-noise estimate of the cosine-weighted sky; it ignores which
// part of the sky the scene hides, so it is biased by the sky visibility), plus the sun
// disk when `sees_sun` (the cubemap has no sun). Procedural: the sky in `direction`,
// with the sun disk when `sees_sun` (unchanged from before the cubemap).
float3 SkyDiffuseMiss(CaelixFrameConstants fc, float3 normal, float3 direction, bool sees_sun)
{
    const float3 sun = sees_sun ? SunDisk(fc, direction) : float3(0.0f, 0.0f, 0.0f);
    CAELIX_BRANCH
    if (fc.sky_cubemap != 0u)
    {
        return SkyIrradianceOverPi(fc, normal) + sun;
    }
    return ProceduralSky(fc, direction) + sun;
}

// The radiance that a SPECULAR bounce ray of a lobe with roughness `roughness` (1 -
// smoothness, 0 for a delta lobe) receives when it misses: the sky at the mip
// roughness * (mip_levels - 1) (the procedural sky ignores it), plus the sun disk when
// `sees_sun`.
float3 SkySpecularMiss(CaelixFrameConstants fc, float3 direction, float roughness, bool sees_sun)
{
    const float3 sun = sees_sun ? SunDisk(fc, direction) : float3(0.0f, 0.0f, 0.0f);
    return SkyRadianceNoSunLod(fc, direction, roughness * fc.sky_max_lod) + sun;
}

// ---------------------------------------------------------------------------
// Sun next-event estimation (kRenderFlagSunNee)

// A direction uniform in the solid angle of the sun cone around sun_dir:
// cos theta = 1 - u1 (1 - cos r), phi = 2 pi u2, pdf = 1 / sun_solid_angle.
// 1 - cos r is taken from sun_solid_angle / (2 pi) (computed in double on the CPU),
// so a small disk keeps its precision; sin theta = sqrt(a (2 - a)), a = 1 - cos theta.
float3 SampleSunDirection(CaelixFrameConstants fc, inout uint rng)
{
    const float one_minus_cos_r = fc.sun_solid_angle * (1.0f / K_TWO_PI);
    const float a = RandomFloat01(rng) * one_minus_cos_r;
    const float phi = RandomFloat01(rng) * K_TWO_PI;
    const float cos_theta = 1.0f - a;
    const float sin_theta = sqrt(max(a * (2.0f - a), 0.0f));
    // Orthonormal basis around the sun direction (Duff et al. 2017, branchless).
    const float3 w = fc.sun_dir.xyz;
    const float sign_z = w.z >= 0.0f ? 1.0f : -1.0f;
    const float p = -1.0f / (sign_z + w.z);
    const float q = w.x * w.y * p;
    const float3 t = float3(1.0f + sign_z * w.x * w.x * p, sign_z * q, -sign_z * w.x);
    const float3 b = float3(q, sign_z + w.y * w.y * p, -w.y);
    return normalize(t * (cos(phi) * sin_theta) + b * (sin(phi) * sin_theta) + w * cos_theta);
}

#endif  // CAELIX_SHADING_HLSL
