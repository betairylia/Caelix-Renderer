// Shading helpers of the trace and composite kernels: block colours (debug views
// and materials), the random number stream, the opaque shading rule of Unity's
// path tracer (CaelixShadeVoxelHit, opaque branch) and the procedural sky.
//
// Ported from Unity (Caelix/Runtime/Rendering/Shaders):
//   Utils/Utils.hlsl                      K_T_MAX, K_RAY_ORIGIN_PUSH_OFF, FresnelReflectAmountOpaque
//   Utils/BlueNoise.hlsl                  RandomFloat01 / RandomUnitVector semantics (inout state)
//   def/CaelixPackedSceneMaterial.hlsl    the packed material decode
//   PathTrace_full/CaelixPathTraceCommon.hlsl  CaelixShadeVoxelHit, CaelixSanitizeRadiance
// Differences, all marked NATIVE: the random stream is a hash + xorshift per pixel
// (Unity samples a spatio-temporal blue-noise texture); a transparent block is
// shaded as an opaque diffuse surface with a fixed tint (the delta chain through
// glass comes in a later step); the sky is procedural (Unity samples a cubemap).
#ifndef CAELIX_SHADING_HLSL
#define CAELIX_SHADING_HLSL

#include "caelix_platform.hlsl"
#include "caelix_record.hlsl"
#include "caelix_frame.hlsl"
#include "caelix_block.hlsl"

static const float K_PI = 3.1415926535f;
static const float K_TWO_PI = 6.283185307f;
// Unity Utils.hlsl: K_RAY_ORIGIN_PUSH_OFF 0.002 (world units along the normal).
static const float K_RAY_ORIGIN_PUSH_OFF = 0.002f;
// Largest finite FP16 value: radiance written to the RGBA16F targets is clamped to it.
static const float kHalfMax = 65504.0f;

static const float3 kAirColor = float3(1.0f, 0.0f, 1.0f);     // air hit: should not happen
static const float3 kGlassTint = float3(0.6f, 0.8f, 1.0f);

// ---------------------------------------------------------------------------
// Block colours

uint HashU32(uint value) {
  value ^= value >> 16;
  value *= 0x7FEB352Du;
  value ^= value >> 15;
  value *= 0x846CA68Bu;
  value ^= value >> 16;
  return value;
}

// A light, saturated-but-soft colour per hash value.
float3 Pastel(uint hash) {
  const float3 rgb = float3(hash & 0xFFu, (hash >> 8) & 0xFFu, (hash >> 16) & 0xFFu) / 255.0f;
  return 0.35f + 0.6f * rgb;
}

// The albedo of a block value (face bits included) for the debug views and the
// G-buffer. Packed encoding: opaque rgb5 (bit 15 set), emissive rgb4 (bits 15..14
// = 01), glass with a fixed tint; block-ID encoding: a pastel hash of the id. Air
// (transparent id 0) is magenta, so a primary ray that reports an air boundary is
// visible.
float3 BlockAlbedo(uint v) {
  if (g_blockEncoding == kEncodingPacked) {
    if ((v & 0xC000u) == 0u) {
      return CaelixTransparentId(v) == 0u ? kAirColor : kGlassTint;
    }
    if ((v & 0x8000u) != 0u) {
      return float3((v >> 10) & 31u, (v >> 5) & 31u, v & 31u) / 31.0f;
    }
    return float3((v >> 10) & 15u, (v >> 6) & 15u, (v >> 2) & 15u) / 15.0f;
  }
  if ((v & 0x8000u) == 0u) {
    const uint id = CaelixTransparentId(v);
    return id == 0u ? kAirColor : Pastel(HashU32(id | 0x10000u));
  }
  return Pastel(HashU32(v & 0x7FFFu));
}

// ---------------------------------------------------------------------------
// Materials

struct CaelixMaterial {
  float3 albedo;
  float3 emission;
  float smoothness;
  float metallic;
  float ior;
};

// The lighting material of a block value. Packed encoding (Unity's
// GetMaterial_Color): opaque rgb5 / 31, emissive rgb4 / 15 with
// emission = albedo * 2^(3 * strength) (CAELIX_PACKED_EMISSION_SCALE = 1),
// smoothness 0, metallic 0, IOR 1.5. Block-ID encoding: the pastel hash colour,
// no emission (the native library has no material table yet).
// NATIVE: a transparent block (glass, also an air boundary) is an opaque diffuse
// surface with kGlassTint and no emission until the glass step ports the delta
// chain.
CaelixMaterial DecodeMaterial(uint v) {
  CaelixMaterial m;
  m.smoothness = 0.0f;
  m.metallic = 0.0f;
  m.ior = 1.5f;
  m.emission = float3(0.0f, 0.0f, 0.0f);
  if (!IsOpaque(int(v))) {
    m.albedo = kGlassTint;
  } else if (g_blockEncoding == kEncodingPacked) {
    if ((v & 0x8000u) != 0u) {
      m.albedo = float3((v >> 10) & 31u, (v >> 5) & 31u, v & 31u) / 31.0f;
    } else {
      m.albedo = float3((v >> 10) & 15u, (v >> 6) & 15u, (v >> 2) & 15u) / 15.0f;
      m.emission = m.albedo * exp2(3.0f * float(v & 3u));
    }
  } else {
    m.albedo = Pastel(HashU32(v & 0x7FFFu));
  }
  return m;
}

// ---------------------------------------------------------------------------
// Random numbers. NATIVE: one 32-bit state per pixel, seeded by a hash of the
// pixel and the frame seed, advanced by xorshift32. Same helper semantics as
// Unity (an inout state, values in [0, 1)).

uint PcgHash(uint value) {
  const uint state = value * 747796405u + 2891336453u;
  const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

uint InitRandom(uint2 pixel, uint frame_seed) {
  const uint state = PcgHash(pixel.x ^ (pixel.y << 16) ^ (frame_seed * 0x9E3779B9u));
  return state != 0u ? state : 0x9E3779B9u;  // xorshift never leaves 0
}

// Uniform in [0, 1): the top 24 bits of the next xorshift32 state.
float RandomFloat01(inout uint state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return float(state >> 8) * (1.0f / 16777216.0f);
}

// Unity BlueNoise.hlsl RandomUnitVector(float2): a uniform direction on the sphere.
float3 RandomUnitVector(float2 sample) {
  const float a = sample.x * K_TWO_PI;
  const float y = sample.y * 2.0f - 1.0f;
  const float sy = sqrt(max(1.0f - y * y, 0.0f));
  return float3(sin(a) * sy, y, cos(a) * sy);
}

float3 RandomUnitVector(inout uint state) {
  const float u = RandomFloat01(state);
  const float v = RandomFloat01(state);
  return RandomUnitVector(float2(u, v));
}

// ---------------------------------------------------------------------------
// Shading

// URP Common.hlsl SafeNormalize: a zero vector stays zero.
float3 SafeNormalize(float3 v) {
  const float dp3 = max(1.175494351e-38f, dot(v, v));
  return v * rsqrt(dp3);
}

// Unity Utils.hlsl: Schlick's approximation.
float FresnelReflectAmountOpaque(float n1, float n2, float3 incident, float3 normal) {
  float r0 = (n1 - n2) / (n1 + n2);
  r0 *= r0;
  const float cos_x = -dot(normal, incident);
  const float x = 1.0f - cos_x;
  const float xx = x * x;
  return r0 + (1.0f - r0) * xx * xx * x;
}

struct CaelixShadeResult {
  float3 albedo;
  float3 emission;
  float3 bounce_origin;
  float3 bounce_direction;
  bool specular_lobe;  // the bounce sampled the specular lobe
  float k;             // lobe probability divisor (1, as in Unity)
};

// Unity CaelixShadeVoxelHit, opaque branch: albedo and emission of the block,
// specular chance lerp(metallic, 1, fresnel * smoothness), cosine-weighted diffuse
// direction normalize(N + unit), the specular direction lerped by smoothness, the
// lobe chosen by one random number, k = 1, origin pushed off along N. The random
// numbers are drawn in Unity's order (lobe, then the unit vector).
CaelixShadeResult ShadeVoxelHit(uint block, float3 hit_position, float3 ray_direction,
                                float3 normal, inout uint rng) {
  const CaelixMaterial material = DecodeMaterial(block);
  CaelixShadeResult shade;
  shade.albedo = material.albedo;
  shade.emission = material.emission;

  const float fresnel = FresnelReflectAmountOpaque(1.0f, material.ior, ray_direction, normal);
  const float specular_chance = lerp(material.metallic, 1.0f, fresnel * material.smoothness);
  const float do_specular = (RandomFloat01(rng) < specular_chance) ? 1.0f : 0.0f;

  float3 diffuse_dir = SafeNormalize(normal + RandomUnitVector(rng));
  diffuse_dir = (dot(diffuse_dir, diffuse_dir) > 0.0f) ? diffuse_dir : normal;
  float3 specular_dir = reflect(ray_direction, normal);
  specular_dir = SafeNormalize(lerp(diffuse_dir, specular_dir, material.smoothness));

  shade.k = 1.0f;
  shade.bounce_origin = hit_position + K_RAY_ORIGIN_PUSH_OFF * normal;
  shade.bounce_direction = lerp(diffuse_dir, specular_dir, do_specular);
  shade.specular_lobe = do_specular > 0.5f;
  return shade;
}

// Unity CaelixSanitizeRadiance (any NaN or Inf channel drops the whole colour), then
// a clamp to the FP16 range of the radiance targets.
float3 SanitizeRadiance(float3 radiance) {
  const float3 safe = any(isnan(radiance) | isinf(radiance)) ? float3(0.0f, 0.0f, 0.0f) : radiance;
  return min(safe, kHalfMax);
}

// ---------------------------------------------------------------------------
// Sky (NATIVE: procedural; Unity samples a PBR sky cubemap)

// Above the horizon lerp(horizon, zenith, sqrt(dir.y)), below it
// lerp(horizon, ground, sqrt(-dir.y)). No sun disk.
float3 SkyRadianceNoSun(CaelixFrameConstants fc, float3 direction) {
  if (direction.y >= 0.0f) {
    return lerp(fc.sky_horizon.xyz, fc.sky_zenith.xyz, sqrt(saturate(direction.y)));
  }
  return lerp(fc.sky_horizon.xyz, fc.sky_ground.xyz, sqrt(saturate(-direction.y)));
}

// The sky plus the sun radiance inside the sun disk (dot(direction, sun_dir) > cos r).
float3 SkyRadiance(CaelixFrameConstants fc, float3 direction) {
  float3 sky = SkyRadianceNoSun(fc, direction);
  if (dot(direction, fc.sun_dir.xyz) > fc.sun_cos_radius) {
    sky += fc.sun_radiance.xyz;
  }
  return sky;
}

// ---------------------------------------------------------------------------
// Sun next-event estimation (kRenderFlagSunNee)

// A direction uniform in the solid angle of the sun cone around sun_dir:
// cos theta = 1 - u1 (1 - cos r), phi = 2 pi u2, pdf = 1 / sun_solid_angle.
// 1 - cos r is taken from sun_solid_angle / (2 pi) (computed in double on the CPU),
// so a small disk keeps its precision; sin theta = sqrt(a (2 - a)), a = 1 - cos theta.
float3 SampleSunDirection(CaelixFrameConstants fc, inout uint rng) {
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
