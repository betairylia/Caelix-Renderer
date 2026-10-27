#pragma once
// The GPU entries of the material tables (docs/MATERIALS.md): CaelixMaterial (the host's view, all
// floats) packed into the 32-byte hot entry and the 32-byte subsurface entry that the kernels read
// (shaders/common/caelix_material.hlsl). Header-only, so the tests can check the packing directly.
#include "caelix/caelix.h"

#include <cstdint>
#include <cstring>

namespace caelix {

// The GPU hot entry: 8 words, halves packed low 16 then high 16.
//   w0 @0  h(albedo.r) | h(albedo.g) << 16
//   w1 @4  h(albedo.b) | h(smoothness) << 16
//   w2 @8  h(emission.r) | h(emission.g) << 16
//   w3 @12 h(emission.b) | h(metallic) << 16
//   w4 @16 h(ior) | h(extinction) << 16
//   w5 @20 h(sss_probability) | (sss_model & 0xFF) << 16 | flags (0) << 24
//   w6 @24 texture_set | connect_group << 16
//   w7 @28 0
constexpr uint32_t kMaterialBytes = 32;
constexpr uint32_t kMaterialAlbedo = 0;
constexpr uint32_t kMaterialEmission = 8;
constexpr uint32_t kMaterialIor = 16;
constexpr uint32_t kMaterialSss = 20;
constexpr uint32_t kMaterialTexture = 24;

// The GPU subsurface entry: 8 words.
//   w0 @0  h(1 - sss_albedo.r) | h(1 - sss_albedo.g) << 16
//   w1 @4  h(1 - sss_albedo.b) | h(sss_anisotropy) << 16
//   w2 @8  sss_steps & 0xFF (bits 8..31 reserved, 0)
//   w3 @12 0
//   w4..w7 @16 h(sss_params[2i]) | h(sss_params[2i + 1]) << 16
constexpr uint32_t kSubsurfaceBytes = 32;
constexpr uint32_t kSubsurfaceAlbedo = 0;
constexpr uint32_t kSubsurfaceSteps = 8;
constexpr uint32_t kSubsurfaceParams = 16;

// The largest finite half.
constexpr float kHalfMax = 65504.0f;

// IEEE binary16 bits of `value`, rounded to nearest even. Magnitudes above the half range clamp to
// the largest finite half (+-65504), NaN gives a quiet NaN (the validated entries have neither).
inline uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    if (magnitude > 0x7F800000u)
    {
        return 0x7E00;
    }
    if (magnitude >= 0x477FF000u)
    {
        // 65520 and above round to infinity in IEEE; clamp to the largest finite half.
        return static_cast<uint16_t>(sign | 0x7BFFu);
    }
    if (magnitude < 0x38800000u)
    {
        // Subnormal half (or zero): the value in units of 2^-24, rounded to nearest even.
        if (magnitude < 0x33000000u)
        {
            return static_cast<uint16_t>(sign);  // below half of the smallest subnormal: +-0
        }
        const uint32_t exponent = magnitude >> 23;
        const uint32_t mantissa = (magnitude & 0x7FFFFFu) | 0x800000u;
        const uint32_t shift = 126u - exponent;  // value / 2^-24 = mantissa * 2^(exponent - 126)
        const uint32_t half = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1u);
        const uint32_t midpoint = 1u << (shift - 1u);
        const uint32_t rounded = half + ((rest > midpoint || (rest == midpoint && (half & 1u) != 0)) ? 1u : 0u);
        return static_cast<uint16_t>(sign | rounded);
    }
    // Normal: rebias the exponent (127 -> 15) and round the 23-bit mantissa to 10 bits. A carry out
    // of the mantissa increments the exponent, which is the correct result.
    const uint32_t rebased = magnitude - ((127u - 15u) << 23);
    const uint32_t half = rebased >> 13;
    const uint32_t rest = rebased & 0x1FFFu;
    const uint32_t rounded = half + ((rest > 0x1000u || (rest == 0x1000u && (half & 1u) != 0)) ? 1u : 0u);
    return static_cast<uint16_t>(sign | rounded);
}

inline uint32_t PackHalf2(float low, float high)
{
    return static_cast<uint32_t>(FloatToHalf(low)) | (static_cast<uint32_t>(FloatToHalf(high)) << 16);
}

// The hot entry of `m` (layout above).
inline void PackMaterialHot(const CaelixMaterial& m, uint32_t out[8])
{
    out[0] = PackHalf2(m.albedo[0], m.albedo[1]);
    out[1] = PackHalf2(m.albedo[2], m.smoothness);
    out[2] = PackHalf2(m.emission[0], m.emission[1]);
    out[3] = PackHalf2(m.emission[2], m.metallic);
    out[4] = PackHalf2(m.ior, m.extinction);
    out[5] = static_cast<uint32_t>(FloatToHalf(m.sss_probability)) | ((m.sss_model & 0xFFu) << 16);
    out[6] = (m.texture_set & 0xFFFFu) | ((m.connect_group & 0xFFFFu) << 16);
    out[7] = 0;
}

// The subsurface entry of `m` (layout above).
inline void PackMaterialSubsurface(const CaelixMaterial& m, uint32_t out[8])
{
    out[0] = PackHalf2(1.0f - m.sss_albedo[0], 1.0f - m.sss_albedo[1]);
    out[1] = PackHalf2(1.0f - m.sss_albedo[2], m.sss_anisotropy);
    out[2] = m.sss_steps & 0xFFu;
    out[3] = 0;
    for (uint32_t i = 0; i < 4; ++i)
    {
        out[4 + i] = PackHalf2(m.sss_params[2 * i], m.sss_params[2 * i + 1]);
    }
}

}  // namespace caelix
