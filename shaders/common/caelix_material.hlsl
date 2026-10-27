// The scene's material tables (Unity's g_Materials, CaelixMaterialTable.hlsl), packed per 16-bit
// block code (docs/MATERIALS.md; the CPU side packs them in lib/src/scene/material_pack.h):
//   hot table (CaelixFrameConstants::materials), 32 bytes per code, halves low 16 then high 16:
//     w0 albedo.r | albedo.g, w1 albedo.b | smoothness, w2 emission.r | emission.g,
//     w3 emission.b | metallic, w4 ior | extinction,
//     w5 sss_probability | sss_model << 16 (8 bits) | flags << 24 (0),
//     w6 texture_set | connect_group << 16 (u16 each), w7 0;
//   subsurface table (CaelixFrameConstants::subsurface_table), 32 bytes per code:
//     w0 (1 - sss_albedo.r) | (1 - sss_albedo.g), w1 (1 - sss_albedo.b) | sss_anisotropy,
//     w2 sss_steps (8 bits), w3 0, w4..w7 sss_params[0..7] in pairs.
// The per-bounce read is one hot entry; the subsurface table is read only by the code that needs it.
// The CPU side builds and uploads both (lib/src/scene/material_table.*); the decode of packed
// colours, the glass palette and the host's table all live there.
//
// GET_MATERIAL(code): a transparent code loses its face bits first (Unity's packed
// GetMaterial_Color does the same), so the entry of the face-free code is read.
#ifndef CAELIX_MATERIAL_HLSL
#define CAELIX_MATERIAL_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_block.hlsl"

struct CaelixMaterial
{
    float3 albedo;
    float smoothness;
    float3 emission;
    float metallic;
    float ior;
    float extinction;
    float sss_probability;  // unused by the kernels so far
    uint sss_model;         // unused by the kernels so far
};

struct CaelixSubsurface
{
    float3 one_minus_albedo;
    float anisotropy;
    uint steps;
    float params[8];
};

static const uint kMaterialBytes = 32;
// Field offsets (macros: see common/caelix_record.hlsl). CaelixGetMaterial loads the 16-byte rows
// at 0 (albedo, smoothness, emission, metallic) and 16 (ior, extinction, sss, texture).
#define kMaterialAlbedo 0
#define kMaterialEmission 8
#define kMaterialIor 16
#define kMaterialSss 20
#define kMaterialTexture 24

static const uint kSubsurfaceBytes = 32;
#define kSubsurfaceAlbedo 0
#define kSubsurfaceSteps 8
#define kSubsurfaceParams 16

// Set by the kernel from CaelixFrameConstants::materials / subsurface_table before the first lookup.
static uint64_t g_materialTable;
static uint64_t g_subsurfaceTable;

// The table index of a raw block code: opaque codes as they are, transparent codes
// without their face bits.
inline uint CaelixMaterialCode(uint blk)
{
    blk &= 0xFFFFu;
    return IsOpaque(int(blk)) ? blk : CaelixTransparentId(blk);
}

// The low and high half of a packed word.
inline float CaelixHalfLo(uint w)
{
    return f16tof32(w & 0xFFFFu);
}

inline float CaelixHalfHi(uint w)
{
    return f16tof32(w >> 16);
}

CaelixMaterial CaelixGetMaterial(uint blk)
{
    const uint64_t address = g_materialTable + uint64_t(CaelixMaterialCode(blk)) * kMaterialBytes;
    const uint4 w0 = CAELIX_LOAD_U32x4(address);           // kMaterialAlbedo = 0
    const uint4 w1 = CAELIX_LOAD_U32x4(address + kMaterialIor);
    CaelixMaterial m;
    m.albedo = float3(CaelixHalfLo(w0.x), CaelixHalfHi(w0.x), CaelixHalfLo(w0.y));
    m.smoothness = CaelixHalfHi(w0.y);
    m.emission = float3(CaelixHalfLo(w0.z), CaelixHalfHi(w0.z), CaelixHalfLo(w0.w));
    m.metallic = CaelixHalfHi(w0.w);
    m.ior = CaelixHalfLo(w1.x);
    m.extinction = CaelixHalfHi(w1.x);
    m.sss_probability = CaelixHalfLo(w1.y);
    m.sss_model = (w1.y >> 16) & 0xFFu;
    return m;
}

// The emission of a table entry (`code` is a table index, see CaelixMaterialCode), without the
// emission scale.
float3 CaelixMaterialEmission(uint64_t table, uint code)
{
    const uint2 w = CAELIX_LOAD_U32x2(table + uint64_t(code) * kMaterialBytes + kMaterialEmission);
    return float3(CaelixHalfLo(w.x), CaelixHalfHi(w.x), CaelixHalfLo(w.y));
}

CaelixSubsurface CaelixGetSubsurface(uint blk)
{
    const uint64_t address = g_subsurfaceTable + uint64_t(CaelixMaterialCode(blk)) * kSubsurfaceBytes;
    const uint4 w0 = CAELIX_LOAD_U32x4(address);  // kSubsurfaceAlbedo = 0
    const uint4 w1 = CAELIX_LOAD_U32x4(address + kSubsurfaceParams);
    CaelixSubsurface s;
    s.one_minus_albedo = float3(CaelixHalfLo(w0.x), CaelixHalfHi(w0.x), CaelixHalfLo(w0.y));
    s.anisotropy = CaelixHalfHi(w0.y);
    s.steps = w0.z & 0xFFu;
    CAELIX_UNROLL
    for (uint i = 0; i < 4; ++i)
    {
        s.params[2 * i] = CaelixHalfLo(w1[i]);
        s.params[2 * i + 1] = CaelixHalfHi(w1[i]);
    }
    return s;
}

#define GET_MATERIAL(blk) CaelixGetMaterial(uint(blk))

#endif  // CAELIX_MATERIAL_HLSL
