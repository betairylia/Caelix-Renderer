// The scene's material table (Unity's g_Materials, CaelixMaterialTable.hlsl): one
// main row (the first 48 bytes of the C ABI struct CaelixMaterial of include/caelix/caelix.h) per 16-bit
// block code, read through the buffer device address CaelixFrameConstants::materials.
// The CPU side builds and uploads it (lib/src/scene/material_table.*); the decode of
// packed colours, the glass palette and the host's table all live there.
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
};

static const uint kMaterialBytes = 48;
// Field offsets (macros: see common/caelix_record.hlsl). The loads take the 16-byte rows (albedo,
// smoothness), (emission, metallic), (ior, extinction, sss_anisotropy, sss_steps). CaelixGetMaterial
// reads the shading fields only; the subsurface walk (render/caelix_subsurface.hlsl, experiment branch
// sss-expr) reads sss_anisotropy and sss_steps itself, and the row (sss_albedo, sss_probability) of a
// code from the separate subsurface table (CaelixFrameConstants::subsurface_table, kSubsurfaceRowBytes
// per code).
#define kMaterialAlbedo 0
#define kMaterialEmission 16
#define kMaterialIor 32
#define kMaterialSssAnisotropy 40
#define kMaterialSssSteps 44
#define kSubsurfaceRowBytes 16

// Set by the kernel from CaelixFrameConstants::materials before the first lookup.
static uint64_t g_materialTable;

// The table index of a raw block code: opaque codes as they are, transparent codes
// without their face bits.
inline uint CaelixMaterialCode(uint blk)
{
    blk &= 0xFFFFu;
    return IsOpaque(int(blk)) ? blk : CaelixTransparentId(blk);
}

CaelixMaterial CaelixGetMaterial(uint blk)
{
    const uint64_t address = g_materialTable + uint64_t(CaelixMaterialCode(blk)) * kMaterialBytes;
    const float4 w0 = asfloat(CAELIX_LOAD_U32x4(address));  // kMaterialAlbedo = 0
    const float4 w1 = asfloat(CAELIX_LOAD_U32x4(address + kMaterialEmission));
    const float2 w2 = asfloat(CAELIX_LOAD_U32x2(address + kMaterialIor));
    CaelixMaterial m;
    m.albedo = w0.xyz;
    m.smoothness = w0.w;
    m.emission = w1.xyz;
    m.metallic = w1.w;
    m.ior = w2.x;
    m.extinction = w2.y;
    return m;
}

#define GET_MATERIAL(blk) CaelixGetMaterial(uint(blk))

#endif  // CAELIX_MATERIAL_HLSL
