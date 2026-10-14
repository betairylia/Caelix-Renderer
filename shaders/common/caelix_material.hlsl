// The scene's material table (Unity's g_Materials, CaelixMaterialTable.hlsl): one
// CaelixMaterial (48 bytes, the C ABI struct of include/caelix/caelix.h) per 16-bit
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

struct CaelixMaterial {
  float3 albedo;
  float smoothness;
  float3 emission;
  float metallic;
  float ior;
  float extinction;
};

static const uint kMaterialBytes = 48;

// Set by the kernel from CaelixFrameConstants::materials before the first lookup.
static uint64_t g_materialTable;

// The table index of a raw block code: opaque codes as they are, transparent codes
// without their face bits.
inline uint CaelixMaterialCode(uint blk) {
  blk &= 0xFFFFu;
  return IsOpaque(int(blk)) ? blk : CaelixTransparentId(blk);
}

CaelixMaterial CaelixGetMaterial(uint blk) {
  const uint64_t address = g_materialTable + uint64_t(CaelixMaterialCode(blk)) * kMaterialBytes;
  const float4 w0 = asfloat(CAELIX_LOAD_U32x4(address));
  const float4 w1 = asfloat(CAELIX_LOAD_U32x4(address + 16));
  const float2 w2 = asfloat(CAELIX_LOAD_U32x2(address + 32));
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
