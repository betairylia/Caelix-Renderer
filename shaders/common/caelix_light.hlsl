// The scene's light list (docs/LIGHT_LIST.md, entry layout docs/LIGHT_FACES.md): the list-light
// rule and the layout of the light table and its entries. Mirrors lib/src/scene/gpu_structs.h
// (LightTableHeader, LightEntry); change both together.
//
// A material is a list light when the luminance of its emission (before emission_scale) is at
// least the scene's min_emission. The finalize kernel writes one light word per AABB row (the
// asuint of the luminance sum of the brick's alive list-light voxels, 0 = none); the compact and
// scan kernels (shaders/scene/caelix_light_*.hlsl) build the table from the light words:
//   header    64 B   uint count; uint dropped; uint capacity; uint pad; float total_weight; pad
//   entries   capacity x 448 B
//   cdf       capacity x 4 B     inclusive prefix sum of the entries' total_weight
// An entry (448 B):
//   0     uint2 face_mask[6][8]  64 B per face f (+X 0, -X 1, +Y 2, -Y 3, +Z 4, -Z 5): bit v of
//                                slice z is set when voxel v is a list light and its face f is open
//                                (on the brick border, or the neighbour inside the brick is empty)
//   384   float face_weight[6]   luminance sum over the set bits of face_mask[f]
//   408   float total_weight     luminance sum over the brick's list-light voxels (the CDF weight)
//   412   uint group; 416 uint id; 420 uint entity; 424 int3 origin; 436 uint pad[3]
// Voxel v = x | y << 3 | z << 6 (x fastest); slice z of a face mask is a uint2 (low half: y 0..3,
// high half: y 4..7; bit x + 8 (y & 3) in the half). The render kernels load the 64 bytes from 384
// on as a LightEntryHeader and a face mask only when they need it.
#ifndef CAELIX_LIGHT_HLSL
#define CAELIX_LIGHT_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_material.hlsl"

static const uint kLightHeaderBytes = 64;
static const uint kLightEntryBytes = 448;
static const uint kLightEntryFaceMaskBytes = 64;   // one face's mask: 8 slices x 8 B
static const uint kLightEntryWeightsOffset = 384;  // face_weight[6], the start of the header part
static const uint kLightEntryTotalOffset = 408;
static const uint kLightEntryOriginOffset = 424;
static const uint kLightHeaderCount = 0;     // byte offsets in the table header
static const uint kLightHeaderDropped = 4;
static const uint kLightHeaderCapacity = 8;
static const uint kLightHeaderTotal = 16;

// Bytes 384..447 of an entry: everything but the face masks.
struct LightEntryHeader
{
    float face_weight[6];  // luminance sum of the set bits of face_mask[f]
    float total_weight;
    uint group;            // scene-wide group index
    uint id;               // dense brick id
    uint entity;
    int3 origin;           // entity-local voxel origin of the brick
};

float CaelixLuminance(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

// The luminance of the emission of block code `code` (the table entry before emission_scale).
// g_materialTable must be set.
float MaterialEmissionLuminance(uint code)
{
    const uint64_t address = g_materialTable + uint64_t(CaelixMaterialCode(code)) * kMaterialBytes + 16;
    return CaelixLuminance(asfloat(CAELIX_LOAD_U32x4(address)).xyz);
}

uint64_t LightEntryAddress(uint64_t table, uint index)
{
    return table + kLightHeaderBytes + uint64_t(index) * kLightEntryBytes;
}

uint64_t LightCdfAddress(uint64_t table, uint capacity, uint index)
{
    return table + kLightHeaderBytes + uint64_t(capacity) * kLightEntryBytes + uint64_t(index) * 4;
}

LightEntryHeader LoadLightEntryHeader(uint64_t address)
{
    const uint4 w0 = CAELIX_LOAD_U32x4(address + kLightEntryWeightsOffset);
    const uint4 w1 = CAELIX_LOAD_U32x4(address + kLightEntryWeightsOffset + 16);
    const uint4 w2 = CAELIX_LOAD_U32x4(address + kLightEntryWeightsOffset + 32);
    const uint4 w3 = CAELIX_LOAD_U32x4(address + kLightEntryWeightsOffset + 48);
    LightEntryHeader h;
    h.face_weight[0] = asfloat(w0.x);
    h.face_weight[1] = asfloat(w0.y);
    h.face_weight[2] = asfloat(w0.z);
    h.face_weight[3] = asfloat(w0.w);
    h.face_weight[4] = asfloat(w1.x);
    h.face_weight[5] = asfloat(w1.y);
    h.total_weight = asfloat(w1.z);
    h.group = w1.w;
    h.id = w2.x;
    h.entity = w2.y;
    h.origin = int3(w2.z, w2.w, w3.x);
    return h;
}

// The mask of face `face` (0..5) of the entry at `address`: slice z in mask[z].
void LoadLightFaceMask(uint64_t address, uint face, out uint2 mask[8])
{
    const uint64_t base = address + uint64_t(face) * kLightEntryFaceMaskBytes;
    CAELIX_UNROLL
    for (uint i = 0; i < 4; ++i)
    {
        const uint4 w = CAELIX_LOAD_U32x4(base + i * 16);
        mask[2 * i] = w.xy;
        mask[2 * i + 1] = w.zw;
    }
}

// Whether voxel `voxel` (0..7 per axis) has its bit set in the mask of face `face`: one 8-byte load
// of the voxel's slice.
bool LightFaceMaskBit(uint64_t address, uint face, uint3 voxel)
{
    const uint2 slice = CAELIX_LOAD_U32x2(address + uint64_t(face) * kLightEntryFaceMaskBytes + voxel.z * 8u);
    const uint half_word = voxel.y < 4u ? slice.x : slice.y;
    return ((half_word >> (voxel.x + 8u * (voxel.y & 3u))) & 1u) != 0u;
}

// The brick's mean emitting direction in entity-local axes, from the face weights: the
// luminance-weighted sum of the open-face directions, normalised; 0 when the sum is (close to) 0,
// i.e. every axis emits the same both ways. The cell fill's orientation term and the MIS proxy
// (which mirrors it) both use this.
float3 LightBrickNormalLocal(LightEntryHeader h)
{
    const float3 n = float3(h.face_weight[0] - h.face_weight[1], h.face_weight[2] - h.face_weight[3],
                            h.face_weight[4] - h.face_weight[5]);
    const float n_length = length(n);
    return n_length > 1e-6f * h.total_weight ? n / n_length : float3(0.0f, 0.0f, 0.0f);
}

#endif  // CAELIX_LIGHT_HLSL
