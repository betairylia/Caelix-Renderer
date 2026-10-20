// The scene's light list (docs/LIGHT_LIST.md): the list-light rule, the layout of the light table
// and its entries, and the octahedral normal code. Mirrors lib/src/scene/gpu_structs.h
// (LightTableHeader, LightEntry); change both together.
//
// A material is a list light when the luminance of its emission (before emission_scale) is at
// least the scene's min_emission. The finalize kernel writes one light word per AABB row (the
// asuint of the luminance sum of the brick's alive list-light voxels, 0 = none); the compact and
// scan kernels (shaders/scene/caelix_light_*.hlsl) build the table from the light words:
//   header    64 B   uint count; uint dropped; uint capacity; uint pad; float total_weight; pad
//   entries   capacity x 128 B   LightEntry
//   cdf       capacity x 4 B     inclusive prefix sum of entry.total_weight
// Voxel v = x | y << 3 | z << 6 (x fastest); word w of the mask is the slice z = w, held as a uint2
// (low half: y 0..3, high half: y 4..7).
#ifndef CAELIX_LIGHT_HLSL
#define CAELIX_LIGHT_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_material.hlsl"

static const uint kLightHeaderBytes = 64;
static const uint kLightEntryBytes = 128;
static const uint kLightHeaderCount = 0;     // byte offsets in the header
static const uint kLightHeaderDropped = 4;
static const uint kLightHeaderCapacity = 8;
static const uint kLightHeaderTotal = 16;

struct LightEntry
{
    uint2 mask[8];           // bit v of slice (v >> 6): voxel v is alive and a list light
    float region_weight[8];  // luminance sum of the set bits of slice w
    float total_weight;
    uint group;              // scene-wide group index
    uint id;                 // dense brick id
    uint entity;
    int3 origin;             // entity-local voxel origin of the brick
    uint mean_normal;        // octahedral 2 x 16-bit snorm (entity-local), 0 = none
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

LightEntry LoadLightEntry(uint64_t address)
{
    LightEntry e;
    CAELIX_UNROLL
    for (uint i = 0; i < 4; ++i)
    {
        const uint4 w = CAELIX_LOAD_U32x4(address + i * 16);
        e.mask[2 * i] = w.xy;
        e.mask[2 * i + 1] = w.zw;
    }
    const uint4 r0 = CAELIX_LOAD_U32x4(address + 64);
    const uint4 r1 = CAELIX_LOAD_U32x4(address + 80);
    e.region_weight[0] = asfloat(r0.x);
    e.region_weight[1] = asfloat(r0.y);
    e.region_weight[2] = asfloat(r0.z);
    e.region_weight[3] = asfloat(r0.w);
    e.region_weight[4] = asfloat(r1.x);
    e.region_weight[5] = asfloat(r1.y);
    e.region_weight[6] = asfloat(r1.z);
    e.region_weight[7] = asfloat(r1.w);
    const uint4 a = CAELIX_LOAD_U32x4(address + 96);
    const uint4 b = CAELIX_LOAD_U32x4(address + 112);
    e.total_weight = asfloat(a.x);
    e.group = a.y;
    e.id = a.z;
    e.entity = a.w;
    e.origin = int3(b.xyz);
    e.mean_normal = b.w;
    return e;
}

// Octahedral code of a unit vector: two 16-bit snorm values (x low, y high). A vector whose code
// would be 0 (+Z exactly) gets 1 instead (a 3e-5 tilt), so that 0 keeps meaning "no normal".
uint EncodeOctahedral(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 p = n.xy;
    if (n.z < 0.0f)
    {
        const float2 s = float2(p.x >= 0.0f ? 1.0f : -1.0f, p.y >= 0.0f ? 1.0f : -1.0f);
        p = (1.0f - abs(p.yx)) * s;
    }
    const int2 q = int2(round(clamp(p, -1.0f, 1.0f) * 32767.0f));
    const uint code = (uint(q.x) & 0xFFFFu) | ((uint(q.y) & 0xFFFFu) << 16);
    return code != 0u ? code : 1u;
}

float3 DecodeOctahedral(uint code)
{
    const int2 q = int2(int(code << 16) >> 16, int(code) >> 16);
    const float2 p = max(float2(q) / 32767.0f, -1.0f);
    float3 n = float3(p, 1.0f - abs(p.x) - abs(p.y));
    if (n.z < 0.0f)
    {
        const float2 s = float2(n.x >= 0.0f ? 1.0f : -1.0f, n.y >= 0.0f ? 1.0f : -1.0f);
        n.xy = (1.0f - abs(n.yx)) * s;
    }
    return normalize(n);
}

#endif  // CAELIX_LIGHT_HLSL
