// Light list sampling of the render kernels (docs/LIGHT_LIST.md, section 2): the light cells
// (world-space ReGIR reservoirs over the scene's light table), the global CDF of the table, the
// point sampling inside a brick entry and the next-event estimation at a diffuse path vertex.
//
// Light cells: a hash table of fc.light_cell_capacity cells in g_light_cells (trace set binding
// 21), 288 bytes each:
//   0    uint key_lo, key_hi    the 64-bit fingerprint of (entity, level, coordinate); key_lo 0 = empty
//   8    uint touched_frame     fc.frame_counter of the newest lookup
//   12   uint entity
//   16   int3 coord             the cell coordinate at `level`: floor(p) >> (3 + 2 level), p entity-local
//   28   uint level             0: 8-voxel cells, 1: 32-voxel cells
//   32   32 x { uint light; float W }   light = entry index of the table, W = the RIS weight (0 = empty)
// The trace kernel looks cells up and inserts missing ones (LightCellLookup); caelix_light_cells.hlsl
// fills the reservoirs of every live cell before the next trace and evicts cells untouched for 64
// frames. The fingerprint is claimed with a 32-bit compare-and-swap on key_lo (64-bit buffer atomics
// are not a device requirement); key_hi is stored after the claim. A lookup that sees key_lo match
// but key_hi not yet written probes on and may insert the same key twice: harmless, every
// reservoir carries its own W.
//
// Include after common/caelix_frame.hlsl, common/caelix_light.hlsl, render/caelix_trace_set.hlsl,
// render/caelix_scene_trace.hlsl and render/caelix_shading.hlsl.
#ifndef CAELIX_LIGHT_SAMPLE_HLSL
#define CAELIX_LIGHT_SAMPLE_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_light.hlsl"
#include "render/caelix_scene_trace.hlsl"
#include "render/caelix_shading.hlsl"

static const uint kLightCellBytes = 288;
static const uint kLightCellReservoirs = 32;
static const uint kLightCellReservoirOffset = 32;
static const uint kLightCellProbes = 8;
// A cell not looked up for this many frames is evicted by the fill pass.
static const uint kLightCellMaxAge = 64;
static const uint kNoLight = 0xFFFFFFFFu;
// Light points closer than this (world units) to the receiver belong to the BSDF path, not to the
// light NEE: TracePath keeps the emission of a list light that a segment hits closer than this,
// and ReSTIR DI gives such points a zero target. Area sampling with a 1/d^2 term and a brick
// choice that ignores the distance make the near field an extremely heavy-tailed estimate
// (bistro: 14 % of the light missing after 512 frames; Artifacts/gi-experiments-20261020/
// light-nee/diagnosis); the BSDF path finds near emitters with a bounded weight. Both sides use
// the same distance from the same pushed-off origin, so the split is unbiased. Every kernel sets
// it from CaelixFrameConstants::light_nee_min_distance before the first use.
static float g_lightNeeMinDistance;

uint64_t SplitMix64(uint64_t x)
{
    x += (uint64_t(0x9E3779B9u) << 32) | uint64_t(0x7F4A7C15u);
    x = (x ^ (x >> 30)) * ((uint64_t(0xBF58476Du) << 32) | uint64_t(0x1CE4E5B9u));
    x = (x ^ (x >> 27)) * ((uint64_t(0x94D049BBu) << 32) | uint64_t(0x133111EBu));
    return x ^ (x >> 31);
}

// The fingerprint of a cell: three rounds of splitmix64 over the packed words; the low word is
// forced non-zero (0 marks an empty slot).
uint2 LightCellKey(uint entity, uint level, int3 coord)
{
    uint64_t h = SplitMix64((uint64_t(entity) << 32) | uint64_t(level));
    h = SplitMix64(h ^ ((uint64_t(asuint(coord.y)) << 32) | uint64_t(asuint(coord.x))));
    h = SplitMix64(h ^ uint64_t(asuint(coord.z)));
    const uint lo = uint(h);
    return uint2(lo != 0u ? lo : 1u, uint(h >> 32));
}

// The level of a path vertex at world position `x` (its pixel footprint at its camera distance).
uint LightCellLevel(CaelixFrameConstants fc, float3 x)
{
    const float footprint = 2.0f * length(x - fc.position.xyz) * fc.tan_half_fov_y / float(fc.size.y);
    return footprint < fc.light_level_footprint ? 0u : 1u;
}

// The cell coordinate of the entity-local point `p` at `level`.
int3 LightCellCoord(float3 p, uint level)
{
    return int3(floor(p)) >> int(3u + 2u * level);
}

// Finds the cell of (entity, level, coord), touching it, or inserts it (empty reservoirs). Returns
// the slot, or -1 when 8 probes found neither the key nor a free slot.
int LightCellLookup(CaelixFrameConstants fc, uint entity, uint level, int3 coord)
{
    const uint2 key = LightCellKey(entity, level, coord);
    const uint mask = fc.light_cell_capacity - 1u;
    CAELIX_LOOP
    for (uint probe = 0; probe < kLightCellProbes; ++probe)
    {
        const uint slot = (key.x + probe) & mask;
        const uint base = slot * kLightCellBytes;
        uint key_lo = g_light_cells.Load(base);
        if (key_lo == 0u)
        {
            uint original;
            g_light_cells.InterlockedCompareExchange(base, 0u, key.x, original);
            if (original == 0u)
            {
                // Inserted. A slot with key 0 has empty reservoirs (zero at creation, cleared on
                // eviction), so the cell samples the global table until the next fill.
                g_light_cells.Store(base + 4, key.y);
                g_light_cells.Store(base + 8, fc.frame_counter);
                g_light_cells.Store(base + 12, entity);
                g_light_cells.Store3(base + 16, asuint(coord));
                g_light_cells.Store(base + 28, level);
                return int(slot);
            }
            key_lo = original;
        }
        if (key_lo == key.x && g_light_cells.Load(base + 4) == key.y)
        {
            if (g_light_cells.Load(base + 8) != fc.frame_counter)
            {
                g_light_cells.Store(base + 8, fc.frame_counter);
            }
            return int(slot);
        }
    }
    return -1;
}

// The smallest index i in [0, count) of the light table's CDF with cdf[i] > target.
uint LightCdfSearch(uint64_t table, uint capacity, uint count, float target)
{
    uint lo = 0;
    uint hi = count - 1u;
    CAELIX_LOOP
    while (lo < hi)
    {
        const uint mid = (lo + hi) >> 1;
        if (asfloat(CAELIX_LOAD_U32(LightCdfAddress(table, capacity, mid))) > target)
        {
            hi = mid;
        }
        else
        {
            lo = mid + 1u;
        }
    }
    return lo;
}

// The position (0..31) of the k-th set bit (k from 0) of `v`, which has more than k set bits.
uint NthSetBit(uint v, uint k)
{
    uint position = 0;
    CAELIX_UNROLL
    for (uint width = 16; width >= 1; width >>= 1)
    {
        const uint low = countbits(v & ((1u << width) - 1u));
        if (k >= low)
        {
            k -= low;
            v >>= width;
            position += width;
        }
    }
    return position;
}

struct LightSample
{
    bool valid;
    float3 position;  // world position on the voxel face
    float3 normal;    // world normal of the face
    float3 radiance;  // the voxel's emission times emission_scale
    float pdf_area;   // probability density per unit area of the point within the brick
    // The indices that name the point (LightPointAt rebuilds it from them; docs/RESTIR_DI.md):
    uint voxel;       // v = x | y << 3 | z << 6 inside the entry's brick
    uint face;        // +X 0, -X 1, +Y 2, -Y 3, +Z 4, -Z 5 (entity-local axes)
    float2 uv;        // the point on the face in (-0.5, 0.5)^2 along (e1, e2) of LightFaceBasis
};

// The world frame of voxel `voxel` of `entry` under the entity's CURRENT transform: the voxel
// centre and the three entity axes (the rotation part of the rows).
struct LightVoxelFrame
{
    float3 centre;
    float3 axes[3];
};

LightVoxelFrame LightVoxelFrameOf(CaelixFrameConstants fc, LightEntry entry, uint3 voxel)
{
    float4 r0;
    float4 r1;
    float4 r2;
    EntityCurRows(fc, entry.entity, r0, r1, r2);
    const float3 centre_local = float3(entry.origin + int3(voxel)) + 0.5f;
    const float4 h = float4(centre_local, 1.0f);
    LightVoxelFrame f;
    f.centre = float3(dot(r0, h), dot(r1, h), dot(r2, h));
    f.axes[0] = float3(r0.x, r1.x, r2.x);
    f.axes[1] = float3(r0.y, r1.y, r2.y);
    f.axes[2] = float3(r0.z, r1.z, r2.z);
    return f;
}

// The outward normal of `face` (+X 0, -X 1, +Y 2, -Y 3, +Z 4, -Z 5) and its basis: e1 = entity axis
// (a + 1) % 3, e2 = entity axis (a + 2) % 3 of the face axis a = face >> 1. SampleLightEntry and
// LightPointAt share it, so a stored uv names the same point in both.
void LightFaceBasis(LightVoxelFrame f, uint face, out float3 normal, out float3 e1, out float3 e2)
{
    const uint axis = face >> 1;
    const float sign = (face & 1u) != 0u ? -1.0f : 1.0f;
    normal = f.axes[axis] * sign;
    e1 = f.axes[(axis + 1u) % 3u];
    e2 = f.axes[(axis + 2u) % 3u];
}

// The point at `uv` on the face (normal, e1, e2) of the voxel of frame `f`.
float3 LightFacePoint(LightVoxelFrame f, float3 normal, float3 e1, float3 e2, float2 uv)
{
    return f.centre + 0.5f * normal + uv.x * e1 + uv.y * e2;
}

// The uv quantization of the ReSTIR DI reservoir (2 x unorm16): u in [0, 1) -> q = floor(u * 65536),
// and back to the centre of cell q in (-0.5, 0.5).
uint LightUvQuantize(float u)
{
    return min(uint(u * 65536.0f), 65535u);
}

float LightUvDequantize(uint q)
{
    return (float(q) + 0.5f) * (1.0f / 65536.0f) - 0.5f;
}

// A point on a list-light voxel of `entry` as seen from the receiver `x_r` (section 2.1): a slice by
// its region weight, a voxel of the slice uniformly, one of the axis faces that face the receiver
// uniformly, and a point uniformly on that face (area 1). Whether the face is covered by a neighbour
// is not tested (the shadow ray finds it). The brick's own probability is not in pdf_area.
// `quantize_uv`: the point is the centre of its unorm16 uv cell (LightUvQuantize), so that the
// ReSTIR DI reservoir, which stores the uv in 2 x 16 bits, names exactly the sampled point.
LightSample SampleLightEntryWith(CaelixFrameConstants fc, LightEntry entry, float3 x_r, bool quantize_uv,
                                 inout uint rng)
{
    LightSample s;
    s.valid = false;
    s.position = float3(0.0f, 0.0f, 0.0f);
    s.normal = float3(0.0f, 0.0f, 0.0f);
    s.radiance = float3(0.0f, 0.0f, 0.0f);
    s.pdf_area = 0.0f;
    s.voxel = 0u;
    s.face = 0u;
    s.uv = float2(0.0f, 0.0f);
    const float u_region = RandomFloat01(rng);
    const float u_voxel = RandomFloat01(rng);
    const float u_face = RandomFloat01(rng);
    const float u_s = RandomFloat01(rng);
    const float u_t = RandomFloat01(rng);
    if (!(entry.total_weight > 0.0f))
    {
        return s;
    }

    // 1. Slice.
    const float target = u_region * entry.total_weight;
    uint slice = 8u;
    float running = 0.0f;
    CAELIX_UNROLL
    for (uint w = 0; w < 8; ++w)
    {
        running += entry.region_weight[w];
        if (slice == 8u && entry.region_weight[w] > 0.0f && running > target)
        {
            slice = w;
        }
    }
    if (slice == 8u)
    {
        // Rounding: the last slice with lights.
        CAELIX_UNROLL
        for (uint w2 = 0; w2 < 8; ++w2)
        {
            if (entry.region_weight[w2] > 0.0f)
            {
                slice = w2;
            }
        }
    }
    if (slice == 8u)
    {
        return s;
    }
    const uint2 bits = entry.mask[slice];
    const uint count_lo = countbits(bits.x);
    const uint n = count_lo + countbits(bits.y);
    if (n == 0u)
    {
        return s;
    }

    // 2. Voxel.
    const uint k = min(n - 1u, uint(u_voxel * float(n)));
    const uint bit = k < count_lo ? NthSetBit(bits.x, k) : 32u + NthSetBit(bits.y, k - count_lo);
    const uint3 voxel = uint3(bit & 7u, bit >> 3, slice);

    // 3. Face: the axis faces whose outward normal points to the receiver's side of the centre.
    const LightVoxelFrame frame = LightVoxelFrameOf(fc, entry, voxel);
    const float3 to_receiver = x_r - frame.centre;
    float signs[3];
    uint faces = 0;
    CAELIX_UNROLL
    for (uint a = 0; a < 3; ++a)
    {
        const float d = dot(frame.axes[a], to_receiver);
        signs[a] = d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
        faces += signs[a] != 0.0f ? 1u : 0u;
    }
    if (faces == 0u)
    {
        return s;
    }
    // The pick-th candidate face (in axis order).
    uint remaining = min(faces - 1u, uint(u_face * float(faces)));
    uint axis = 0;
    CAELIX_UNROLL
    for (uint a2 = 0; a2 < 3; ++a2)
    {
        if (signs[a2] != 0.0f)
        {
            if (remaining == 0u)
            {
                axis = a2;
                remaining = 0xFFFFFFFFu;  // found: no later face is taken
            }
            else if (remaining != 0xFFFFFFFFu)
            {
                remaining -= 1u;
            }
        }
    }
    const uint face = axis * 2u + (signs[axis] < 0.0f ? 1u : 0u);
    float3 face_normal;
    float3 e1;
    float3 e2;
    LightFaceBasis(frame, face, face_normal, e1, e2);

    // 4. Point, radiance and density.
    s.uv = quantize_uv ? float2(LightUvDequantize(LightUvQuantize(u_s)), LightUvDequantize(LightUvQuantize(u_t)))
                       : float2(u_s - 0.5f, u_t - 0.5f);
    s.position = LightFacePoint(frame, face_normal, e1, e2, s.uv);
    s.normal = face_normal;
    const uint value = BrickVoxelValue(fc, entry.group, entry.id, voxel);
    s.radiance = GET_MATERIAL(value).emission * g_emissionScale;
    s.pdf_area = (entry.region_weight[slice] / entry.total_weight) * (1.0f / float(n)) * (1.0f / float(faces));
    s.voxel = bit | (slice << 6);
    s.face = face;
    s.valid = s.pdf_area > 0.0f;
    return s;
}

LightSample SampleLightEntry(CaelixFrameConstants fc, LightEntry entry, float3 x_r, inout uint rng)
{
    return SampleLightEntryWith(fc, entry, x_r, false, rng);
}

// A light point rebuilt from the indices that a LightSample named (ReSTIR DI, docs/RESTIR_DI.md):
// voxel v (x | y << 3 | z << 6) of `entry`, its face and the uv on it, under the entity's current
// transform. The same frame, basis and point expressions as SampleLightEntryWith.
struct LightPoint
{
    float3 position;
    float3 normal;
    float3 radiance;
};

LightPoint LightPointAt(CaelixFrameConstants fc, LightEntry entry, uint voxel_index, uint face, float2 uv)
{
    const uint3 voxel = uint3(voxel_index & 7u, (voxel_index >> 3) & 7u, (voxel_index >> 6) & 7u);
    const LightVoxelFrame frame = LightVoxelFrameOf(fc, entry, voxel);
    float3 e1;
    float3 e2;
    LightPoint p;
    LightFaceBasis(frame, min(face, 5u), p.normal, e1, e2);
    p.position = LightFacePoint(frame, p.normal, e1, e2, uv);
    const uint value = BrickVoxelValue(fc, entry.group, entry.id, voxel);
    p.radiance = GET_MATERIAL(value).emission * g_emissionScale;
    return p;
}

// Next-event estimation on the light list at a diffuse path vertex (section 2.3): the receiver
// x_r (the pushed-off vertex position) with world normal n_r, on brick group `group`, entity-local
// position `object_point` (the hit point moved a little into the air, for the cell coordinate).
// The brick comes from one of the 32 reservoirs of the vertex's light cell when it holds one, else
// from the global CDF; the point from SampleLightEntry; one shadow ray. Returns the demodulated
// incident radiance of the Lambert lobe (the albedo is in the caller's throughput, as for the sun
// NEE), 0 without a usable sample.
float3 LightNextEvent(CaelixFrameConstants fc, float3 x_r, float3 n_r, uint group, float3 object_point,
                      inout uint rng)
{
    const float u_reservoir = RandomFloat01(rng);
    const float u_global = RandomFloat01(rng);
    const uint64_t table = fc.light_table;
    const uint count = CAELIX_LOAD_U32(table + kLightHeaderCount);
    const float total = asfloat(CAELIX_LOAD_U32(table + kLightHeaderTotal));
    if (count == 0u || !(total > 0.0f))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    const uint capacity = CAELIX_LOAD_U32(table + kLightHeaderCapacity);

    uint light = kNoLight;
    float W_brick = 0.0f;
    CAELIX_BRANCH
    if (fc.light_cells != 0u)
    {
        const uint level = LightCellLevel(fc, x_r);
        const int slot = LightCellLookup(fc, GroupEntity(fc, group), level, LightCellCoord(object_point, level));
        if (slot >= 0)
        {
            const uint r = min(kLightCellReservoirs - 1u, uint(u_reservoir * float(kLightCellReservoirs)));
            const uint2 reservoir =
                g_light_cells.Load2(uint(slot) * kLightCellBytes + kLightCellReservoirOffset + r * 8u);
            if (asfloat(reservoir.y) > 0.0f && reservoir.x < count)
            {
                light = reservoir.x;
                W_brick = asfloat(reservoir.y);
            }
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

    const LightSample s = SampleLightEntry(fc, entry, x_r, rng);
    if (!s.valid || !(W_brick > 0.0f))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    const float3 d = s.position - x_r;
    const float d2 = dot(d, d);
    if (!(d2 > 0.0f))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    const float distance = sqrt(d2);
    if (distance < g_lightNeeMinDistance)
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    const float3 w = d / distance;
    const float cos_r = dot(n_r, w);
    const float cos_l = -dot(s.normal, w);
    if (!(cos_r > 0.0f) || !(cos_l > 0.0f))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    // The point lies on a voxel face: the ray stops 2 push-offs short of it (as GiVisible).
    RayDesc shadow;
    shadow.Origin = x_r;
    shadow.Direction = w;
    shadow.TMin = 0.0f;
    shadow.TMax = distance - 2.0f * K_RAY_ORIGIN_PUSH_OFF;
    if (shadow.TMax > 0.0f && TraceShadow(fc, shadow))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return s.radiance * ((cos_r * (1.0f / K_PI)) * (cos_l / d2) / s.pdf_area * W_brick);
}

#endif  // CAELIX_LIGHT_SAMPLE_HLSL
