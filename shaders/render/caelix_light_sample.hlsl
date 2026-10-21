// Light list sampling of the render kernels (docs/LIGHT_LIST.md, section 2): the light cells
// (world-space ReGIR reservoirs over the scene's light table), the global CDF of the table, the
// point sampling inside a brick entry, the next-event estimation at a diffuse path vertex and its
// MIS with the BSDF path (docs/LIGHT_MIS.md: the density proxy, LightNeeMisWeight, LightListCanSample).
//
// Light cells: a hash table of fc.light_cell_capacity cells in g_light_cells (trace set binding
// 21), 296 bytes each (288 before step 3 of the light list):
//   0    uint key_lo, key_hi    the 64-bit fingerprint of (entity, level, coordinate); key_lo 0 = empty
//   8    uint touched_frame     fc.frame_counter of the newest lookup
//   12   uint entity
//   16   int3 coord             the cell coordinate at `level`: floor(p) >> (3 + 2 level), p entity-local
//   28   uint level             0: 8-voxel cells, 1: 32-voxel cells
//   32   float C_prev           the cell normalisation C_c of the fill before the latest one (the MIS
//                               proxy reads this one; 0 = none yet: the global proxy)
//   36   float C_cur            C_c of the latest fill: the mean of w_sum / K over the 32 reservoirs
//   40   32 x { uint light; float W }   light = entry index of the table, W = the RIS weight (0 = empty)
// The trace kernel looks cells up and inserts missing ones (LightCellLookup); caelix_light_cells.hlsl
// fills the reservoirs of every live cell before the next trace and evicts cells untouched for 64
// frames. The fingerprint is claimed with a 32-bit compare-and-swap on key_lo (64-bit buffer atomics
// are not a device requirement); key_hi is stored after the claim. A lookup that sees key_lo match
// but key_hi not yet written probes on and may insert the same key twice: harmless, every
// reservoir carries its own W. A trace never reads the C_c computed from the reservoirs it samples
// (the fill moves C_cur to C_prev before writing C_cur): a ratio of two estimates from the same
// reservoirs would be biased.
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

static const uint kLightCellBytes = 296;
static const uint kLightCellReservoirs = 32;
static const uint kLightCellCPrevOffset = 32;
static const uint kLightCellCCurOffset = 36;
static const uint kLightCellReservoirOffset = 40;
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

// The point `p` and the direction `v` of an entity under its rows (EntityCurRows) in world space.
float3 RowsTransform(float4 r0, float4 r1, float4 r2, float3 p)
{
    const float4 h = float4(p, 1.0f);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

float3 RowsRotate(float4 r0, float4 r1, float4 r2, float3 v)
{
    return float3(dot(r0.xyz, v), dot(r1.xyz, v), dot(r2.xyz, v));
}

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
                // eviction), so the cell samples the global table until the next fill. C_prev and
                // C_cur start at 0 (the MIS proxy is the global one until the second fill).
                g_light_cells.Store(base + 4, key.y);
                g_light_cells.Store(base + 8, fc.frame_counter);
                g_light_cells.Store(base + 12, entity);
                g_light_cells.Store3(base + 16, asuint(coord));
                g_light_cells.Store(base + 28, level);
                g_light_cells.Store2(base + kLightCellCPrevOffset, uint2(0u, 0u));
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

// The world frame of a light brick under the entity's CURRENT transform: the brick's origin corner
// (entity-local origin) and the three entity axes (the rotation part of the rows).
struct LightBrickFrame
{
    float3 origin;
    float3 axes[3];
};

LightBrickFrame LightBrickFrameOf(CaelixFrameConstants fc, LightEntryHeader header)
{
    float4 r0;
    float4 r1;
    float4 r2;
    EntityCurRows(fc, header.entity, r0, r1, r2);
    LightBrickFrame b;
    b.origin = RowsTransform(r0, r1, r2, float3(header.origin));
    b.axes[0] = float3(r0.x, r1.x, r2.x);
    b.axes[1] = float3(r0.y, r1.y, r2.y);
    b.axes[2] = float3(r0.z, r1.z, r2.z);
    return b;
}

// The candidate faces of a brick (origin and axes, world; the entry's two extent words) seen from the
// receiver `x_r`, as a 6-bit mask (bit f = face f, +X 0, -X 1, +Y 2, -Y 3, +Z 4, -Z 5;
// docs/LIGHT_FACES.md section 3): per axis a, t_a = dot(axis_a, x_r - origin) in voxel units (the
// brick spans 0..8); +a is a candidate when t_a > min_row + 1 (the lowest +a face plane of the
// face's mask), -a when t_a < max_row (the highest -a face plane); a face with an empty mask (extent
// 0xFF) never is. Both faces of an axis are candidates for a receiver between the two planes. Every
// face that a receiver can see (cos_l > 0) is a candidate: a +a face at row r has its plane at
// r + 1 >= min_row + 1, a -a face at r <= max_row. A face on the brick border whose planes all lie
// beyond the receiver (a top-row voxel's +Y face seen from below) is not. The sampler and the MIS
// proxy both call this function.
uint LightCandidateFaceMask(float3 origin, float3 axis0, float3 axis1, float3 axis2, uint extent0, uint extent1,
                            float3 x_r)
{
    const float3 d = x_r - origin;
    const float3 t = float3(dot(axis0, d), dot(axis1, d), dot(axis2, d));
    uint mask = 0u;
    CAELIX_UNROLL
    for (uint a = 0; a < 3; ++a)
    {
        const uint lowest = LightFaceExtent(extent0, extent1, 2u * a);        // lowest +a face row
        const uint highest = LightFaceExtent(extent0, extent1, 2u * a + 1u);  // highest -a face row
        mask |= (lowest != 0xFFu && t[a] > float(lowest) + 1.0f) ? 1u << (2u * a) : 0u;
        mask |= (highest != 0xFFu && t[a] < float(highest)) ? 2u << (2u * a) : 0u;
    }
    return mask;
}

// The sum of the face weights over the candidate faces of `mask` (in face order, so that the sampler
// and the proxy add the same numbers in the same order).
float LightCandidateWeightSum(float face_weight[6], uint mask)
{
    float sum = 0.0f;
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        sum += ((mask >> f) & 1u) != 0u ? face_weight[f] : 0.0f;
    }
    return sum;
}

// The description of a list-light point for the MIS proxy (docs/LIGHT_MIS.md section 1,
// docs/LIGHT_FACES.md section 4), the same for every point of one face of one voxel: the brick's
// world frame (origin corner and the three entity axes) and the entry's two face extent words (for
// the candidate faces of a receiver), the six face weights, the point's face and 1 / n_face
// (n_face = the set bits of that face's mask), the entry's total weight, and the brick's world
// centre (origin + 4 along each axis) and mean normal (world, LightBrickNormalLocal rotated; length 0
// without one).
struct LightMisDesc
{
    float3 origin;
    float3 axes[3];
    uint face_extent[2];
    float face_weight[6];
    uint face;
    float inv_n_face;
    float brick_total;
    float3 brick_centre;
    float3 brick_normal;
};

LightMisDesc LightMisDescZero()
{
    LightMisDesc d;
    d.origin = float3(0.0f, 0.0f, 0.0f);
    d.axes[0] = float3(0.0f, 0.0f, 0.0f);
    d.axes[1] = float3(0.0f, 0.0f, 0.0f);
    d.axes[2] = float3(0.0f, 0.0f, 0.0f);
    d.face_extent[0] = 0xFFFFFFFFu;  // every face empty
    d.face_extent[1] = 0xFFFFFFFFu;
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        d.face_weight[f] = 0.0f;
    }
    d.face = 0u;
    d.inv_n_face = 0.0f;
    d.brick_total = 0.0f;
    d.brick_centre = float3(0.0f, 0.0f, 0.0f);
    d.brick_normal = float3(0.0f, 0.0f, 0.0f);
    return d;
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
    LightMisDesc mis; // the point's MIS description (LightMisDescOf of the entry, the voxel and the face)
};

// The world frame of one voxel of a brick: the voxel centre and the three entity axes.
struct LightVoxelFrame
{
    float3 centre;
    float3 axes[3];
};

LightVoxelFrame LightVoxelFrameIn(LightBrickFrame b, uint3 voxel)
{
    const float3 c = float3(voxel) + 0.5f;
    LightVoxelFrame f;
    f.centre = b.origin + b.axes[0] * c.x + b.axes[1] * c.y + b.axes[2] * c.z;
    f.axes[0] = b.axes[0];
    f.axes[1] = b.axes[1];
    f.axes[2] = b.axes[2];
    return f;
}

LightVoxelFrame LightVoxelFrameOf(CaelixFrameConstants fc, LightEntryHeader header, uint3 voxel)
{
    return LightVoxelFrameIn(LightBrickFrameOf(fc, header), voxel);
}

// The MIS description of a point on face `face` of the entry's brick (frame `b`), `n_face` = the set
// bits of that face's mask. SampleLightEntryWith (NEE side) and LightMisDescOf (BSDF side, DI
// resolve) both build it here from the same frame function.
LightMisDesc LightMisDescWith(LightEntryHeader header, LightBrickFrame b, uint face, uint n_face)
{
    LightMisDesc d;
    d.origin = b.origin;
    d.axes[0] = b.axes[0];
    d.axes[1] = b.axes[1];
    d.axes[2] = b.axes[2];
    d.face_extent[0] = header.face_extent[0];
    d.face_extent[1] = header.face_extent[1];
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        d.face_weight[f] = header.face_weight[f];
    }
    d.face = min(face, 5u);
    d.inv_n_face = n_face > 0u ? 1.0f / float(n_face) : 0.0f;
    d.brick_total = header.total_weight;
    d.brick_centre = b.origin + (b.axes[0] + b.axes[1] + b.axes[2]) * 4.0f;
    const float3 n_local = LightBrickNormalLocal(header);
    d.brick_normal = b.axes[0] * n_local.x + b.axes[1] * n_local.y + b.axes[2] * n_local.z;
    return d;
}

// The number of set bits of the mask of face `face` of the entry at `address` (4 loads).
uint LightFaceCount(uint64_t address, uint face)
{
    uint2 mask[8];
    LoadLightFaceMask(address, face, mask);
    uint n = 0u;
    CAELIX_UNROLL
    for (uint z = 0; z < 8; ++z)
    {
        n += countbits(mask[z].x) + countbits(mask[z].y);
    }
    return n;
}

// The MIS description of a point on face `face` of the entry at `address` (header `header`), the
// BSDF side (TracePath) and the ReSTIR DI resolve. `voxel` does not enter the description: every
// point of one face set has the same in-brick density.
LightMisDesc LightMisDescOf(CaelixFrameConstants fc, LightEntryHeader header, uint64_t address, uint3 voxel,
                            uint face)
{
    const uint f = min(face, 5u);
    return LightMisDescWith(header, LightBrickFrameOf(fc, header), f, LightFaceCount(address, f));
}

// The sum of the face weights of the candidate faces of `d`'s brick seen from the receiver `x_r`
// (LightCandidateFaceMask, exactly as SampleLightEntryWith computes it).
float LightCandidateWeight(LightMisDesc d, float3 x_r)
{
    return LightCandidateWeightSum(d.face_weight, LightCandidateFaceMask(d.origin, d.axes[0], d.axes[1], d.axes[2],
                                                                         d.face_extent[0], d.face_extent[1], x_r));
}

// The per-vertex half of the light NEE's density proxy (docs/LIGHT_MIS.md section 1): the
// UNJITTERED light cell of the vertex, when it has a normalisation from an earlier fill, else the
// global CDF. A deterministic function of the vertex, so that the NEE (LightNextEvent) and the BSDF
// segment that leaves the vertex (TracePath) evaluate the same density.
struct LightNeeProxyContext
{
    bool cell;           // the cell proxy (a cell with C_prev > 0); false: the global proxy
    float3 cell_centre;  // world centre of the cell (as the fill computes it)
    float near_d2;       // (cell half size + 4)^2: the fill's distance floor
    float C;             // the cell's C_prev
    float total;         // the light table's total weight; 0 without a table or without entries
};

// The context of cell `slot` (-1 = none) of (entity, level, coord); `total` = the table's total.
LightNeeProxyContext LightNeeProxyOfSlot(CaelixFrameConstants fc, int slot, uint entity, uint level, int3 coord,
                                         float total)
{
    LightNeeProxyContext ctx;
    ctx.cell = false;
    ctx.cell_centre = float3(0.0f, 0.0f, 0.0f);
    ctx.near_d2 = 0.0f;
    ctx.C = 0.0f;
    ctx.total = total;
    if (slot < 0)
    {
        return ctx;
    }
    const float c_prev = asfloat(g_light_cells.Load(uint(slot) * kLightCellBytes + kLightCellCPrevOffset));
    if (!(c_prev > 0.0f))
    {
        return ctx;
    }
    const float cell_size = float(8u << (2u * level));
    float4 r0;
    float4 r1;
    float4 r2;
    EntityCurRows(fc, entity, r0, r1, r2);
    ctx.cell = true;
    ctx.cell_centre = RowsTransform(r0, r1, r2, (float3(coord) + 0.5f) * cell_size);
    ctx.near_d2 = (0.5f * cell_size + 4.0f) * (0.5f * cell_size + 4.0f);
    ctx.C = c_prev;
    return ctx;
}

// The proxy context of the vertex x_r (pushed-off position) on brick group `group`, entity-local
// point `object_point` (LightCellPoint): the unjittered cell lookup (it inserts the cell when it is
// missing, as every lookup does). LightNextEvent builds the same context from the same lookup.
LightNeeProxyContext LightNeeProxyAt(CaelixFrameConstants fc, float3 x_r, uint group, float3 object_point)
{
    float total = 0.0f;
    if (fc.light_table != 0u)
    {
        const uint count = CAELIX_LOAD_U32(fc.light_table + kLightHeaderCount);
        const float table_total = asfloat(CAELIX_LOAD_U32(fc.light_table + kLightHeaderTotal));
        total = (count != 0u && table_total > 0.0f) ? table_total : 0.0f;
    }
    int slot = -1;
    uint entity = 0;
    uint level = 0;
    int3 coord = int3(0, 0, 0);
    if (fc.light_cells != 0u && total > 0.0f)
    {
        level = LightCellLevel(fc, x_r);
        entity = GroupEntity(fc, group);
        coord = LightCellCoord(object_point, level);
        slot = LightCellLookup(fc, entity, level, coord);
    }
    return LightNeeProxyOfSlot(fc, slot, entity, level, coord, total);
}

// The proxy area density of the light NEE at the point of `d` seen from the receiver `x_r`
// (docs/LIGHT_MIS.md section 1, docs/LIGHT_FACES.md section 4). The in-brick density is the
// sampler's: face_weight[face] / W_cand / n_face, W_cand = LightCandidateWeight(d, x_r) (0 when no
// candidate face has weight: the NEE cannot sample the brick from there). With a cell: p_hat_c(b) /
// C_c times the in-brick density, p_hat_c(b) = total_b * G, G and orient as in the fill kernel
// (caelix_light_cells.hlsl). Without one: the global CDF, total_b / total times the in-brick density.
float LightNeeProxyPdfArea(LightNeeProxyContext ctx, LightMisDesc d, float3 x_r)
{
    const float w_cand = LightCandidateWeight(d, x_r);
    const float in_brick = w_cand > 0.0f ? d.face_weight[d.face] * d.inv_n_face / w_cand : 0.0f;
    // The entry's weight times the in-brick density (the step 1 proxy's pdf_base / n_faces).
    const float weighted = d.brick_total * in_brick;
    if (ctx.cell)
    {
        const float3 to_cell = ctx.cell_centre - d.brick_centre;
        const float d2 = dot(to_cell, to_cell);
        float orient = 1.0f;
        if (any(d.brick_normal != 0.0f) && d2 > 1e-12f)
        {
            orient = max(0.1f, dot(d.brick_normal, to_cell * rsqrt(d2)));
        }
        const float G = orient / max(d2, ctx.near_d2);
        return G * weighted / ctx.C;
    }
    return ctx.total > 0.0f ? weighted / ctx.total : 0.0f;
}

// The balance heuristic between the light NEE and the cosine-weighted diffuse bounce at the
// receiver (x_r, n_r) for the light point of `d` in direction `w` (unit) at distance `d_light`;
// cos_l = the light face's cosine toward the receiver. `nee_side`: the NEE's weight
// p_sa / (p_sa + p_bsdf), else the BSDF's p_bsdf / (p_sa + p_bsdf), with p_sa = the proxy area
// density * d^2 / cos_l and p_bsdf = cos_r / pi. Both sides call this function with the same
// context and description, so the two weights of a point sum to 1 in every branch: a non-finite
// or degenerate p_sa (cos_l <= 0) counts as +inf (NEE 1, BSDF 0), both densities 0 as NEE 0, BSDF 1.
float LightNeeMisWeight(LightNeeProxyContext ctx, LightMisDesc d, float3 x_r, float3 n_r, float3 w, float d_light,
                        float cos_l, bool nee_side)
{
    const float p_area = LightNeeProxyPdfArea(ctx, d, x_r);
    const float p_sa = cos_l > 0.0f ? p_area * (d_light * d_light) / cos_l : 0.0f;
    const float p_bsdf = max(0.0f, dot(n_r, w)) * (1.0f / K_PI);
    if (!(cos_l > 0.0f) || !(p_sa <= 3.0e38f))
    {
        return nee_side ? 1.0f : 0.0f;
    }
    const float sum = p_sa + p_bsdf;
    if (!(sum > 0.0f))
    {
        return nee_side ? 0.0f : 1.0f;
    }
    return (nee_side ? p_sa : p_bsdf) / sum;
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

// A point on a list-light voxel of the entry at `address` (header `header`) as seen from the receiver
// `x_r` (docs/LIGHT_FACES.md section 3): a face direction among the brick's candidate faces
// (LightCandidateFaceMask) with probability face_weight[f] / W_cand, a voxel uniformly among the set
// bits of that face's mask (the list-light voxels whose face f is open inside the brick), and a point
// uniformly on that face (area 1). A face covered inside the brick is never chosen; a face on the
// brick border covered by a neighbour brick can be when the receiver lies beyond its plane (the
// shadow ray finds it). Between the lowest +a and the highest -a face plane of an axis both faces
// are candidates, and a chosen face that points away from the receiver gives cos_l <= 0: a wasted
// sample, not a bias. pdf_area = face_weight[f] / W_cand / n_f; the brick's own
// probability is not in it. Four random numbers in every branch (face, voxel, u, v).
// `quantize_uv`: the point is the centre of its unorm16 uv cell (LightUvQuantize), so that the
// ReSTIR DI reservoir, which stores the uv in 2 x 16 bits, names exactly the sampled point.
LightSample SampleLightEntryWith(CaelixFrameConstants fc, LightEntryHeader header, uint64_t address, float3 x_r,
                                 bool quantize_uv, inout uint rng)
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
    s.mis = LightMisDescZero();
    const float u_face = RandomFloat01(rng);
    const float u_voxel = RandomFloat01(rng);
    const float u_s = RandomFloat01(rng);
    const float u_t = RandomFloat01(rng);
    if (!(header.total_weight > 0.0f))
    {
        return s;
    }

    // 1. Face: a candidate by its weight.
    const LightBrickFrame brick = LightBrickFrameOf(fc, header);
    const uint candidates = LightCandidateFaceMask(brick.origin, brick.axes[0], brick.axes[1], brick.axes[2],
                                                   header.face_extent[0], header.face_extent[1], x_r);
    const float w_cand = LightCandidateWeightSum(header.face_weight, candidates);
    if (!(w_cand > 0.0f))
    {
        return s;  // every candidate face is covered: the NEE cannot reach this brick from x_r
    }
    const float target = u_face * w_cand;
    uint face = 6u;
    uint last = 6u;
    float running = 0.0f;
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        const float w = ((candidates >> f) & 1u) != 0u ? header.face_weight[f] : 0.0f;
        running += w;
        if (w > 0.0f)
        {
            last = f;
            if (face == 6u && running > target)
            {
                face = f;
            }
        }
    }
    if (face == 6u)
    {
        face = last;  // rounding: the last candidate with weight
    }

    // 2. Voxel: the k-th set bit of the face's mask, slice by slice, half word by half word.
    uint2 mask[8];
    LoadLightFaceMask(address, face, mask);
    uint n = 0u;
    CAELIX_UNROLL
    for (uint z = 0; z < 8; ++z)
    {
        n += countbits(mask[z].x) + countbits(mask[z].y);
    }
    if (n == 0u)
    {
        return s;
    }
    uint k = min(n - 1u, uint(u_voxel * float(n)));
    uint slice = 8u;
    uint half_index = 0u;
    uint word = 0u;
    CAELIX_UNROLL
    for (uint z2 = 0; z2 < 8; ++z2)
    {
        CAELIX_UNROLL
        for (uint h = 0; h < 2; ++h)
        {
            const uint bits = h == 0u ? mask[z2].x : mask[z2].y;
            const uint c = countbits(bits);
            if (slice == 8u)
            {
                if (k < c)
                {
                    slice = z2;
                    half_index = h;
                    word = bits;
                }
                else
                {
                    k -= c;
                }
            }
        }
    }
    if (slice == 8u)
    {
        return s;
    }
    // Bit x + 8 (y & 3) of the half word -> x + 8 y in the slice.
    const uint bit = NthSetBit(word, k) + 32u * half_index;
    const uint3 voxel = uint3(bit & 7u, bit >> 3, slice);

    // 3. Point, radiance and density.
    const LightVoxelFrame frame = LightVoxelFrameIn(brick, voxel);
    float3 face_normal;
    float3 e1;
    float3 e2;
    LightFaceBasis(frame, face, face_normal, e1, e2);
    s.uv = quantize_uv ? float2(LightUvDequantize(LightUvQuantize(u_s)), LightUvDequantize(LightUvQuantize(u_t)))
                       : float2(u_s - 0.5f, u_t - 0.5f);
    s.position = LightFacePoint(frame, face_normal, e1, e2, s.uv);
    s.normal = face_normal;
    const uint value = BrickVoxelValue(fc, header.group, header.id, voxel);
    s.radiance = GET_MATERIAL(value).emission * g_emissionScale;
    s.pdf_area = (header.face_weight[face] / w_cand) * (1.0f / float(n));
    s.voxel = bit | (slice << 6);
    s.face = face;
    s.mis = LightMisDescWith(header, brick, face, n);
    s.valid = s.pdf_area > 0.0f;
    return s;
}

LightSample SampleLightEntry(CaelixFrameConstants fc, LightEntryHeader header, uint64_t address, float3 x_r,
                             inout uint rng)
{
    return SampleLightEntryWith(fc, header, address, x_r, false, rng);
}

// A light point rebuilt from the indices that a LightSample named (ReSTIR DI, docs/RESTIR_DI.md):
// voxel v (x | y << 3 | z << 6) of the entry's brick, its face and the uv on it, under the entity's
// current transform. The same frame, basis and point expressions as SampleLightEntryWith.
struct LightPoint
{
    float3 position;
    float3 normal;
    float3 radiance;
};

LightPoint LightPointAt(CaelixFrameConstants fc, LightEntryHeader header, uint voxel_index, uint face, float2 uv)
{
    const uint3 voxel = uint3(voxel_index & 7u, (voxel_index >> 3) & 7u, (voxel_index >> 6) & 7u);
    const LightVoxelFrame frame = LightVoxelFrameOf(fc, header, voxel);
    float3 e1;
    float3 e2;
    LightPoint p;
    LightFaceBasis(frame, min(face, 5u), p.normal, e1, e2);
    p.position = LightFacePoint(frame, p.normal, e1, e2, uv);
    const uint value = BrickVoxelValue(fc, header.group, header.id, voxel);
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
// With kGiFlagLightCellJitter (section 2.5) the cell is looked up at `object_point` moved by a
// uniform random offset in (-0.5, 0.5) cells per axis: the pixels of one brick then read up to 8
// cells instead of the same one, so the variance of the cell contents (one refill of 32 reservoirs
// per frame) turns from a per-cell flicker into per-pixel noise. Unbiased: every cell's reservoirs
// are unbiased estimators for any receiver (W does not depend on the receiver).
// `mis` (kGiFlagLightMis, docs/LIGHT_MIS.md): the contribution is multiplied by the balance
// heuristic weight LightNeeMisWeight(nee_side = true), and `ctx` receives the vertex's proxy context
// (the unjittered cell, LightNeeProxyAt's lookup; with the jitter on, the sampling cell is a second
// lookup) for the BSDF segment that leaves the vertex. Without `mis`, `ctx` is the global context
// and the lookups and random numbers are the same as before step 3. The MIS draws no random number.
float3 LightNextEvent(CaelixFrameConstants fc, float3 x_r, float3 n_r, uint group, float3 object_point, bool mis,
                      inout uint rng, out LightNeeProxyContext ctx)
{
    const float u_reservoir = RandomFloat01(rng);
    const float u_global = RandomFloat01(rng);
    const uint64_t table = fc.light_table;
    const uint count = CAELIX_LOAD_U32(table + kLightHeaderCount);
    const float total = asfloat(CAELIX_LOAD_U32(table + kLightHeaderTotal));
    ctx = LightNeeProxyOfSlot(fc, -1, 0u, 0u, int3(0, 0, 0), (count != 0u && total > 0.0f) ? total : 0.0f);
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
        const uint entity = GroupEntity(fc, group);
        const bool jitter = (fc.gi_flags & kGiFlagLightCellJitter) != 0u;
        int slot = -1;
        // The proxy's cell is the unjittered one; without the jitter it is also the sampling cell.
        if (mis || !jitter)
        {
            const int3 coord = LightCellCoord(object_point, level);
            slot = LightCellLookup(fc, entity, level, coord);
            if (mis)
            {
                ctx = LightNeeProxyOfSlot(fc, slot, entity, level, coord, total);
            }
        }
        if (jitter)
        {
            const float3 u_jitter = float3(RandomFloat01(rng), RandomFloat01(rng), RandomFloat01(rng));
            const float3 cell_point = object_point + (u_jitter - 0.5f) * float(8u << (2u * level));
            slot = LightCellLookup(fc, entity, level, LightCellCoord(cell_point, level));
        }
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
    const bool from_cdf = light == kNoLight;
    if (from_cdf)
    {
        light = LightCdfSearch(table, capacity, count, u_global * total);
    }
    const uint64_t address = LightEntryAddress(table, light);
    const LightEntryHeader entry = LoadLightEntryHeader(address);
    if (from_cdf)
    {
        // The global CDF: the brick's probability is total_b / total.
        W_brick = entry.total_weight > 0.0f ? total / entry.total_weight : 0.0f;
    }
    const LightSample s = SampleLightEntry(fc, entry, address, x_r, rng);
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
    const float3 contribution = s.radiance * ((cos_r * (1.0f / K_PI)) * (cos_l / d2) / s.pdf_area * W_brick);
    return mis ? contribution * LightNeeMisWeight(ctx, s.mis, x_r, n_r, w, distance, cos_l, true) : contribution;
}

// Whether the light NEE can produce the point that the ray of `hit` found on a list light
// (docs/LIGHT_MIS.md sections 2 and 4, docs/LIGHT_FACES.md section 4): the hit brick has an entry in
// the light table (the scene's brick -> entry map, fc.light_indices: entry index + 1 per AABB row,
// GroupInfo word 6 = aabb_row_base) and the hit voxel's bit is set in the mask of the HIT face. False
// for a brick the table dropped (beyond its capacity), a voxel whose material is no longer a list
// light, or a face the entry counts as covered: the BSDF path keeps their emission in full, since no
// NEE samples them. On true: the entry's header and address, the voxel (0..7 per axis inside the
// entry's brick: the hit point moved a quarter voxel against the object normal) and the hit face
// (+X 0, -X 1, +Y 2, -Y 3, +Z 4, -Z 5, entity-local axes).
bool LightListCanSample(CaelixFrameConstants fc, TraceResult hit, out LightEntryHeader header, out uint64_t address,
                        out uint3 voxel, out uint face)
{
    header = (LightEntryHeader)0;
    address = 0u;
    voxel = uint3(0u, 0u, 0u);
    face = 0u;
    if (fc.light_indices == 0u || fc.light_table == 0u)
    {
        return false;
    }
    const uint aabb_row_base = CAELIX_LOAD_U32(fc.group_infos + uint64_t(hit.group) * 32 + 24);
    const uint index = CAELIX_LOAD_U32(fc.light_indices + (uint64_t(aabb_row_base) + hit.id) * 4);
    const uint count = CAELIX_LOAD_U32(fc.light_table + kLightHeaderCount);
    if (index == 0u || index - 1u >= count)
    {
        return false;
    }
    address = LightEntryAddress(fc.light_table, index - 1u);
    header = LoadLightEntryHeader(address);
    const float3 object_normal = float3(UnpackObjectNormal(hit.attrib));
    const int3 v = int3(floor(hit.object_hit - 0.25f * object_normal)) - header.origin;
    if (any(v < 0) || any(v > 7))
    {
        return false;
    }
    voxel = uint3(v);
    const uint axis = abs(object_normal.x) > 0.5f ? 0u : (abs(object_normal.y) > 0.5f ? 1u : 2u);
    const float component = axis == 0u ? object_normal.x : (axis == 1u ? object_normal.y : object_normal.z);
    face = axis * 2u + (component < 0.0f ? 1u : 0u);
    return LightFaceMaskBit(address, face, voxel);
}

#endif  // CAELIX_LIGHT_SAMPLE_HLSL
