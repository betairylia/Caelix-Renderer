// Light list sampling of the render kernels (docs/LIGHT_NEE.md): the light cells (world-space
// ReGIR reservoirs over the scene's light table, section 3), the global CDF of the table, the point
// sampling inside a brick entry (section 5.4), the next-event estimation at a diffuse path vertex
// (section 5) and its MIS with the BSDF path (section 6: the density proxy, LightNeeMisWeight,
// LightListCanSample).
//
// Light cells: a hash table of fc.light_cell_capacity cells in g_light_cells (trace set binding
// 21), 64 + 8 R bytes each (R = fc.light_cell_reservoirs reservoirs, a power of two in 1 .. 64,
// default 8; 128 B at R 8; LightCellBytes):
//   0    uint key_lo, key_hi    the 64-bit fingerprint of (entity, level, coordinate, face); key_lo 0 = empty
//   8    uint touched_frame     fc.frame_counter of the newest lookup
//   12   uint entity
//   16   int3 coord             the cell coordinate at `level`: floor(p) >> (5 + 2 level), p entity-local
//   28   uint level             0: 32-voxel cells, 1: 128-voxel cells
//   32   float C_prev           the cell normalisation C_c of the fill before the latest one (the MIS
//                               proxy reads this one; 0 = none yet: the global proxy)
//   36   float C_cur            C_c of the latest fill: the mean of w_sum / K over the R reservoirs
//                               (kGiFlagLightCellTemporal: the running mean of dedicated draws)
//   40   float M                kGiFlagLightCellTemporal: the candidate count behind every reservoir of
//                               the cell (uniform over the R: all threads merge the same way); 0 = no history
//   44   float C_n              kGiFlagLightCellTemporal: the draw count behind C_cur; 0 = none
//   48   uint sun_tests         kGiFlagNeeCellVisibility: sun shadow rays traced from receivers of the
//                               cell (TracePath, NextEventAtVertex; atomics), decayed by 1/8 per fill;
//                               0 at insert and eviction
//   52   uint face              the receiver face of the key (kGiFlagLightCellOriented: 0..5 = +X -X +Y
//                               -Y +Z -Z of the receivers' world normal, GiFaceIndex), else
//                               kLightCellFaceNone (6); written by the insert, read by the fill
//   56   uint sun_visible       kGiFlagNeeCellVisibility: of those, the rays that found the sun
//                               unoccluded (same decay); the combined NEE's sun hint is
//                               max(kNeeSunHintFloor, sun_visible / sun_tests) (docs/LIGHT_NEE.md section 10)
//   60   4 bytes of padding
//   64   R x { uint light; float W }   the reservoirs: light = the entry index of a brick of the light
//                               table (kNoLight = empty), W = its unbiased contribution weight (0 =
//                               empty); the vertex samples a point inside the brick
// After the last cell (byte fc.light_cell_capacity x LightCellBytes, LightCellCounterBase) the buffer holds 16
// bytes of usage counters, zeroed by the renderer before every fill and copied to the host after the
// trace (CaelixGiStats):
//   0    uint live              cells with a key after this frame's eviction (fill, thread 0)
//   4    uint active            of those, looked up by the previous frame's trace (fill, thread 0)
//   8    uint inserted          cells inserted by this frame's lookups (LightCellLookup)
//   12   uint failed            lookups that found neither the key nor a free slot in 8 probes
// The trace kernel looks cells up and inserts missing ones (LightCellLookup); caelix_light_cells.hlsl
// fills the reservoirs of every live cell before the next trace and evicts cells untouched for 64
// frames. The fingerprint is claimed with a 32-bit compare-and-swap on key_lo (64-bit buffer atomics
// are not a device requirement); key_hi is stored after the claim. A lookup that sees key_lo match
// and key_hi still 0 (not yet written) takes that cell, so that a thread that lost the CAS does not
// insert the same key again further on. A trace never reads the C_c computed from the reservoirs it
// samples (the fill moves C_cur to C_prev before writing C_cur): a ratio of two estimates from the
// same reservoirs would be biased.
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
#include "render/caelix_restir_gi.hlsl"

// The cell edge at level 0 is 1 << kLightCellBaseShift voxels (32); each level is 4x larger.
static const uint kLightCellBaseShift = 5u;
static const uint kLightCellHeaderBytes = 64;
static const uint kLightCellReservoirBytes = 8;
static const uint kLightCellCPrevOffset = 32;
static const uint kLightCellCCurOffset = 36;
static const uint kLightCellMOffset = 40;
static const uint kLightCellCNOffset = 44;
static const uint kLightCellSunTestsOffset = 48;
static const uint kLightCellFaceOffset = 52;
static const uint kLightCellSunVisibleOffset = 56;
static const uint kLightCellReservoirOffset = 64;
// The header face of an unoriented cell (kGiFlagLightCellOriented off): no receiver face in the key.
static const uint kLightCellFaceNone = 6u;
// The floor of the receiver orientation term of an oriented cell (LightCellReceiverOrient): never 0,
// because a receiver may read a cell of another orientation or position (the jittered lookup, a
// merged slot) and every reservoir must stay an unbiased contribution weight for it; the floor costs
// noise there (W up to 10x), not bias.
static const float kLightCellOrientFloor = 0.1f;
// The floor of the sun hint of the combined NEE (kGiFlagNeeCellVisibility, docs/LIGHT_NEE.md section
// 10): the sun candidate's selection weight is multiplied by max(this, visible / tests) of the
// sampling cell, never 0, so that a visible sun keeps a positive selection probability (unbiased).
static const float kNeeSunHintFloor = 0.05f;
static const uint kLightCellProbes = 8;
// A cell not looked up for this many frames is evicted by the fill pass.
static const uint kLightCellMaxAge = 64;
// An empty reservoir (no entry index).
static const uint kNoLight = 0xFFFFFFFFu;
// The usage counters after the last cell (byte offsets from LightCellCounterBase).
static const uint kLightCellCounterLive = 0;
static const uint kLightCellCounterActive = 4;
static const uint kLightCellCounterInserted = 8;
static const uint kLightCellCounterFailed = 12;
// Light points closer than this (world units) to the receiver belong to the BSDF path, not to the
// light NEE (docs/LIGHT_NEE.md section 2.6): TracePath keeps the emission of a list light that a
// segment hits closer than this. Area sampling with a 1/d^2 term and a brick choice that ignores the
// distance make the near field an extremely heavy-tailed estimate (bistro: 14 % of the light missing
// after 512 frames; Artifacts/gi-experiments-20261020/light-nee/diagnosis); the BSDF path finds near
// emitters with a bounded weight. Both sides use the same distance from the same pushed-off origin,
// so the split is unbiased. Every kernel sets it from CaelixFrameConstants::light_nee_min_distance
// before the first use.
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

// The fingerprint of a cell: three rounds of splitmix64 over the packed words, a fourth over the
// receiver face of an oriented cell (face 0..5; kLightCellFaceNone keeps the unoriented hash); the
// low word is forced non-zero (0 marks an empty slot).
uint2 LightCellKey(uint entity, uint level, int3 coord, uint face)
{
    uint64_t h = SplitMix64((uint64_t(entity) << 32) | uint64_t(level));
    h = SplitMix64(h ^ ((uint64_t(asuint(coord.y)) << 32) | uint64_t(asuint(coord.x))));
    h = SplitMix64(h ^ uint64_t(asuint(coord.z)));
    if (face != kLightCellFaceNone)
    {
        h = SplitMix64(h ^ uint64_t(face));
    }
    const uint lo = uint(h);
    return uint2(lo != 0u ? lo : 1u, uint(h >> 32));
}

// The reservoirs per cell R (fc.light_cell_reservoirs, a power of two in 1 .. 64). A cell is
// 64 + 8 R bytes: the header and R x 8 B reservoirs.
uint LightCellBytes(CaelixFrameConstants fc)
{
    return kLightCellHeaderBytes + kLightCellReservoirBytes * fc.light_cell_reservoirs;
}

// The edge of a cell at `level` in voxels (entity-local units).
float LightCellSize(uint level)
{
    return float(1u << (kLightCellBaseShift + 2u * level));
}

// The byte offset of the usage counters in g_light_cells (after the last cell).
uint LightCellCounterBase(CaelixFrameConstants fc)
{
    return fc.light_cell_capacity * LightCellBytes(fc);
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
    return int3(floor(p)) >> int(kLightCellBaseShift + 2u * level);
}

// The receiver face of the cell key of a vertex with world normal `n_r` (docs/LIGHT_NEE.md section
// 3.1): with kGiFlagLightCellOriented the face index of the normal (GiFaceIndex, 0..5), else
// kLightCellFaceNone.
uint LightCellFaceOf(CaelixFrameConstants fc, float3 n_r)
{
    return (fc.gi_flags & kGiFlagLightCellOriented) != 0u ? GiFaceIndex(n_r) : kLightCellFaceNone;
}

// The world normal of a cell's receiver face (GiFaceNormal; the face comes from the world normal, so
// no entity rotation), the zero vector for kLightCellFaceNone (an unoriented cell).
float3 LightCellFaceNormal(uint face)
{
    return face < kLightCellFaceNone ? GiFaceNormal(face) : float3(0.0f, 0.0f, 0.0f);
}

// The receiver orientation term of an oriented cell (docs/LIGHT_NEE.md section 4.2) for a light at
// world position `light`: max(kLightCellOrientFloor, dot(n_cell, direction from the cell's BACK plane
// point to the light)), where the back plane point is the cell centre moved half a cell against the
// face normal (`near_d2` = (half + 4)^2 as in the fill, so half = sqrt(near_d2) - 4). Every receiver of
// the cell lies in front of that plane, so a light at a receiver's own height (a lamp standing on the
// floor the receivers are part of) keeps a positive cosine instead of the floor; the term is an upper
// bound of the receivers' cosines in height, never 0. 1 for an unoriented cell (n_cell 0).
float LightCellReceiverOrient(float3 n_cell, float3 cell_centre, float near_d2, float3 light)
{
    if (any(n_cell != 0.0f))
    {
        const float half_cell = sqrt(near_d2) - 4.0f;
        const float3 to_light = light - (cell_centre - half_cell * n_cell);
        const float d2 = dot(to_light, to_light);
        return d2 > 1e-12f ? max(kLightCellOrientFloor, dot(n_cell, to_light) * rsqrt(d2)) : 1.0f;
    }
    return 1.0f;
}

// Finds the cell of (entity, level, coord, face), touching it, or inserts it (empty reservoirs).
// Returns the slot, or -1 when 8 probes found neither the key nor a free slot. Counts each insert and
// each failed lookup in the usage counters (kLightCellCounterInserted, kLightCellCounterFailed).
// `face`: LightCellFaceOf of the receiver (kLightCellFaceNone without kGiFlagLightCellOriented).
int LightCellLookup(CaelixFrameConstants fc, uint entity, uint level, int3 coord, uint face)
{
    const uint2 key = LightCellKey(entity, level, coord, face);
    const uint mask = fc.light_cell_capacity - 1u;
    CAELIX_LOOP
    for (uint probe = 0; probe < kLightCellProbes; ++probe)
    {
        const uint slot = (key.x + probe) & mask;
        const uint base = slot * LightCellBytes(fc);
        uint key_lo = g_light_cells.Load(base);
        if (key_lo == 0u)
        {
            uint original;
            g_light_cells.InterlockedCompareExchange(base, 0u, key.x, original);
            if (original == 0u)
            {
                // Inserted. A slot with key 0 has empty reservoirs (zero at creation, cleared on
                // eviction), so the cell samples the global table until the next fill. C_prev,
                // C_cur, M and C_n start at 0 (the MIS proxy is the global one until the second
                // fill; no temporal history). The sun visibility counters start at 0.
                g_light_cells.Store(base + 4, key.y);
                g_light_cells.Store(base + 8, fc.frame_counter);
                g_light_cells.Store(base + 12, entity);
                g_light_cells.Store3(base + 16, asuint(coord));
                g_light_cells.Store(base + 28, level);
                g_light_cells.Store4(base + kLightCellCPrevOffset, uint4(0u, 0u, 0u, 0u));
                g_light_cells.Store4(base + kLightCellSunTestsOffset, uint4(0u, face, 0u, 0u));
                g_light_cells.InterlockedAdd(LightCellCounterBase(fc) + kLightCellCounterInserted, 1u);
                return int(slot);
            }
            key_lo = original;
        }
        // A high word of 0 is a pending insert (key_hi is stored after the claim) and counts as a
        // match: it stops a thread that lost the CAS from inserting a duplicate further on. A low-word
        // collision with a different high word while the insert is pending merges two cells
        // (harmless: every reservoir carries its own W).
        const uint key_hi = key_lo == key.x ? g_light_cells.Load(base + 4) : 0u;
        if (key_lo == key.x && (key_hi == key.y || key_hi == 0u))
        {
            if (g_light_cells.Load(base + 8) != fc.frame_counter)
            {
                g_light_cells.Store(base + 8, fc.frame_counter);
            }
            return int(slot);
        }
    }
    g_light_cells.InterlockedAdd(LightCellCounterBase(fc) + kLightCellCounterFailed, 1u);
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
// docs/LIGHT_NEE.md section 5.4): per axis a, t_a = dot(axis_a, x_r - origin) in voxel units (the
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

// The description of a list-light point for the MIS proxy (docs/LIGHT_NEE.md section 6.1), the same
// for every point of one face of one voxel: the brick's
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

// The MIS description of a point on face `face` of the entry's brick (frame `b`), `n_face` = the set
// bits of that face's mask. SampleLightEntry (NEE side) and LightMisDescOf (BSDF side) both build it
// here from the same frame function.
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
// BSDF side (TracePath). `voxel` does not enter the description: every point of one face set has the
// same in-brick density.
LightMisDesc LightMisDescOf(CaelixFrameConstants fc, LightEntryHeader header, uint64_t address, uint3 voxel,
                            uint face)
{
    const uint f = min(face, 5u);
    return LightMisDescWith(header, LightBrickFrameOf(fc, header), f, LightFaceCount(address, f));
}

// The sum of the face weights of the candidate faces of `d`'s brick seen from the receiver `x_r`
// (LightCandidateFaceMask, exactly as SampleLightEntry computes it).
float LightCandidateWeight(LightMisDesc d, float3 x_r)
{
    return LightCandidateWeightSum(d.face_weight, LightCandidateFaceMask(d.origin, d.axes[0], d.axes[1], d.axes[2],
                                                                         d.face_extent[0], d.face_extent[1], x_r));
}

// The per-vertex half of the light NEE's density proxy (docs/LIGHT_NEE.md section 6.1): the
// UNJITTERED light cell of the vertex, when it has a normalisation from an earlier fill, else the
// global CDF. A deterministic function of the vertex, so that the NEE (LightNextEvent) and the BSDF
// segment that leaves the vertex (TracePath) evaluate the same density.
struct LightNeeProxyContext
{
    bool cell;           // the cell proxy (a cell with C_prev > 0); false: the global proxy
    float3 cell_centre;  // world centre of the cell (as the fill computes it)
    float3 cell_normal;  // kGiFlagLightCellOriented: the world normal of the cell's receiver face (the
                         // fill's receiver orientation term); zero = an unoriented cell or no cell
    float near_d2;       // (cell half size + 4)^2: the fill's distance floor
    float C;             // the cell's C_prev
    float total;         // the light table's total weight; 0 without a table or without entries
};

// The context of cell `slot` (-1 = none) of (entity, level, coord, face); `total` = the table's total.
// The cell normal comes from the lookup's `face` (as the centre from its coordinate), so that the
// context is a function of the vertex.
LightNeeProxyContext LightNeeProxyOfSlot(CaelixFrameConstants fc, int slot, uint entity, uint level, int3 coord,
                                         uint face, float total)
{
    LightNeeProxyContext ctx;
    ctx.cell = false;
    ctx.cell_centre = float3(0.0f, 0.0f, 0.0f);
    ctx.cell_normal = float3(0.0f, 0.0f, 0.0f);
    ctx.near_d2 = 0.0f;
    ctx.C = 0.0f;
    ctx.total = total;
    if (slot < 0)
    {
        return ctx;
    }
    const float c_prev = asfloat(g_light_cells.Load(uint(slot) * LightCellBytes(fc) + kLightCellCPrevOffset));
    if (!(c_prev > 0.0f))
    {
        return ctx;
    }
    const float cell_size = LightCellSize(level);
    float4 r0;
    float4 r1;
    float4 r2;
    EntityCurRows(fc, entity, r0, r1, r2);
    ctx.cell = true;
    ctx.cell_centre = RowsTransform(r0, r1, r2, (float3(coord) + 0.5f) * cell_size);
    ctx.cell_normal = LightCellFaceNormal(face);
    ctx.near_d2 = (0.5f * cell_size + 4.0f) * (0.5f * cell_size + 4.0f);
    ctx.C = c_prev;
    return ctx;
}

// The proxy area density of the light NEE at the point of `d` seen from the receiver `x_r`
// (docs/LIGHT_NEE.md section 6.1). The in-brick density is the
// sampler's: face_weight[face] / W_cand / n_face, W_cand = LightCandidateWeight(d, x_r) (0 when no
// candidate face has weight: the NEE cannot sample the brick from there). With a cell: p_hat_c(b) /
// C_c times the in-brick density, p_hat_c(b) = total_b * G, G and orient as in the fill kernel
// (caelix_light_cells.hlsl), times the receiver orientation term of an oriented cell
// (LightCellReceiverOrient of ctx.cell_normal toward the brick centre, docs/LIGHT_NEE.md section
// 4.2). Without one: the global CDF, total_b / total times the in-brick density.
float LightNeeProxyPdfArea(LightNeeProxyContext ctx, LightMisDesc d, float3 x_r)
{
    const float w_cand = LightCandidateWeight(d, x_r);
    const float in_brick = w_cand > 0.0f ? d.face_weight[d.face] * d.inv_n_face / w_cand : 0.0f;
    // The entry's weight times the in-brick density.
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
        // The fill's receiver orientation term (1 for an unoriented cell): toward the brick centre.
        const float orient_r = LightCellReceiverOrient(ctx.cell_normal, ctx.cell_centre, ctx.near_d2, d.brick_centre);
        const float G = orient * orient_r / max(d2, ctx.near_d2);
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
// (a + 1) % 3, e2 = entity axis (a + 2) % 3 of the face axis a = face >> 1 (SampleLightEntry).
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

// A point on a list-light voxel of the entry at `address` (header `header`) as seen from the receiver
// `x_r` (docs/LIGHT_NEE.md section 5.4): a face direction among the brick's candidate faces
// (LightCandidateFaceMask) with probability face_weight[f] / W_cand, a voxel uniformly among the set
// bits of that face's mask (the list-light voxels whose face f is open inside the brick), and a point
// uniformly on that face (area 1). A face covered inside the brick is never chosen; a face on the
// brick border covered by a neighbour brick can be when the receiver lies beyond its plane (the
// shadow ray finds it). Between the lowest +a and the highest -a face plane of an axis both faces
// are candidates, and a chosen face that points away from the receiver gives cos_l <= 0: a wasted
// sample, not a bias. pdf_area = face_weight[f] / W_cand / n_f; the brick's own
// probability is not in it. Four random numbers in every branch (face, voxel, u, v).
LightSample SampleLightEntry(CaelixFrameConstants fc, LightEntryHeader header, uint64_t address, float3 x_r,
                             inout uint rng)
{
    LightSample s;
    s.valid = false;
    s.position = float3(0.0f, 0.0f, 0.0f);
    s.normal = float3(0.0f, 0.0f, 0.0f);
    s.radiance = float3(0.0f, 0.0f, 0.0f);
    s.pdf_area = 0.0f;
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
    s.position = LightFacePoint(frame, face_normal, e1, e2, float2(u_s - 0.5f, u_t - 0.5f));
    s.normal = face_normal;
    const uint value = BrickVoxelValue(fc, header.group, header.id, voxel);
    s.radiance = GET_MATERIAL(value).emission * g_emissionScale;
    s.pdf_area = (header.face_weight[face] / w_cand) * (1.0f / float(n));
    s.mis = LightMisDescWith(header, brick, face, n);
    s.valid = s.pdf_area > 0.0f;
    return s;
}

// A brick of the vertex's light sample: from reservoir `u_reservoir` of the sampling cell `slot` when
// it holds a valid one (W_brick = the reservoir's W), else from the global CDF at `u_global` (the
// brick's probability is total_b / total, so W_brick = total / total_b). `entry` and `address` name
// the brick's table entry; W_brick 0 = no usable brick.
void LightPickBrick(CaelixFrameConstants fc, int slot, uint count, float total, uint capacity, float u_reservoir,
                    float u_global, out LightEntryHeader entry, out uint64_t address, out float W_brick)
{
    const uint64_t table = fc.light_table;
    uint light = kNoLight;
    W_brick = 0.0f;
    if (slot >= 0)
    {
        const uint r = min(fc.light_cell_reservoirs - 1u, uint(u_reservoir * float(fc.light_cell_reservoirs)));
        const uint2 reservoir = g_light_cells.Load2(uint(slot) * LightCellBytes(fc) + kLightCellReservoirOffset +
                                                    r * kLightCellReservoirBytes);
        // kNoLight is never below count (the table holds at most 2^20 entries).
        if (asfloat(reservoir.y) > 0.0f && reservoir.x < count)
        {
            light = reservoir.x;
            W_brick = asfloat(reservoir.y);
        }
    }
    if (light == kNoLight)
    {
        light = LightCdfSearch(table, capacity, count, u_global * total);
        address = LightEntryAddress(table, light);
        entry = LoadLightEntryHeader(address);
        W_brick = entry.total_weight > 0.0f ? total / entry.total_weight : 0.0f;
        return;
    }
    address = LightEntryAddress(table, light);
    entry = LoadLightEntryHeader(address);
}

// One next-event candidate before its shadow ray: the demodulated incident radiance it would add
// when unoccluded (W and the MIS weight included), the ray toward it and its length.
struct NeeCandidate
{
    bool valid;
    float3 direction;
    float t_max;          // shadow ray TMax (kRayTMax for the sun)
    float3 contribution;  // unoccluded demodulated radiance of the Lambert lobe
};

NeeCandidate NeeCandidateNone()
{
    NeeCandidate c;
    c.valid = false;
    c.direction = float3(0.0f, 0.0f, 0.0f);
    c.t_max = 0.0f;
    c.contribution = float3(0.0f, 0.0f, 0.0f);
    return c;
}

// Whether the candidate `c` is visible from `origin`: a shadow ray of length c.t_max (none with
// kGiFlagDebugNeeNoShadow, timing only: every candidate counts as visible). A candidate with
// t_max <= 0 is never visible. Shadow rays treat glass as opaque (TraceShadow).
bool NeeVisible(CaelixFrameConstants fc, float3 origin, NeeCandidate c)
{
    if (c.t_max <= 0.0f)
    {
        return false;
    }
    if ((fc.gi_flags & kGiFlagDebugNeeNoShadow) != 0u)
    {
        return true;
    }
    RayDesc shadow;
    shadow.Origin = origin;
    shadow.Direction = c.direction;
    shadow.TMin = 0.0f;
    shadow.TMax = c.t_max;
    return !TraceShadow(fc, shadow);
}

// The resampled light NEE of LightNextEventSample (docs/LIGHT_NEE.md section 5.3, K =
// fc.light_nee_candidates >= 2): K candidates, each drawn exactly as a one-sample NEE draws its sample
// (a brick from a reservoir of the sampling cell `slot`, else from the global CDF; then a point in the
// brick). Candidate i is the point y_i with the unbiased contribution weight W_i = W_brick / pdf_area.
// The target at the vertex is the unoccluded contribution p_hat(y) = luminance(L_e) cos_r cos_l / d^2,
// times w_nee when `mis`; it is 0 inside the near field (distance < g_lightNeeMinDistance), where the
// NEE's integrand is 0 too (the near field is the BSDF path's), and 0 where cos_r or cos_l <= 0. One
// candidate y is kept with probability w_i / w_sum, w_i = p_hat(y_i) W_i, and W_out = w_sum / (K
// p_hat(y)); the caller traces one shadow ray to y. RIS with the equal input weights 1/K and unbiased
// input contribution weights: (y, W_out) is an unbiased contribution weight for any integrand whose
// support lies in the target's support. The candidates read the same R reservoirs of one cell and are
// therefore correlated; that is harmless, since the estimator is linear in the inputs' contribution
// weights. Seven random numbers per candidate (reservoir, global, the point's four, select), the
// jitter's three before in the caller. Returns the kept candidate (its unoccluded contribution).
NeeCandidate LightNextEventResampled(CaelixFrameConstants fc, float3 x_r, float3 n_r, int slot, uint count,
                                     float total, uint capacity, bool mis, LightNeeProxyContext ctx, inout uint rng)
{
    const uint K = min(fc.light_nee_candidates, 32u);
    float w_sum = 0.0f;
    // The kept candidate (scalars, not the whole LightSample, to limit register pressure).
    float3 kept_radiance = float3(0.0f, 0.0f, 0.0f);
    float3 kept_w = float3(0.0f, 0.0f, 0.0f);
    float kept_distance = 0.0f;
    float kept_d2 = 0.0f;
    float kept_cos_r = 0.0f;
    float kept_cos_l = 0.0f;
    float kept_p_hat = 0.0f;
    float kept_w_nee = 1.0f;
    CAELIX_LOOP
    for (uint i = 0u; i < K; ++i)
    {
        const float u_reservoir = RandomFloat01(rng);
        const float u_global = RandomFloat01(rng);
        LightEntryHeader entry;
        uint64_t address;
        float W_brick;
        LightPickBrick(fc, slot, count, total, capacity, u_reservoir, u_global, entry, address, W_brick);
        const LightSample s = SampleLightEntry(fc, entry, address, x_r, rng);
        const float u_select = RandomFloat01(rng);
        if (!s.valid || !(W_brick > 0.0f))
        {
            continue;
        }
        const float3 d = s.position - x_r;
        const float d2 = dot(d, d);
        const float distance = sqrt(d2);
        if (!(d2 > 0.0f) || distance < g_lightNeeMinDistance)
        {
            continue;  // target 0: the near field is the BSDF path's
        }
        const float3 w = d / distance;
        const float cos_r = dot(n_r, w);
        const float cos_l = -dot(s.normal, w);
        if (!(cos_r > 0.0f) || !(cos_l > 0.0f))
        {
            continue;
        }
        float p_hat = CaelixLuminance(s.radiance) * cos_r * cos_l / d2;
        float w_nee = 1.0f;
        if (mis)
        {
            w_nee = LightNeeMisWeight(ctx, s.mis, x_r, n_r, w, distance, cos_l, true);
            p_hat *= w_nee;
        }
        const float w_i = p_hat * W_brick / s.pdf_area;
        if (!(w_i > 0.0f))
        {
            continue;
        }
        w_sum += w_i;
        if (u_select * w_sum < w_i)
        {
            kept_radiance = s.radiance;
            kept_w = w;
            kept_distance = distance;
            kept_d2 = d2;
            kept_cos_r = cos_r;
            kept_cos_l = cos_l;
            kept_p_hat = p_hat;
            kept_w_nee = w_nee;
        }
    }
    NeeCandidate c = NeeCandidateNone();
    if (!(w_sum > 0.0f) || !(kept_p_hat > 0.0f))
    {
        return c;
    }
    const float W_out = w_sum / (float(K) * kept_p_hat);
    // The point lies on a voxel face: the ray stops 2 push-offs short of it (as GiVisible).
    c.direction = kept_w;
    c.t_max = kept_distance - 2.0f * K_RAY_ORIGIN_PUSH_OFF;
    c.contribution = kept_radiance * ((kept_cos_r * (1.0f / K_PI)) * (kept_cos_l / kept_d2) * W_out);
    if (mis)
    {
        c.contribution *= kept_w_nee;
    }
    c.valid = c.t_max > 0.0f;
    return c;
}

// The light NEE candidate at a diffuse path vertex (docs/LIGHT_NEE.md section 5), before its shadow
// ray: the receiver x_r (the pushed-off vertex position) with world normal n_r, on brick group
// `group`, entity-local position `object_point` (the hit point moved a little into the air, for the
// cell coordinate). The light sample is a brick from one of the R reservoirs of the vertex's light
// cell when it holds a valid one, else from the global CDF (LightPickBrick), then a point in the brick
// by SampleLightEntry (W = W_brick / pdf_area). The candidate's contribution is the demodulated
// incident radiance of the Lambert lobe when unoccluded (the albedo is in the caller's throughput, as
// for the sun NEE); invalid without a usable sample.
// With kGiFlagLightCellJitter (section 5.1) the cell is looked up at `object_point` moved by a
// uniform random offset in (-0.5, 0.5) cells per axis: the pixels of one brick then read up to 8
// cells instead of the same one, so the variance of the cell contents (one refill per frame) turns
// from a per-cell flicker into per-pixel noise. Unbiased: every cell's reservoirs are unbiased
// estimators for any receiver (W does not depend on the receiver).
// `mis` (kGiFlagLightMis, section 6): the contribution is multiplied by the balance heuristic weight
// LightNeeMisWeight(nee_side = true), and `ctx` receives the vertex's proxy context (the unjittered
// cell; with the jitter on, the sampling cell is a second lookup) for the BSDF segment that leaves
// the vertex. Without `mis`, `ctx` is the global context. The MIS draws no random number.
// Per-vertex resampling (fc.light_nee_candidates >= 2, section 5.3): after the same lookups (the
// unjittered one for the proxy context, the jittered one for the sampling cell, the three jitter
// numbers once) LightNextEventResampled draws K candidates from the sampling cell (or the global CDF)
// and keeps one by the exact target at the vertex (times w_nee with `mis`); the first two random
// numbers of this function are then unused. With K = 1 the path below runs.
// `sample_slot`: the cell slot the sample came from (-1 = none: no cells, or a failed lookup);
// `hint_slot`: the UNJITTERED cell's slot when that lookup ran (`mis` or no jitter), else sample_slot
// (section 10: the sun visibility counters are written there).
NeeCandidate LightNextEventSample(CaelixFrameConstants fc, float3 x_r, float3 n_r, uint group, float3 object_point,
                                  bool mis, inout uint rng, out LightNeeProxyContext ctx, out int hint_slot,
                                  out int sample_slot)
{
    const float u_reservoir = RandomFloat01(rng);
    const float u_global = RandomFloat01(rng);
    const uint64_t table = fc.light_table;
    const uint count = CAELIX_LOAD_U32(table + kLightHeaderCount);
    const float total = asfloat(CAELIX_LOAD_U32(table + kLightHeaderTotal));
    ctx = LightNeeProxyOfSlot(fc, -1, 0u, 0u, int3(0, 0, 0), kLightCellFaceNone,
                              (count != 0u && total > 0.0f) ? total : 0.0f);
    hint_slot = -1;
    sample_slot = -1;
    NeeCandidate c = NeeCandidateNone();
    if (count == 0u || !(total > 0.0f))
    {
        return c;
    }
    const uint capacity = CAELIX_LOAD_U32(table + kLightHeaderCapacity);

    int slot = -1;
    CAELIX_BRANCH
    if (fc.light_cells != 0u)
    {
        const uint level = LightCellLevel(fc, x_r);
        const uint entity = GroupEntity(fc, group);
        // kGiFlagLightCellOriented: both lookups use the receiver face of n_r (the jitter moves only
        // the point).
        const uint face = LightCellFaceOf(fc, n_r);
        const bool jitter = (fc.gi_flags & kGiFlagLightCellJitter) != 0u;
        // The proxy's cell is the unjittered one; without the jitter it is also the sampling cell.
        if (mis || !jitter)
        {
            const int3 coord = LightCellCoord(object_point, level);
            slot = LightCellLookup(fc, entity, level, coord, face);
            hint_slot = slot;
            if (mis)
            {
                ctx = LightNeeProxyOfSlot(fc, slot, entity, level, coord, face, total);
            }
        }
        if (jitter)
        {
            const float3 u_jitter = float3(RandomFloat01(rng), RandomFloat01(rng), RandomFloat01(rng));
            const float3 cell_point = object_point + (u_jitter - 0.5f) * LightCellSize(level);
            slot = LightCellLookup(fc, entity, level, LightCellCoord(cell_point, level), face);
            if (!mis)
            {
                hint_slot = slot;
            }
        }
    }
    sample_slot = slot;

    // Per-vertex resampling (section 5.3): K >= 2 candidates of the same sampling cell, one ray.
    CAELIX_BRANCH
    if (fc.light_nee_candidates >= 2u)
    {
        return LightNextEventResampled(fc, x_r, n_r, slot, count, total, capacity, mis, ctx, rng);
    }

    LightEntryHeader entry;
    uint64_t address;
    float W_brick;
    LightPickBrick(fc, slot, count, total, capacity, u_reservoir, u_global, entry, address, W_brick);
    const LightSample s = SampleLightEntry(fc, entry, address, x_r, rng);
    if (!s.valid || !(W_brick > 0.0f))
    {
        return c;
    }
    const float3 d = s.position - x_r;
    const float d2 = dot(d, d);
    const float distance = sqrt(d2);
    if (!(d2 > 0.0f) || distance < g_lightNeeMinDistance)
    {
        return c;  // the near field is the BSDF path's
    }
    const float3 w = d / distance;
    const float cos_r = dot(n_r, w);
    const float cos_l = -dot(s.normal, w);
    if (!(cos_r > 0.0f) || !(cos_l > 0.0f))
    {
        return c;
    }
    // The point lies on a voxel face: the ray stops 2 push-offs short of it (as GiVisible).
    c.direction = w;
    c.t_max = distance - 2.0f * K_RAY_ORIGIN_PUSH_OFF;
    c.contribution = s.radiance * ((cos_r * (1.0f / K_PI)) * (cos_l / d2) / s.pdf_area * W_brick);
    if (mis)
    {
        c.contribution *= LightNeeMisWeight(ctx, s.mis, x_r, n_r, w, distance, cos_l, true);
    }
    c.valid = c.t_max > 0.0f;
    return c;
}

// Next-event estimation on the light list at a diffuse path vertex: LightNextEventSample and its
// shadow ray (NeeVisible). Returns the demodulated incident radiance of the Lambert lobe, 0 without a
// usable sample or when the point is occluded.
float3 LightNextEvent(CaelixFrameConstants fc, float3 x_r, float3 n_r, uint group, float3 object_point, bool mis,
                      inout uint rng, out LightNeeProxyContext ctx)
{
    int hint_slot;
    int sample_slot;
    const NeeCandidate c =
        LightNextEventSample(fc, x_r, n_r, group, object_point, mis, rng, ctx, hint_slot, sample_slot);
    return c.valid && NeeVisible(fc, x_r, c) ? c.contribution : float3(0.0f, 0.0f, 0.0f);
}

// Whether the light NEE can produce the point that the ray of `hit` found on a list light
// (docs/LIGHT_NEE.md sections 2.7 and 6.3): the hit brick has an entry in
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
