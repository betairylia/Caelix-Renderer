// Light list sampling of the render kernels (docs/LIGHT_LIST.md, section 2): the light cells
// (world-space ReGIR reservoirs over the scene's light table), the global CDF of the table, the
// point sampling inside a brick entry, the next-event estimation at a diffuse path vertex and its
// MIS with the BSDF path (docs/LIGHT_MIS.md: the density proxy, LightNeeMisWeight, LightListCanSample).
//
// Light cells: a hash table of fc.light_cell_capacity cells in g_light_cells (trace set binding
// 21), 64 + 24 R bytes each (R = fc.light_cell_reservoirs reservoirs and push entries, 1 .. 64,
// default 8; 256 B at R 8; LightCellBytes). Before 2026-10-22 R was the constant 8; 288 bytes before
// step 3 of the light list, 296 before the temporal cell reuse, 304 before the push-back of shadow ray
// results, 832 with 32 reservoirs and 32 push entries before the oriented cells, docs/LIGHT_LIST.md
// sections 2.7 and 2.8:
//   0    uint key_lo, key_hi    the 64-bit fingerprint of (entity, level, coordinate, face); key_lo 0 = empty
//   8    uint touched_frame     fc.frame_counter of the newest lookup
//   12   uint entity
//   16   int3 coord             the cell coordinate at `level`: floor(p) >> (5 + 2 level), p entity-local
//   28   uint level             0: 32-voxel cells, 1: 128-voxel cells (8 / 32 before 2026-10-22)
//   32   float C_prev           the cell normalisation C_c of the fill before the latest one (the MIS
//                               proxy reads this one; 0 = none yet: the global proxy)
//   36   float C_cur            C_c of the latest fill: the mean of w_sum / K over the R reservoirs
//                               (kGiFlagLightCellTemporal: the running mean of dedicated draws)
//   40   float M                kGiFlagLightCellTemporal: the candidate count behind every reservoir of
//                               the cell (uniform over the R: all threads merge the same way); 0 = no history
//   44   float C_n              kGiFlagLightCellTemporal: the draw count behind C_cur; 0 = none
//   48   uint push_count        kGiFlagLightCellPush: pushes claimed since the latest fill (may exceed R)
//   52   uint face              the receiver face of the key (kGiFlagLightCellOriented: 0..5 = +X -X +Y
//                               -Y +Z -Z of the receivers' world normal, GiFaceIndex), else
//                               kLightCellFaceNone (6); written by the insert, read by the fill
//   56   8 bytes of padding
//   64   R x { uint light; float W }   the reservoirs: light = a light sample word (below), W = its
//                               unbiased contribution weight (0 = empty)
//   64 + 8 R   R x { uint word; float W; float visible; uint pad }   kGiFlagLightCellPush: the push ring,
//                               the light samples that the trace's vertices drew from this cell
//                               (LightCellClaimPush, LightCellStorePush); the fill merges entry r
//                               into reservoir r
// A light sample word (LightCellWord, LightCellDecode): bits 0..19 the entry index in the light table
// (the table holds at most 2^20 entries), bits 20..28 the voxel v = x | y << 3 | z << 6 inside the
// entry's brick, bits 29..31 the face: 0..5 a voxel-face sample (a point uniform on that face of that
// voxel, area 1), 6 = kLightCellFaceBrick a brick-level sample (the vertex samples a point inside the
// brick; voxel bits 0), 7 only in kNoLight (empty). The fill's fresh candidates are brick-level;
// voxel-face samples enter the cells only through the push ring.
// After the last cell (byte fc.light_cell_capacity x LightCellBytes, LightCellCounterBase) the buffer holds 32
// bytes of usage counters, zeroed by the renderer before every fill and copied to the host after the
// trace (CaelixGiStats):
//   0    uint live              cells with a key after this frame's eviction (fill, thread 0)
//   4    uint active            of those, looked up by the previous frame's trace (fill, thread 0)
//   8    uint inserted          cells inserted by this frame's lookups (LightCellLookup)
//   12   uint failed            lookups that found neither the key nor a free slot in 8 probes
//   16   uint pushed            pushes of the previous frame's trace that got a ring slot (fill, thread 0)
//   20   uint push_dropped      pushes of the previous frame's trace beyond the R slots (fill, thread 0)
//   24   8 bytes of padding
// The trace kernel looks cells up and inserts missing ones (LightCellLookup); caelix_light_cells.hlsl
// fills the reservoirs of every live cell before the next trace and evicts cells untouched for 64
// frames. The fingerprint is claimed with a 32-bit compare-and-swap on key_lo (64-bit buffer atomics
// are not a device requirement); key_hi is stored after the claim. A lookup that sees key_lo match
// and key_hi still 0 (not yet written) takes that cell: before 2026-10-21 it probed on and inserted
// the same key again, up to 8 times (the shroom forest had twice as many live cells as used ones and
// a full table). A trace never reads the C_c computed from the reservoirs it samples
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
#include "render/caelix_restir_gi.hlsl"

// The cell edge at level 0 is 1 << kLightCellBaseShift voxels (32); each level is 4x larger.
static const uint kLightCellBaseShift = 5u;
static const uint kLightCellHeaderBytes = 64;
static const uint kLightCellReservoirBytes = 8;
static const uint kLightCellCPrevOffset = 32;
static const uint kLightCellCCurOffset = 36;
static const uint kLightCellMOffset = 40;
static const uint kLightCellCNOffset = 44;
static const uint kLightCellPushCountOffset = 48;
static const uint kLightCellReservoirOffset = 64;
static const uint kLightCellFaceOffset = 52;
static const uint kLightCellPushBytes = 16;
// The header face of an unoriented cell (kGiFlagLightCellOriented off): no receiver face in the key.
static const uint kLightCellFaceNone = 6u;
// The floor of the receiver orientation term of an oriented cell (LightCellReceiverOrient): never 0,
// because a receiver may read a cell of another orientation or position (the jittered lookup, a
// merged slot) and every reservoir must stay an unbiased contribution weight for it; the floor costs
// noise there (W up to 10x), not bias.
static const float kLightCellOrientFloor = 0.1f;
// The light sample word of a cell reservoir or push entry (see above).
static const uint kLightCellEntryMask = 0xFFFFFu;
static const uint kLightCellVoxelShift = 20u;
static const uint kLightCellFaceShift = 29u;
static const uint kLightCellFaceBrick = 6u;
// The fill's target factor of a voxel-face sample that a push of this frame saw occluded
// (caelix_light_cells.hlsl): a floor, not 0, so that the support stays full. A sample occluded
// from one receiver but visible from another gets a W up to 10x larger there: noise, not bias.
static const float kLightOccludedTarget = 0.1f;
static const uint kLightCellProbes = 8;
// A cell not looked up for this many frames is evicted by the fill pass.
static const uint kLightCellMaxAge = 64;
static const uint kNoLight = 0xFFFFFFFFu;
// The usage counters after the last cell (byte offsets from LightCellCounterBase).
static const uint kLightCellCounterLive = 0;
static const uint kLightCellCounterActive = 4;
static const uint kLightCellCounterInserted = 8;
static const uint kLightCellCounterFailed = 12;
static const uint kLightCellCounterPushed = 16;
static const uint kLightCellCounterPushDropped = 20;
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

// The reservoirs per cell R (fc.light_cell_reservoirs, a power of two in 1 .. 64); the push ring has
// R entries as well. A cell is 64 + 24 R bytes: the header, R x 8 B reservoirs, R x 16 B push entries.
uint LightCellBytes(CaelixFrameConstants fc)
{
    return kLightCellHeaderBytes + (kLightCellReservoirBytes + kLightCellPushBytes) * fc.light_cell_reservoirs;
}

// The byte offset of the push ring inside a cell (after the reservoirs).
uint LightCellPushOffset(CaelixFrameConstants fc)
{
    return kLightCellReservoirOffset + kLightCellReservoirBytes * fc.light_cell_reservoirs;
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

// The receiver face of the cell key of a vertex with world normal `n_r` (docs/LIGHT_LIST.md section
// 2.7): with kGiFlagLightCellOriented the face index of the normal (GiFaceIndex, 0..5, the index that
// ReSTIR DI stores for its receiver), else kLightCellFaceNone.
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

// The receiver orientation term of an oriented cell (docs/LIGHT_LIST.md section 2.7) for a light at
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
                // C_cur, M, C_n and push_count start at 0 (the MIS proxy is the global one until the
                // second fill; no temporal history; a push of this trace that claimed a slot before
                // this store is lost, which does not depend on the sample it carried).
                g_light_cells.Store(base + 4, key.y);
                g_light_cells.Store(base + 8, fc.frame_counter);
                g_light_cells.Store(base + 12, entity);
                g_light_cells.Store3(base + 16, asuint(coord));
                g_light_cells.Store(base + 28, level);
                g_light_cells.Store4(base + kLightCellCPrevOffset, uint4(0u, 0u, 0u, 0u));
                g_light_cells.Store2(base + kLightCellPushCountOffset, uint2(0u, face));
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

// The light sample word of (entry, voxel v = x | y << 3 | z << 6, face 0..5 or kLightCellFaceBrick).
uint LightCellWord(uint entry, uint voxel, uint face)
{
    return (entry & kLightCellEntryMask) | ((voxel & 511u) << kLightCellVoxelShift) | (face << kLightCellFaceShift);
}

// Decodes a light sample word of a table with `count` entries: false for kNoLight (face 7) and for an
// entry index beyond the table.
bool LightCellDecode(uint word, uint count, out uint entry, out uint voxel, out uint face)
{
    entry = word & kLightCellEntryMask;
    voxel = (word >> kLightCellVoxelShift) & 511u;
    face = word >> kLightCellFaceShift;
    return word != kNoLight && face <= kLightCellFaceBrick && entry < count;
}

// The voxel (0..7 per axis) of v = x | y << 3 | z << 6.
uint3 LightVoxel3(uint v)
{
    return uint3(v & 7u, (v >> 3) & 7u, (v >> 6) & 7u);
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

// The union of LightCandidateFaceMask over every receiver in a box (centre `centre`, world half axes
// half0..half2): the same plane tests with t_a replaced by its maximum (+a) or minimum (-a) over the
// box, t_a(centre) +- sum_k |dot(axis_a, half_k)| (exact for a box under any rotation). The push-back
// of a vertex's light sample (LightNextEvent) compares the vertex's own mask with this one over the
// cell's receiver region.
uint LightCandidateFaceMaskOverRegion(float3 origin, float3 axis0, float3 axis1, float3 axis2, uint extent0,
                                      uint extent1, float3 centre, float3 half0, float3 half1, float3 half2)
{
    const float3 d = centre - origin;
    float3 axes[3];
    axes[0] = axis0;
    axes[1] = axis1;
    axes[2] = axis2;
    uint mask = 0u;
    CAELIX_UNROLL
    for (uint a = 0; a < 3; ++a)
    {
        const float t = dot(axes[a], d);
        const float reach = abs(dot(axes[a], half0)) + abs(dot(axes[a], half1)) + abs(dot(axes[a], half2));
        const uint lowest = LightFaceExtent(extent0, extent1, 2u * a);        // lowest +a face row
        const uint highest = LightFaceExtent(extent0, extent1, 2u * a + 1u);  // highest -a face row
        mask |= (lowest != 0xFFu && t + reach > float(lowest) + 1.0f) ? 1u << (2u * a) : 0u;
        mask |= (highest != 0xFFu && t - reach < float(highest)) ? 2u << (2u * a) : 0u;
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
    uint face_mask;   // SampleLightEntryWith: the candidate faces it chose from; LightSampleFaceOf: 0
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
    bool vertex;         // cell and kGiFlagLightProxyVertex: the vertex form of the cell proxy (a frame
                         // constant: every place that rebuilds a context sets it from fc.gi_flags)
    float3 cell_centre;  // world centre of the cell (as the fill computes it)
    float3 cell_normal;  // kGiFlagLightCellOriented: the world normal of the cell's receiver face (the
                         // fill's receiver orientation term); zero = an unoriented cell or no cell
    float near_d2;       // (cell half size + 4)^2: the fill's distance floor
    float C;             // the cell's C_prev
    float total;         // the light table's total weight; 0 without a table or without entries
};

// The context of cell `slot` (-1 = none) of (entity, level, coord, face); `total` = the table's total.
// The cell normal comes from the lookup's `face` (as the centre from its coordinate), so that the
// context is a function of the vertex; ReSTIR DI rebuilds the same normal from its stored face index.
LightNeeProxyContext LightNeeProxyOfSlot(CaelixFrameConstants fc, int slot, uint entity, uint level, int3 coord,
                                         uint face, float total)
{
    LightNeeProxyContext ctx;
    ctx.cell = false;
    ctx.vertex = false;
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
    ctx.vertex = (fc.gi_flags & kGiFlagLightProxyVertex) != 0u;
    ctx.cell_centre = RowsTransform(r0, r1, r2, (float3(coord) + 0.5f) * cell_size);
    ctx.cell_normal = LightCellFaceNormal(face);
    ctx.near_d2 = (0.5f * cell_size + 4.0f) * (0.5f * cell_size + 4.0f);
    ctx.C = c_prev;
    return ctx;
}

// The proxy context of the vertex x_r (pushed-off position) with world normal `n_r` on brick group
// `group`, entity-local point `object_point` (LightCellPoint): the unjittered cell lookup (it inserts
// the cell when it is missing, as every lookup does; the receiver face of n_r with
// kGiFlagLightCellOriented). LightNextEvent builds the same context from the same lookup.
LightNeeProxyContext LightNeeProxyAt(CaelixFrameConstants fc, float3 x_r, float3 n_r, uint group, float3 object_point)
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
    const uint face = LightCellFaceOf(fc, n_r);
    if (fc.light_cells != 0u && total > 0.0f)
    {
        level = LightCellLevel(fc, x_r);
        entity = GroupEntity(fc, group);
        coord = LightCellCoord(object_point, level);
        slot = LightCellLookup(fc, entity, level, coord, face);
    }
    return LightNeeProxyOfSlot(fc, slot, entity, level, coord, face, total);
}

// The proxy area density of the light NEE at the point of `d` seen from the receiver `x_r`
// (docs/LIGHT_MIS.md section 1, docs/LIGHT_FACES.md section 4). The in-brick density is the
// sampler's: face_weight[face] / W_cand / n_face, W_cand = LightCandidateWeight(d, x_r) (0 when no
// candidate face has weight: the NEE cannot sample the brick from there). With a cell: p_hat_c(b) /
// C_c times the in-brick density, p_hat_c(b) = total_b * G, G and orient as in the fill kernel
// (caelix_light_cells.hlsl), times the receiver orientation term of an oriented cell
// (LightCellReceiverOrient of ctx.cell_normal toward the brick centre, docs/LIGHT_LIST.md section
// 2.7). Without one: the global CDF, total_b / total times the in-brick density.
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
// With ctx.vertex (kGiFlagLightProxyVertex, docs/LIGHT_MIS.md "Vertex proxy") the area density is
// the vertex form instead of LightNeeProxyPdfArea's cell form.
float LightNeeMisWeight(LightNeeProxyContext ctx, LightMisDesc d, float3 x_r, float3 n_r, float3 w, float d_light,
                        float cos_l, bool nee_side)
{
    float p_area;
    if (ctx.vertex)
    {
        // Vertex form (kGiFlagLightProxyVertex): the voxel-face weight times the receiver's own geometry
        // term over the cell's C: L_f * cos_r * cos_l / max(d^2, 1) / C, 0 when no candidate face has weight.
        const float w_cand = LightCandidateWeight(d, x_r);
        const float L_f = d.face_weight[d.face] * d.inv_n_face;
        const float cos_r = max(0.0f, dot(n_r, w));
        p_area = (w_cand > 0.0f && ctx.C > 0.0f)
                     ? L_f * cos_r * cos_l / max(d_light * d_light, 1.0f) / ctx.C
                     : 0.0f;
    }
    else
    {
        p_area = LightNeeProxyPdfArea(ctx, d, x_r);
    }
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
    s.face_mask = 0u;
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
    s.face_mask = candidates;
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

// The point of a voxel-face sample (a cell reservoir word with face 0..5): voxel v = x | y << 3 |
// z << 6 of the entry's brick, face `face`, a point uniform on that face (pdf_area 1: the word's W is
// already the contribution weight per unit area). The same frame, basis, point and uv quantization as
// SampleLightEntryWith; the MIS description is the brick-level one of that face. Valid iff the
// voxel's bit is set in the face's mask. Four random numbers, as SampleLightEntryWith (the first two
// unused), so that the stream does not depend on the word.
LightSample LightSampleFaceOf(CaelixFrameConstants fc, LightEntryHeader header, uint64_t address, uint voxel_index,
                              uint face, bool quantize_uv, inout uint rng)
{
    RandomFloat01(rng);
    RandomFloat01(rng);
    const float u_s = RandomFloat01(rng);
    const float u_t = RandomFloat01(rng);
    const uint3 voxel = LightVoxel3(voxel_index);
    const uint f = min(face, 5u);
    const LightBrickFrame brick = LightBrickFrameOf(fc, header);
    const LightVoxelFrame frame = LightVoxelFrameIn(brick, voxel);
    float3 face_normal;
    float3 e1;
    float3 e2;
    LightFaceBasis(frame, f, face_normal, e1, e2);
    LightSample s;
    s.uv = quantize_uv ? float2(LightUvDequantize(LightUvQuantize(u_s)), LightUvDequantize(LightUvQuantize(u_t)))
                       : float2(u_s - 0.5f, u_t - 0.5f);
    s.position = LightFacePoint(frame, face_normal, e1, e2, s.uv);
    s.normal = face_normal;
    s.radiance = GET_MATERIAL(BrickVoxelValue(fc, header.group, header.id, voxel)).emission * g_emissionScale;
    s.pdf_area = 1.0f;
    s.voxel = voxel_index & 511u;
    s.face = f;
    s.mis = LightMisDescWith(header, brick, f, LightFaceCount(address, f));
    s.face_mask = 0u;
    s.valid = LightFaceMaskBit(address, f, voxel);
    return s;
}

// The candidate faces of `header`'s brick over the receiver region of the cell (entity, level,
// coord): the cell box grown by half a cell (the jittered lookup, section 2.5) and one voxel (the
// object offset of the lookup point and the push-off) on every side, i.e. world half axes of
// cell_size + 1 along the entity's axes. Every receiver that reads the cell lies in it.
uint LightCellRegionFaceMask(CaelixFrameConstants fc, uint entity, uint level, int3 coord, LightEntryHeader header)
{
    const float cell_size = LightCellSize(level);
    float4 c0;
    float4 c1;
    float4 c2;
    EntityCurRows(fc, entity, c0, c1, c2);
    const float3 centre = RowsTransform(c0, c1, c2, (float3(coord) + 0.5f) * cell_size);
    const float h = cell_size + 1.0f;
    const LightBrickFrame b = LightBrickFrameOf(fc, header);
    return LightCandidateFaceMaskOverRegion(b.origin, b.axes[0], b.axes[1], b.axes[2], header.face_extent[0],
                                            header.face_extent[1], centre, float3(c0.x, c1.x, c2.x) * h,
                                            float3(c0.y, c1.y, c2.y) * h, float3(c0.z, c1.z, c2.z) * h);
}

// Claims a push of cell `slot` (kGiFlagLightCellPush) and returns its index: the first R claims of
// a frame get a ring entry (R = fc.light_cell_reservoirs), later ones are dropped (the fill counts
// both from push_count). A vertex claims BEFORE it draws its light sample: which vertices win the R
// entries then depends on the paths' timing up to the claim, never on the sample (a claim after the
// shadow ray lets the vertices without a ray win, a sample-dependent choice: 0.36 % dark in the
// 4-light box).
uint LightCellClaimPush(CaelixFrameConstants fc, uint slot)
{
    uint index;
    g_light_cells.InterlockedAdd(slot * LightCellBytes(fc) + kLightCellPushCountOffset, 1u, index);
    return index;
}

// Writes the claimed push `index` of cell `slot` (no-op beyond the R ring entries).
void LightCellStorePush(CaelixFrameConstants fc, uint slot, uint index, uint word, float W, bool visible)
{
    if (index < fc.light_cell_reservoirs)
    {
        g_light_cells.Store4(slot * LightCellBytes(fc) + LightCellPushOffset(fc) + index * kLightCellPushBytes,
                             uint4(word, asuint(W), asuint(visible ? 1.0f : 0.0f), 0u));
    }
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

// The resampled light NEE of LightNextEvent (section 2.6, K = fc.light_nee_candidates >= 2): K
// candidates, each drawn exactly as a one-sample NEE draws its sample (a reservoir of the sampling
// cell `slot`, else the global CDF; then a point), the same loop as DiInitialReservoir. Candidate i
// is the point y_i with the unbiased contribution weight W_i = W_brick / pdf_area. The target at the
// vertex is the unoccluded contribution p_hat(y) = luminance(L_e) cos_r cos_l / d^2, times w_nee when
// `mis`; it is 0 inside the near field (distance < g_lightNeeMinDistance), where the NEE's integrand
// is 0 too (the near field is the BSDF path's), and 0 where cos_r or cos_l <= 0. One candidate y is
// kept with probability w_i / w_sum, w_i = p_hat(y_i) W_i, and W_out = w_sum / (K p_hat(y)); one shadow
// ray to y. RIS with the equal input weights 1/K and unbiased input contribution weights: (y, W_out)
// is an unbiased contribution weight for any integrand whose support lies in the target's support.
// The candidates read the same 8 reservoirs of one cell and are therefore correlated; that is
// harmless, since the estimator is linear in the inputs' contribution weights. Five random numbers per
// candidate (reservoir, global, the point's four, select), the jitter's three before in the caller.
// Push-back is excluded by the validation (CAELIX_GI_FLAG_LIGHT_CELL_PUSH needs K = 1).
float3 LightNextEventResampled(CaelixFrameConstants fc, float3 x_r, float3 n_r, int slot, uint count, float total,
                               uint capacity, bool mis, LightNeeProxyContext ctx, inout uint rng)
{
    const uint64_t table = fc.light_table;
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
        uint light = kNoLight;
        float W_brick = 0.0f;
        uint cell_voxel = 0u;
        uint cell_face = kLightCellFaceBrick;
        if (slot >= 0)
        {
            const uint cell_r =
                min(fc.light_cell_reservoirs - 1u, uint(u_reservoir * float(fc.light_cell_reservoirs)));
            const uint2 reservoir = g_light_cells.Load2(uint(slot) * LightCellBytes(fc) + kLightCellReservoirOffset +
                                                        cell_r * kLightCellReservoirBytes);
            uint entry_index;
            uint voxel;
            uint face;
            if (asfloat(reservoir.y) > 0.0f && LightCellDecode(reservoir.x, count, entry_index, voxel, face))
            {
                light = entry_index;
                W_brick = asfloat(reservoir.y);
                cell_voxel = voxel;
                cell_face = face;
            }
        }
        uint64_t address = 0u;
        LightEntryHeader entry = (LightEntryHeader)0;
        if (light != kNoLight)
        {
            address = LightEntryAddress(table, light);
            entry = LoadLightEntryHeader(address);
            // A voxel-face word whose voxel bit is clear (the scene changed): no sample from the cell.
            if (cell_face < kLightCellFaceBrick && !LightFaceMaskBit(address, cell_face, LightVoxel3(cell_voxel)))
            {
                light = kNoLight;
            }
        }
        if (light == kNoLight)
        {
            // The global CDF: the brick's probability is total_b / total.
            cell_face = kLightCellFaceBrick;
            light = LightCdfSearch(table, capacity, count, u_global * total);
            address = LightEntryAddress(table, light);
            entry = LoadLightEntryHeader(address);
            W_brick = entry.total_weight > 0.0f ? total / entry.total_weight : 0.0f;
        }
        // A voxel-face word: a point uniform on its face (pdf_area 1, W the reservoir's); else a point
        // of the brick (W_brick / pdf_area).
        LightSample s;
        if (cell_face < kLightCellFaceBrick)
        {
            s = LightSampleFaceOf(fc, entry, address, cell_voxel, cell_face, false, rng);
        }
        else
        {
            s = SampleLightEntry(fc, entry, address, x_r, rng);
        }
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
    if (!(w_sum > 0.0f) || !(kept_p_hat > 0.0f))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    const float W_out = w_sum / (float(K) * kept_p_hat);
    // The point lies on a voxel face: the ray stops 2 push-offs short of it (as GiVisible).
    RayDesc shadow;
    shadow.Origin = x_r;
    shadow.Direction = kept_w;
    shadow.TMin = 0.0f;
    shadow.TMax = kept_distance - 2.0f * K_RAY_ORIGIN_PUSH_OFF;
    if (shadow.TMax > 0.0f && TraceShadow(fc, shadow))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    float3 contribution = kept_radiance * ((kept_cos_r * (1.0f / K_PI)) * (kept_cos_l / kept_d2) * W_out);
    if (mis)
    {
        contribution *= kept_w_nee;
    }
    return contribution;
}

// Next-event estimation on the light list at a diffuse path vertex (section 2.3): the receiver
// x_r (the pushed-off vertex position) with world normal n_r, on brick group `group`, entity-local
// position `object_point` (the hit point moved a little into the air, for the cell coordinate).
// The light sample comes from one of the 8 reservoirs of the vertex's light cell when it holds a
// valid one, else from the global CDF: a brick-level sample (a brick, then a point by
// SampleLightEntry, W = W_brick / pdf_area) or a voxel-face sample (a point uniform on that face by
// LightSampleFaceOf, W = the reservoir's W, pdf_area 1; a word whose voxel bit is clear counts as no
// sample from the cell). One shadow ray. Returns the demodulated incident radiance of the Lambert lobe
// (the albedo is in the caller's throughput, as for the sun NEE), 0 without a usable sample.
// With kGiFlagLightCellJitter (section 2.5) the cell is looked up at `object_point` moved by a
// uniform random offset in (-0.5, 0.5) cells per axis: the pixels of one brick then read up to 8
// cells instead of the same one, so the variance of the cell contents (one refill of 8 reservoirs
// per frame) turns from a per-cell flicker into per-pixel noise. Unbiased: every cell's reservoirs
// are unbiased estimators for any receiver (W does not depend on the receiver).
// `mis` (kGiFlagLightMis, docs/LIGHT_MIS.md): the contribution is multiplied by the balance
// heuristic weight LightNeeMisWeight(nee_side = true), and `ctx` receives the vertex's proxy context
// (the unjittered cell, LightNeeProxyAt's lookup; with the jitter on, the sampling cell is a second
// lookup) for the BSDF segment that leaves the vertex. Without `mis`, `ctx` is the global context
// and the lookups and random numbers are the same as before step 3. The MIS draws no random number.
// Push-back (kGiFlagLightCellPush, docs/LIGHT_LIST.md section 2.2): when the sampling cell exists,
// the vertex claims a push (LightCellClaimPush) before it draws its sample and writes it once,
// whatever happened to the sample (so that neither the presence of a ring entry nor the vertex that
// fills it depends on the sample): a voxel-face reservoir sample its own word and W;
// a brick-level sample (reservoir or global CDF) its voxel-face word with W_brick / pdf_area when the
// vertex's candidate faces equal their union over the cell's receiver region
// (LightCellRegionFaceMask: then the point's density covers every face any receiver of the cell can
// see), else its brick-level word with W_brick; nothing usable (no brick weight) an empty entry.
// visible = 0 only when the shadow ray was traced and hit.
// Per-vertex resampling (fc.light_nee_candidates >= 2, section 2.6): after the same lookups (the
// unjittered one for the proxy context, the jittered one for the sampling cell, the three jitter
// numbers once) LightNextEventResampled draws K candidates from the sampling cell (or the global CDF),
// keeps one by the exact target at the vertex (times w_nee with `mis`) and traces one shadow ray; the
// first two random numbers of this function are then unused. With K = 1 the path below runs
// unchanged (the same random stream as before section 2.6). No push-back with K >= 2 (validation).
float3 LightNextEvent(CaelixFrameConstants fc, float3 x_r, float3 n_r, uint group, float3 object_point, bool mis,
                      inout uint rng, out LightNeeProxyContext ctx)
{
    const float u_reservoir = RandomFloat01(rng);
    const float u_global = RandomFloat01(rng);
    const uint64_t table = fc.light_table;
    const uint count = CAELIX_LOAD_U32(table + kLightHeaderCount);
    const float total = asfloat(CAELIX_LOAD_U32(table + kLightHeaderTotal));
    ctx = LightNeeProxyOfSlot(fc, -1, 0u, 0u, int3(0, 0, 0), kLightCellFaceNone,
                              (count != 0u && total > 0.0f) ? total : 0.0f);
    if (count == 0u || !(total > 0.0f))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    const uint capacity = CAELIX_LOAD_U32(table + kLightHeaderCapacity);

    int slot = -1;
    uint entity = 0u;
    uint level = 0u;
    int3 sample_coord = int3(0, 0, 0);
    CAELIX_BRANCH
    if (fc.light_cells != 0u)
    {
        level = LightCellLevel(fc, x_r);
        entity = GroupEntity(fc, group);
        // kGiFlagLightCellOriented: both lookups use the receiver face of n_r (the jitter moves only
        // the point).
        const uint face = LightCellFaceOf(fc, n_r);
        const bool jitter = (fc.gi_flags & kGiFlagLightCellJitter) != 0u;
        // The proxy's cell is the unjittered one; without the jitter it is also the sampling cell.
        if (mis || !jitter)
        {
            sample_coord = LightCellCoord(object_point, level);
            slot = LightCellLookup(fc, entity, level, sample_coord, face);
            if (mis)
            {
                ctx = LightNeeProxyOfSlot(fc, slot, entity, level, sample_coord, face, total);
            }
        }
        if (jitter)
        {
            const float3 u_jitter = float3(RandomFloat01(rng), RandomFloat01(rng), RandomFloat01(rng));
            const float3 cell_point = object_point + (u_jitter - 0.5f) * LightCellSize(level);
            sample_coord = LightCellCoord(cell_point, level);
            slot = LightCellLookup(fc, entity, level, sample_coord, face);
        }
    }

    // Per-vertex resampling (section 2.6): K >= 2 candidates of the same sampling cell, one ray.
    CAELIX_BRANCH
    if (fc.light_nee_candidates >= 2u)
    {
        return LightNextEventResampled(fc, x_r, n_r, slot, count, total, capacity, mis, ctx, rng);
    }

    uint light = kNoLight;
    float W_cell = 0.0f;
    uint cell_voxel = 0u;
    uint cell_face = kLightCellFaceBrick;
    uint push_index = kNoLight;  // kGiFlagLightCellPush: the claimed push of the sampling cell
    if (slot >= 0)
    {
        if ((fc.gi_flags & kGiFlagLightCellPush) != 0u)
        {
            push_index = LightCellClaimPush(fc, uint(slot));
        }
        const uint r = min(fc.light_cell_reservoirs - 1u, uint(u_reservoir * float(fc.light_cell_reservoirs)));
        const uint2 reservoir = g_light_cells.Load2(uint(slot) * LightCellBytes(fc) + kLightCellReservoirOffset +
                                                    r * kLightCellReservoirBytes);
        uint entry_index;
        uint voxel;
        uint face;
        if (asfloat(reservoir.y) > 0.0f && LightCellDecode(reservoir.x, count, entry_index, voxel, face))
        {
            light = entry_index;
            W_cell = asfloat(reservoir.y);
            cell_voxel = voxel;
            cell_face = face;
        }
    }
    uint64_t address = 0u;
    LightEntryHeader entry = (LightEntryHeader)0;
    if (light != kNoLight)
    {
        address = LightEntryAddress(table, light);
        entry = LoadLightEntryHeader(address);
        // A voxel-face word whose voxel is no longer a list light with that face open (the scene
        // changed): no sample from the cell.
        if (cell_face < kLightCellFaceBrick && !LightFaceMaskBit(address, cell_face, LightVoxel3(cell_voxel)))
        {
            light = kNoLight;
        }
    }
    float W_brick = W_cell;
    if (light == kNoLight)
    {
        // The global CDF: the brick's probability is total_b / total.
        cell_face = kLightCellFaceBrick;
        light = LightCdfSearch(table, capacity, count, u_global * total);
        address = LightEntryAddress(table, light);
        entry = LoadLightEntryHeader(address);
        W_brick = entry.total_weight > 0.0f ? total / entry.total_weight : 0.0f;
    }
    const bool face_sample = cell_face < kLightCellFaceBrick;
    LightSample s;
    if (face_sample)
    {
        s = LightSampleFaceOf(fc, entry, address, cell_voxel, cell_face, false, rng);
    }
    else
    {
        s = SampleLightEntry(fc, entry, address, x_r, rng);
    }

    float3 contribution = float3(0.0f, 0.0f, 0.0f);
    bool occluded = false;
    if (s.valid && W_brick > 0.0f)
    {
        const float3 d = s.position - x_r;
        const float d2 = dot(d, d);
        const float distance = sqrt(d2);
        if (d2 > 0.0f && distance >= g_lightNeeMinDistance)
        {
            const float3 w = d / distance;
            const float cos_r = dot(n_r, w);
            const float cos_l = -dot(s.normal, w);
            if (cos_r > 0.0f && cos_l > 0.0f)
            {
                // The point lies on a voxel face: the ray stops 2 push-offs short of it (as GiVisible).
                RayDesc shadow;
                shadow.Origin = x_r;
                shadow.Direction = w;
                shadow.TMin = 0.0f;
                shadow.TMax = distance - 2.0f * K_RAY_ORIGIN_PUSH_OFF;
                occluded = shadow.TMax > 0.0f && TraceShadow(fc, shadow);
                if (!occluded)
                {
                    contribution =
                        s.radiance * ((cos_r * (1.0f / K_PI)) * (cos_l / d2) / s.pdf_area * W_brick);
                    if (mis)
                    {
                        contribution *= LightNeeMisWeight(ctx, s.mis, x_r, n_r, w, distance, cos_l, true);
                    }
                }
            }
        }
    }

    CAELIX_BRANCH
    if (push_index < fc.light_cell_reservoirs)
    {
        uint word = kNoLight;
        float W_push = 0.0f;
        if (face_sample)
        {
            word = LightCellWord(light, cell_voxel, cell_face);
            W_push = W_brick;
        }
        else if (W_brick > 0.0f)
        {
            const uint region = LightCellRegionFaceMask(fc, entity, level, sample_coord, entry);
            if (s.valid && s.face_mask == region)
            {
                word = LightCellWord(light, s.voxel, s.face);
                W_push = W_brick / s.pdf_area;
            }
            else
            {
                word = LightCellWord(light, 0u, kLightCellFaceBrick);
                W_push = W_brick;
            }
        }
        LightCellStorePush(fc, uint(slot), push_index, word, W_push, !occluded);
    }
    return contribution;
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
