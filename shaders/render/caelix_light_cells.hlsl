// The light cells of the light NEE (docs/LIGHT_NEE.md sections 3, 4 and 6), shared by the fill kernel
// (render/caelix_light_cell_fill.hlsl), the trace kernel (render/caelix_path_trace.hlsl) and the test
// probes: the cell geometry (receiver box and normal), the cell header, the MIS weight and the
// density proxy. Every function takes its tables as arguments (no frame constants), so a probe can run
// it. The layouts are common/caelix_light_cells_layout.hlsl.
//
// A cell is keyed by (entity, entity life, level, coordinate, receiver face). Cells are entity-local:
// the cell point of a vertex is its entity-local hit point moved kLightCellPointOffset voxels into the
// air along the face normal; its coordinate at level L is floor(p) >> (5 + 2 L). The receiver face is
// the entity-local face of the hit voxel, so every receiver of a cell has the world normal R e_face.
// The receiver BOX of a cell (the fill's draws, the MIS proxy) is the cell inflated by half a cell
// (the jittered lookups read the cell of a point moved by up to half a cell) plus
// kLightCellBoxMarginVoxels (the receiver itself lies a quarter voxel below the cell point), as the
// world AABB of the box under the entity's current transform: every receiver that can read the cell
// lies inside it.
#ifndef CAELIX_LIGHT_CELLS_HLSL
#define CAELIX_LIGHT_CELLS_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_light_cells_layout.hlsl"
#include "scene/caelix_light_tree.hlsl"

// The cell point: the hit point moved this far (voxels, entity-local) into the air along the face.
static const float kLightCellPointOffset = 0.25f;
// Lamp points closer than this (world units) to the receiver belong to the BSDF path, on both sides
// of the MIS (docs/LIGHT_NEE.md section 4.6): area sampling with a 1 / d^2 term is extremely
// heavy-tailed near a lamp, the BSDF path finds near emitters with a bounded weight.
static const float kLightNeeNearField = 8.0f;
// The floor of the sun hint of the sun / lamp selection: a visible sun keeps a positive probability.
static const float kLightNeeSunHintFloor = 0.05f;

uint64_t LightCellAddress(uint64_t payloads, uint slot)
{
    return payloads + uint64_t(slot) * kLightCellBytes;
}

// The edge of a level-`level` cell in voxels (32, 128).
float LightCellEdge(uint level)
{
    return float(1u << (kLightCellBaseShift + kLightCellLevelShift * level));
}

// The coordinate of the entity-local cell point p at `level`.
int3 LightCellCoord(float3 p, uint level)
{
    return int3(floor(p)) >> int(kLightCellBaseShift + kLightCellLevelShift * level);
}

// The coordinate of the cell that a vertex reads: its entity-local hit point `object_hit` on the face with
// the entity-local normal `object_normal`, moved kLightCellPointOffset into the air, then (`jitter`) by
// (u - 0.5) cells per axis for the uniform numbers u in [0, 1). The trace (LightCellAtVertex) and the
// tests call it.
int3 LightCellCoordOfHit(float3 object_hit, float3 object_normal, float3 u, bool jitter, uint level)
{
    float3 p = object_hit + kLightCellPointOffset * object_normal;
    if (jitter)
    {
        p += (u - 0.5f) * LightCellEdge(level);
    }
    return LightCellCoord(p, level);
}

// The key words of a cell (the hash table's payload key, kLightCellKeyWords of them).
void LightCellKeyWords(uint entity, uint life, int3 coord, uint level, uint face,
                       out uint words[kHashMaxKeyWords])
{
    words[0] = entity;
    words[1] = life;
    words[2] = asuint(coord.x);
    words[3] = asuint(coord.y);
    words[4] = asuint(coord.z);
    words[5] = level;
    words[6] = face;
    words[7] = 0u;
}

// The receiver region of a cell in world space: the box every receiver of the cell lies in, and the
// receivers' normal.
struct LightCellRegion
{
    float3 lo;
    float3 hi;
    float3 normal;
};

LightCellRegion LightCellRegionOf(uint64_t entity_infos, uint entity, int3 coord, uint level, uint face)
{
    const float edge = LightCellEdge(level);
    const float inflate = 0.5f * edge + float(kLightCellBoxMarginVoxels);
    const float3 local_lo = float3(coord) * edge - inflate;
    const float3 local_hi = float3(coord + 1) * edge + inflate;
    const LtFrame frame = LtEntityFrame(entity_infos, entity);
    const float3 centre = mul(frame.rotation, 0.5f * (local_lo + local_hi)) + frame.translation;
    // The AABB of the rotated box: |R| times the half size.
    const float3 half_size = mul(abs(frame.rotation), 0.5f * (local_hi - local_lo));
    LightCellRegion region;
    region.lo = centre - half_size;
    region.hi = centre + half_size;
    region.normal = mul(frame.rotation, float3(FaceDirection(face)));
    return region;
}

// The receiver region of the cell at `cell` (its header's entity, coordinate, level and face).
LightCellRegion LightCellRegionOfHeader(uint64_t entity_infos, uint64_t cell)
{
    const uint entity = CAELIX_LOAD_U32(cell + kLcEntity);
    const int3 coord = asint(uint3(CAELIX_LOAD_U32(cell + kLcCoord), CAELIX_LOAD_U32(cell + kLcCoord + 4),
                                   CAELIX_LOAD_U32(cell + kLcCoord + 8)));
    const uint level = CAELIX_LOAD_U32(cell + kLcLevel);
    const uint face = CAELIX_LOAD_U32(cell + kLcFace);
    return LightCellRegionOf(entity_infos, entity, coord, level, face);
}

// A light brick reference of a reservoir (group, entry, generation).
LightBrickRef LightCellReservoirRef(uint4 reservoir)
{
    LightBrickRef brick;
    brick.group = reservoir.x;
    brick.entry = reservoir.y;
    brick.generation = reservoir.w;
    return brick;
}

// The balance heuristic between the light NEE and the cosine-weighted diffuse bounce at the receiver
// with world normal n_r, for a lamp point with proxy area density p_area in the unit direction w at
// distance d; cos_l = the lamp face's cosine toward the receiver. `nee_side`: the NEE's weight
// p_sa / (p_sa + p_bsdf), else the BSDF's p_bsdf / (p_sa + p_bsdf), with p_sa = p_area d^2 / cos_l and
// p_bsdf = cos_r / pi. Both sides call it with the same density, so the two weights of a point sum to
// 1 in every branch: a non-finite or degenerate p_sa (cos_l <= 0) counts as +inf (NEE 1, BSDF 0),
// both densities 0 as NEE 0, BSDF 1.
float LightNeeMisWeight(float p_area, float3 n_r, float3 w, float d, float cos_l, bool nee_side)
{
    const float p_sa = cos_l > 0.0f ? p_area * (d * d) / cos_l : 0.0f;
    const float p_bsdf = max(0.0f, dot(n_r, w)) * (1.0f / 3.1415926535f);
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

// The light NEE's density proxy at a lamp point (docs/LIGHT_NEE.md section 6): the fill's target of
// the point's brick (the box importance of its leaf summary for the cell's region, read from the leaf
// array) over the cell's normalisation C, times the point's in-brick density. 0 when C <= 0 or the
// brick has no leaf (the NEE never draws it: the BSDF keeps the full emission). Both MIS sides call it.
float LightNeeProxyArea(LightTree t, LightCellRegion region, float norm, LightBrickRef brick, float in_brick)
{
    if (!(norm > 0.0f))
    {
        return 0.0f;
    }
    return LightTreeLeafImportanceBox(t, brick, region.lo, region.hi, region.normal) / norm * in_brick;
}

#endif  // CAELIX_LIGHT_CELLS_HLSL
