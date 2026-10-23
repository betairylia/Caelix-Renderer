// Probe kernel of caelix_light_nee_test (tests/light_nee_test.cpp): runs the light cells' and the light
// tree's functions of the light NEE (render/caelix_light_cells.hlsl, scene/caelix_light_tree.hlsl).
//   mode 0 region:  thread i < count: RegionQuery at input + 64 i (entity, face, level, jitter flag,
//                   the jitter numbers u, the entity-local hit point) -> the cell of the hit
//                   (LightCellCoordOfHit) and its region (LightCellRegionOf): output + 48 i =
//                   float3 lo, float3 hi, float3 normal, int3 coord (the test checks that the region
//                   holds the receiver and has its normal)
//   mode 1 support: thread i < count: SupportQuery at input + 64 i (a box receiver lo, hi, normal, a point
//                   in front of the light face and the face: group, voxel in group, face, block) -> the
//                   face's brick (LightTreeFindLightFace), its box pdf (LightTreeBrickPdfBox) and its target
//                   (LightTreeLeafImportanceBox): output + 16 i = (found, has leaf, pdf, target)
//   mode 2 mis:     thread i < count: MisQuery at input2 + 64 (i / draws) (the cell region, its C, a
//                   receiver and its normal) and seed: one brick from LightTreeSampleBrickBox for the
//                   region and a point by LightBrickSamplePoint, its NEE-side proxy and weight; the point
//                   found again as a BSDF hit would find it (LightTreeFindLightFace) and its BSDF-side
//                   proxy and weight: output + 32 i = (ok, same brick, pdf_in_brick, in_brick of the hit,
//                   p_area NEE side, p_area BSDF side, w_nee, w_bsdf)
#include "render/caelix_light_cells.hlsl"

struct LightNeeProbePush
{
    uint64_t descriptor;
    uint64_t input;
    uint64_t input2;
    uint64_t output;
    uint mode;
    uint count;
    uint draws;
    uint seed;
};

CAELIX_PUSH_CONSTANTS(LightNeeProbePush, g_push);

uint ProbeHash(uint x)
{
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

float3 LoadFloat3(uint64_t address)
{
    return asfloat(
        uint3(CAELIX_LOAD_U32(address), CAELIX_LOAD_U32(address + 4), CAELIX_LOAD_U32(address + 8)));
}

void StoreFloat4(uint64_t address, float4 v)
{
    CAELIX_STORE_U32x4(address, asuint(v));
}

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    const uint i = dispatch_id.x;
    if (i >= g_push.count)
    {
        return;
    }
    LightTree tree;
    const bool open = LightTreeOpen(g_push.descriptor, tree);
    const uint64_t entity_infos = LtDescriptorU64(g_push.descriptor, kLtDescEntityInfos);
    if (g_push.mode == 0)
    {
        // RegionQuery: uint entity, face, level, jitter; float3 u; pad; float3 object_hit; pad.
        const uint64_t q = g_push.input + uint64_t(i) * 64;
        const uint4 head = CAELIX_LOAD_U32x4(q);
        const float3 u = LoadFloat3(q + 16);
        const float3 hit = LoadFloat3(q + 32);
        const int3 coord = LightCellCoordOfHit(hit, float3(FaceDirection(head.y)), u, head.w != 0u, head.z);
        const LightCellRegion region = LightCellRegionOf(entity_infos, head.x, coord, head.z, head.y);
        const uint64_t o = g_push.output + uint64_t(i) * 48;
        StoreFloat4(o, float4(region.lo, region.hi.x));
        StoreFloat4(o + 16, float4(region.hi.yz, region.normal.xy));
        CAELIX_STORE_U32x4(o + 32, uint4(asuint(region.normal.z), asuint(coord)));
        return;
    }
    if (g_push.mode == 1)
    {
        // SupportQuery: float3 lo; x.x; float3 hi; x.y; float3 normal; x.z; uint group, voxel (packed
        // x | y << 10 | z << 20), face, block. x lies in front of the face (the family's weight at x is
        // positive there, so the face is found).
        const uint64_t q = g_push.input + uint64_t(i) * 64;
        const float3 lo = LoadFloat3(q);
        const float3 hi = LoadFloat3(q + 16);
        const float3 n = LoadFloat3(q + 32);
        const float3 front =
            asfloat(uint3(CAELIX_LOAD_U32(q + 12), CAELIX_LOAD_U32(q + 28), CAELIX_LOAD_U32(q + 44)));
        const uint4 face = CAELIX_LOAD_U32x4(q + 48);
        float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);
        LightBrickRef brick;
        uint leaf;
        float in_brick;
        const uint3 voxel = LtUnpackVoxel(face.y);
        if (open && LightTreeFindLightFace(tree, front, face.x, voxel, face.z, face.w, brick, leaf, in_brick))
        {
            result.x = 1.0f;
            result.y = leaf != kLtNone ? 1.0f : 0.0f;
            result.z = LightTreeBrickPdfBox(tree, lo, hi, n, brick);
            result.w = LightTreeLeafImportanceBox(tree, brick, lo, hi, n);
        }
        StoreFloat4(g_push.output + uint64_t(i) * 16, result);
        return;
    }
    // MisQuery: float3 lo; float norm; float3 hi; pad; float3 normal; pad; float3 x_r; pad (the receiver
    // normal is the region's).
    const uint64_t q = g_push.input2 + uint64_t(i / max(g_push.draws, 1u)) * 64;
    LightCellRegion region;
    region.lo = LoadFloat3(q);
    const float norm = asfloat(CAELIX_LOAD_U32(q + 12));
    region.hi = LoadFloat3(q + 16);
    region.normal = LoadFloat3(q + 32);
    const float3 x_r = LoadFloat3(q + 48);
    const float3 n_r = region.normal;
    uint rng = ProbeHash(i * 2654435761u + g_push.seed);
    float4 a = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 b = float4(0.0f, 0.0f, 0.0f, 0.0f);
    LightBrickRef brick;
    float pdf;
    uint leaf;
    float target;
    LightPoint p;
    const bool drawn = open && LightTreeSampleBrickBox(tree, region.lo, region.hi, region.normal, rng, brick,
                                                       pdf, leaf, target);
    if (drawn && LightBrickSamplePoint(tree, brick, x_r, n_r, rng, p) && p.pdf > 0.0f)
    {
        const float3 d = p.position - x_r;
        const float distance = length(d);
        const float3 w = d / distance;
        const float cos_l = -dot(p.normal, w);
        const float p_nee = LightNeeProxyArea(tree, region, norm, brick, p.pdf);
        // The BSDF side: the hit's voxel in its group and its face, found again.
        const uint3 voxel =
            LightBrickVoxelOrigin(tree, brick) + uint3(p.voxel & 7, (p.voxel >> 3) & 7, p.voxel >> 6);
        LightBrickRef found;
        uint found_leaf;
        float in_brick;
        const bool hit = LightTreeFindLightFace(tree, x_r, brick.group, voxel, p.face, p.block, found,
                                                found_leaf, in_brick);
        const float p_bsdf = hit ? LightNeeProxyArea(tree, region, norm, found, in_brick) : 0.0f;
        a.x = 1.0f;
        a.y = hit && found.entry == brick.entry && found.generation == brick.generation ? 1.0f : 0.0f;
        a.z = p.pdf;
        a.w = in_brick;
        b.x = p_nee;
        b.y = p_bsdf;
        b.z = LightNeeMisWeight(p_nee, n_r, w, distance, cos_l, true);
        b.w = LightNeeMisWeight(p_bsdf, n_r, w, distance, cos_l, false);
    }
    StoreFloat4(g_push.output + uint64_t(i) * 32, a);
    StoreFloat4(g_push.output + uint64_t(i) * 32 + 16, b);
}
