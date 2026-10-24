// Probe kernel of caelix_light_nee_test (tests/light_nee_test.cpp): runs the light tree's functions of the
// light NEE and of a region receiver (light/caelix_light_tree.hlsl) and the light NEE's MIS weight
// (render/caelix_light_nee.hlsl).
//   mode 0 support: thread i < count: SupportQuery at input + 64 i (a box receiver lo, hi, normal, a point
//                   in front of the light face and the face: group, voxel in group, face, block) -> the
//                   face's brick (LightTreeFindLightFace), its box pdf (LightTreeBrickPdfBox) and its target
//                   (LightTreeLeafImportanceBox): output + 16 i = (found, has leaf, pdf, target)
//   mode 1 mis:     thread i < count: MisQuery at input + 32 (i / draws) (a receiver point and its normal)
//                   and seed: one lamp point drawn as the trace's LampCandidate draws it (LightTreeSample)
//                   and its NEE-side weight; the point found again as a BSDF hit would find it
//                   (LightTreeFindLightFace) and its BSDF-side pdf (LightTreeBrickPdf x in_brick) and
//                   weight: output + 32 i = (ok, same brick, pdf of the draw, pdf of the BSDF side,
//                   w_nee, w_bsdf, 0, 0)
//   mode 2 weight:  thread i < count: WeightQuery at input + 48 i (p_area, d, cos_l, the receiver normal
//                   n and the unit direction w) -> LightNeeMisWeight on both sides: output + 16 i =
//                   (w_nee, w_bsdf, 0, 0)
#include "light/caelix_light_tree.hlsl"
#include "render/caelix_light_nee.hlsl"

struct LightNeeProbePush
{
    uint64_t descriptor;
    uint64_t input;
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
    if (g_push.mode == 0)
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
    if (g_push.mode == 1)
    {
        // MisQuery: float3 x_r; pad; float3 n_r; pad.
        const uint64_t q = g_push.input + uint64_t(i / max(g_push.draws, 1u)) * 32;
        const float3 x_r = LoadFloat3(q);
        const float3 n_r = LoadFloat3(q + 16);
        uint rng = ProbeHash(i * 2654435761u + g_push.seed);
        float4 a = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 b = float4(0.0f, 0.0f, 0.0f, 0.0f);
        LightPoint p;
        if (open && LightTreeSample(tree, x_r, n_r, rng, p) && p.pdf > 0.0f)
        {
            const float3 d = p.position - x_r;
            const float distance = length(d);
            const float3 w = d / distance;
            const float cos_l = -dot(p.normal, w);
            // The BSDF side: the hit's voxel in its group and its face, found again.
            const uint3 voxel = LightBrickVoxelOrigin(tree, p.brick) +
                                uint3(p.voxel & 7, (p.voxel >> 3) & 7, p.voxel >> 6);
            LightBrickRef found;
            uint found_leaf;
            float in_brick;
            const bool hit = LightTreeFindLightFace(tree, x_r, p.brick.group, voxel, p.face, p.block, found,
                                                    found_leaf, in_brick);
            const float p_bsdf = hit ? LightTreeBrickPdf(tree, x_r, n_r, found) * in_brick : 0.0f;
            a.x = 1.0f;
            a.y = hit && found.entry == p.brick.entry && found.generation == p.brick.generation ? 1.0f : 0.0f;
            a.z = p.pdf;
            a.w = p_bsdf;
            b.x = LightNeeMisWeight(p.pdf, n_r, w, distance, cos_l, true);
            b.y = LightNeeMisWeight(p_bsdf, n_r, w, distance, cos_l, false);
        }
        StoreFloat4(g_push.output + uint64_t(i) * 32, a);
        StoreFloat4(g_push.output + uint64_t(i) * 32 + 16, b);
        return;
    }
    // WeightQuery: float p_area, d, cos_l; pad; float3 n; pad; float3 w; pad.
    const uint64_t q = g_push.input + uint64_t(i) * 48;
    const float3 head = LoadFloat3(q);
    const float3 n = LoadFloat3(q + 16);
    const float3 w = LoadFloat3(q + 32);
    StoreFloat4(g_push.output + uint64_t(i) * 16,
                float4(LightNeeMisWeight(head.x, n, w, head.y, head.z, true),
                       LightNeeMisWeight(head.x, n, w, head.y, head.z, false), 0.0f, 0.0f));
}
