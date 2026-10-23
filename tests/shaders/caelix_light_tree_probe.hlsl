// Probe kernel of caelix_light_tree_test (tests/light_tree_test.cpp): runs the light tree's device
// API (scene/caelix_light_tree.hlsl) and copies GPU memory for the test's readback.
//   mode 0 copy:   thread i < count: output[i] = the uint4 at input + 16 i (any device address)
//   mode 1 sample: thread i < count: receiver i / draws (input2, 32 bytes: float4 position,
//                  float4 normal); one LightTreeSample, the pdf evaluations of its result, and one
//                  LightTreeSampleBrick + LightBrickSamplePoint draw; 32 words to output + 128 i
//                  (layout: SampleRecord in the test)
//   mode 2 pdf:    thread i < count: query uint4 at input + 16 i (receiver, group, voxel in group
//                  x | y << 10 | z << 20, face | block << 16) -> float LightTreePointPdf at output + 4 i
//   mode 3 bench:  thread i < count: `draws` LightTreeSample draws for receiver i % receivers
//                  (input2); the sum of the pdfs to output + 4 i
//   mode 4 draw:   thread i < count: one LightTreeSample for receiver i / draws; uint4 (group, voxel in
//                  group, face, ok) to output + 16 i
#include "scene/caelix_light_tree.hlsl"

struct ProbePush
{
    uint64_t descriptor;
    uint64_t input;
    uint64_t input2;
    uint64_t output;
    uint mode;
    uint count;
    uint draws;
    uint seed;
    uint receivers;  // mode 3: receivers in input2
    uint pad;
};

CAELIX_PUSH_CONSTANTS(ProbePush, g_push);

uint Hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

void LoadReceiver(uint index, out float3 x, out float3 n)
{
    const uint64_t address = g_push.input2 + uint64_t(index) * 32;
    x = asfloat(CAELIX_LOAD_U32x4(address)).xyz;
    n = asfloat(CAELIX_LOAD_U32x4(address + 16)).xyz;
}

uint3 VoxelInGroup(LightTree t, LightPoint p)
{
    return LightBrickVoxelOrigin(t, p.brick) + uint3(p.voxel & 7, (p.voxel >> 3) & 7, p.voxel >> 6);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    const uint i = dispatch_id.x;
    if (i >= g_push.count)
    {
        return;
    }
    if (g_push.mode == 0)
    {
        const uint4 words = CAELIX_LOAD_U32x4(g_push.input + uint64_t(i) * 16);
        CAELIX_STORE_U32x4(g_push.output + uint64_t(i) * 16, words);
        return;
    }
    LightTree tree;
    const bool open = LightTreeOpen(g_push.descriptor, tree);
    if (g_push.mode == 1)
    {
        float3 x;
        float3 n;
        LoadReceiver(i / g_push.draws, x, n);
        uint rng = Hash(i * 2654435761u + g_push.seed);
        uint words[32];
        for (uint w = 0; w < 32; ++w)
        {
            words[w] = 0;
        }
        if (open)
        {
            LightPoint p;
            const bool ok = LightTreeSample(tree, x, n, rng, p);
            words[0] = ok ? 1u : 0u;
            words[1] = p.brick.group;
            words[2] = p.brick.entry;
            words[3] = p.brick.generation;
            words[4] = p.voxel;
            words[5] = p.face;
            words[6] = p.block;
            words[7] = asuint(p.pdf);
            words[8] = asuint(p.position.x);
            words[9] = asuint(p.position.y);
            words[10] = asuint(p.position.z);
            words[11] = asuint(p.normal.x);
            words[12] = asuint(p.normal.y);
            words[13] = asuint(p.normal.z);
            if (ok)
            {
                const uint3 v = VoxelInGroup(tree, p);
                words[14] = LtPackVoxel(v);
                words[15] = asuint(LightTreePointPdf(tree, x, n, p.brick.group, v, p.face, p.block));
                words[16] = asuint(LightTreeBrickPdf(tree, x, n, p.brick));
                words[17] = LightBrickRefValid(tree, p.brick) ? 1u : 0u;
            }
            LightBrickRef brick;
            float pdf_brick;
            if (LightTreeSampleBrick(tree, x, n, rng, brick, pdf_brick))
            {
                LightPoint q;
                const bool ok2 = LightBrickSamplePoint(tree, brick, x, n, rng, q);
                words[18] = ok2 ? 1u : 0u;
                words[19] = asuint(pdf_brick);
                words[20] = asuint(LightTreeBrickPdf(tree, x, n, brick));
                words[21] = asuint(q.pdf);
                if (ok2)
                {
                    const uint3 v = VoxelInGroup(tree, q);
                    words[22] = asuint(LightTreePointPdf(tree, x, n, brick.group, v, q.face, q.block));
                    words[23] = brick.group;
                    words[24] = LtPackVoxel(v);
                    words[25] = q.face;
                }
            }
        }
        const uint64_t destination = g_push.output + uint64_t(i) * 128;
        for (uint k = 0; k < 32; k += 4)
        {
            const uint4 four = uint4(words[k], words[k + 1], words[k + 2], words[k + 3]);
            CAELIX_STORE_U32x4(destination + k * 4, four);
        }
        return;
    }
    if (g_push.mode == 2)
    {
        const uint4 query = CAELIX_LOAD_U32x4(g_push.input + uint64_t(i) * 16);
        float3 x;
        float3 n;
        LoadReceiver(query.x, x, n);
        const uint3 v = LtUnpackVoxel(query.z);
        const float pdf =
            open ? LightTreePointPdf(tree, x, n, query.y, v, query.w & 0xFFFFu, query.w >> 16) : 0.0f;
        CAELIX_STORE_U32(g_push.output + uint64_t(i) * 4, asuint(pdf));
        return;
    }
    if (g_push.mode == 4)
    {
        float3 x;
        float3 n;
        LoadReceiver(i / g_push.draws, x, n);
        uint rng = Hash(i * 2654435761u + g_push.seed);
        uint4 result = uint4(0, 0, 0, 0);
        LightPoint p;
        if (open && LightTreeSample(tree, x, n, rng, p))
        {
            result = uint4(p.brick.group, LtPackVoxel(VoxelInGroup(tree, p)), p.face, 1u);
        }
        CAELIX_STORE_U32x4(g_push.output + uint64_t(i) * 16, result);
        return;
    }
    // Bench.
    float3 x;
    float3 n;
    LoadReceiver(i % g_push.receivers, x, n);
    uint rng = Hash(i);
    float sum = 0.0f;
    if (open)
    {
        for (uint d = 0; d < g_push.draws; ++d)
        {
            LightPoint p;
            if (LightTreeSample(tree, x, n, rng, p))
            {
                sum += p.pdf;
            }
        }
    }
    CAELIX_STORE_U32(g_push.output + uint64_t(i) * 4, asuint(sum));
}
