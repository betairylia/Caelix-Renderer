// Sky SH kernel: projects the host's sky cubemap (CaelixRenderOptions::sky) onto the 9
// real spherical harmonics of bands 0..2 (CaelixShBasis), convolved with the clamped
// cosine lobe (Ramamoorthi and Hanrahan 2001), so that the trace kernel evaluates the
// irradiance E(n) of a normal n with 9 loads (SkyIrradianceOverPi, diffuse bounces
// that miss). One workgroup of 256 threads, dispatched before the trace in the
// lighting modes when the host gave a cubemap (the renderer orders its write before
// the trace with a barrier).
//
// Sampling: 6 faces x 32 x 32 texel centres (6144 directions, 24 per thread) at the
// mip fc.sky_sh_lod (the mip whose face is about 32 px). Texel (face, x, y), y = row
// from the top of the face, s = (x + 0.5) / 32 * 2 - 1, t = (y + 0.5) / 32 * 2 - 1,
// has the direction of the Vulkan cube face selection inverted:
//   +X (1, -t, -s)   -X (-1, -t, s)   +Y (s, 1, t)   -Y (s, -1, -t)   +Z (s, -t, 1)   -Z (-s, -t, -1)
// and the solid angle w = (2 / 32)^2 / (1 + s^2 + t^2)^1.5. The weights are summed too
// and the result is renormalized by 4 pi / sum: the discretization error of the solid
// angles cancels, and a constant sky L projects exactly (E(n) = pi L for every n).
//
// Output: 9 float4 (rgb = coefficient, a = 0) at fc.sky_sh, already multiplied by the
// band's cosine-lobe factor A_l (pi, 2 pi / 3, pi / 4), so E(n) = sum c_i Y_i(n).
//
// Descriptors: set 0 binding 0 the cubemap (TextureCube, a sampled image), binding 1
// its sampler (linear, mipmapped, clamp to edge). The push constant is the device
// address of the frame constants (same block as the trace).
#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"
// The sky bindings of this kernel's set (caelix_shading.hlsl declares them).
#define CAELIX_SKY_BINDING_IMAGE 0
#define CAELIX_SKY_BINDING_SAMPLER 1
#include "render/caelix_shading.hlsl"

CAELIX_PUSH_CONSTANTS(CaelixTracePush, g_push);

static const uint kShThreads = 256;
static const uint kShFaceSize = 32;
static const uint kShSamples = 6 * kShFaceSize * kShFaceSize;  // 6144

groupshared float3 g_reduce[kShThreads];

// The world direction of the centre of texel (x, y) of `face` (not normalized).
float3 CubeTexelDirection(uint face, float s, float t)
{
    switch (face)
    {
        case 0u:
            return float3(1.0f, -t, -s);
        case 1u:
            return float3(-1.0f, -t, s);
        case 2u:
            return float3(s, 1.0f, t);
        case 3u:
            return float3(s, -1.0f, -t);
        case 4u:
            return float3(s, -t, 1.0f);
        default:
            return float3(-s, -t, -1.0f);
    }
}

// Sums `value` over the workgroup; thread 0 receives the total (the other threads a
// partial sum). Every thread must call it (barriers).
float3 WorkgroupSum(uint thread, float3 value)
{
    g_reduce[thread] = value;
    GroupMemoryBarrierWithGroupSync();
    CAELIX_UNROLL
    for (uint stride = kShThreads / 2; stride > 0; stride >>= 1)
    {
        if (thread < stride)
        {
            g_reduce[thread] += g_reduce[thread + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float3 total = g_reduce[0];
    // The next call overwrites g_reduce: every thread must have read it first.
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    const CaelixFrameConstants fc = CAELIX_LOAD_STRUCT(CaelixFrameConstants, g_push.frame_constants, 16);
    const uint thread = thread_id.x;

    float3 sums[9];
    CAELIX_UNROLL
    for (uint k = 0; k < 9; ++k)
    {
        sums[k] = float3(0.0f, 0.0f, 0.0f);
    }
    float weight_sum = 0.0f;
    const float texel = 2.0f / float(kShFaceSize);
    CAELIX_LOOP
    for (uint i = thread; i < kShSamples; i += kShThreads)
    {
        const uint face = i / (kShFaceSize * kShFaceSize);
        const uint x = i % kShFaceSize;
        const uint y = (i / kShFaceSize) % kShFaceSize;
        const float s = (float(x) + 0.5f) * texel - 1.0f;
        const float t = (float(y) + 0.5f) * texel - 1.0f;
        const float3 direction = normalize(CubeTexelDirection(face, s, t));
        const float r2 = 1.0f + s * s + t * t;
        const float weight = texel * texel / (r2 * sqrt(r2));
        const float3 radiance = g_sky.SampleLevel(g_sky_sampler, direction, fc.sky_sh_lod).rgb;
        float basis[9];
        CaelixShBasis(direction, basis);
        CAELIX_UNROLL
        for (uint k = 0; k < 9; ++k)
        {
            sums[k] += radiance * (basis[k] * weight);
        }
        weight_sum += weight;
    }

    // 4 pi / total weight: the exact solid angle of the sphere over the sampled one.
    const float total_weight = WorkgroupSum(thread, float3(weight_sum, 0.0f, 0.0f)).x;
    const float normalization = 4.0f * K_PI / total_weight;
    CAELIX_UNROLL
    for (uint k = 0; k < 9; ++k)
    {
        const float3 total = WorkgroupSum(thread, sums[k]);
        if (thread == 0u)
        {
            // A_l of the clamped cosine: pi (l = 0), 2 pi / 3 (l = 1), pi / 4 (l = 2).
            const float band = k == 0u ? K_PI : (k < 4u ? 2.0f * K_PI / 3.0f : 0.25f * K_PI);
            const float3 coefficient = total * (band * normalization);
            CAELIX_STORE_U32x4(fc.sky_sh + uint64_t(k) * 16, asuint(float4(coefficient, 0.0f)));
        }
    }
}
