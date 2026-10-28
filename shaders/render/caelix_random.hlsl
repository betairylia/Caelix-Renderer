// The per-pixel random stream of the render kernels (PcgHash, InitRandom, RandomFloat01,
// RandomUnitVector) and the circle constants K_PI and K_TWO_PI. render/caelix_shading.hlsl includes
// this file; render/caelix_scene_trace.hlsl includes it for the subsurface part of the transmissive
// shadow ray (docs/SHADOW_TRANSMISSION.md).
//
// Ported from Unity (Caelix/Runtime/Rendering/Shaders):
//   Utils/BlueNoise.hlsl                  RandomFloat01 / RandomUnitVector semantics (inout state)
#ifndef CAELIX_RANDOM_HLSL
#define CAELIX_RANDOM_HLSL

static const float K_PI = 3.1415926535f;
static const float K_TWO_PI = 6.283185307f;

// Random numbers. NATIVE: one 32-bit state per pixel, seeded by a hash of the
// pixel and the frame seed, advanced by xorshift32. Same helper semantics as
// Unity (an inout state, values in [0, 1)).

uint PcgHash(uint value)
{
    const uint state = value * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint InitRandom(uint2 pixel, uint frame_seed)
{
    const uint state = PcgHash(pixel.x ^ (pixel.y << 16) ^ (frame_seed * 0x9E3779B9u));
    return state != 0u ? state : 0x9E3779B9u;  // xorshift never leaves 0
}

// Uniform in [0, 1): the top 24 bits of the next xorshift32 state.
float RandomFloat01(inout uint state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(state >> 8) * (1.0f / 16777216.0f);
}

// Unity BlueNoise.hlsl RandomUnitVector(float2): a uniform direction on the sphere.
float3 RandomUnitVector(float2 sample)
{
    const float a = sample.x * K_TWO_PI;
    const float y = sample.y * 2.0f - 1.0f;
    const float sy = sqrt(max(1.0f - y * y, 0.0f));
    return float3(sin(a) * sy, y, cos(a) * sy);
}

float3 RandomUnitVector(inout uint state)
{
    const float u = RandomFloat01(state);
    const float v = RandomFloat01(state);
    return RandomUnitVector(float2(u, v));
}

#endif  // CAELIX_RANDOM_HLSL
