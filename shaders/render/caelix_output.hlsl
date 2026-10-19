// The output encode shared by the kernels that write the host colour target (the composite
// stage of caelix_composite_set.hlsl and caelix_upscale_encode.hlsl).
#ifndef CAELIX_OUTPUT_HLSL
#define CAELIX_OUTPUT_HLSL

// Linear -> sRGB (IEC 61966-2-1) of a colour clamped to [0, 1], for UNORM colour targets.
float3 LinearToSrgb(float3 c)
{
    c = saturate(c);
    const float3 low = c * 12.92f;
    const float3 high = 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
    return float3(c.x <= 0.0031308f ? low.x : high.x, c.y <= 0.0031308f ? low.y : high.y,
                  c.z <= 0.0031308f ? low.z : high.z);
}

#endif  // CAELIX_OUTPUT_HLSL
