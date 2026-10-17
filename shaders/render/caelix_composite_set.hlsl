// The composite descriptor set, shared by caelix_composite.hlsl and
// caelix_cross_resolve.hlsl (one set layout, one set per frame slot), and the output
// stage both use: the sRGB encode and NRD's validation overlay.
//
// Set 0: binding 0 the host colour target (no format qualifier, write only),
// bindings 1..8 internal targets: albedo (a = coverage), deterministic, diffuse,
// specular, the reference accumulation, NRD's validation layer, the motion (w = the
// delta checkerboard flag) and the resolve image (the composite's linear output when
// the cross resolve runs); binding 9 the sky cubemap (sampled image: the host's
// cubemap or the renderer's fallback cube) and binding 10 its sampler, declared by
// render/caelix_shading.hlsl through CAELIX_SKY_BINDING_IMAGE / _SAMPLER, which this
// file defines: include it BEFORE caelix_shading.hlsl. The push constant is the device
// address of the frame constants (same block as the trace).
#ifndef CAELIX_COMPOSITE_SET_HLSL
#define CAELIX_COMPOSITE_SET_HLSL

#define CAELIX_SKY_BINDING_IMAGE 9
#define CAELIX_SKY_BINDING_SAMPLER 10

#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"

CAELIX_PUSH_CONSTANTS(CaelixTracePush, g_push);
CAELIX_BINDING(0, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_color;
CAELIX_BINDING(1, 0) CAELIX_FORMAT("rgba8") RWTexture2D<float4> g_int_albedo;
CAELIX_BINDING(2, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_deterministic;
CAELIX_BINDING(3, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_diff;
CAELIX_BINDING(4, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_spec;
CAELIX_BINDING(5, 0) CAELIX_FORMAT("rgba32f") RWTexture2D<float4> g_int_accum;
CAELIX_BINDING(6, 0) CAELIX_FORMAT("rgba8") RWTexture2D<float4> g_int_validation;
CAELIX_BINDING(7, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_motion;
CAELIX_BINDING(8, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_resolve;

float3 LinearToSrgb(float3 c)
{
    c = saturate(c);
    const float3 low = c * 12.92f;
    const float3 high = 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
    return float3(c.x <= 0.0031308f ? low.x : high.x, c.y <= 0.0031308f ? low.y : high.y,
                  c.z <= 0.0031308f ? low.z : high.z);
}

// The last stage before the host target: linear -> sRGB for a UNORM target, then (path
// trace mode, kViewNrdValidation, denoised frame) NRD's display-ready RGBA8 validation
// layer blended over the encoded colour.
float3 CaelixOutputColor(CaelixFrameConstants fc, uint2 pixel, float3 color)
{
    if (fc.output_srgb != 0u)
    {
        color = LinearToSrgb(color);
    }
    CAELIX_BRANCH
    if (fc.mode == kModePathTrace && fc.debug_view == kViewNrdValidation && fc.denoised != 0u)
    {
        const float4 layer = g_int_validation[pixel];
        color = lerp(color, layer.rgb, layer.a);
    }
    return color;
}

#endif  // CAELIX_COMPOSITE_SET_HLSL
