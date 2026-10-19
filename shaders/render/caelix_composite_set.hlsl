// The composite descriptor set, shared by caelix_composite.hlsl and
// caelix_cross_resolve.hlsl (one set layout, one set per frame slot), and the output
// stage both use: the sRGB encode and NRD's validation overlay.
//
// Set 0: binding 0 the host colour target (no format qualifier, write only),
// bindings 1..8 internal targets: albedo (a = coverage), deterministic, diffuse,
// specular, the reference accumulation, NRD's validation layer, the motion (w = the
// delta checkerboard flag) and the resolve image (the composite's linear output when
// the cross resolve runs); binding 9 the upscale input (fc.upscale: the linear, exposed
// colour that the upscaler reads, written instead of the colour target); binding 10 the
// sky cubemap (sampled image: the host's cubemap or the renderer's fallback cube) and
// binding 11 its sampler, declared by render/caelix_shading.hlsl through
// CAELIX_SKY_BINDING_IMAGE / _SAMPLER, which this file defines: include it BEFORE
// caelix_shading.hlsl. The push constant is the device address of the frame constants
// (same block as the trace).
#ifndef CAELIX_COMPOSITE_SET_HLSL
#define CAELIX_COMPOSITE_SET_HLSL

#define CAELIX_SKY_BINDING_IMAGE 10
#define CAELIX_SKY_BINDING_SAMPLER 11

#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"
#include "render/caelix_output.hlsl"

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
CAELIX_BINDING(9, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_upscale_in;

// The value an upscaled frame stores in the upscale input: the linear colour (clamped to the FP16
// range) and the coverage; no sRGB encode, no validation overlay, no flip.
float4 CaelixUpscaleInput(float3 color, float coverage)
{
    return float4(min(color, 65504.0f), coverage);
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
