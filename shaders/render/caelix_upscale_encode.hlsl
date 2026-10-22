// Upscale encode kernel (frames with an upscaler, lib/src/render/renderer_render.cpp): writes the
// host colour target at the OUTPUT size from the upscale source, one thread per output pixel.
//
//   source = g_src[p * upscale_src_size / output_size]       (integer nearest: the upscaler's
//            output at the output size, or the render-size upscale input on the fallback)
//   coverage = g_int_albedo[p * size / output_size].a        (nearest, render size)
//   colour target[p] = (linear -> sRGB of source.rgb for a UNORM target, coverage)
//
// No NRD validation overlay (it does not exist at the output size). The motion debug view of the
// path trace mode is written as it is (no sRGB encode, as without an upscaler). kRenderFlagFlipY:
// the target row is output_size.y - 1 - y.
//
// Descriptors (the encode set): binding 0 the host colour target (no format qualifier), binding 1
// the source (RGBA16F), binding 2 the internal albedo (RGBA8, a = coverage). The frame constants
// are the uniform buffer g_frame (set 1, binding 0, common/caelix_frame_ubo.hlsl); no push
// constants.
#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_frame_ubo.hlsl"
#include "render/caelix_output.hlsl"

CAELIX_BINDING(0, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_color;
CAELIX_BINDING(1, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_src;
CAELIX_BINDING(2, 0) CAELIX_FORMAT("rgba8") RWTexture2D<float4> g_int_albedo;

[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    const CaelixFrameConstants fc = g_frame;
    if (any(thread_id.xy >= fc.output_size))
    {
        return;
    }
    const uint2 pixel = thread_id.xy;
    const uint2 src_pixel = min(pixel * fc.upscale_src_size / fc.output_size, fc.upscale_src_size - 1u);
    const uint2 render_pixel = min(pixel * fc.size / fc.output_size, fc.size - 1u);
    const float3 source = g_src[src_pixel].rgb;
    const float coverage = g_int_albedo[render_pixel].a > 0.5f ? 1.0f : 0.0f;
    const bool motion_view = fc.mode == kModePathTrace && fc.debug_view == kViewMotion;
    const float3 color = (fc.output_srgb != 0u && !motion_view) ? LinearToSrgb(source) : source;
    const uint2 out_pixel =
        (fc.flags & kRenderFlagFlipY) != 0u ? uint2(pixel.x, fc.output_size.y - 1u - pixel.y) : pixel;
    g_color[out_pixel] = float4(color, coverage);
}
