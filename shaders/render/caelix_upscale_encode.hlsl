// Upscale encode kernel (frames with an upscaler or with the per-voxel colour pass,
// lib/src/render/renderer_render.cpp): writes the host colour target at the OUTPUT size from the upscale
// source, one thread per output pixel.
//
//   source = g_src[p * upscale_src_size / output_size]       (integer nearest: the upscaler's
//            output at the output size, or the render-size upscale input on the fallback and
//            without an upscaler, where both sizes equal size and the mapping is the identity)
//   coverage = g_int_albedo[p * size / output_size].a        (nearest, render size)
//   fc.voxel_color (experiment, docs/VOXEL_COLOR_EXPERIMENT.md): source = the mean colour of the slot
//            that caelix_voxel_accumulate.hlsl stored for p (g_voxel_slots[p]) when it is valid
//   colour target[p] = (linear -> sRGB of source.rgb for a UNORM target, coverage)
//
// No NRD validation overlay (it does not exist at the output size). The motion debug view of the
// path trace mode is written as it is (no sRGB encode, as without an upscaler). kRenderFlagFlipY:
// the target row is output_size.y - 1 - y.
//
// Descriptors (the encode set): binding 0 the host colour target (no format qualifier), binding 1
// the source (RGBA16F), binding 2 the internal albedo (RGBA8, a = coverage), binding 3 the voxel slot
// image (R32_UINT, output size) and binding 4 the voxel hash table (common/caelix_voxel_color.hlsl);
// 3 and 4 are read only with fc.voxel_color (PARTIALLY_BOUND: not written otherwise). The frame
// constants are the uniform buffer g_frame (set 1, binding 0, common/caelix_frame_ubo.hlsl); no push
// constants.
#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_frame_ubo.hlsl"
#include "common/caelix_voxel_color.hlsl"
#include "render/caelix_output.hlsl"

CAELIX_BINDING(0, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_color;
CAELIX_BINDING(1, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_src;
CAELIX_BINDING(2, 0) CAELIX_FORMAT("rgba8") RWTexture2D<float4> g_int_albedo;
CAELIX_BINDING(3, 0) CAELIX_FORMAT("r32ui") RWTexture2D<uint> g_voxel_slots;
CAELIX_BINDING(4, 0) StructuredBuffer<uint> g_voxel_hash;

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
    float3 source = g_src[src_pixel].rgb;
    CAELIX_BRANCH
    if (fc.voxel_color != 0u)
    {
        const uint slot = g_voxel_slots[pixel];
        if (slot != kVoxelSlotInvalid)
        {
            const uint base = slot * kVoxelSlotWords;
            const uint count = g_voxel_hash[base + kVoxelSlotCount];
            const uint3 sum_lo = uint3(g_voxel_hash[base + kVoxelSlotSumLo], g_voxel_hash[base + kVoxelSlotSumLo + 1u],
                                       g_voxel_hash[base + kVoxelSlotSumLo + 2u]);
            const uint3 sum_hi = uint3(g_voxel_hash[base + kVoxelSlotSumHi], g_voxel_hash[base + kVoxelSlotSumHi + 1u],
                                       g_voxel_hash[base + kVoxelSlotSumHi + 2u]);
            source = VoxelSlotMean(count, sum_lo, sum_hi);  // count >= 1: this pixel added to the slot
        }
    }
    const float coverage = g_int_albedo[render_pixel].a > 0.5f ? 1.0f : 0.0f;
    const bool motion_view = fc.mode == kModePathTrace && fc.debug_view == kViewMotion;
    const float3 color = (fc.output_srgb != 0u && !motion_view) ? LinearToSrgb(source) : source;
    const uint2 out_pixel =
        (fc.flags & kRenderFlagFlipY) != 0u ? uint2(pixel.x, fc.output_size.y - 1u - pixel.y) : pixel;
    g_color[out_pixel] = float4(color, coverage);
}
