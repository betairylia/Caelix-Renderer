// Voxel id kernel (frames with an upscaler and a voxel id target, lib/src/render/renderer_render.cpp):
// writes the host voxel id target at the OUTPUT size, one thread per output pixel, so that the key
// image pairs with the upscaled colour. Without an upscaler the trace kernel writes the target.
//
//   ray      = from fc.position along CaelixCamera's pinhole mapping of the output pixel CENTRE
//              (no jitter: the primary trace stays jittered, this ray does not)
//   target[p] = VoxelIdHash of the first hit (transparent voxels included), or 0xFFFFFFFF in every
//              channel for a miss and for a hit farther than fc.voxel_id_tmax (0 = no limit)
//
// kRenderFlagFlipY: the target row is output_size.y - 1 - y.
//
// Descriptors: the trace set (the renderer binds the frame slot's trace set): binding 0 the TLAS
// (render/caelix_scene_trace.hlsl), binding 7 the voxel id target (no format qualifier: R32_UINT or
// R32G32_UINT); the other bindings are PARTIALLY_BOUND and unused here. The frame constants are the
// uniform buffer g_frame (set 1, binding 0, common/caelix_frame_ubo.hlsl); no push constants. It
// reads only the TLAS and the scene tables, so it needs no barrier after the trace dispatch.
#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_frame_ubo.hlsl"
#include "render/caelix_scene_trace.hlsl"

CAELIX_BINDING(7, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<uint4> g_voxel_id;

[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    const CaelixFrameConstants fc = g_frame;
    if (any(thread_id.xy >= fc.output_size))
    {
        return;
    }
    // The brick DDA reads the face bits of transparent codes by the scene's encoding.
    g_blockEncoding = fc.encoding;
    const uint2 pixel = thread_id.xy;
    // The pinhole mapping of CaelixPrimaryDirection at the output size, without the jitter: the
    // same expression, so a render without an upscaler and jitter 0 at this size traces the same
    // rays.
    const float2 sample_position = float2(float(pixel.x) + 0.5f, float(pixel.y) + 0.5f);
    const float2 ndc = float2(sample_position.x / float(fc.output_size.x) * 2.0f - 1.0f,
                              1.0f - sample_position.y / float(fc.output_size.y) * 2.0f);
    RayDesc ray;
    ray.Origin = fc.position.xyz;
    ray.Direction = normalize(fc.right.xyz * (ndc.x * fc.tan_half_fov_y * fc.aspect) +
                              fc.up.xyz * (ndc.y * fc.tan_half_fov_y) + fc.forward.xyz);
    ray.TMin = 0.0f;
    ray.TMax = 1e30f;

    const TraceResult hit = TraceScene(fc, ray);
    const uint2 key = hit.hit && (fc.voxel_id_tmax <= 0.0f || hit.T <= fc.voxel_id_tmax)
                          ? VoxelIdHash(fc, hit)
                          : uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
    const uint2 out_pixel =
        (fc.flags & kRenderFlagFlipY) != 0u ? uint2(pixel.x, fc.output_size.y - 1u - pixel.y) : pixel;
    g_voxel_id[out_pixel] = uint4(key, 0u, 0u);
}
