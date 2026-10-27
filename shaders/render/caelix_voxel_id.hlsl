// The output-size pass (kernel caelix_voxel_id; frames with an upscaler and a voxel id target or an
// output-size depth target, lib/src/render/renderer_render.cpp): writes those host targets at the
// OUTPUT size, one thread per output pixel, so that they pair with the upscaled colour. Without an
// upscaler (and for a render-size depth target) the trace kernel writes them.
//
//   ray         = from fc.position along CaelixCamera's pinhole mapping of the output pixel CENTRE
//                 (no jitter: the primary trace stays jittered, this ray does not)
//   voxel_id[p] = VoxelIdHash of the first hit (transparent voxels included), or 0xFFFFFFFF in every
//                 channel for a miss and for a hit farther than fc.voxel_id_tmax (0 = no limit);
//                 written when kTargetVoxelId is set
//   depth[p]    = the device depth of the same first hit (no voxel_id_tmax, no delta chain), the
//                 trace kernel's formula: rel = direction * T, view = (rel . right, rel . up,
//                 rel . forward, 1), depth = dot(projection[2], view) / dot(projection[3], view);
//                 fc.miss_depth on a miss; written when kTargetDepthOutput is set
//
// kRenderFlagFlipY: the target row is output_size.y - 1 - y.
//
// Descriptors: the trace set (the renderer binds the frame slot's trace set): binding 0 the TLAS
// (render/caelix_scene_trace.hlsl), binding 4 the depth target (no format qualifier: R32_SFLOAT),
// binding 7 the voxel id target (no format qualifier: R32_UINT or R32G32_UINT); the other bindings are
// PARTIALLY_BOUND and unused here. The frame constants are the uniform buffer g_frame (set 1,
// binding 0, common/caelix_frame_ubo.hlsl); no push constants. It reads only the TLAS and the scene
// tables, so it needs no barrier after the trace dispatch.
#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_frame_ubo.hlsl"
#include "render/caelix_scene_trace.hlsl"

CAELIX_BINDING(4, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_depth;
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
    const uint2 out_pixel =
        (fc.flags & kRenderFlagFlipY) != 0u ? uint2(pixel.x, fc.output_size.y - 1u - pixel.y) : pixel;
    CAELIX_BRANCH
    if ((fc.target_mask & kTargetVoxelId) != 0u)
    {
        const uint2 key = hit.hit && (fc.voxel_id_tmax <= 0.0f || hit.T <= fc.voxel_id_tmax)
                              ? VoxelIdHash(fc, hit)
                              : uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
        g_voxel_id[out_pixel] = uint4(key, 0u, 0u);
    }
    CAELIX_BRANCH
    if ((fc.target_mask & kTargetDepthOutput) != 0u)
    {
        // The trace kernel's device depth (caelix_path_trace.hlsl), from the same expressions.
        float device_depth = fc.miss_depth;
        if (hit.hit)
        {
            const float3 rel = ray.Direction * hit.T;
            const float view_z = dot(rel, fc.forward.xyz);
            const float4 view = float4(dot(rel, fc.right.xyz), dot(rel, fc.up.xyz), view_z, 1.0f);
            device_depth = dot(fc.projection[2], view) / dot(fc.projection[3], view);
        }
        g_depth[out_pixel] = float4(device_depth, 0.0f, 0.0f, 0.0f);
    }
}
