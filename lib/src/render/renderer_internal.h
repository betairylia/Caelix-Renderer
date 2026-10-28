#pragma once
// Tables and helpers shared by the renderer_*.cpp translation units. Not part of
// the Renderer interface: include it from those files only.
#include "render/frame_constants.h"
#include "render/renderer.h"

namespace caelix {

const VkImageSubresourceRange kColorRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

// The renderer's one NRD denoiser (REBLUR_DIFFUSE_SPECULAR).
constexpr nrd::Identifier kReblurIdentifier = 1;

// REBLUR's hit distance normalization (A, B, C), NRD's defaults (3, 0.1, 20): the
// trace kernel's REBLUR_FrontEnd_GetNormHitDist and ReblurSettings use the same.
inline nrd::ReblurHitDistanceParameters HitDistanceParameters()
{
    return nrd::ReblurHitDistanceParameters{};
}

// The targets of CaelixRenderTargets in descriptor binding order (binding = index + 1).
// `formats` lists the accepted formats; all VK_FORMAT_UNDEFINED = any storage format.
struct TargetSpec
{
    const char* name;
    uint32_t bit;
    VkFormat formats[4];  // unused entries are VK_FORMAT_UNDEFINED
};
constexpr TargetSpec kTargetSpecs[] = {
    {"colour", kTargetColor, {}},
    {"albedo", kTargetAlbedo, {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT}},
    {"normal", kTargetNormal, {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM}},
    // With an upscaler the depth target has the render or the output size (kTargetDepthIndex).
    {"depth", kTargetDepth, {VK_FORMAT_R32_SFLOAT}},
    // RGBA formats also receive the delta checkerboard flag in w.
    {"motion", kTargetMotion, {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT}},
    {"view_z", kTargetViewZ, {VK_FORMAT_R32_SFLOAT}},
    // With an upscaler the voxel id target has the output size (kTargetVoxelIdIndex).
    {"voxel_id", kTargetVoxelId, {VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT}},
};
// The indices of the depth and voxel id targets in kTargetSpecs (binding index + 1).
constexpr uint32_t kTargetDepthIndex = 3;
constexpr uint32_t kTargetVoxelIdIndex = 6;

// The internal targets in Renderer::Internal order. The formats match the
// image_format qualifiers of the kernels. The two light hint images have the size
// ceil(size / patch) (Renderer::EnsureInternalTargets), the others the render size.
struct InternalSpec
{
    const char* name;
    VkFormat format;
};
constexpr InternalSpec kInternalSpecs[] = {
    {"view_z", VK_FORMAT_R32_SFLOAT},
    {"normal_roughness", VK_FORMAT_A2B10G10R10_UNORM_PACK32},
    {"motion", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"albedo", VK_FORMAT_R8G8B8A8_UNORM},
    {"deterministic", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"diff_in", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"spec_in", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"depth", VK_FORMAT_R32_SFLOAT},
    {"rr_diffuse_albedo", VK_FORMAT_R8G8B8A8_UNORM},
    {"rr_specular_albedo", VK_FORMAT_R8G8B8A8_UNORM},
    {"rr_normal_roughness", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"rr_hit_dist", VK_FORMAT_R16_SFLOAT},
    {"light_hint_a", VK_FORMAT_R8G8_UNORM},
    {"light_hint_b", VK_FORMAT_R8G8_UNORM},
    {"accum", VK_FORMAT_R32G32B32A32_SFLOAT},
    {"diff_out", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"spec_out", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"validation", VK_FORMAT_R8G8B8A8_UNORM},
    {"resolve", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"upscale_in", VK_FORMAT_R16G16B16A16_SFLOAT},
};

// NRD's (and DLSS RR's) world-to-view matrix of `camera`: column-major, column vectors
// (view = M * world), rows right, up, forward (the camera's view space, +z forward,
// left-handed like the projection), translation -dot(axis, position). NGX's "row major, left
// multiplication" matrices have the same memory layout.
inline void WorldToView(const CaelixCamera& camera, float m[16])
{
    const float* axes[3] = {camera.right, camera.up, camera.forward};
    for (uint32_t row = 0; row < 3; ++row)
    {
        const float* axis = axes[row];
        m[0 * 4 + row] = axis[0];
        m[1 * 4 + row] = axis[1];
        m[2 * 4 + row] = axis[2];
        m[3 * 4 + row] = -(axis[0] * camera.position[0] + axis[1] * camera.position[1] +
                           axis[2] * camera.position[2]);
    }
    m[3] = 0.0f;
    m[7] = 0.0f;
    m[11] = 0.0f;
    m[15] = 1.0f;
}

// True when CaelixCamera::projection is reverse Z: the device depth (clip.z / clip.w) of a point
// at view z = 2 near is below the depth at z = near (near = camera.near_z, at least 1e-3).
inline bool ProjectionDepthInverted(const CaelixCamera& camera)
{
    const float* m = camera.projection;  // column-major: row r, column c = m[c * 4 + r]
    const auto depth = [m](float z) {
        const float w = m[11] * z + m[15];
        return w != 0.0f ? (m[10] * z + m[14]) / w : 0.0f;
    };
    const float z = camera.near_z > 1e-3f ? camera.near_z : 1e-3f;
    return depth(2.0f * z) < depth(z);
}

}  // namespace caelix
