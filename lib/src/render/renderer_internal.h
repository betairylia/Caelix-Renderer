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
inline nrd::ReblurHitDistanceParameters HitDistanceParameters() {
  return nrd::ReblurHitDistanceParameters{};
}

// The targets of CaelixRenderTargets in descriptor binding order (binding = index + 1).
// `formats` lists the accepted formats; all VK_FORMAT_UNDEFINED = any storage format.
struct TargetSpec {
  const char* name;
  uint32_t bit;
  VkFormat formats[4];  // unused entries are VK_FORMAT_UNDEFINED
};
constexpr TargetSpec kTargetSpecs[] = {
    {"colour", kTargetColor, {}},
    {"albedo", kTargetAlbedo, {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT}},
    {"normal", kTargetNormal, {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM}},
    {"depth", kTargetDepth, {VK_FORMAT_R32_SFLOAT}},
    // RGBA formats also receive the delta checkerboard flag in w.
    {"motion", kTargetMotion,
     {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
      VK_FORMAT_R32G32B32A32_SFLOAT}},
    {"view_z", kTargetViewZ, {VK_FORMAT_R32_SFLOAT}},
};

// The internal targets in Renderer::Internal order. The formats match the
// image_format qualifiers of the kernels.
struct InternalSpec {
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
    {"accum", VK_FORMAT_R32G32B32A32_SFLOAT},
    {"diff_out", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"spec_out", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"validation", VK_FORMAT_R8G8B8A8_UNORM},
    {"resolve", VK_FORMAT_R16G16B16A16_SFLOAT},
};

}  // namespace caelix
