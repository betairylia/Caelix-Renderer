#pragma once
// The generic upscaler interface of the renderer (CaelixRenderOptions::upscaler). No NGX types:
// renderer.h and every file it includes see only this header. The DLSS implementation is in
// dlss_upscaler.h / .cpp (CAELIX_DLSS builds only).
#include "core/vk_common.h"

#include <cstdint>

namespace caelix {

enum class UpscalerKind : uint32_t
{
    None = 0,                   // CAELIX_UPSCALER_NONE
    DlssSuperResolution = 1,    // CAELIX_UPSCALER_DLSS_SR
    DlssRayReconstruction = 2,  // CAELIX_UPSCALER_DLSS_RR
};

// What an upscaler instance is created for. Any change of a field creates a new instance (the
// history is lost).
struct UpscalerDesc
{
    UpscalerKind kind = UpscalerKind::None;
    uint32_t quality = 0;  // CaelixUpscaleQuality
    uint32_t preset = 0;   // raw NGX preset hint, 0 = default
    uint32_t render_width = 0;
    uint32_t render_height = 0;
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    bool depth_inverted = false;  // the depth input is reverse Z (near 1, far 0)
    bool auto_exposure = false;   // SR: NGX auto exposure instead of the fixed exposure 1
};

inline bool operator==(const UpscalerDesc& a, const UpscalerDesc& b)
{
    return a.kind == b.kind && a.quality == b.quality && a.preset == b.preset &&
           a.render_width == b.render_width && a.render_height == b.render_height &&
           a.output_width == b.output_width && a.output_height == b.output_height &&
           a.depth_inverted == b.depth_inverted && a.auto_exposure == b.auto_exposure;
}

// One image of an evaluate: 2D, mip 0, layer 0. Every input is in VK_IMAGE_LAYOUT_GENERAL when
// Evaluate is recorded and is back in GENERAL after it; the output too.
struct UpscalerImage
{
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
};

// The inputs of one frame. Render-size images except the output (output size).
struct UpscalerInputs
{
    UpscalerImage color;   // linear HDR colour, already exposed
    UpscalerImage depth;   // device depth of the host projection
    UpscalerImage motion;  // previous pixel - current pixel, render pixels (xy)
    UpscalerImage output;  // written by the upscaler
    // Ray Reconstruction only:
    UpscalerImage diffuse_albedo;
    UpscalerImage specular_albedo;
    UpscalerImage normal_roughness;       // world normal xyz, linear roughness w
    UpscalerImage specular_hit_distance;  // world distance primary surface -> first bounce hit
    float jitter[2] = {0.0f, 0.0f};       // render pixels, y down, in [-0.5, 0.5]
    bool reset = false;                   // drop the history
    float world_to_view[16] = {};         // column-major, column vectors (= NGX "row major, left multiplication")
    float view_to_clip[16] = {};          // CaelixCamera::projection as it is
    float frame_time_ms = 0.0f;           // time since the previous evaluate, 0 = unknown
};

// One upscaler instance (one feature of one size and quality). Destroy it only when no pending GPU
// work uses it.
class Upscaler
{
public:
    virtual ~Upscaler() = default;
    virtual const UpscalerDesc& desc() const = 0;
    // Records the barriers around the evaluate (inputs GENERAL -> SHADER_READ_ONLY_OPTIMAL and back,
    // the output stays GENERAL) and the evaluate itself. The after-barriers are recorded also when
    // the evaluate fails. Returns false after an error (logged once; failed() stays true).
    virtual bool Evaluate(VkCommandBuffer cmd, const UpscalerInputs& in) = 0;
    virtual bool failed() const = 0;
};

}  // namespace caelix
