// Renderer: the light NEE options, the choice per render call and the stats (docs/LIGHT_NEE.md section 7).
#include "render/renderer.h"

#include "core/log.h"
#include "render/frame_constants.h"
#include "scene/scene.h"

namespace caelix {
namespace {

constexpr uint32_t kKnownFlags = CAELIX_LIGHT_NEE_FLAG_NO_MIS | CAELIX_LIGHT_NEE_FLAG_NO_VISIBILITY_HINT;
// The visibility hint's patch for CaelixLightNeeOptions::hint_patch 0 (docs/LIGHT_NEE.md section 12.3
// chooses it by measurement).
constexpr uint32_t kDefaultHintPatch = 2;

}  // namespace

CaelixResult Renderer::SetLightNeeOptions(const CaelixLightNeeOptions& options)
{
    constexpr const char* kFunction = "caelix_renderer_set_light_nee_options";
    if (options.enabled > 1)
    {
        CAELIX_LOGE("%s: enabled must be 0 or 1 (%u)", kFunction, options.enabled);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if ((options.flags & ~kKnownFlags) != 0)
    {
        CAELIX_LOGE("%s: unknown flags 0x%x", kFunction, options.flags & ~kKnownFlags);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.hint_patch != 0 && options.hint_patch != 1 && options.hint_patch != 2 && options.hint_patch != 4)
    {
        CAELIX_LOGE("%s: hint_patch must be 0 (the default), 1, 2 or 4 (%u)", kFunction, options.hint_patch);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.reserved != 0)
    {
        CAELIX_LOGE("%s: reserved must be 0 (%u)", kFunction, options.reserved);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    light_nee_options_ = options;
    return CAELIX_OK;
}

uint32_t Renderer::LightNeeHintShift() const
{
    const uint32_t patch = light_nee_options_.hint_patch != 0 ? light_nee_options_.hint_patch : kDefaultHintPatch;
    return patch == 4 ? 2u : (patch == 2 ? 1u : 0u);
}

uint32_t Renderer::LightNeeFlags(bool lighting)
{
    light_nee_running_ = false;
    light_nee_hint_ = false;
    if (light_nee_options_.enabled == 0 || !lighting)
    {
        return 0;
    }
    if (scene_.light_tree_address() == 0)
    {
        if (!light_nee_no_tree_logged_)
        {
            CAELIX_LOGW("light NEE is enabled but the scene has no light tree "
                        "(caelix_scene_set_light_options): it does not run");
            light_nee_no_tree_logged_ = true;
        }
        return 0;
    }
    light_nee_running_ = true;
    const bool mis = (light_nee_options_.flags & CAELIX_LIGHT_NEE_FLAG_NO_MIS) == 0;
    light_nee_hint_ = (light_nee_options_.flags & CAELIX_LIGHT_NEE_FLAG_NO_VISIBILITY_HINT) == 0;
    return kLightNeeOn | (mis ? kLightNeeMis : 0u) | (light_nee_hint_ ? kLightNeeHint : 0u);
}

void Renderer::GetLightNeeStats(CaelixLightNeeStats& out) const
{
    out = CaelixLightNeeStats{};
    out.enabled = light_nee_options_.enabled;
    out.running = light_nee_running_ ? 1u : 0u;
    // The patch the internal targets were sized for in the newest lighting render.
    out.hint_patch = light_nee_hint_ ? 1u << internal_hint_shift_ : 0u;
}

}  // namespace caelix
