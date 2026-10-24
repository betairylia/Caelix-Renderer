// Renderer: the light NEE options, the choice per render call and the stats (docs/LIGHT_NEE.md section 7).
#include "render/renderer.h"

#include "core/log.h"
#include "render/frame_constants.h"
#include "scene/scene.h"

#include <iterator>

namespace caelix {
namespace {

constexpr uint32_t kKnownFlags = CAELIX_LIGHT_NEE_FLAG_NO_MIS;

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
    for (uint32_t i = 0; i < std::size(options.reserved); ++i)
    {
        if (options.reserved[i] != 0)
        {
            CAELIX_LOGE("%s: reserved[%u] must be 0 (%u)", kFunction, i, options.reserved[i]);
            return CAELIX_ERROR_INVALID_ARGUMENT;
        }
    }
    light_nee_options_ = options;
    return CAELIX_OK;
}

uint32_t Renderer::LightNeeFlags(bool lighting)
{
    light_nee_running_ = false;
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
    return kLightNeeOn | (mis ? kLightNeeMis : 0u);
}

void Renderer::GetLightNeeStats(CaelixLightNeeStats& out) const
{
    out = CaelixLightNeeStats{};
    out.enabled = light_nee_options_.enabled;
    out.running = light_nee_running_ ? 1u : 0u;
}

}  // namespace caelix
