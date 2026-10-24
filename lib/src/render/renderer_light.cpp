// Renderer: the light NEE options, the light cells' lifetime and stats (docs/LIGHT_NEE.md section 7).
#include "render/renderer.h"

#include "core/log.h"
#include "scene/scene.h"

#include <cmath>
#include <utility>

namespace caelix {
namespace {

// CaelixLightNeeOptions defaults (0 = these) and ranges.
constexpr uint32_t kDefaultCellCapacity = 262144;
constexpr uint32_t kMinCellCapacity = 1024;
constexpr float kDefaultFootprint = 0.25f;
// Mode "off" (owner choice 2026-10-24 after the fill bench, docs/LIGHT_NEE.md section 11.3): white
// frame-to-frame noise for the denoiser, at the higher fill cost.
constexpr uint32_t kDefaultReservoirMode = kLightCellModeOff;
constexpr uint32_t kDefaultCandidates = 1;
constexpr uint32_t kMaxCandidates = 2;
constexpr uint32_t kDefaultMaxAge = 128;
constexpr uint32_t kMaxMaxAge = 1u << 20;
constexpr uint32_t kKnownFlags = CAELIX_LIGHT_NEE_FLAG_NO_JITTER | CAELIX_LIGHT_NEE_FLAG_NO_MIS |
                                 CAELIX_LIGHT_NEE_FLAG_NO_SUN_HINT | CAELIX_LIGHT_NEE_FLAG_NO_CONTROLLER |
                                 CAELIX_LIGHT_NEE_FLAG_LOOKUP_ONLY;
static_assert(kLightCellModeOff == CAELIX_LIGHT_NEE_RESERVOIRS_OFF);
static_assert(kLightCellModeRotate == CAELIX_LIGHT_NEE_RESERVOIRS_ROTATE);

}  // namespace

CaelixResult Renderer::SetLightNeeOptions(const CaelixLightNeeOptions& options)
{
    constexpr const char* kFunction = "caelix_renderer_set_light_nee_options";
    if (options.enabled > 1)
    {
        CAELIX_LOGE("%s: enabled must be 0 or 1 (%u)", kFunction, options.enabled);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.cell_capacity != 0 &&
        (options.cell_capacity < kMinCellCapacity || options.cell_capacity > kHashMaxPayloads))
    {
        CAELIX_LOGE("%s: cell_capacity %u is outside %u .. %u (0 = %u)", kFunction, options.cell_capacity,
                    kMinCellCapacity, kHashMaxPayloads, kDefaultCellCapacity);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(options.footprint) || options.footprint < 0.0f)
    {
        CAELIX_LOGE("%s: footprint %g must be finite and >= 0", kFunction,
                    static_cast<double>(options.footprint));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.reservoir_mode > CAELIX_LIGHT_NEE_RESERVOIRS_ROTATE)
    {
        CAELIX_LOGE("%s: unknown reservoir_mode %u", kFunction, options.reservoir_mode);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.candidates > kMaxCandidates)
    {
        CAELIX_LOGE("%s: candidates %u is above %u", kFunction, options.candidates, kMaxCandidates);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.max_age > kMaxMaxAge)
    {
        CAELIX_LOGE("%s: max_age %u is above %u", kFunction, options.max_age, kMaxMaxAge);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if ((options.flags & ~kKnownFlags) != 0)
    {
        CAELIX_LOGE("%s: unknown flags 0x%x", kFunction, options.flags & ~kKnownFlags);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    light_nee_options_ = options;
    LightCells::Settings& s = light_nee_settings_;
    s.payload_capacity = options.cell_capacity != 0 ? options.cell_capacity : kDefaultCellCapacity;
    s.footprint = options.footprint > 0.0f ? options.footprint : kDefaultFootprint;
    s.reservoir_mode = options.reservoir_mode != 0 ? options.reservoir_mode : kDefaultReservoirMode;
    s.candidates = options.candidates != 0 ? options.candidates : kDefaultCandidates;
    s.max_age = options.max_age != 0 ? options.max_age : kDefaultMaxAge;
    s.flags = options.flags;
    return CAELIX_OK;
}

LightCells* Renderer::EnsureLightCells(bool lighting, uint64_t frame)
{
    light_nee_running_ = false;
    const bool wanted = light_nee_options_.enabled != 0;
    // A new light tree (disabled and enabled again: new rows whose generations start over) could match
    // the generation of a reservoir of the old one: the cells start over with it.
    const VkDeviceAddress tree = scene_.light_tree_address();
    const bool new_tree = tree != 0 && tree != light_cells_tree_;
    const bool retire = light_cells_ != nullptr &&
                        (!wanted || new_tree ||
                         light_cells_->payload_capacity() != light_nee_settings_.payload_capacity);
    if (retire)
    {
        // Released once its last frame is safe; a later enable starts with empty cells.
        retired_light_cells_.push_back(RetiredLightCells{light_cells_last_frame_, std::move(light_cells_)});
    }
    if (!wanted || !lighting)
    {
        return nullptr;
    }
    if (tree == 0)
    {
        if (!light_nee_no_tree_logged_)
        {
            CAELIX_LOGW("light NEE is enabled but the scene has no light tree "
                        "(caelix_scene_set_light_options): it does not run");
            light_nee_no_tree_logged_ = true;
        }
        return nullptr;
    }
    if (light_cells_ == nullptr)
    {
        light_cells_ = std::make_unique<LightCells>(device_, light_nee_settings_.payload_capacity);
        light_cells_tree_ = tree;
    }
    light_cells_last_frame_ = frame;
    light_nee_running_ = true;
    return light_cells_.get();
}

void Renderer::GetLightNeeStats(CaelixLightNeeStats& out) const
{
    out = CaelixLightNeeStats{};
    out.enabled = light_nee_options_.enabled;
    if (light_cells_ == nullptr)
    {
        return;
    }
    light_cells_->GetStats(out);
    out.running = light_nee_running_ ? 1u : 0u;
    out.footprint = light_cells_->footprint(light_nee_settings_);
}

}  // namespace caelix
