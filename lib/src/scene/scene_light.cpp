// The scene's light tree: caelix_scene_set_light_options, caelix_scene_get_light_stats and the
// scene-side helpers of the flush. The tree itself is lib/src/light/light_tree.*.
#include "scene/scene.h"

#include "core/log.h"

#include <cmath>

namespace caelix {
namespace {

constexpr float kDefaultMinEmission = 20.0f;
constexpr uint64_t kDefaultHeapBytes = 16ull << 20;
constexpr uint64_t kMaxHeapBytes = kLtMaxHeapUnits * kLtUnitBytes;

}  // namespace

CaelixResult Scene::SetLightOptions(const CaelixLightOptions& options)
{
    constexpr const char* kFunction = "caelix_scene_set_light_options";
    if (options.enabled > 1)
    {
        CAELIX_LOGE("%s: enabled %u is not 0 or 1", kFunction, options.enabled);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(options.min_emission) || options.min_emission < 0.0f)
    {
        CAELIX_LOGE("%s: min_emission %g must be finite and >= 0", kFunction,
                    static_cast<double>(options.min_emission));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.heap_bytes > kMaxHeapBytes)
    {
        CAELIX_LOGE("%s: heap_bytes %llu is above the maximum of %llu", kFunction,
                    static_cast<unsigned long long>(options.heap_bytes),
                    static_cast<unsigned long long>(kMaxHeapBytes));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    const float min_emission = options.min_emission > 0.0f ? options.min_emission : kDefaultMinEmission;
    if (options.enabled == 0)
    {
        if (light_ != nullptr)
        {
            // The GPU may still read it in the frames already flushed.
            retired_light_.push_back(RetiredLightTree{has_flushed_ ? last_frame_ : 0, std::move(light_)});
        }
        return CAELIX_OK;
    }
    if (light_ != nullptr)
    {
        light_->SetMinEmission(min_emission);
        return CAELIX_OK;
    }
    const uint64_t heap_bytes = options.heap_bytes != 0 ? options.heap_bytes : kDefaultHeapBytes;
    std::unique_ptr<LightTree> tree =
        std::make_unique<LightTree>(device_, group_shift_, encoding_, min_emission, heap_bytes);
    // The current scene, through the same hooks as later changes.
    for (uint32_t id = 0; id < entities_.size(); ++id)
    {
        const Entity& entity = entities_[id];
        if (!entity.alive)
        {
            continue;
        }
        tree->OnEntityAdded(id, entity.motion);
    }
    for (uint32_t index = 0; index < groups_.size(); ++index)
    {
        if (groups_[index].alive)
        {
            tree->OnGroupCreated(index);
        }
    }
    light_ = std::move(tree);
    return CAELIX_OK;
}

void Scene::GetLightStats(CaelixLightStats& out) const
{
    out = CaelixLightStats{};
    if (light_ != nullptr)
    {
        light_->GetStats(out);
    }
}

LightSceneTables Scene::LightTables() const
{
    LightSceneTables tables;
    tables.page_table = page_table_.address();
    tables.slot_tables = slot_tables_.address();
    tables.reverse_tables = reverse_tables_.address();
    tables.group_infos = group_infos_.address();
    tables.entity_infos = entity_infos_.address();
    tables.grids = grids_.address();
    tables.materials = material_buffer_.address();
    tables.records_per_page_shift = records_per_page_shift();
    tables.record_bytes = layout_.record_bytes;
    tables.group_capacity = static_cast<uint32_t>(groups_.size());
    tables.entity_capacity = static_cast<uint32_t>(entities_.size());
    return tables;
}

void Scene::RequeueAllResident()
{
    for (uint32_t index = 0; index < groups_.size(); ++index)
    {
        Group& group = groups_[index];
        if (!group.alive)
        {
            continue;
        }
        for (uint16_t id = 0; id < group.slots.size(); ++id)
        {
            if (group.slots[id].state == kSlotResident)
            {
                QueueFinalize(index, id);
            }
        }
    }
}

}  // namespace caelix
