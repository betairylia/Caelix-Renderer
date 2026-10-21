// The light list (docs/LIGHT_LIST.md): options and stats, the light words' capacity, the light
// table's allocation and its rebuild (shaders/scene/caelix_light_compact.hlsl and
// caelix_light_scan.hlsl) inside Scene::Flush.
#include "scene/scene.h"

#include "core/log.h"
#include "rhi/shader_loader.h"
#include "scene/scene_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace caelix {
namespace {

// CaelixLightOptions defaults (a zero field takes them). 20 = luminance of a packed emissive voxel
// of strength 2 (64 x albedo) with an albedo luminance of 0.3125 (see the "As built" section of
// docs/LIGHT_LIST.md).
constexpr float kDefaultMinEmission = 20.0f;
constexpr uint32_t kDefaultLightMaxBricks = 65536;
// The near-field split (docs/LIGHT_LIST.md "Near field"): 8 voxels, validated on bistro and Cornell.
constexpr float kDefaultNeeMinDistance = 8.0f;
constexpr uint32_t kMaxLightMaxBricks = 1u << 20;
// The header's count, dropped, capacity and pad: one readback slot.
constexpr uint32_t kLightReadbackBytes = 16;

}  // namespace

void Scene::CreateLightObjects()
{
    const VkDevice device = device_.device();
    light_min_emission_ = kDefaultMinEmission;
    light_max_bricks_ = kDefaultLightMaxBricks;
    light_nee_min_distance_ = kDefaultNeeMinDistance;
    // Binding 0: the light table (a storage buffer; the compact kernel counts with atomics).
    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                         nullptr};
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;
    const VkResult result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &light_set_layout_);
    if (result != VK_SUCCESS)
    {
        light_set_layout_ = VK_NULL_HANDLE;
        CAELIX_LOGE("vkCreateDescriptorSetLayout (light table) failed: %d", static_cast<int>(result));
        throw VulkanError("vkCreateDescriptorSetLayout failed", result);
    }
    const std::vector<uint32_t> compact_spirv = LoadShader("caelix_light_compact");
    light_compact_ =
        ComputePipeline(device, compact_spirv, sizeof(LightPush), "caelix_light_compact", light_set_layout_);
    const std::vector<uint32_t> scan_spirv = LoadShader("caelix_light_scan");
    light_scan_ = ComputePipeline(device, scan_spirv, sizeof(LightPush), "caelix_light_scan", light_set_layout_);
    light_readback_ = GpuBuffer::HostReadback(device_.allocator(), device,
                                              static_cast<VkDeviceSize>(kLightReadbackSlots) * kLightReadbackBytes);
}

void Scene::DestroyLightTable(LightTableGpu& table)
{
    if (table.pool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_.device(), table.pool, nullptr);  // frees the set too
    }
    table.pool = VK_NULL_HANDLE;
    table.set = VK_NULL_HANDLE;
    table.buffer.Reset();
    table.capacity = 0;
}

void Scene::DestroyLightObjects()
{
    // The device is idle (caelix_scene_destroy contract).
    DestroyLightTable(light_table_);
    for (RetiredLightTable& retired : retired_light_tables_)
    {
        DestroyLightTable(retired.table);
    }
    retired_light_tables_.clear();
    light_compact_ = ComputePipeline();
    light_scan_ = ComputePipeline();
    if (light_set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_.device(), light_set_layout_, nullptr);
        light_set_layout_ = VK_NULL_HANDLE;
    }
}

void Scene::ReserveLightWords()
{
    light_words_.Reserve(aabbs_.bytes() / kAabbRowBytes * sizeof(uint32_t));
    light_indices_.Reserve(aabbs_.bytes() / kAabbRowBytes * sizeof(uint32_t));
}

CaelixResult Scene::SetLightOptions(const CaelixLightOptions& options)
{
    constexpr const char* kFunction = "caelix_scene_set_light_options";
    if (!std::isfinite(options.min_emission) || options.min_emission < 0.0f)
    {
        CAELIX_LOGE("%s: min_emission %g must be finite and >= 0", kFunction,
                    static_cast<double>(options.min_emission));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(options.nee_min_distance) || options.nee_min_distance < 0.0f)
    {
        CAELIX_LOGE("%s: nee_min_distance %g must be finite and >= 0", kFunction,
                    static_cast<double>(options.nee_min_distance));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.max_bricks > kMaxLightMaxBricks)
    {
        CAELIX_LOGE("%s: max_bricks %u is above the maximum %u", kFunction, options.max_bricks, kMaxLightMaxBricks);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    // The table is opt-in: a host that never calls this function (the Unity plugin today) pays
    // nothing beyond the light words and the entry map.
    lights_enabled_ = true;
    lights_dirty_ = true;
    const float min_emission = options.min_emission != 0.0f ? options.min_emission : kDefaultMinEmission;
    const uint32_t max_bricks = options.max_bricks != 0 ? options.max_bricks : kDefaultLightMaxBricks;
    light_nee_min_distance_ =
        options.nee_min_distance != 0.0f ? options.nee_min_distance : kDefaultNeeMinDistance;
    if (min_emission != light_min_emission_ || max_bricks != light_max_bricks_)
    {
        light_min_emission_ = min_emission;
        light_max_bricks_ = max_bricks;
        // The next flush rebuilds the table (the threshold of the entries, the new capacity). Nothing
        // to rebuild before the first brick: the table is created by the first rebuild.
        lights_dirty_ = lights_dirty_ || light_table_.buffer.valid();
    }
    return CAELIX_OK;
}

void Scene::GetLightStats(CaelixLightStats& out) const
{
    out = CaelixLightStats{};
    out.bricks = light_stats_count_;
    out.bricks_dropped = light_stats_dropped_;
    out.table_bytes = light_table_.buffer.size() + light_words_.bytes() + light_indices_.bytes();
    out.rebuilds = light_rebuilds_;
}

void Scene::EnsureLightTable(uint64_t frame)
{
    // A rebuild may be due in this flush: the table is dirty, or bricks wait for finalization, or
    // groups for staging (their range moves and freed ids make it dirty during the flush).
    const bool rebuild_possible = lights_dirty_ || !dirty_.empty() || !dirty_groups_.empty();
    if (!lights_enabled_ || !rebuild_possible ||
        (light_table_.buffer.valid() && light_table_.capacity == light_max_bricks_))
    {
        return;
    }
    const VkDevice device = device_.device();
    LightTableGpu fresh;
    fresh.capacity = light_max_bricks_;
    fresh.buffer = GpuBuffer::DeviceLocal(device_.allocator(), device, LightTableBytes(light_max_bricks_));
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkResult result = vkCreateDescriptorPool(device, &pool_info, nullptr, &fresh.pool);
    if (result != VK_SUCCESS)
    {
        fresh.pool = VK_NULL_HANDLE;
        CAELIX_LOGE("vkCreateDescriptorPool (light table) failed: %d", static_cast<int>(result));
        throw VulkanError("vkCreateDescriptorPool failed", result);
    }
    VkDescriptorSetAllocateInfo allocate_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate_info.descriptorPool = fresh.pool;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &light_set_layout_;
    result = vkAllocateDescriptorSets(device, &allocate_info, &fresh.set);
    if (result != VK_SUCCESS)
    {
        DestroyLightTable(fresh);
        CAELIX_LOGE("vkAllocateDescriptorSets (light table) failed: %d", static_cast<int>(result));
        throw VulkanError("vkAllocateDescriptorSets failed", result);
    }
    const VkDescriptorBufferInfo buffer_info{fresh.buffer.buffer(), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = fresh.set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    if (light_table_.buffer.valid())
    {
        retired_light_tables_.push_back(RetiredLightTable{frame, std::move(light_table_)});
        light_table_ = LightTableGpu{};
    }
    light_table_ = std::move(fresh);
    light_table_built_ = false;
    CAELIX_LOGI("scene: light table created for %u bricks (%.1f MiB)", light_table_.capacity,
                static_cast<double>(light_table_.buffer.size()) / (1024.0 * 1024.0));
}

void Scene::RecordLightRebuild(VkCommandBuffer cmd, uint64_t frame)
{
    const VkBuffer table = light_table_.buffer.buffer();
    // Count and dropped start at 0 (the capacity and total are written by the scan).
    vkCmdFillBuffer(cmd, table, 0, 16, 0);
    MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    LightPush push{};
    push.group_infos = group_infos_.address();
    push.slot_tables = slot_tables_.address();
    push.page_table = page_table_.address();
    push.light_words = light_words_.address();
    push.light_indices = light_indices_.address();
    push.materials = material_buffer_.address();
    push.group_count = static_cast<uint32_t>(groups_.size());
    push.max_bricks = light_table_.capacity;
    push.records_per_page_shift = records_per_page_shift();
    push.record_bytes = layout_.record_bytes;
    push.group_shift = group_shift_;
    push.encoding = encoding_;
    push.min_emission = light_min_emission_;
    // Both kernels have the same pipeline layout: the set stays bound across the two pipelines.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, light_compact_.layout(), 0, 1, &light_table_.set,
                            0, nullptr);
    for (uint32_t base = 0; base < push.group_count; base += kMaxDispatchGroups)
    {
        push.group_base = base;
        light_compact_.Dispatch(cmd, &push, sizeof(push), std::min(kMaxDispatchGroups, push.group_count - base));
    }
    MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    push.group_base = 0;
    light_scan_.Dispatch(cmd, &push, sizeof(push), 1);

    // The header's first 16 bytes -> the readback slot of this frame (CaelixLightStats).
    MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    const uint32_t slot = light_readback_next_;
    light_readback_next_ = (light_readback_next_ + 1) % kLightReadbackSlots;
    const VkBufferCopy region{0, static_cast<VkDeviceSize>(slot) * kLightReadbackBytes, kLightReadbackBytes};
    vkCmdCopyBuffer(cmd, table, light_readback_.buffer(), 1, &region);
    MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    light_readback_frames_[slot] = frame;
    light_table_built_ = true;
    lights_dirty_ = false;
    ++light_rebuilds_;
}

void Scene::RecycleLights()
{
    // The newest rebuild whose frame has completed.
    uint32_t newest = kLightReadbackSlots;
    for (uint32_t slot = 0; slot < kLightReadbackSlots; ++slot)
    {
        const uint64_t frame = light_readback_frames_[slot];
        if (frame != 0 && frame <= safe_frame_ && frame > light_stats_frame_ &&
            (newest == kLightReadbackSlots || frame > light_readback_frames_[newest]))
        {
            newest = slot;
        }
    }
    if (newest != kLightReadbackSlots)
    {
        const VkDeviceSize offset = static_cast<VkDeviceSize>(newest) * kLightReadbackBytes;
        light_readback_.InvalidateHostReads(offset, kLightReadbackBytes);
        uint32_t header[4];
        std::memcpy(header, light_readback_.mapped() + offset, sizeof(header));
        light_stats_count_ = header[0];
        light_stats_dropped_ = header[1];
        light_stats_frame_ = light_readback_frames_[newest];
    }
    for (RetiredLightTable& retired : retired_light_tables_)
    {
        if (retired.frame <= safe_frame_)
        {
            DestroyLightTable(retired.table);
        }
    }
    std::erase_if(retired_light_tables_,
                  [](const RetiredLightTable& retired) { return !retired.table.buffer.valid(); });
}

}  // namespace caelix
