// The light cells of one renderer: the pool, the per-frame maintenance and fill, the controller and
// the stats.
#include "render/light_cells.h"

#include "core/log.h"
#include "rhi/shader_loader.h"

#include <algorithm>

namespace caelix {
namespace {

// The controller (docs/LIGHT_NEE.md section 3.5): the footprint scale shrinks by kShrink when the live
// cells pass kHighFill x capacity or inserts found no payload, grows by kGrow below kLowFill x capacity,
// and stays in [kMinScale, 1] (it only coarsens: never above the configured footprint).
constexpr float kHighFill = 0.85f;
constexpr float kLowFill = 0.70f;
constexpr float kShrink = 0.9f;
constexpr float kGrow = 1.1f;
constexpr float kMinScale = 0.25f;

}  // namespace

LightCells::LightCells(Device& device, uint32_t payload_capacity)
    : device_(device)
{
    HashTable::Desc desc;
    desc.name = "light cells";
    desc.payload_capacity = payload_capacity;
    desc.payload_stride = kLightCellBytes;
    desc.key_words = kLightCellKeyWords;
    table_ = std::make_unique<HashTable>(device, desc);
    payloads_ = GpuBuffer::DeviceLocal(device.allocator(), device.device(),
                                       uint64_t{payload_capacity} * kLightCellBytes);
    fill_ = ComputePipeline(device.device(), LoadShader("caelix_light_cell_fill"), sizeof(LightCellFillPush),
                            "caelix_light_cell_fill");
    fill_timer_ = GpuTimer(device);
    CAELIX_LOGI("light cells: %u cells, %llu bytes", payload_capacity,
                static_cast<unsigned long long>(bytes()));
}

void LightCells::Record(VkCommandBuffer cmd, VkDeviceAddress light_tree, const Settings& settings,
                        uint32_t frame_seed, uint64_t frame_number, uint64_t safe_frame)
{
    ++frame_;
    if (!payloads_cleared_)
    {
        // Every word of a new pool is 0 (state free): the reservoirs and headers of a cell are written
        // by the fill before the trace reads them, the zeros only keep the memory defined.
        vkCmdFillBuffer(cmd, payloads_.buffer(), 0, VK_WHOLE_SIZE, 0);
        payloads_cleared_ = true;
    }
    // Cells per fill workgroup (caelix_light_cell_fill): 1 in mode "off", kLightCellRotation in "rotate".
    const uint32_t cells_per_group =
        settings.reservoir_mode == kLightCellModeRotate ? kLightCellRotation : 1u;
    table_->RecordMaintenance(cmd, payloads_.address(), frame_, settings.max_age, cells_per_group,
                              frame_number, safe_frame);

    LightCellFillPush push{};
    push.payloads = payloads_.address();
    push.active = table_->active_address();
    push.dispatch = table_->dispatch_address();
    push.counters = table_->counters_address();
    push.light_tree = light_tree;
    push.frame_seed = frame_seed;
    push.mode = settings.reservoir_mode;
    push.candidates = settings.candidates;
    last_mode_ = settings.reservoir_mode;
    last_candidates_ = settings.candidates;
    const bool timed = fill_timer_.Begin(cmd, frame_number, safe_frame);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fill_.pipeline());
    vkCmdPushConstants(cmd, fill_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatchIndirect(cmd, table_->dispatch_buffer(), 0);
    if (timed)
    {
        fill_timer_.End(cmd);
    }
    // The fill's cells -> the trace's reads, lookups and inserts.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

void LightCells::RecordReadback(VkCommandBuffer cmd, uint64_t frame_number)
{
    table_->RecordReadback(cmd, frame_number);
}

void LightCells::Resolve(uint64_t safe_frame, const Settings& settings)
{
    table_->Resolve(safe_frame);
    fill_timer_.Resolve(safe_frame);
    const uint64_t counters_frame = table_->counters_frame();
    if ((settings.flags & CAELIX_LIGHT_NEE_FLAG_NO_CONTROLLER) != 0)
    {
        scale_ = 1.0f;
        return;
    }
    if (counters_frame == 0 || counters_frame < controller_frame_ + kControllerFrames)
    {
        return;
    }
    controller_frame_ = counters_frame;
    const HashCounters& c = table_->counters();
    const float capacity = static_cast<float>(table_->payload_capacity());
    const float before = scale_;
    if (static_cast<float>(c.live) > kHighFill * capacity || c.failed_payload > 0)
    {
        scale_ = std::max(kMinScale, scale_ * kShrink);
    }
    else if (static_cast<float>(c.live) < kLowFill * capacity)
    {
        scale_ = std::min(1.0f, scale_ * kGrow);
    }
    if (scale_ != before)
    {
        CAELIX_LOGD("light cells: footprint scale %.3f -> %.3f (live %u of %u, failed payload %u)",
                    static_cast<double>(before), static_cast<double>(scale_), c.live,
                    table_->payload_capacity(), c.failed_payload);
    }
}

void LightCells::GetStats(CaelixLightNeeStats& out) const
{
    const HashCounters& c = table_->counters();
    out.live_cells = c.live;
    out.active_cells = c.active;
    out.inserted = c.inserted;
    out.pending = c.pending;
    out.failed_slot = c.failed_slot;
    out.failed_payload = c.failed_payload;
    out.evicted = c.evicted;
    out.rebuild_dropped = c.rebuild_dropped;
    out.fresh_draws = c.user[(kLcCounterFreshDraws - kHashCounterUser) / 4];
    out.normalisation_draws = c.user[(kLcCounterNormDraws - kHashCounterUser) / 4];
    out.stale_refreshed = c.user[(kLcCounterStaleRefreshed - kHashCounterUser) / 4];
    out.cell_capacity = table_->payload_capacity();
    out.reservoir_mode = last_mode_;
    out.candidates = last_candidates_;
    out.controller_scale = scale_;
    out.maintenance_ms = table_->maintenance_ms();
    out.fill_ms = fill_timer_.latest_ms();
    out.frame_number = table_->counters_frame();
    out.cell_bytes = bytes();
}

}  // namespace caelix
