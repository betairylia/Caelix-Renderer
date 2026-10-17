#include "rhi/gpu_timer.h"

#include "core/log.h"
#include "rhi/gpu_buffer.h"

#include <utility>

namespace caelix {

GpuTimer::GpuTimer(const Device& device)
    : device_(device.device())
{
    const uint32_t bits = device.timestamp_valid_bits();
    if (bits == 0)
    {
        return;
    }
    period_ns_ = device.timestamp_period_ns();
    mask_ = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = 2 * kSlots;
    const VkResult result = vkCreateQueryPool(device_, &info, nullptr, &pool_);
    if (result != VK_SUCCESS)
    {
        pool_ = VK_NULL_HANDLE;
        CAELIX_LOGE("vkCreateQueryPool failed: %d", static_cast<int>(result));
        throw VulkanError("vkCreateQueryPool failed", result);
    }
}

GpuTimer::~GpuTimer()
{
    Reset();
}

GpuTimer::GpuTimer(GpuTimer&& other) noexcept
{
    *this = std::move(other);
}

GpuTimer& GpuTimer::operator=(GpuTimer&& other) noexcept
{
    if (this != &other)
    {
        Reset();
        device_ = std::exchange(other.device_, VK_NULL_HANDLE);
        pool_ = std::exchange(other.pool_, VK_NULL_HANDLE);
        period_ns_ = other.period_ns_;
        mask_ = other.mask_;
        slots_ = other.slots_;
        next_slot_ = other.next_slot_;
        open_slot_ = other.open_slot_;
        history_ = other.history_;
        history_next_ = other.history_next_;
        latest_frame_ = other.latest_frame_;
        latest_ms_ = other.latest_ms_;
    }
    return *this;
}

void GpuTimer::Reset()
{
    if (pool_ != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(device_, pool_, nullptr);
    }
    pool_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

bool GpuTimer::Begin(VkCommandBuffer cmd, uint64_t frame, uint64_t safe_frame)
{
    open_slot_ = kSlots;
    if (pool_ == VK_NULL_HANDLE)
    {
        return false;
    }
    Resolve(safe_frame);
    const uint32_t slot = next_slot_;
    if (slots_[slot].pending)
    {
        return false;  // its frame is still in flight: this span is not measured
    }
    next_slot_ = (next_slot_ + 1) % kSlots;
    slots_[slot] = Slot{frame, true};
    open_slot_ = slot;
    vkCmdResetQueryPool(cmd, pool_, 2 * slot, 2);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool_, 2 * slot);
    return true;
}

void GpuTimer::End(VkCommandBuffer cmd)
{
    if (open_slot_ >= kSlots)
    {
        return;
    }
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool_, 2 * open_slot_ + 1);
    open_slot_ = kSlots;
}

void GpuTimer::Resolve(uint64_t safe_frame)
{
    if (pool_ == VK_NULL_HANDLE)
    {
        return;
    }
    for (uint32_t slot = 0; slot < kSlots; ++slot)
    {
        Slot& entry = slots_[slot];
        if (!entry.pending || entry.frame > safe_frame || slot == open_slot_)
        {
            continue;
        }
        entry.pending = false;
        uint64_t ticks[2] = {};
        // The frame has completed, so both queries are available: no WAIT flag.
        const VkResult result =
            vkGetQueryPoolResults(device_, pool_, 2 * slot, 2, sizeof(ticks), ticks, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT);
        if (result != VK_SUCCESS)
        {
            CAELIX_LOGW("vkGetQueryPoolResults of frame %llu returned %d; the GPU time is dropped",
                        static_cast<unsigned long long>(entry.frame), static_cast<int>(result));
            continue;
        }
        const uint64_t delta = (ticks[1] - ticks[0]) & mask_;
        const float ms = static_cast<float>(static_cast<double>(delta) * period_ns_ * 1e-6);
        history_[history_next_] = Result{entry.frame, ms};
        history_next_ = (history_next_ + 1) % kHistory;
        if (entry.frame >= latest_frame_)
        {
            latest_frame_ = entry.frame;
            latest_ms_ = ms;
        }
    }
}

bool GpuTimer::MsForFrame(uint64_t frame, float& ms) const
{
    for (const Result& result : history_)
    {
        if (result.frame == frame && frame != 0)
        {
            ms = result.ms;
            return true;
        }
    }
    return false;
}

}  // namespace caelix
