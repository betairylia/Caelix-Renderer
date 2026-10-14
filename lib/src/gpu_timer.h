#pragma once
#include "device.h"

#include <array>
#include <cstdint>

namespace caelix {

// GPU time of one recorded span per frame, measured with two timestamp queries.
//
// The queries live in a ring of kSlots pairs. Begin() takes the next pair for a
// frame: when that pair still belongs to a frame that is not safe yet, the span
// of this frame is not measured (Begin returns false). Resolve(safe) reads every
// pair whose frame is <= safe; the results stay available for kHistory frames.
// The queries are reset with vkCmdResetQueryPool in the recording command
// buffer, so no host query reset feature is needed. Without timestamp support
// on the queue (valid bits 0) every call does nothing.
class GpuTimer {
 public:
  static constexpr uint32_t kSlots = 8;
  static constexpr uint32_t kHistory = 16;

  GpuTimer() = default;
  // Throws VulkanError when the query pool cannot be created.
  explicit GpuTimer(const Device& device);
  ~GpuTimer();
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;
  GpuTimer(GpuTimer&& other) noexcept;
  GpuTimer& operator=(GpuTimer&& other) noexcept;

  // Starts the span of `frame`: resets the slot's two queries and writes the
  // first timestamp (ALL_COMMANDS: written when every earlier command has
  // completed). Returns false when nothing was recorded.
  bool Begin(VkCommandBuffer cmd, uint64_t frame, uint64_t safe_frame);
  // Writes the second timestamp of the span started by the last Begin() that
  // returned true.
  void End(VkCommandBuffer cmd);
  // Reads back the spans of every frame <= safe_frame.
  void Resolve(uint64_t safe_frame);

  // The milliseconds of `frame` when it was measured and resolved.
  bool MsForFrame(uint64_t frame, float& ms) const;
  // The newest resolved frame (0 = none) and its milliseconds.
  uint64_t latest_frame() const { return latest_frame_; }
  float latest_ms() const { return latest_ms_; }

 private:
  struct Slot {
    uint64_t frame = 0;  // frame of the recorded span
    bool pending = false;
  };
  struct Result {
    uint64_t frame = 0;
    float ms = 0.0f;
  };

  void Reset();

  VkDevice device_ = VK_NULL_HANDLE;
  VkQueryPool pool_ = VK_NULL_HANDLE;
  float period_ns_ = 1.0f;
  uint64_t mask_ = 0;
  std::array<Slot, kSlots> slots_{};
  uint32_t next_slot_ = 0;
  uint32_t open_slot_ = kSlots;  // slot of the span between Begin and End
  std::array<Result, kHistory> history_{};
  uint32_t history_next_ = 0;
  uint64_t latest_frame_ = 0;
  float latest_ms_ = 0.0f;
};

}  // namespace caelix
