#pragma once
#include "gpu_buffer.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace caelix {

// One slot table row: 16 bytes, one per dense brick id of a render group.
struct SlotRow {
  uint32_t pos_state;     // bits 0..23 group-local position, bits 24..31 SlotState
  uint32_t summary;       // 0 for now
  uint32_t record_index;  // RecordPool index, 0xFFFFFFFF when not resident
  uint32_t last_used;     // 0 for now
};
static_assert(sizeof(SlotRow) == 16);

enum SlotState : uint8_t {
  kSlotFree = 0,
  kSlotResident = 1,
  kSlotSummaryOnly = 2,
  kSlotRequested = 3,
};

// A range of rows in the slot table buffer.
struct SlotRange {
  uint32_t offset = 0;  // in rows
  uint32_t rows = 0;    // power of two, >= SlotTablePool::kMinRows
  bool valid() const { return rows != 0; }
};

// All slot tables of a scene in one device-local buffer.
//
// Ranges come from power-of-two size classes (in rows, minimum kMinRows), with
// one LIFO free list per class and a bump pointer at the end. Freed ranges are
// never merged or split: a range freed in one class is reused only by that
// class. This keeps allocation O(1) and offsets stable; the cost is some
// fragmentation when groups shrink and other groups grow.
//
// The buffer grows by doubling. Growth allocates the new buffer at once; the
// copy of the old contents (same offsets, so no range changes) is recorded at
// the next flush, before any other copy, and the old buffer is destroyed after
// that flush's frame has completed.
class SlotTablePool {
 public:
  static constexpr uint32_t kMinRows = 16;
  static constexpr uint32_t kRowBytes = sizeof(SlotRow);
  static constexpr uint32_t kInitialRows = 1u << 16;  // 1 MiB

  SlotTablePool() = default;
  SlotTablePool(VmaAllocator allocator, VkDevice device);

  // Throws VulkanError (or std::bad_alloc) when the buffer cannot grow. On
  // throw, the pool is unchanged.
  SlotRange Allocate(uint32_t rows);
  void Free(SlotRange range);

  VkBuffer buffer() const { return buffer_.buffer(); }
  VkDeviceAddress address() const { return buffer_.address(); }
  uint64_t bytes() const { return buffer_.size(); }

  // The buffer whose contents must be copied into buffer() at the next flush,
  // if the pool grew since the last flush. The caller records a copy of
  // `growth_source->size()` bytes at offset 0 and keeps the source alive until
  // that frame has completed.
  std::optional<GpuBuffer> TakeGrowthSource();

 private:
  void Grow(uint64_t min_rows);

  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  GpuBuffer buffer_;
  // Holds the GPU-valid contents while a growth copy is pending.
  std::optional<GpuBuffer> growth_source_;
  uint64_t capacity_rows_ = 0;
  uint64_t bump_rows_ = 0;
  std::vector<std::vector<uint32_t>> free_by_class_;  // offsets, LIFO
};

}  // namespace caelix
