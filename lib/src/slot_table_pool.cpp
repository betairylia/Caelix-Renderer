#include "slot_table_pool.h"

#include "log.h"

#include <bit>
#include <new>
#include <utility>

namespace caelix {

SlotTablePool::SlotTablePool(VmaAllocator allocator, VkDevice device)
    : allocator_(allocator), device_(device) {
  buffer_ = GpuBuffer::DeviceLocal(allocator_, device_,
                                   static_cast<VkDeviceSize>(kInitialRows) * kRowBytes);
  capacity_rows_ = kInitialRows;
}

SlotRange SlotTablePool::Allocate(uint32_t rows) {
  const uint32_t size = std::bit_ceil(rows < kMinRows ? kMinRows : rows);
  const uint32_t size_class = static_cast<uint32_t>(std::countr_zero(size / kMinRows));
  if (size_class < free_by_class_.size() && !free_by_class_[size_class].empty()) {
    const uint32_t offset = free_by_class_[size_class].back();
    free_by_class_[size_class].pop_back();
    return SlotRange{offset, size};
  }
  // Make the free list of this class exist now, so Free() can index it directly.
  if (size_class >= free_by_class_.size()) {
    free_by_class_.resize(size_class + 1);
  }
  if (bump_rows_ + size > capacity_rows_) {
    Grow(bump_rows_ + size);
  }
  const SlotRange range{static_cast<uint32_t>(bump_rows_), size};
  bump_rows_ += size;
  return range;
}

void SlotTablePool::Free(SlotRange range) {
  if (!range.valid()) {
    return;
  }
  const uint32_t size_class = static_cast<uint32_t>(std::countr_zero(range.rows / kMinRows));
  free_by_class_[size_class].push_back(range.offset);
}

void SlotTablePool::Grow(uint64_t min_rows) {
  uint64_t new_rows = capacity_rows_;
  while (new_rows < min_rows) {
    new_rows *= 2;
  }
  // Row offsets are uint32_t.
  if (new_rows > (uint64_t{1} << 32)) {
    CAELIX_LOGE("slot table pool: row offsets would exceed 32 bits");
    throw std::bad_alloc();
  }
  GpuBuffer grown =
      GpuBuffer::DeviceLocal(allocator_, device_, static_cast<VkDeviceSize>(new_rows) * kRowBytes);
  if (!growth_source_.has_value()) {
    // buffer_ holds the GPU-valid contents; keep it as the copy source.
    growth_source_ = std::move(buffer_);
  }
  // Else buffer_ was created after the last flush, so no recorded command uses
  // it: it can be destroyed now, and growth_source_ stays the copy source.
  buffer_ = std::move(grown);
  CAELIX_LOGD("slot table pool: grew from %llu to %llu rows",
              static_cast<unsigned long long>(capacity_rows_),
              static_cast<unsigned long long>(new_rows));
  capacity_rows_ = new_rows;
}

std::optional<GpuBuffer> SlotTablePool::TakeGrowthSource() {
  std::optional<GpuBuffer> source = std::move(growth_source_);
  growth_source_.reset();
  return source;
}

}  // namespace caelix
