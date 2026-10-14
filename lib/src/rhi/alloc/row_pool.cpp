#include "rhi/alloc/row_pool.h"

#include "core/log.h"

#include <bit>
#include <new>
#include <utility>

namespace caelix {

GrowableBuffer::GrowableBuffer(VmaAllocator allocator, VkDevice device,
                               VkDeviceSize initial_bytes, VkBufferUsageFlags extra_usage)
    : allocator_(allocator), device_(device), extra_usage_(extra_usage) {
  buffer_ = GpuBuffer::DeviceLocal(allocator_, device_, initial_bytes, extra_usage_);
}

void GrowableBuffer::Reserve(VkDeviceSize bytes) {
  VkDeviceSize new_bytes = buffer_.size();
  if (new_bytes >= bytes) {
    return;
  }
  while (new_bytes < bytes) {
    new_bytes *= 2;
  }
  GpuBuffer grown = GpuBuffer::DeviceLocal(allocator_, device_, new_bytes, extra_usage_);
  if (!growth_source_.has_value()) {
    // buffer_ holds the GPU-valid contents; keep it as the copy source.
    growth_source_ = std::move(buffer_);
  }
  // Else buffer_ was created after the last flush, so no recorded command uses
  // it: it can be destroyed now, and growth_source_ stays the copy source.
  buffer_ = std::move(grown);
}

std::optional<GpuBuffer> GrowableBuffer::TakeGrowthSource() {
  std::optional<GpuBuffer> source = std::move(growth_source_);
  growth_source_.reset();
  return source;
}

RowPool::RowPool(VmaAllocator allocator, VkDevice device, uint32_t row_bytes, uint32_t min_rows,
                 uint32_t initial_rows, const char* name, VkBufferUsageFlags extra_usage)
    : storage_(allocator, device, static_cast<VkDeviceSize>(initial_rows) * row_bytes,
               extra_usage),
      name_(name),
      row_bytes_(row_bytes),
      min_rows_(min_rows),
      capacity_rows_(initial_rows) {}

RowRange RowPool::Allocate(uint32_t rows) {
  const uint32_t size = std::bit_ceil(rows < min_rows_ ? min_rows_ : rows);
  const uint32_t size_class = static_cast<uint32_t>(std::countr_zero(size / min_rows_));
  if (size_class < free_by_class_.size() && !free_by_class_[size_class].empty()) {
    const uint32_t offset = free_by_class_[size_class].back();
    free_by_class_[size_class].pop_back();
    return RowRange{offset, size};
  }
  if (bump_rows_ + size > capacity_rows_) {
    uint64_t new_rows = capacity_rows_;
    while (new_rows < bump_rows_ + size) {
      new_rows *= 2;
    }
    // Row offsets are uint32_t.
    if (new_rows > (uint64_t{1} << 32)) {
      CAELIX_LOGE("%s pool: row offsets would exceed 32 bits", name_);
      throw std::bad_alloc();
    }
    storage_.Reserve(static_cast<VkDeviceSize>(new_rows) * row_bytes_);
    CAELIX_LOGD("%s pool: grew from %llu to %llu rows", name_,
                static_cast<unsigned long long>(capacity_rows_),
                static_cast<unsigned long long>(new_rows));
    capacity_rows_ = new_rows;
  }
  // Make the free list of this class exist now, so Free() can index it directly.
  if (size_class >= free_by_class_.size()) {
    free_by_class_.resize(size_class + 1);
  }
  const RowRange range{static_cast<uint32_t>(bump_rows_), size};
  bump_rows_ += size;
  return range;
}

void RowPool::Free(RowRange range) {
  if (!range.valid()) {
    return;
  }
  const uint32_t size_class = static_cast<uint32_t>(std::countr_zero(range.rows / min_rows_));
  free_by_class_[size_class].push_back(range.offset);
}

}  // namespace caelix
