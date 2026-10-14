#pragma once
#include "gpu_buffer.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace caelix {

// A device-local buffer that grows by doubling.
//
// Growth allocates the new buffer at once; the copy of the old contents (same
// offsets) is recorded by the owner at the next flush: TakeGrowthSource()
// returns the buffer that holds the GPU-valid contents, the owner records a copy
// of its whole size into buffer() and keeps it alive until that frame has
// completed. A buffer that does not need its old contents (for example a list
// that is rewritten every flush) just keeps the source alive.
class GrowableBuffer {
 public:
  GrowableBuffer() = default;
  GrowableBuffer(VmaAllocator allocator, VkDevice device, VkDeviceSize initial_bytes);

  // Makes the buffer hold at least `bytes`. Throws VulkanError (or
  // std::bad_alloc); the buffer is unchanged then.
  void Reserve(VkDeviceSize bytes);

  VkBuffer buffer() const { return buffer_.buffer(); }
  VkDeviceAddress address() const { return buffer_.address(); }
  uint64_t bytes() const { return buffer_.size(); }

  // The buffer whose contents must be copied into buffer() at the next flush, if
  // the buffer grew since the last call.
  std::optional<GpuBuffer> TakeGrowthSource();

 private:
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  GpuBuffer buffer_;
  // Holds the GPU-valid contents while a growth copy is pending.
  std::optional<GpuBuffer> growth_source_;
};

// A range of rows in a RowPool.
struct RowRange {
  uint32_t offset = 0;  // in rows
  uint32_t rows = 0;    // power of two, >= the pool's minimum
  bool valid() const { return rows != 0; }
  bool operator==(const RowRange&) const = default;
};

// Ranges of fixed-size rows in one GrowableBuffer.
//
// Ranges come from power-of-two size classes (in rows, minimum `min_rows`),
// with one LIFO free list per class and a bump pointer at the end. Freed ranges
// are never merged or split: a range freed in one class is reused only by that
// class. This keeps allocation O(1) and offsets stable; the cost is some
// fragmentation when ranges shrink and others grow. Growth of the buffer keeps
// every offset (see GrowableBuffer).
class RowPool {
 public:
  RowPool() = default;
  // `min_rows` and `initial_rows` are powers of two, initial_rows >= min_rows.
  RowPool(VmaAllocator allocator, VkDevice device, uint32_t row_bytes, uint32_t min_rows,
          uint32_t initial_rows, const char* name);

  // Throws VulkanError (or std::bad_alloc) when the buffer cannot grow. On
  // throw, the pool is unchanged.
  RowRange Allocate(uint32_t rows);
  void Free(RowRange range);

  uint32_t row_bytes() const { return row_bytes_; }
  uint32_t min_rows() const { return min_rows_; }
  VkBuffer buffer() const { return storage_.buffer(); }
  VkDeviceAddress address() const { return storage_.address(); }
  uint64_t bytes() const { return storage_.bytes(); }
  VkDeviceSize ByteOffset(uint32_t row) const {
    return static_cast<VkDeviceSize>(row) * row_bytes_;
  }
  std::optional<GpuBuffer> TakeGrowthSource() { return storage_.TakeGrowthSource(); }

 private:
  GrowableBuffer storage_;
  const char* name_ = "";
  uint32_t row_bytes_ = 0;
  uint32_t min_rows_ = 0;
  uint64_t capacity_rows_ = 0;
  uint64_t bump_rows_ = 0;
  std::vector<std::vector<uint32_t>> free_by_class_;  // offsets, LIFO
};

}  // namespace caelix
