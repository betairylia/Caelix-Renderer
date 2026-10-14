#include "rhi/alloc/record_pool.h"

#include "core/log.h"

#include <bit>
#include <new>

namespace caelix {

RecordPool::RecordPool(VmaAllocator allocator, VkDevice device, uint32_t records_per_page,
                       uint32_t record_bytes)
    : allocator_(allocator),
      device_(device),
      records_per_page_(records_per_page),
      page_shift_(static_cast<uint32_t>(std::countr_zero(records_per_page))),
      record_bytes_(record_bytes) {}

uint32_t RecordPool::Allocate() {
  // 1. A free slot of any page.
  while (!pages_with_free_.empty()) {
    Page& page = pages_[pages_with_free_.back()];
    if (page.free_slots.empty()) {
      page.in_free_list = false;
      pages_with_free_.pop_back();
      continue;
    }
    const uint32_t slot = page.free_slots.back();
    page.free_slots.pop_back();
    return (pages_with_free_.back() << page_shift_) | slot;
  }
  // 2. Bump the high water of the last page.
  if (!pages_.empty() && pages_.back().high_water < records_per_page_) {
    const uint32_t page = static_cast<uint32_t>(pages_.size() - 1);
    return (page << page_shift_) | pages_.back().high_water++;
  }
  // 3. A new page. Its last index must stay below kInvalidIndex.
  const uint64_t page = pages_.size();
  if ((((page + 1) << page_shift_) - 1) >= kInvalidIndex) {
    CAELIX_LOGE("record pool: the 32-bit record index space is full");
    throw std::bad_alloc();
  }
  Page new_page;
  new_page.buffer = GpuBuffer::DeviceLocal(allocator_, device_, page_bytes());
  new_page.high_water = 1;
  pages_.push_back(std::move(new_page));
  CAELIX_LOGD("record pool: page %llu created (%llu bytes)",
              static_cast<unsigned long long>(page),
              static_cast<unsigned long long>(page_bytes()));
  return static_cast<uint32_t>(page << page_shift_);
}

void RecordPool::Free(uint32_t index) {
  const uint32_t page_index = page_of(index);
  Page& page = pages_[page_index];
  page.free_slots.push_back(slot_of(index));
  if (!page.in_free_list) {
    page.in_free_list = true;
    pages_with_free_.push_back(page_index);
  }
}

}  // namespace caelix
