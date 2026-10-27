#pragma once
#include "rhi/gpu_buffer.h"

#include <cstdint>
#include <vector>

namespace caelix {

// GPU storage for brick records: a list of device-local page buffers of
// `records_per_page` records each. A record index is global:
// `page << page_shift | slot`. Records never move; pages are only added, never
// freed before the pool is destroyed.
//
// Free(index) makes the record reusable IMMEDIATELY, before the GPU has stopped
// reading it. That is safe because all GPU work on the pool runs on one queue in
// submission order, and every flush starts with a barrier from all earlier
// commands (shader reads included) to the writes of the record scatter kernel that
// overwrite a reused record. An async transfer queue would break this assumption: frees would then
// have to wait until the frame that last read the record has completed.
class RecordPool
{
public:
    static constexpr uint32_t kInvalidIndex = 0xFFFFFFFFu;

    RecordPool() = default;
    RecordPool(VmaAllocator allocator, VkDevice device, uint32_t records_per_page,
               uint32_t record_bytes);

    // Returns a free record index; creates a page when every page is full.
    // Throws VulkanError (or std::bad_alloc) when a page cannot be created.
    uint32_t Allocate();
    void Free(uint32_t index);

    uint32_t page_of(uint32_t index) const
    {
        return index >> page_shift_;
    }
    uint32_t slot_of(uint32_t index) const
    {
        return index & (records_per_page_ - 1);
    }

    VkBuffer PageBuffer(uint32_t index) const
    {
        return pages_[page_of(index)].buffer.buffer();
    }
    VkDeviceSize PageOffset(uint32_t index) const
    {
        return static_cast<VkDeviceSize>(slot_of(index)) * record_bytes_;
    }
    VkDeviceAddress PageAddress(uint32_t page) const
    {
        return pages_[page].buffer.address();
    }

    uint32_t page_count() const
    {
        return static_cast<uint32_t>(pages_.size());
    }
    uint64_t page_bytes() const
    {
        return static_cast<uint64_t>(records_per_page_) * record_bytes_;
    }
    uint64_t total_bytes() const
    {
        return page_bytes() * pages_.size();
    }

private:
    struct Page
    {
        GpuBuffer buffer;
        std::vector<uint32_t> free_slots;  // LIFO
        uint32_t high_water = 0;           // slots [0, high_water) were handed out at least once
        bool in_free_list = false;         // listed in pages_with_free_
    };

    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t records_per_page_ = 0;
    uint32_t page_shift_ = 0;
    uint32_t record_bytes_ = 0;
    std::vector<Page> pages_;
    // Pages that may have free slots. An entry can be stale (its list emptied);
    // Allocate drops stale entries as it finds them.
    std::vector<uint32_t> pages_with_free_;
};

}  // namespace caelix
