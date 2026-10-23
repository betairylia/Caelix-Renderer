#pragma once
#include "rhi/alloc/row_pool.h"
#include "scene/gpu_structs.h"

#include <cstdint>

namespace caelix {

// All slot tables of a scene live in one RowPool of 16-byte rows. Every group
// has a power-of-two range of rows that doubles when its dense ids run out; the
// AABB rows of a group (24 bytes, same capacity) follow the same ranges.
using SlotRange = RowRange;
using SlotTablePool = RowPool;

constexpr uint32_t kSlotMinRows = 16;
constexpr uint32_t kSlotInitialRows = 1u << 16;  // 1 MiB of slot rows

inline SlotTablePool MakeSlotTablePool(VmaAllocator allocator, VkDevice device)
{
    return SlotTablePool(allocator, device, kSlotRowBytes, kSlotMinRows, kSlotInitialRows,
                         "slot table");
}

}  // namespace caelix
