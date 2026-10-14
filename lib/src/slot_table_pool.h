#pragma once
#include "row_pool.h"

#include <cstdint>

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

// All slot tables of a scene live in one RowPool of 16-byte rows. Every group
// has a power-of-two range of rows that doubles when its dense ids run out; the
// AABB rows of a group (24 bytes, same capacity) follow the same ranges.
using SlotRange = RowRange;
using SlotTablePool = RowPool;

constexpr uint32_t kSlotRowBytes = sizeof(SlotRow);
constexpr uint32_t kSlotMinRows = 16;
constexpr uint32_t kSlotInitialRows = 1u << 16;  // 1 MiB of slot rows

inline SlotTablePool MakeSlotTablePool(VmaAllocator allocator, VkDevice device) {
  return SlotTablePool(allocator, device, kSlotRowBytes, kSlotMinRows, kSlotInitialRows,
                       "slot table");
}

}  // namespace caelix
