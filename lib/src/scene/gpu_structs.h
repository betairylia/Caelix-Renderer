#pragma once
#include <cstddef>
#include <cstdint>

namespace caelix {

// CPU mirrors of the tables that shaders/scene/caelix_finalize.hlsl reads. The
// HLSL side is shaders/common/caelix_record.hlsl; change both together.

// One slot table row: 16 bytes, one per dense brick id of a render group (slot_table_pool.h).
struct SlotRow
{
    uint32_t pos_state;     // bits 0..23 group-local position, bits 24..31 SlotState
    uint32_t summary;       // 0 for now
    uint32_t record_index;  // RecordPool index, kInvalidRecord when not resident
    uint32_t last_used;     // 0 for now
};
constexpr uint32_t kSlotRowBytes = 16;
static_assert(sizeof(SlotRow) == kSlotRowBytes);
constexpr uint32_t kSlotRowPosMask = 0x00FFFFFFu;  // SlotRow::pos_state
constexpr uint32_t kSlotRowStateShift = 24;
constexpr uint32_t kInvalidRecord = 0xFFFFFFFFu;  // = RecordPool::kInvalidIndex

enum SlotState : uint8_t
{
    kSlotFree = 0,
    kSlotResident = 1,
    kSlotSummaryOnly = 2,
    kSlotRequested = 3,
};

// One brick to finalize: one workgroup each. 16 bytes.
struct DirtyEntry
{
    uint32_t group_index;
    uint32_t id;
    uint32_t record_index;
    uint32_t pad;
};
constexpr uint32_t kDirtyEntryBytes = 16;
static_assert(sizeof(DirtyEntry) == kDirtyEntryBytes);

// One render group, indexed by the scene-wide group index. 32 bytes.
struct GroupInfo
{
    int32_t coord[3];  // group coordinate = brick key >> group_shift
    uint32_t entity;
    uint32_t slot_row_base;  // first slot table row
    uint32_t reverse_base;   // first u16 entry of the reverse table
    uint32_t aabb_row_base;  // first AABB row
    uint32_t high_water;     // dense ids in [0, high_water) were handed out
};
// Named offsets (shaders/common/caelix_record.hlsl).
constexpr uint32_t kGroupInfoCoord = 0;
constexpr uint32_t kGroupInfoEntity = 12;
constexpr uint32_t kGroupInfoSlotRowBase = 16;
constexpr uint32_t kGroupInfoReverseBase = 20;
constexpr uint32_t kGroupInfoAabbRowBase = 24;
constexpr uint32_t kGroupInfoHighWater = 28;
constexpr uint32_t kGroupInfoBytes = 32;
static_assert(offsetof(GroupInfo, coord) == kGroupInfoCoord);
static_assert(offsetof(GroupInfo, entity) == kGroupInfoEntity);
static_assert(offsetof(GroupInfo, slot_row_base) == kGroupInfoSlotRowBase);
static_assert(offsetof(GroupInfo, reverse_base) == kGroupInfoReverseBase);
static_assert(offsetof(GroupInfo, aabb_row_base) == kGroupInfoAabbRowBase);
static_assert(offsetof(GroupInfo, high_water) == kGroupInfoHighWater);
static_assert(sizeof(GroupInfo) == kGroupInfoBytes);

// One entity, indexed by the entity id. 128 bytes. The finalize kernel reads the
// first 32 bytes; the transform rows are for the trace kernels.
struct EntityInfo
{
    int32_t grid_min[3];   // group coordinate of grid cell (0, 0, 0)
    uint32_t grid_base;    // first u32 cell of the group grid
    int32_t grid_dims[3];  // cells per axis, powers of two; 0 = no grid
    uint32_t pad;
    // Object-to-world rows (3x4, row-major: rotation | position) of the current
    // transform, then of the previous frame's transform. prev differs from cur for
    // exactly one flush after a transform change.
    float cur_rows[3][4];
    float prev_rows[3][4];
};
constexpr uint32_t kEntityInfoGridMin = 0;
constexpr uint32_t kEntityInfoGridBase = 12;
constexpr uint32_t kEntityInfoGridDims = 16;
constexpr uint32_t kEntityInfoCurRow0 = 32;
constexpr uint32_t kEntityInfoCurRow1 = 48;
constexpr uint32_t kEntityInfoCurRow2 = 64;
constexpr uint32_t kEntityInfoPrevRow0 = 80;
constexpr uint32_t kEntityInfoPrevRow1 = 96;
constexpr uint32_t kEntityInfoPrevRow2 = 112;
constexpr uint32_t kEntityInfoBytes = 128;
static_assert(offsetof(EntityInfo, grid_min) == kEntityInfoGridMin);
static_assert(offsetof(EntityInfo, grid_base) == kEntityInfoGridBase);
static_assert(offsetof(EntityInfo, grid_dims) == kEntityInfoGridDims);
static_assert(offsetof(EntityInfo, cur_rows) == kEntityInfoCurRow0);
static_assert(offsetof(EntityInfo, cur_rows) + sizeof(EntityInfo::cur_rows[0]) == kEntityInfoCurRow1);
static_assert(offsetof(EntityInfo, cur_rows) + 2 * sizeof(EntityInfo::cur_rows[0]) == kEntityInfoCurRow2);
static_assert(offsetof(EntityInfo, prev_rows) == kEntityInfoPrevRow0);
static_assert(offsetof(EntityInfo, prev_rows) + sizeof(EntityInfo::prev_rows[0]) == kEntityInfoPrevRow1);
static_assert(offsetof(EntityInfo, prev_rows) + 2 * sizeof(EntityInfo::prev_rows[0]) == kEntityInfoPrevRow2);
static_assert(sizeof(EntityInfo) == kEntityInfoBytes);

// Push constants of the finalize kernel. 104 bytes.
struct FinalizePush
{
    uint64_t dirty_list;
    uint64_t page_table;
    uint64_t slot_tables;
    uint64_t reverse_tables;
    uint64_t group_infos;
    uint64_t entity_infos;
    uint64_t group_grids;
    uint64_t aabb_pool;
    uint32_t dirty_count;
    uint32_t records_per_page_shift;
    uint32_t record_bytes;
    uint32_t group_shift;
    uint32_t encoding;
    uint32_t pad;
    uint64_t light_codes;  // light tree: 65536-bit light code set; 0 = no light tree
    uint64_t light_group_rows;  // light tree: GroupLightRow by group index
};
static_assert(offsetof(FinalizePush, dirty_count) == 64);
static_assert(offsetof(FinalizePush, light_codes) == 88);
static_assert(sizeof(FinalizePush) == 104);
static_assert(sizeof(FinalizePush) <= 128);  // the guaranteed minimum maxPushConstantsSize

// One AABB row: float3 min, float3 max (entity-local block coordinates). All NaN
// = inactive primitive.
constexpr uint32_t kAabbRowBytes = 24;
constexpr uint32_t kQuietNaNBits = 0x7FC00000u;
constexpr uint16_t kNoBrickId = 0xFFFF;
constexpr uint32_t kNoGroup = 0xFFFFFFFFu;
// Workgroups per vkCmdDispatch (maxComputeWorkGroupCount[0] is at least 65535).
constexpr uint32_t kMaxDispatchGroups = 65535;

}  // namespace caelix
