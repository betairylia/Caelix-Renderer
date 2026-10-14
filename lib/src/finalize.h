#pragma once
#include <cstdint>

namespace caelix {

// CPU mirrors of the tables that shaders/caelix_finalize.hlsl reads. The HLSL
// side is shaders/caelix_record.hlsl; change both together.

// One brick to finalize: one workgroup each. 16 bytes.
struct DirtyEntry {
  uint32_t group_index;
  uint32_t id;
  uint32_t record_index;
  uint32_t pad;
};
static_assert(sizeof(DirtyEntry) == 16);

// One render group, indexed by the scene-wide group index. 32 bytes.
struct GroupInfo {
  int32_t coord[3];        // group coordinate = brick key >> group_shift
  uint32_t entity;
  uint32_t slot_row_base;  // first slot table row
  uint32_t reverse_base;   // first u16 entry of the reverse table
  uint32_t aabb_row_base;  // first AABB row
  uint32_t high_water;     // dense ids in [0, high_water) were handed out
};
static_assert(sizeof(GroupInfo) == 32);

// One entity, indexed by the entity id. 128 bytes. The finalize kernel reads the
// first 32 bytes; the transform rows are for the trace kernels.
struct EntityInfo {
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
static_assert(sizeof(EntityInfo) == 128);

// Push constants of the finalize kernel. 88 bytes.
struct FinalizePush {
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
};
static_assert(sizeof(FinalizePush) == 88);
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
