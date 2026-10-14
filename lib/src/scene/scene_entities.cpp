// Scene creation, entities, the group grid, groups, transforms and inspection.
#include "scene/scene.h"

#include "core/log.h"
#include "rhi/shader_loader.h"
#include "scene/scene_internal.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iterator>
#include <new>

namespace caelix {
namespace {

constexpr uint64_t kDefaultStagingBytes = 64ull << 20;
constexpr uint64_t kMinStagingBytes = 4ull << 20;
constexpr uint32_t kDefaultRecordsPerPage = 1u << 18;
constexpr uint32_t kDefaultGroupShift = 4;
constexpr uint32_t kMaxGroupShift = 8;
// Dense brick ids are 16-bit and 0xFFFF means "none", so a group holds at most
// 1 << 15 bricks in this version.
constexpr uint32_t kMaxImplementedGroupShift = 5;
constexpr uint32_t kDefaultBlockBytes = 2;
constexpr uint32_t kEncodingPacked = 0;
constexpr uint32_t kEncodingBlockId = 1;

// Brick keys must fit 21 bits per axis.
constexpr int32_t kKeyLimit = 1 << 20;

// Initial sizes of the growable GPU tables (they double on demand).
constexpr uint32_t kAabbInitialRows = 1u << 16;           // 1.5 MiB
constexpr uint32_t kReverseInitialRows = 1u << 19;        // 1 MiB of u16 entries
constexpr uint32_t kGridMinRows = 8;
constexpr uint32_t kGridInitialRows = 1u << 16;           // 256 KiB of u32 cells
constexpr VkDeviceSize kGroupInfoInitialBytes = 1024 * sizeof(GroupInfo);
constexpr VkDeviceSize kEntityInfoInitialBytes = 64 * sizeof(EntityInfo);
constexpr VkDeviceSize kPageTableInitialBytes = 64 * sizeof(uint64_t);
constexpr VkDeviceSize kDirtyListInitialBytes = 4096 * sizeof(DirtyEntry);

// A group grid larger than this (64 MiB of cells) is refused: the entity's groups
// are spread too far apart for a dense grid.
constexpr uint64_t kMaxGridCells = 1ull << 24;

// Acceleration structures.
constexpr uint64_t kDefaultScratchBytes = 128ull << 20;
constexpr VkDeviceSize kInstancesInitialBytes = 1024 * kInstanceBytes;

uint64_t PackGroupKey(int32_t gx, int32_t gy, int32_t gz) {
  constexpr uint64_t mask = (1ull << 21) - 1;
  const uint64_t x = static_cast<uint64_t>(static_cast<int64_t>(gx) + kKeyLimit) & mask;
  const uint64_t y = static_cast<uint64_t>(static_cast<int64_t>(gy) + kKeyLimit) & mask;
  const uint64_t z = static_cast<uint64_t>(static_cast<int64_t>(gz) + kKeyLimit) & mask;
  return x | (y << 21) | (z << 42);
}

}  // namespace

Scene::Scene(Device& device) : device_(device) {}

Scene::~Scene() { DestroyAccelerationStructures(); }

CaelixResult Scene::Create(Device& device, const CaelixSceneDesc* desc,
                           std::unique_ptr<Scene>& out) {
  out.reset();
  const CaelixSceneDesc defaults{};
  const CaelixSceneDesc& d = desc != nullptr ? *desc : defaults;

  const uint64_t staging_bytes = d.staging_bytes != 0 ? d.staging_bytes : kDefaultStagingBytes;
  const uint32_t records_per_page =
      d.records_per_page != 0 ? d.records_per_page : kDefaultRecordsPerPage;
  const uint32_t group_shift = d.group_shift != 0 ? d.group_shift : kDefaultGroupShift;
  const uint32_t block_bytes = d.block_bytes != 0 ? d.block_bytes : kDefaultBlockBytes;
  const uint64_t scratch_bytes = d.scratch_bytes != 0 ? d.scratch_bytes : kDefaultScratchBytes;

  if (staging_bytes < kMinStagingBytes) {
    CAELIX_LOGE("caelix_scene_create: staging_bytes %llu is below the minimum of 4 MiB",
                static_cast<unsigned long long>(staging_bytes));
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (!std::has_single_bit(records_per_page)) {
    CAELIX_LOGE("caelix_scene_create: records_per_page %u is not a power of two",
                records_per_page);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (group_shift > kMaxGroupShift) {
    CAELIX_LOGE("caelix_scene_create: group_shift %u is above the maximum of %u", group_shift,
                kMaxGroupShift);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (group_shift > kMaxImplementedGroupShift) {
    CAELIX_LOGE("caelix_scene_create: group_shift %u is not implemented (max %u in this version)",
                group_shift, kMaxImplementedGroupShift);
    return CAELIX_ERROR_UNSUPPORTED;
  }
  if (block_bytes == 4) {
    CAELIX_LOGE("caelix_scene_create: block_bytes 4 is not implemented in this version");
    return CAELIX_ERROR_UNSUPPORTED;
  }
  if (block_bytes != 2) {
    CAELIX_LOGE("caelix_scene_create: block_bytes %u is not valid (2 or 4)", block_bytes);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (d.encoding != kEncodingPacked && d.encoding != kEncodingBlockId) {
    CAELIX_LOGE("caelix_scene_create: encoding %u is not valid (0 packed, 1 block-ID)",
                d.encoding);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }

  const RecordLayout layout = RecordLayout::ForBlockBytes(block_bytes);
  VkPhysicalDeviceVulkan13Properties vk13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &vk13;
  vkGetPhysicalDeviceProperties2(device.physical_device(), &properties);
  const uint64_t page_bytes = static_cast<uint64_t>(records_per_page) * layout.record_bytes;
  if (page_bytes > vk13.maxBufferSize || staging_bytes > vk13.maxBufferSize ||
      scratch_bytes > vk13.maxBufferSize) {
    CAELIX_LOGE("caelix_scene_create: a record page (%llu bytes), the staging ring (%llu bytes) or "
                "the build scratch (%llu bytes) exceeds the device's maxBufferSize (%llu bytes)",
                static_cast<unsigned long long>(page_bytes),
                static_cast<unsigned long long>(staging_bytes),
                static_cast<unsigned long long>(scratch_bytes),
                static_cast<unsigned long long>(vk13.maxBufferSize));
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }

  const VmaAllocator allocator = device.allocator();
  const VkDevice vk_device = device.device();
  const uint32_t bricks_per_group = 1u << (3 * group_shift);
  std::unique_ptr<Scene> scene(new Scene(device));
  scene->layout_ = layout;
  scene->group_shift_ = group_shift;
  scene->bricks_per_group_ = bricks_per_group;
  scene->encoding_ = d.encoding;
  scene->records_ = RecordPool(allocator, vk_device, records_per_page, layout.record_bytes);
  scene->slot_tables_ = MakeSlotTablePool(allocator, vk_device);
  // The AABB rows are the BLAS geometry: build input.
  scene->aabbs_ = RowPool(allocator, vk_device, kAabbRowBytes, kSlotMinRows, kAabbInitialRows,
                          "AABB",
                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  scene->reverse_tables_ =
      RowPool(allocator, vk_device, sizeof(uint16_t), bricks_per_group,
              std::max(kReverseInitialRows, bricks_per_group), "reverse table");
  scene->grids_ =
      RowPool(allocator, vk_device, sizeof(uint32_t), kGridMinRows, kGridInitialRows, "group grid");
  scene->group_infos_ = GrowableBuffer(allocator, vk_device, kGroupInfoInitialBytes);
  scene->entity_infos_ = GrowableBuffer(allocator, vk_device, kEntityInfoInitialBytes);
  scene->page_table_ = GrowableBuffer(allocator, vk_device, kPageTableInitialBytes);
  scene->dirty_buffer_ = GrowableBuffer(allocator, vk_device, kDirtyListInitialBytes);
  scene->staging_ = StagingRing(allocator, vk_device, staging_bytes);
  scene->materials_ = std::make_unique<MaterialTable>(d.encoding);
  scene->material_buffer_ = GpuBuffer::DeviceLocal(allocator, vk_device, MaterialTable::kBytes);
  scene->as_pool_.Init(allocator, vk_device);
  scene->scratch_ = GpuBuffer::DeviceLocalWithUsage(
      allocator, vk_device, scratch_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  scene->instances_ =
      GrowableBuffer(allocator, vk_device, kInstancesInitialBytes,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  const std::vector<uint32_t> finalize_spirv = LoadShader("caelix_finalize");
  scene->finalize_ = ComputePipeline(vk_device, finalize_spirv, sizeof(FinalizePush),
                                     "caelix_finalize");
  scene->flush_timer_ = GpuTimer(device);
  CAELIX_LOGI("scene created: staging %llu bytes, %u records per page (%llu bytes), group %u^3 "
              "bricks, record %u bytes, %s encoding, build scratch %llu bytes",
              static_cast<unsigned long long>(staging_bytes), records_per_page,
              static_cast<unsigned long long>(page_bytes), 1u << group_shift,
              layout.record_bytes, d.encoding == kEncodingPacked ? "packed" : "block-ID",
              static_cast<unsigned long long>(scratch_bytes));
  out = std::move(scene);
  return CAELIX_OK;
}

// ---------------------------------------------------------------------------
// Entities

void Scene::ListEntity(CaelixEntityId id) {
  Entity& entity = entities_[id];
  if (!entity.in_dirty_list) {
    entity.in_dirty_list = true;
    dirty_entities_.push_back(id);
  }
}

CaelixResult Scene::AddEntity(const CaelixEntityDesc& desc, CaelixEntityId& out_id) {
  uint32_t id;
  if (!free_entities_.empty()) {
    id = free_entities_.back();
    free_entities_.pop_back();
  } else {
    if (entities_.size() >= CAELIX_INVALID_ENTITY) {
      CAELIX_LOGE("caelix_scene_add_entity: too many entities");
      return CAELIX_ERROR_OUT_OF_MEMORY;
    }
    entity_infos_.Reserve((entities_.size() + 1) * sizeof(EntityInfo));
    entities_.emplace_back();
    id = static_cast<uint32_t>(entities_.size() - 1);
  }
  Entity& entity = entities_[id];
  std::memcpy(entity.guid, desc.guid, sizeof(entity.guid));
  std::memcpy(entity.position, desc.position, sizeof(entity.position));
  std::memcpy(entity.rotation, desc.rotation, sizeof(entity.rotation));
  TransformRows(entity.position, entity.rotation, entity.cur_rows);
  std::memcpy(entity.prev_rows, entity.cur_rows, sizeof(entity.prev_rows));
  entity.transform_changed = false;
  entity.groups.clear();
  entity.alive = true;
  std::fill(std::begin(entity.grid_min), std::end(entity.grid_min), 0);
  std::fill(std::begin(entity.grid_dims), std::end(entity.grid_dims), 0);
  entity.grid_range = RowRange{};
  entity.grid.clear();
  entity.has_bounds = false;
  entity.grid_fill = false;
  entity.grid_dirty_cells.clear();
  entity.info_dirty = true;
  // in_dirty_list keeps its value: the id may still be listed from its previous life.
  ListEntity(id);
  ++entities_alive_;
  out_id = id;
  return CAELIX_OK;
}

CaelixResult Scene::RemoveEntity(CaelixEntityId id) {
  Entity& entity = entities_[id];
  for (const auto& [key, group_index] : entity.groups) {
    Group& group = groups_[group_index];
    for (const BrickSlotCpu& slot : group.slots) {
      if (slot.state == kSlotResident) {
        records_.Free(slot.record_index);
        --bricks_resident_;
      }
    }
    bricks_known_ -= group.brick_count;
    std::fill(group.id_by_pos.begin(), group.id_by_pos.end(), Group::kNoId);
    ReleaseGroup(group_index);
  }
  entity.groups.clear();
  FreeAfterFlush(grids_, entity.grid_range);
  entity.grid_range = RowRange{};
  entity.grid.clear();
  entity.grid_dirty_cells.clear();
  std::fill(std::begin(entity.grid_dims), std::end(entity.grid_dims), 0);
  entity.has_bounds = false;
  entity.grid_fill = false;
  entity.info_dirty = false;
  entity.transform_changed = false;
  entity.alive = false;
  free_entities_.push_back(id);
  tlas_dirty_ = true;
  --entities_alive_;
  return CAELIX_OK;
}

CaelixResult Scene::SetEntityTransform(CaelixEntityId id, const float position[3],
                                       const float rotation[4]) {
  Entity& entity = entities_[id];
  std::memcpy(entity.position, position, sizeof(entity.position));
  std::memcpy(entity.rotation, rotation, sizeof(entity.rotation));
  // The next flush moves cur to prev, restages the info row and rebuilds the TLAS.
  entity.transform_changed = true;
  ListEntity(id);
  tlas_dirty_ = true;
  return CAELIX_OK;
}

// ---------------------------------------------------------------------------
// Group grid

bool Scene::GridContains(const Entity& entity, int32_t gx, int32_t gy, int32_t gz) const {
  const int32_t c[3] = {gx, gy, gz};
  for (int a = 0; a < 3; ++a) {
    if (entity.grid_dims[a] == 0 || c[a] < entity.grid_min[a] ||
        c[a] >= entity.grid_min[a] + entity.grid_dims[a]) {
      return false;
    }
  }
  return true;
}

Scene::GridShape Scene::GrownGridShape(const Entity& entity, int32_t gx, int32_t gy,
                                       int32_t gz) const {
  const int32_t c[3] = {gx, gy, gz};
  GridShape shape{};
  shape.cells = 1;
  for (int a = 0; a < 3; ++a) {
    const int32_t old_min = entity.grid_min[a];
    const int32_t old_dims = entity.grid_dims[a];
    if (old_dims > 0 && c[a] >= old_min && c[a] < old_min + old_dims) {
      shape.min[a] = old_min;  // this axis already covers the new group
      shape.dims[a] = old_dims;
    } else {
      const int64_t lo = entity.has_bounds ? std::min(entity.bounds_min[a], c[a]) : c[a];
      const int64_t hi = entity.has_bounds ? std::max(entity.bounds_max[a], c[a]) : c[a];
      int64_t dims = static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(hi - lo + 1)));
      if (old_dims > 0 && dims <= old_dims) {
        dims = static_cast<int64_t>(old_dims) * 2;  // grow geometrically
      }
      // The slack goes to the side the grid grows towards.
      const bool grows_down = old_dims > 0 && c[a] < old_min;
      shape.min[a] = static_cast<int32_t>(grows_down ? hi - dims + 1 : lo);
      shape.dims[a] = static_cast<int32_t>(dims);
    }
    shape.cells *= static_cast<uint64_t>(shape.dims[a]);
  }
  if (shape.cells > kMaxGridCells) {
    CAELIX_LOGE("entity group grid of %d x %d x %d cells exceeds the limit of %llu cells: the "
                "entity's bricks are spread too far apart",
                shape.dims[0], shape.dims[1], shape.dims[2],
                static_cast<unsigned long long>(kMaxGridCells));
    throw std::bad_alloc();
  }
  return shape;
}

uint32_t Scene::GridCell(const Entity& entity, int32_t gx, int32_t gy, int32_t gz) const {
  const uint32_t x = static_cast<uint32_t>(gx - entity.grid_min[0]);
  const uint32_t y = static_cast<uint32_t>(gy - entity.grid_min[1]);
  const uint32_t z = static_cast<uint32_t>(gz - entity.grid_min[2]);
  return x + static_cast<uint32_t>(entity.grid_dims[0]) *
                 (y + static_cast<uint32_t>(entity.grid_dims[1]) * z);
}

void Scene::RebuildGrid(Entity& entity, const GridShape& shape, RowRange range) {
  std::copy(std::begin(shape.min), std::end(shape.min), std::begin(entity.grid_min));
  std::copy(std::begin(shape.dims), std::end(shape.dims), std::begin(entity.grid_dims));
  entity.grid.assign(shape.cells, kNoGroup);
  entity.grid_range = range;
  entity.grid_fill = true;
  entity.grid_dirty_cells.clear();
  for (const auto& [key, group_index] : entity.groups) {
    const Group& group = groups_[group_index];
    const uint32_t cell = GridCell(entity, group.gx, group.gy, group.gz);
    entity.grid[cell] = group_index;
    entity.grid_dirty_cells.push_back(cell);
  }
  entity.info_dirty = true;
}

void Scene::SetGridCell(Entity& entity, int32_t gx, int32_t gy, int32_t gz, uint32_t value) {
  const uint32_t cell = GridCell(entity, gx, gy, gz);
  entity.grid[cell] = value;
  entity.grid_dirty_cells.push_back(cell);
}

// ---------------------------------------------------------------------------
// Keys and groups

bool Scene::KeyInRange(const CaelixBrickKey& key) {
  return key.x >= -kKeyLimit && key.x < kKeyLimit && key.y >= -kKeyLimit && key.y < kKeyLimit &&
         key.z >= -kKeyLimit && key.z < kKeyLimit;
}

Scene::Location Scene::Locate(const CaelixBrickKey& key) const {
  const int32_t mask = (1 << group_shift_) - 1;
  Location loc;
  // Arithmetic right shift = floor division for negative keys (C++20 defines it).
  loc.gx = key.x >> group_shift_;
  loc.gy = key.y >> group_shift_;
  loc.gz = key.z >> group_shift_;
  loc.group_key = PackGroupKey(loc.gx, loc.gy, loc.gz);
  const uint32_t bx = static_cast<uint32_t>(key.x & mask);
  const uint32_t by = static_cast<uint32_t>(key.y & mask);
  const uint32_t bz = static_cast<uint32_t>(key.z & mask);
  loc.pos = bx | (by << group_shift_) | (bz << (2 * group_shift_));
  return loc;
}

void Scene::ListGroup(uint32_t group_index) {
  Group& group = groups_[group_index];
  if (!group.in_dirty_list) {
    group.in_dirty_list = true;
    dirty_groups_.push_back(group_index);
  }
}

uint32_t Scene::CreateGroup(CaelixEntityId entity_id, const Location& loc, SlotRange slot_rows,
                            RowRange aabb_rows, RowRange reverse_range) {
  uint32_t index;
  if (!free_groups_.empty()) {
    index = free_groups_.back();
    free_groups_.pop_back();
  } else {
    groups_.emplace_back();
    index = static_cast<uint32_t>(groups_.size() - 1);
  }
  Group& group = groups_[index];
  group.entity = entity_id;
  group.gx = loc.gx;
  group.gy = loc.gy;
  group.gz = loc.gz;
  group.key = loc.group_key;
  if (group.id_by_pos.size() != bricks_per_group_) {
    group.id_by_pos.assign(bricks_per_group_, Group::kNoId);
  }
  group.slots.clear();
  group.free_ids.clear();
  group.dirty_ids.clear();
  group.reverse_dirty_words.clear();
  group.nan_ids.clear();
  group.slot_rows = slot_rows;
  group.aabb_rows = aabb_rows;
  group.aabb_gpu_rows = RowRange{};
  group.reverse_range = reverse_range;
  group.brick_count = 0;
  group.full_rewrite = false;
  group.info_dirty = true;
  group.reverse_fill = true;
  group.alive = true;
  group.blas = AsBlock{};
  group.blas_handle = VK_NULL_HANDLE;
  group.blas_address = 0;
  group.blas_primitive_count = 0;
  group.blas_dirty = false;
  group.finalize_wait_serial = 0;
  // in_dirty_list and in_blas_list keep their values: the index may still be listed
  // from its previous life.
  ListGroup(index);

  Entity& entity = entities_[entity_id];
  entity.groups.emplace(loc.group_key, index);
  const int32_t c[3] = {loc.gx, loc.gy, loc.gz};
  for (int a = 0; a < 3; ++a) {
    entity.bounds_min[a] = entity.has_bounds ? std::min(entity.bounds_min[a], c[a]) : c[a];
    entity.bounds_max[a] = entity.has_bounds ? std::max(entity.bounds_max[a], c[a]) : c[a];
  }
  entity.has_bounds = true;
  SetGridCell(entity, loc.gx, loc.gy, loc.gz, index);
  ListEntity(entity_id);
  ++groups_alive_;
  return index;
}

void Scene::ReleaseGroup(uint32_t group_index) {
  Group& group = groups_[group_index];
  Entity& entity = entities_[group.entity];
  if (GridContains(entity, group.gx, group.gy, group.gz)) {
    SetGridCell(entity, group.gx, group.gy, group.gz, kNoGroup);
    ListEntity(group.entity);
  }
  // The GPU may still read the rows of the last flushed frame; the ranges are
  // reused only after this window's flush (see Flush).
  FreeAfterFlush(slot_tables_, group.slot_rows);
  FreeAfterFlush(aabbs_, group.aabb_rows);
  FreeAfterFlush(reverse_tables_, group.reverse_range);
  group.slot_rows = SlotRange{};
  group.aabb_rows = RowRange{};
  group.aabb_gpu_rows = RowRange{};
  group.reverse_range = RowRange{};
  group.slots.clear();
  group.free_ids.clear();
  group.dirty_ids.clear();
  group.reverse_dirty_words.clear();
  group.nan_ids.clear();
  group.brick_count = 0;
  group.full_rewrite = false;
  group.info_dirty = false;
  group.reverse_fill = false;
  group.alive = false;
  // The current TLAS refers to the BLAS: it is released after the next TLAS build.
  if (group.blas_handle != VK_NULL_HANDLE) {
    retired_blas_.push_back(RetiredBlas{group.blas, group.blas_handle});
    --blas_count_;
    tlas_dirty_ = true;
  }
  group.blas = AsBlock{};
  group.blas_handle = VK_NULL_HANDLE;
  group.blas_address = 0;
  group.blas_primitive_count = 0;
  group.blas_dirty = false;
  free_groups_.push_back(group_index);
  --groups_alive_;
}

// ---------------------------------------------------------------------------
// Transforms

void Scene::TransformRows(const float position[3], const float rotation[4], float rows[3][4]) {
  float x = rotation[0];
  float y = rotation[1];
  float z = rotation[2];
  float w = rotation[3];
  // Scale 1: the quaternion is normalized (an invalid one becomes the identity).
  const float norm = std::sqrt(x * x + y * y + z * z + w * w);
  if (norm > 0.0f && std::isfinite(norm)) {
    x /= norm;
    y /= norm;
    z /= norm;
    w /= norm;
  } else {
    x = y = z = 0.0f;
    w = 1.0f;
  }
  rows[0][0] = 1.0f - 2.0f * (y * y + z * z);
  rows[0][1] = 2.0f * (x * y - w * z);
  rows[0][2] = 2.0f * (x * z + w * y);
  rows[0][3] = position[0];
  rows[1][0] = 2.0f * (x * y + w * z);
  rows[1][1] = 1.0f - 2.0f * (x * x + z * z);
  rows[1][2] = 2.0f * (y * z - w * x);
  rows[1][3] = position[1];
  rows[2][0] = 2.0f * (x * z - w * y);
  rows[2][1] = 2.0f * (y * z + w * x);
  rows[2][2] = 1.0f - 2.0f * (x * x + y * y);
  rows[2][3] = position[2];
}

// ---------------------------------------------------------------------------
// Inspection

void Scene::GetStats(CaelixSceneStats& out) const {
  out = CaelixSceneStats{};
  out.entities = entities_alive_;
  out.groups = groups_alive_;
  out.bricks_resident = bricks_resident_;
  out.bricks_known = bricks_known_;
  out.record_bytes = records_.total_bytes();
  out.slot_table_bytes = slot_tables_.bytes();
  out.staging_bytes = staging_.capacity();
  out.uploaded_bricks_total = uploaded_bricks_total_;
  out.pending_bricks = pending_bricks_;
  out.record_pages = records_.page_count();
  out.finalized_bricks_total = finalized_bricks_total_;
  out.reverse_table_bytes = reverse_tables_.bytes();
  out.aabb_bytes = aabbs_.bytes();
  out.grid_bytes = grids_.bytes();
  out.blas_count = blas_count_;
  out.blas_bytes = as_pool_.allocated_bytes(AsKind::kBlas);
  out.tlas_bytes = as_pool_.allocated_bytes(AsKind::kTlas);
  out.scratch_bytes = scratch_.size();
  out.as_builds_total = as_builds_total_;
  uint64_t pending = tlas_dirty_ ? 1 : 0;
  for (const uint32_t group_index : blas_pending_) {
    const Group& group = groups_[group_index];
    if (group.alive && group.blas_dirty) {
      ++pending;
    }
  }
  out.as_builds_pending = pending;
  out.material_table_bytes = material_buffer_.size();
}

CaelixResult Scene::QueryBrick(CaelixEntityId entity_id, CaelixBrickKey key,
                               CaelixBrickInfo& out) const {
  out = CaelixBrickInfo{};
  if (!KeyInRange(key)) {
    CAELIX_LOGE("caelix_scene_query_brick: key (%d, %d, %d) is out of range", key.x, key.y, key.z);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  const Entity& entity = entities_[entity_id];
  const Location loc = Locate(key);
  const auto found = entity.groups.find(loc.group_key);
  if (found == entity.groups.end()) {
    return CAELIX_OK;
  }
  const Group& group = groups_[found->second];
  const uint16_t id = group.id_by_pos[loc.pos];
  if (id == Group::kNoId) {
    return CAELIX_OK;
  }
  const BrickSlotCpu& slot = group.slots[id];
  if (slot.state != kSlotResident) {
    return CAELIX_OK;
  }
  out.resident = 1;
  out.group_index = found->second;
  out.group_local_id = id;
  out.record_index = slot.record_index;
  out.record_page_address = records_.PageAddress(records_.page_of(slot.record_index));
  out.record_page_buffer = HandleToU64(records_.PageBuffer(slot.record_index));
  out.record_page_offset_bytes = records_.PageOffset(slot.record_index);
  out.slot_table_buffer = HandleToU64(slot_tables_.buffer());
  out.slot_row_offset_bytes = slot_tables_.ByteOffset(group.slot_rows.offset + id);
  out.aabb_buffer = HandleToU64(aabbs_.buffer());
  out.aabb_row_offset_bytes = aabbs_.ByteOffset(group.aabb_rows.offset + id);
  out.blas_address = group.blas_address;
  out.blas_handle = HandleToU64(group.blas_handle);
  out.entity_info_buffer = HandleToU64(entity_infos_.buffer());
  out.entity_info_offset_bytes = static_cast<uint64_t>(entity_id) * sizeof(EntityInfo);
  return CAELIX_OK;
}

uint32_t Scene::records_per_page_shift() const {
  return static_cast<uint32_t>(
      std::countr_zero(static_cast<uint32_t>(records_.page_bytes() / layout_.record_bytes)));
}

void Scene::GetTlas(uint64_t& handle, uint64_t& address) const {
  handle = HandleToU64(tlas_handle_);
  address = tlas_address_;
}

}  // namespace caelix
