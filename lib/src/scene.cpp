#include "scene.h"

#include "log.h"
#include "shader_loader.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <new>
#include <optional>

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

// Staging alignment. Records are 8-byte multiples, so consecutive records stay
// contiguous in the ring and their copies merge. vkCmdCopyBuffer itself has no
// offset alignment rule.
constexpr uint64_t kRecordAlignment = 8;
constexpr uint64_t kRowAlignment = 16;

// vkCmdCopyBuffer regions per call.
constexpr size_t kMaxRegionsPerCopy = 4096;

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

// Neighbour directions in face order: +X, -X, +Y, -Y, +Z, -Z.
constexpr int32_t kDirections[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                       {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};

uint64_t PackGroupKey(int32_t gx, int32_t gy, int32_t gz) {
  constexpr uint64_t mask = (1ull << 21) - 1;
  const uint64_t x = static_cast<uint64_t>(static_cast<int64_t>(gx) + kKeyLimit) & mask;
  const uint64_t y = static_cast<uint64_t>(static_cast<int64_t>(gy) + kKeyLimit) & mask;
  const uint64_t z = static_cast<uint64_t>(static_cast<int64_t>(gz) + kKeyLimit) & mask;
  return x | (y << 21) | (z << 42);
}

SlotRow ToRow(const BrickSlotCpu& slot) {
  SlotRow row{};
  row.pos_state = (slot.pos & 0x00FFFFFFu) | (static_cast<uint32_t>(slot.state) << 24);
  row.summary = 0;
  row.record_index = slot.record_index;
  row.last_used = 0;
  return row;
}

void MemoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                   VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
  VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  barrier.srcStageMask = src_stage;
  barrier.srcAccessMask = src_access;
  barrier.dstStageMask = dst_stage;
  barrier.dstAccessMask = dst_access;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(cmd, &dependency);
}

// Transfer writes (copies and fills) -> later transfer reads and writes.
void TransferBarrier(VkCommandBuffer cmd) {
  MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

template <typename T>
void SortUnique(std::vector<T>& values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

// Consecutive u32 words [first, last] of a table, restaged with one copy region.
struct WordRun {
  uint32_t first;
  uint32_t last;
};

// Groups sorted, unique word indices into runs that also cover gaps of up to
// kMaxGapWords clean words; the CPU holds the current value of every word. Every
// vkCmdCopyBuffer region has a fixed GPU cost that is much higher than copying a
// few hundred bytes: with one 4-byte region per dirty reverse table word, the
// 8K citadel load recorded 200K regions and took 0.55 s instead of 0.3 s. A merge
// stages at most 4 * kMaxGapWords extra bytes and saves one region.
constexpr uint32_t kMaxGapWords = 128;

void BuildRuns(const std::vector<uint32_t>& words, std::vector<WordRun>& runs) {
  runs.clear();
  if (words.empty()) {
    return;
  }
  WordRun run{words[0], words[0]};
  for (size_t i = 1; i < words.size(); ++i) {
    if (words[i] - run.last <= kMaxGapWords + 1) {
      run.last = words[i];
    } else {
      runs.push_back(run);
      run = WordRun{words[i], words[i]};
    }
  }
  runs.push_back(run);
}

uint64_t RunWords(const std::vector<WordRun>& runs) {
  uint64_t words = 0;
  for (const WordRun& run : runs) {
    words += static_cast<uint64_t>(run.last) - run.first + 1;
  }
  return words;
}

}  // namespace

Scene::Scene(Device& device) : device_(device) {}

Scene::~Scene() = default;

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
  if (page_bytes > vk13.maxBufferSize || staging_bytes > vk13.maxBufferSize) {
    CAELIX_LOGE("caelix_scene_create: a record page (%llu bytes) or the staging ring (%llu bytes) "
                "exceeds the device's maxBufferSize (%llu bytes)",
                static_cast<unsigned long long>(page_bytes),
                static_cast<unsigned long long>(staging_bytes),
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
  scene->aabbs_ =
      RowPool(allocator, vk_device, kAabbRowBytes, kSlotMinRows, kAabbInitialRows, "AABB");
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
  const std::vector<uint32_t> finalize_spirv = LoadShader("caelix_finalize");
  scene->finalize_ = ComputePipeline(vk_device, finalize_spirv, sizeof(FinalizePush),
                                     "caelix_finalize");
  CAELIX_LOGI("scene created: staging %llu bytes, %u records per page (%llu bytes), group %u^3 "
              "bricks, record %u bytes, %s encoding",
              static_cast<unsigned long long>(staging_bytes), records_per_page,
              static_cast<unsigned long long>(page_bytes), 1u << group_shift,
              layout.record_bytes, d.encoding == kEncodingPacked ? "packed" : "block-ID");
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
  entity.alive = false;
  free_entities_.push_back(id);
  --entities_alive_;
  return CAELIX_OK;
}

CaelixResult Scene::SetEntityTransform(CaelixEntityId id, const float position[3],
                                       const float rotation[4]) {
  Entity& entity = entities_[id];
  std::memcpy(entity.position, position, sizeof(entity.position));
  std::memcpy(entity.rotation, rotation, sizeof(entity.rotation));
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

void Scene::FreeAfterFlush(RowPool& pool, RowRange range) {
  if (range.valid()) {
    ranges_to_free_after_flush_.push_back(RangeToFree{&pool, range});
  }
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
  // in_dirty_list keeps its value: the index may still be listed from its previous life.
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
  free_groups_.push_back(group_index);
  --groups_alive_;
}

void Scene::MarkDirty(uint32_t group_index, uint16_t id) {
  Group& group = groups_[group_index];
  if (!group.full_rewrite) {
    group.dirty_ids.push_back(id);
  }
  ListGroup(group_index);
}

void Scene::MarkReverseDirty(Group& group, uint32_t pos) {
  group.reverse_dirty_words.push_back(pos >> 1);
}

Scene::Placed Scene::PlaceBrick(CaelixEntityId entity_id, const Location& loc) {
  Entity& entity = entities_[entity_id];
  const auto found = entity.groups.find(loc.group_key);
  uint32_t group_index = found != entity.groups.end() ? found->second : 0xFFFFFFFFu;
  const uint32_t pages_before = records_.page_count();

  // Replace an existing brick: same id, same record, unchanged row.
  if (found != entity.groups.end()) {
    Group& group = groups_[group_index];
    const uint16_t id = group.id_by_pos[loc.pos];
    if (id != Group::kNoId) {
      BrickSlotCpu& slot = group.slots[id];
      if (slot.state != kSlotResident) {
        // A known brick without a record (states for later versions).
        const uint32_t record = records_.Allocate();
        try {
          page_table_.Reserve(static_cast<VkDeviceSize>(records_.page_count()) * sizeof(uint64_t));
        } catch (...) {
          records_.Free(record);
          throw;
        }
        slot.record_index = record;
        slot.state = kSlotResident;
        ++bricks_resident_;
        MarkDirty(group_index, id);
      }
      return Placed{slot.record_index, group_index, id};
    }
  }

  // A new brick. GPU allocations first: when one throws, nothing has changed.
  const uint32_t record = records_.Allocate();
  SlotRange new_slots{};
  RowRange new_aabbs{};
  RowRange new_reverse{};
  RowRange new_grid{};
  std::optional<GridShape> grid_shape;
  try {
    if (records_.page_count() != pages_before) {
      page_table_.Reserve(static_cast<VkDeviceSize>(records_.page_count()) * sizeof(uint64_t));
    }
    if (found == entity.groups.end()) {
      new_slots = slot_tables_.Allocate(kSlotMinRows);
      new_aabbs = aabbs_.Allocate(kSlotMinRows);
      new_reverse = reverse_tables_.Allocate(bricks_per_group_);
      if (free_groups_.empty()) {
        group_infos_.Reserve((groups_.size() + 1) * sizeof(GroupInfo));
      }
      if (!GridContains(entity, loc.gx, loc.gy, loc.gz)) {
        grid_shape = GrownGridShape(entity, loc.gx, loc.gy, loc.gz);
        new_grid = grids_.Allocate(static_cast<uint32_t>(grid_shape->cells));
      }
    } else {
      const Group& group = groups_[group_index];
      if (group.free_ids.empty() && group.slots.size() == group.slot_rows.rows) {
        new_slots = slot_tables_.Allocate(group.slot_rows.rows * 2);
        new_aabbs = aabbs_.Allocate(group.slot_rows.rows * 2);
      }
    }
  } catch (...) {
    slot_tables_.Free(new_slots);
    aabbs_.Free(new_aabbs);
    reverse_tables_.Free(new_reverse);
    grids_.Free(new_grid);
    records_.Free(record);
    throw;
  }

  if (found == entity.groups.end()) {
    if (grid_shape.has_value()) {
      FreeAfterFlush(grids_, entity.grid_range);
      RebuildGrid(entity, *grid_shape, new_grid);
      ListEntity(entity_id);
    }
    group_index = CreateGroup(entity_id, loc, new_slots, new_aabbs, new_reverse);
  } else if (new_slots.valid()) {
    // The ranges double. The old ranges are freed after this window's flush;
    // every slot row is restaged into the new range and the AABB rows are copied
    // over on the GPU at that flush.
    Group& group = groups_[group_index];
    FreeAfterFlush(slot_tables_, group.slot_rows);
    FreeAfterFlush(aabbs_, group.aabb_rows);
    group.slot_rows = new_slots;
    group.aabb_rows = new_aabbs;
    group.full_rewrite = true;
    group.dirty_ids.clear();
    group.info_dirty = true;
  }

  Group& group = groups_[group_index];
  uint16_t id;
  if (!group.free_ids.empty()) {
    id = group.free_ids.back();
    group.free_ids.pop_back();
  } else {
    id = static_cast<uint16_t>(group.slots.size());
    group.slots.emplace_back();
    group.info_dirty = true;  // the high water grew
  }
  group.id_by_pos[loc.pos] = id;
  group.slots[id] = BrickSlotCpu{loc.pos, kSlotResident, record};
  ++group.brick_count;
  ++bricks_resident_;
  ++bricks_known_;
  MarkDirty(group_index, id);
  MarkReverseDirty(group, loc.pos);
  return Placed{record, group_index, id};
}

// ---------------------------------------------------------------------------
// Finalize queue

void Scene::QueueFinalize(uint32_t group_index, uint16_t id) {
  BrickSlotCpu& slot = groups_[group_index].slots[id];
  if (slot.finalize_epoch == finalize_epoch_) {
    return;
  }
  slot.finalize_epoch = finalize_epoch_;
  dirty_.push_back(DirtyBrick{group_index, id});
}

void Scene::QueueNeighbors(CaelixEntityId entity_id, const CaelixBrickKey& key,
                           uint32_t group_index) {
  const Entity& entity = entities_[entity_id];
  const int32_t edge = 1 << group_shift_;
  const int32_t local[3] = {key.x & (edge - 1), key.y & (edge - 1), key.z & (edge - 1)};
  const int32_t stride[3] = {1, edge, edge * edge};
  const Location own = Locate(key);
  for (int d = 0; d < 6; ++d) {
    const int axis = d >> 1;
    const int32_t step = (d & 1) != 0 ? -1 : 1;
    uint32_t neighbour_group = 0xFFFFFFFFu;
    uint32_t pos = 0;
    const int32_t next = local[axis] + step;
    if (group_index != 0xFFFFFFFFu && next >= 0 && next < edge) {
      // Same group: no lookup.
      neighbour_group = group_index;
      pos = static_cast<uint32_t>(static_cast<int32_t>(own.pos) + step * stride[axis]);
    } else {
      const CaelixBrickKey neighbour{key.x + kDirections[d][0], key.y + kDirections[d][1],
                                     key.z + kDirections[d][2]};
      if (!KeyInRange(neighbour)) {
        continue;
      }
      const Location loc = Locate(neighbour);
      const auto found = entity.groups.find(loc.group_key);
      if (found == entity.groups.end()) {
        continue;
      }
      neighbour_group = found->second;
      pos = loc.pos;
    }
    const Group& group = groups_[neighbour_group];
    const uint16_t id = group.id_by_pos[pos];
    if (id != Group::kNoId && group.slots[id].state == kSlotResident) {
      QueueFinalize(neighbour_group, id);
    }
  }
}

// ---------------------------------------------------------------------------
// Upload and removal

void Scene::WriteRecord(uint8_t* dst, uint32_t pos, const uint8_t* blocks) const {
  // Build the record in local memory, then copy it in one go: the staging memory
  // may be write-combined.
  uint32_t record[2 + 16 + 256];
  static_assert(sizeof(record) == 1096);
  uint16_t values[RecordLayout::kBlocksPerBrick];
  std::memcpy(values, blocks, sizeof(values));  // the host pointer may be unaligned
  record[0] = pos & 0xFFFFu;  // the finalize kernel fills bits 16..23 and words 1..17
  for (uint32_t w = 1; w < layout_.payload_first_word(); ++w) {
    record[w] = 0;
  }
  uint32_t* payload = record + layout_.payload_first_word();
  // Voxel index i = x + 8y + 64z. For even x, i/2 = (x >> 1) + 4y + 32z, and the
  // pair (i, i + 1) shares one word: even x high, odd x low.
  for (uint32_t i = 0; i < RecordLayout::kBlocksPerBrick; i += 2) {
    payload[i / 2] = (static_cast<uint32_t>(values[i]) << 16) | values[i + 1];
  }
  std::memcpy(dst, record, layout_.record_bytes);
}

CaelixResult Scene::UploadBricks(CaelixEntityId entity, uint32_t count, const CaelixBrickKey* keys,
                                 const void* blocks, uint32_t& consumed) {
  consumed = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (!KeyInRange(keys[i])) {
      CAELIX_LOGE("caelix_scene_upload_bricks: key %u (%d, %d, %d) is out of range", i, keys[i].x,
                  keys[i].y, keys[i].z);
      return CAELIX_ERROR_INVALID_ARGUMENT;
    }
  }
  const uint8_t* block_bytes = static_cast<const uint8_t*>(blocks);
  const size_t brick_bytes = static_cast<size_t>(RecordLayout::kBlocksPerBrick) * layout_.block_bytes;
  for (uint32_t i = 0; i < count; ++i) {
    StagingRing::Span span;
    if (!staging_.TryAllocate(layout_.record_bytes, kRecordAlignment,
                              StagingRing::Purpose::kRecords, span)) {
      break;
    }
    const Location loc = Locate(keys[i]);
    const Placed placed = PlaceBrick(entity, loc);
    WriteRecord(span.data, loc.pos, block_bytes + i * brick_bytes);
    pending_.push_back(PendingCopy{span.offset, records_.PageBuffer(placed.record),
                                   records_.PageOffset(placed.record), layout_.record_bytes});
    // The brick and its resident neighbours are finalized at the next flush.
    QueueFinalize(placed.group_index, placed.id);
    QueueNeighbors(entity, keys[i], placed.group_index);
    ++uploaded_bricks_total_;
    ++pending_bricks_;
    ++consumed;
  }
  return CAELIX_OK;
}

CaelixResult Scene::RemoveBricks(CaelixEntityId entity_id, uint32_t count,
                                 const CaelixBrickKey* keys) {
  for (uint32_t i = 0; i < count; ++i) {
    if (!KeyInRange(keys[i])) {
      CAELIX_LOGE("caelix_scene_remove_bricks: key %u (%d, %d, %d) is out of range", i, keys[i].x,
                  keys[i].y, keys[i].z);
      return CAELIX_ERROR_INVALID_ARGUMENT;
    }
  }
  Entity& entity = entities_[entity_id];
  for (uint32_t i = 0; i < count; ++i) {
    const Location loc = Locate(keys[i]);
    const auto found = entity.groups.find(loc.group_key);
    if (found == entity.groups.end()) {
      continue;
    }
    const uint32_t group_index = found->second;
    Group& group = groups_[group_index];
    const uint16_t id = group.id_by_pos[loc.pos];
    if (id == Group::kNoId) {
      continue;
    }
    BrickSlotCpu& slot = group.slots[id];
    if (slot.state == kSlotResident) {
      // Reusable at once; see RecordPool for why that is safe on one queue.
      records_.Free(slot.record_index);
      --bricks_resident_;
    }
    slot = BrickSlotCpu{loc.pos, kSlotFree, RecordPool::kInvalidIndex};
    group.id_by_pos[loc.pos] = Group::kNoId;
    group.free_ids.push_back(id);
    --group.brick_count;
    --bricks_known_;
    if (group.brick_count == 0) {
      entity.groups.erase(found);
      ReleaseGroup(group_index);
    } else {
      MarkDirty(group_index, id);
      MarkReverseDirty(group, loc.pos);
      group.nan_ids.push_back(id);  // the freed id's AABB row becomes inactive
    }
    // The neighbours lose a neighbour: they are finalized again.
    QueueNeighbors(entity_id, keys[i], group.brick_count > 0 ? group_index : 0xFFFFFFFFu);
  }
  return CAELIX_OK;
}

// ---------------------------------------------------------------------------
// Flush

void Scene::PrepareGroupAabbs(Group& group) {
  if (group.reverse_fill) {
    // A new reverse range: every position "none" before the used words are staged.
    fills_.push_back(PendingFill{reverse_tables_.buffer(),
                                 reverse_tables_.ByteOffset(group.reverse_range.offset),
                                 reverse_tables_.ByteOffset(group.reverse_range.rows),
                                 0xFFFFFFFFu});
    group.reverse_fill = false;
  }
  if (group.aabb_rows == group.aabb_gpu_rows) {
    return;
  }
  // The AABB range is new or grew: copy the rows the GPU holds, and make every
  // other row inactive (NaN), so free ids never show up as primitives.
  uint32_t kept = 0;
  if (group.aabb_gpu_rows.valid()) {
    kept = group.aabb_gpu_rows.rows;
    aabb_moves_.push_back(VkBufferCopy{aabbs_.ByteOffset(group.aabb_gpu_rows.offset),
                                       aabbs_.ByteOffset(group.aabb_rows.offset),
                                       aabbs_.ByteOffset(kept)});
  }
  fills_.push_back(PendingFill{aabbs_.buffer(), aabbs_.ByteOffset(group.aabb_rows.offset + kept),
                               aabbs_.ByteOffset(group.aabb_rows.rows - kept), kQuietNaNBits});
  group.aabb_gpu_rows = group.aabb_rows;
}

bool Scene::EnsureNanRow() {
  if (nan_row_valid_) {
    return true;
  }
  StagingRing::Span span;
  if (!staging_.TryAllocate(kAabbRowBytes, 8, StagingRing::Purpose::kRows, span)) {
    return false;
  }
  const uint32_t nan_row[6] = {kQuietNaNBits, kQuietNaNBits, kQuietNaNBits,
                               kQuietNaNBits, kQuietNaNBits, kQuietNaNBits};
  std::memcpy(span.data, nan_row, sizeof(nan_row));
  nan_row_offset_ = span.offset;
  nan_row_valid_ = true;
  return true;
}

bool Scene::StageGroup(uint32_t group_index, Group& group) {
  if (!group.full_rewrite) {
    SortUnique(group.dirty_ids);
  }
  SortUnique(group.reverse_dirty_words);
  std::vector<WordRun> reverse_runs;
  BuildRuns(group.reverse_dirty_words, reverse_runs);
  if (!group.nan_ids.empty() && !EnsureNanRow()) {
    return false;
  }
  const uint64_t slot_rows = group.full_rewrite ? group.slots.size() : group.dirty_ids.size();
  const uint64_t slot_bytes = slot_rows * kSlotRowBytes;
  const uint64_t info_bytes = group.info_dirty ? sizeof(GroupInfo) : 0;
  const uint64_t reverse_bytes = RunWords(reverse_runs) * sizeof(uint32_t);
  const uint64_t total = slot_bytes + info_bytes + reverse_bytes;
  StagingRing::Span span;
  if (total > 0 &&
      !staging_.TryAllocate(total, kRowAlignment, StagingRing::Purpose::kRows, span)) {
    return false;
  }

  // Slot rows.
  const VkBuffer table = slot_tables_.buffer();
  const VkDeviceSize range_base = slot_tables_.ByteOffset(group.slot_rows.offset);
  for (uint64_t k = 0; k < slot_rows; ++k) {
    const uint16_t id = group.full_rewrite ? static_cast<uint16_t>(k) : group.dirty_ids[k];
    const SlotRow row = ToRow(group.slots[id]);
    std::memcpy(span.data + k * kSlotRowBytes, &row, sizeof(row));
    if (!group.full_rewrite) {
      pending_.push_back(PendingCopy{span.offset + k * kSlotRowBytes, table,
                                     range_base + static_cast<VkDeviceSize>(id) * kSlotRowBytes,
                                     kSlotRowBytes});
    }
  }
  if (group.full_rewrite && slot_rows > 0) {
    pending_.push_back(PendingCopy{span.offset, table, range_base, slot_bytes});
  }

  // GroupInfo row.
  uint64_t cursor = slot_bytes;
  if (group.info_dirty) {
    GroupInfo info{};
    info.coord[0] = group.gx;
    info.coord[1] = group.gy;
    info.coord[2] = group.gz;
    info.entity = group.entity;
    info.slot_row_base = group.slot_rows.offset;
    info.reverse_base = group.reverse_range.offset;
    info.aabb_row_base = group.aabb_rows.offset;
    info.high_water = static_cast<uint32_t>(group.slots.size());
    std::memcpy(span.data + cursor, &info, sizeof(info));
    pending_.push_back(PendingCopy{span.offset + cursor, group_infos_.buffer(),
                                   static_cast<VkDeviceSize>(group_index) * sizeof(GroupInfo),
                                   sizeof(GroupInfo)});
    cursor += sizeof(GroupInfo);
  }

  // Reverse table words: two u16 entries per word, the even position in the low half.
  const VkDeviceSize reverse_base = reverse_tables_.ByteOffset(group.reverse_range.offset);
  for (const WordRun& run : reverse_runs) {
    const uint64_t run_begin = cursor;
    for (uint32_t word = run.first; word <= run.last; ++word) {
      const uint32_t value = static_cast<uint32_t>(group.id_by_pos[2 * word]) |
                             (static_cast<uint32_t>(group.id_by_pos[2 * word + 1]) << 16);
      std::memcpy(span.data + cursor, &value, sizeof(value));
      cursor += sizeof(uint32_t);
    }
    pending_.push_back(PendingCopy{span.offset + run_begin, reverse_tables_.buffer(),
                                   reverse_base + static_cast<VkDeviceSize>(run.first) * 4,
                                   cursor - run_begin});
  }

  // NaN rows of freed ids.
  for (const uint16_t id : group.nan_ids) {
    pending_.push_back(PendingCopy{nan_row_offset_, aabbs_.buffer(),
                                   aabbs_.ByteOffset(group.aabb_rows.offset + id), kAabbRowBytes});
  }

  group.full_rewrite = false;
  group.info_dirty = false;
  group.dirty_ids.clear();
  group.reverse_dirty_words.clear();
  group.nan_ids.clear();
  return true;
}

bool Scene::StageEntity(CaelixEntityId id, Entity& entity) {
  SortUnique(entity.grid_dirty_cells);
  std::vector<WordRun> cell_runs;
  BuildRuns(entity.grid_dirty_cells, cell_runs);
  const uint64_t info_bytes = entity.info_dirty ? sizeof(EntityInfo) : 0;
  const uint64_t cell_bytes = RunWords(cell_runs) * sizeof(uint32_t);
  const uint64_t total = info_bytes + cell_bytes;
  if (total == 0) {
    return true;
  }
  StagingRing::Span span;
  if (!staging_.TryAllocate(total, kRowAlignment, StagingRing::Purpose::kRows, span)) {
    return false;
  }
  uint64_t cursor = 0;
  if (entity.info_dirty) {
    EntityInfo info{};
    for (int a = 0; a < 3; ++a) {
      info.grid_min[a] = entity.grid_min[a];
      info.grid_dims[a] = entity.grid_dims[a];
    }
    info.grid_base = entity.grid_range.offset;
    std::memcpy(span.data, &info, sizeof(info));
    pending_.push_back(PendingCopy{span.offset, entity_infos_.buffer(),
                                   static_cast<VkDeviceSize>(id) * sizeof(EntityInfo),
                                   sizeof(EntityInfo)});
    cursor += sizeof(EntityInfo);
  }
  for (const WordRun& run : cell_runs) {
    const uint64_t bytes = (static_cast<uint64_t>(run.last) - run.first + 1) * sizeof(uint32_t);
    std::memcpy(span.data + cursor, &entity.grid[run.first], bytes);
    pending_.push_back(PendingCopy{span.offset + cursor, grids_.buffer(),
                                   grids_.ByteOffset(entity.grid_range.offset + run.first), bytes});
    cursor += bytes;
  }
  entity.info_dirty = false;
  entity.grid_dirty_cells.clear();
  return true;
}

bool Scene::StagePageTable() {
  const uint32_t pages = records_.page_count();
  if (pages_staged_ == pages) {
    return true;
  }
  const uint64_t count = pages - pages_staged_;
  StagingRing::Span span;
  if (!staging_.TryAllocate(count * sizeof(uint64_t), 8, StagingRing::Purpose::kRows, span)) {
    return false;
  }
  for (uint32_t page = pages_staged_; page < pages; ++page) {
    const uint64_t address = records_.PageAddress(page);
    std::memcpy(span.data + (page - pages_staged_) * sizeof(uint64_t), &address, sizeof(address));
  }
  pending_.push_back(PendingCopy{span.offset, page_table_.buffer(),
                                 static_cast<VkDeviceSize>(pages_staged_) * sizeof(uint64_t),
                                 count * sizeof(uint64_t)});
  pages_staged_ = pages;
  return true;
}

uint32_t Scene::StageDirtyList() {
  if (dirty_.empty()) {
    return 0;
  }
  // GPU allocation first, so a throw leaves the queue untouched.
  dirty_buffer_.Reserve(dirty_.size() * sizeof(DirtyEntry));

  // Drop entries whose brick was removed (or whose slot was reused and queued
  // again) and duplicates: a valid entry still carries the current epoch.
  uint32_t next_epoch = finalize_epoch_ + 1;
  if (next_epoch == 0) {
    next_epoch = 1;
  }
  std::vector<DirtyBrick> valid;
  valid.reserve(dirty_.size());
  for (const DirtyBrick& brick : dirty_) {
    if (brick.group_index >= groups_.size()) {
      continue;
    }
    Group& group = groups_[brick.group_index];
    if (!group.alive || brick.id >= group.slots.size()) {
      continue;
    }
    BrickSlotCpu& slot = group.slots[brick.id];
    if (slot.state != kSlotResident || slot.finalize_epoch != finalize_epoch_) {
      continue;
    }
    slot.finalize_epoch = next_epoch;  // seen: later duplicates are dropped
    valid.push_back(brick);
  }

  // As many entries as the ring holds; the rest waits for the next flush.
  uint64_t fit = valid.size();
  StagingRing::Span span;
  while (fit > 0 && !staging_.TryAllocate(fit * sizeof(DirtyEntry), kRowAlignment,
                                          StagingRing::Purpose::kRows, span)) {
    fit /= 2;
  }
  for (uint64_t k = 0; k < fit; ++k) {
    BrickSlotCpu& slot = groups_[valid[k].group_index].slots[valid[k].id];
    const DirtyEntry entry{valid[k].group_index, valid[k].id, slot.record_index, 0};
    std::memcpy(span.data + k * sizeof(DirtyEntry), &entry, sizeof(entry));
    slot.finalize_epoch = 0;  // finalized at this flush
  }
  if (fit > 0) {
    pending_.push_back(
        PendingCopy{span.offset, dirty_buffer_.buffer(), 0, fit * sizeof(DirtyEntry)});
  }
  if (fit < valid.size()) {
    CAELIX_LOGD("caelix_scene_flush: the staging ring is full; %llu of %zu bricks wait for the "
                "next flush to be finalized",
                static_cast<unsigned long long>(valid.size() - fit), valid.size());
  }
  // The rest keeps its queue stamp: it equals the new epoch.
  dirty_.assign(valid.begin() + static_cast<ptrdiff_t>(fit), valid.end());
  finalize_epoch_ = next_epoch;
  finalized_bricks_total_ += fit;
  return static_cast<uint32_t>(fit);
}

void Scene::RecordCopies(VkCommandBuffer cmd) {
  // Sort by destination. The sort is stable, so among copies to the same
  // destination (a brick uploaded twice, or a record freed and reused in one
  // window) the last one staged stays last.
  std::stable_sort(pending_.begin(), pending_.end(),
                   [](const PendingCopy& a, const PendingCopy& b) {
                     const uint64_t da = HandleToU64(a.dst);
                     const uint64_t db = HandleToU64(b.dst);
                     return da != db ? da < db : a.dst_offset < b.dst_offset;
                   });
  // Keep only the last copy per destination: regions of one vkCmdCopyBuffer must
  // not overlap, and the newest data must win.
  size_t kept = 0;
  for (size_t i = 0; i < pending_.size(); ++i) {
    if (kept > 0 && pending_[kept - 1].dst == pending_[i].dst &&
        pending_[kept - 1].dst_offset == pending_[i].dst_offset) {
      pending_[kept - 1] = pending_[i];
    } else {
      pending_[kept++] = pending_[i];
    }
  }
  pending_.resize(kept);

  // Merge runs that are contiguous in both the ring and the destination, then
  // record one vkCmdCopyBuffer per destination buffer (chunked).
  const VkBuffer staging = staging_.buffer();
  std::vector<VkBufferCopy> regions;
  regions.reserve(std::min<size_t>(pending_.size(), kMaxRegionsPerCopy));
  size_t i = 0;
  while (i < pending_.size()) {
    const VkBuffer dst = pending_[i].dst;
    regions.clear();
    for (; i < pending_.size() && pending_[i].dst == dst; ++i) {
      const PendingCopy& copy = pending_[i];
      if (!regions.empty()) {
        VkBufferCopy& last = regions.back();
        if (last.dstOffset + last.size == copy.dst_offset &&
            last.srcOffset + last.size == copy.src_offset) {
          last.size += copy.size;
          continue;
        }
        if (regions.size() == kMaxRegionsPerCopy) {
          vkCmdCopyBuffer(cmd, staging, dst, static_cast<uint32_t>(regions.size()),
                          regions.data());
          regions.clear();
        }
      }
      regions.push_back(VkBufferCopy{copy.src_offset, copy.dst_offset, copy.size});
    }
    if (!regions.empty()) {
      vkCmdCopyBuffer(cmd, staging, dst, static_cast<uint32_t>(regions.size()), regions.data());
    }
  }
}

void Scene::Recycle(uint64_t safe_frame) {
  safe_frame_ = std::max(safe_frame_, safe_frame);
  staging_.Recycle(safe_frame_);
  std::erase_if(deferred_buffers_,
                [this](const DeferredBuffer& deferred) { return deferred.frame <= safe_frame_; });
}

CaelixResult Scene::Flush(const CaelixFrameDesc& desc) {
  if (desc.safe_frame_number >= desc.frame_number) {
    CAELIX_LOGE("caelix_scene_flush: safe_frame_number %llu must be below frame_number %llu",
                static_cast<unsigned long long>(desc.safe_frame_number),
                static_cast<unsigned long long>(desc.frame_number));
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (has_flushed_ && desc.frame_number <= last_frame_) {
    CAELIX_LOGE("caelix_scene_flush: frame_number %llu is not above the previous flush's %llu",
                static_cast<unsigned long long>(desc.frame_number),
                static_cast<unsigned long long>(last_frame_));
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  const VkCommandBuffer cmd = HandleFromU64<VkCommandBuffer>(desc.command_buffer);
  const uint64_t frame = desc.frame_number;

  // 1. Release what the GPU has finished with.
  Recycle(desc.safe_frame_number);

  // 2. GPU-side table work that needs no staging space: AABB range moves and
  //    NaN fills, reverse table and group grid fills of new ranges.
  for (const uint32_t group_index : dirty_groups_) {
    Group& group = groups_[group_index];
    if (group.alive) {
      PrepareGroupAabbs(group);
    }
  }
  for (const CaelixEntityId id : dirty_entities_) {
    Entity& entity = entities_[id];
    if (entity.alive && entity.grid_fill) {
      fills_.push_back(PendingFill{grids_.buffer(), grids_.ByteOffset(entity.grid_range.offset),
                                   grids_.ByteOffset(entity.grid_range.rows), kNoGroup});
      entity.grid_fill = false;
    }
  }

  // 3. Stage the tables: page table, then per group (slot rows, info row, reverse
  //    words, NaN rows) and per entity (info row, grid cells). A group or entity is
  //    staged whole or not at all; when the ring is full, the rest stays dirty for
  //    the next flush, and the kernel does not run (it needs consistent tables).
  bool ring_full = !StagePageTable();
  {
    std::vector<uint32_t> still_dirty;
    for (const uint32_t group_index : dirty_groups_) {
      Group& group = groups_[group_index];
      if (!group.alive) {
        group.in_dirty_list = false;
        continue;
      }
      if (ring_full || !StageGroup(group_index, group)) {
        ring_full = true;
        still_dirty.push_back(group_index);
        continue;
      }
      group.in_dirty_list = false;
    }
    dirty_groups_.swap(still_dirty);
  }
  {
    std::vector<CaelixEntityId> still_dirty;
    for (const CaelixEntityId id : dirty_entities_) {
      Entity& entity = entities_[id];
      if (!entity.alive) {
        entity.in_dirty_list = false;
        continue;
      }
      if (ring_full || !StageEntity(id, entity)) {
        ring_full = true;
        still_dirty.push_back(id);
        continue;
      }
      entity.in_dirty_list = false;
    }
    dirty_entities_.swap(still_dirty);
  }
  if (ring_full) {
    CAELIX_LOGW("caelix_scene_flush: the staging ring is full; %zu groups and %zu entities keep "
                "dirty table rows for the next flush, and finalization waits",
                dirty_groups_.size(), dirty_entities_.size());
  }

  // 4. The finalize list.
  const uint32_t dispatch_count = ring_full ? 0 : StageDirtyList();

  // 5. Pool growth since the last flush: the old contents are copied first, at
  //    the same offsets. The dirty list buffer is rewritten every flush and needs
  //    no copy.
  struct GrowthCopy {
    GpuBuffer source;
    VkBuffer destination;
  };
  std::vector<GrowthCopy> growth;
  const auto take_growth = [&](std::optional<GpuBuffer> source, VkBuffer destination) {
    if (source.has_value()) {
      growth.push_back(GrowthCopy{std::move(*source), destination});
    }
  };
  take_growth(slot_tables_.TakeGrowthSource(), slot_tables_.buffer());
  take_growth(aabbs_.TakeGrowthSource(), aabbs_.buffer());
  take_growth(reverse_tables_.TakeGrowthSource(), reverse_tables_.buffer());
  take_growth(grids_.TakeGrowthSource(), grids_.buffer());
  take_growth(group_infos_.TakeGrowthSource(), group_infos_.buffer());
  take_growth(entity_infos_.TakeGrowthSource(), entity_infos_.buffer());
  take_growth(page_table_.TakeGrowthSource(), page_table_.buffer());
  if (std::optional<GpuBuffer> old = dirty_buffer_.TakeGrowthSource(); old.has_value()) {
    deferred_buffers_.push_back(DeferredBuffer{frame, std::move(*old)});
  }

  if (!pending_.empty() || !growth.empty() || !fills_.empty() || !aabb_moves_.empty() ||
      dispatch_count > 0) {
    staging_.FlushHostWrites();

    // 6. Earlier commands (shader reads of reused records and rows included)
    //    complete before the transfers write.
    MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                  VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // 7. Growth copies.
    if (!growth.empty()) {
      for (GrowthCopy& copy : growth) {
        const VkBufferCopy region{0, 0, copy.source.size()};
        vkCmdCopyBuffer(cmd, copy.source.buffer(), copy.destination, 1, &region);
        deferred_buffers_.push_back(DeferredBuffer{frame, std::move(copy.source)});
      }
      TransferBarrier(cmd);
    }

    // 8. AABB range moves and fills of new ranges. They come before the staged
    //    copies, which may write NaN rows or table words into the same ranges.
    if (!fills_.empty() || !aabb_moves_.empty()) {
      for (size_t i = 0; i < aabb_moves_.size(); i += kMaxRegionsPerCopy) {
        const size_t n = std::min(kMaxRegionsPerCopy, aabb_moves_.size() - i);
        vkCmdCopyBuffer(cmd, aabbs_.buffer(), aabbs_.buffer(), static_cast<uint32_t>(n),
                        aabb_moves_.data() + i);
      }
      for (const PendingFill& fill : fills_) {
        vkCmdFillBuffer(cmd, fill.buffer, fill.offset, fill.size, fill.value);
      }
      TransferBarrier(cmd);
    }

    // 9. Staging -> pools and tables.
    RecordCopies(cmd);

    // 10. Finalize the dirty bricks, then later commands see everything.
    if (dispatch_count > 0) {
      MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
      FinalizePush push{};
      push.page_table = page_table_.address();
      push.slot_tables = slot_tables_.address();
      push.reverse_tables = reverse_tables_.address();
      push.group_infos = group_infos_.address();
      push.entity_infos = entity_infos_.address();
      push.group_grids = grids_.address();
      push.aabb_pool = aabbs_.address();
      push.records_per_page_shift = static_cast<uint32_t>(
          std::countr_zero(static_cast<uint32_t>(records_.page_bytes() / layout_.record_bytes)));
      push.record_bytes = layout_.record_bytes;
      push.group_shift = group_shift_;
      push.encoding = encoding_;
      for (uint32_t base = 0; base < dispatch_count; base += kMaxDispatchGroups) {
        const uint32_t groups = std::min(kMaxDispatchGroups, dispatch_count - base);
        push.dirty_list = dirty_buffer_.address() + static_cast<uint64_t>(base) * sizeof(DirtyEntry);
        push.dirty_count = groups;
        finalize_.Dispatch(cmd, &push, sizeof(push), groups);
      }
      MemoryBarrier(cmd,
                    VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    } else {
      MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    }
  }

  // 11. Close the window.
  staging_.CloseFrame(frame);
  pending_.clear();
  fills_.clear();
  aabb_moves_.clear();
  nan_row_valid_ = false;
  pending_bricks_ = 0;
  for (const RangeToFree& item : ranges_to_free_after_flush_) {
    item.pool->Free(item.range);
  }
  ranges_to_free_after_flush_.clear();
  last_frame_ = frame;
  has_flushed_ = true;
  return CAELIX_OK;
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
  return CAELIX_OK;
}

}  // namespace caelix
