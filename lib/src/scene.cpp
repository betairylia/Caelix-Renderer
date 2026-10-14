#include "scene.h"

#include "log.h"

#include <algorithm>
#include <bit>
#include <cstring>
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

// Brick keys must fit 21 bits per axis.
constexpr int32_t kKeyLimit = 1 << 20;

// Staging alignment. Records are 8-byte multiples, so consecutive records stay
// contiguous in the ring and their copies merge. vkCmdCopyBuffer itself has no
// offset alignment rule.
constexpr uint64_t kRecordAlignment = 8;
constexpr uint64_t kRowAlignment = 16;

// vkCmdCopyBuffer regions per call.
constexpr size_t kMaxRegionsPerCopy = 4096;

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

  std::unique_ptr<Scene> scene(new Scene(device));
  scene->layout_ = layout;
  scene->group_shift_ = group_shift;
  scene->bricks_per_group_ = 1u << (3 * group_shift);
  scene->records_ =
      RecordPool(device.allocator(), device.device(), records_per_page, layout.record_bytes);
  scene->slot_tables_ = SlotTablePool(device.allocator(), device.device());
  scene->staging_ = StagingRing(device.allocator(), device.device(), staging_bytes);
  CAELIX_LOGI("scene created: staging %llu bytes, %u records per page (%llu bytes), group %u^3 "
              "bricks, record %u bytes",
              static_cast<unsigned long long>(staging_bytes), records_per_page,
              static_cast<unsigned long long>(page_bytes), 1u << group_shift,
              layout.record_bytes);
  out = std::move(scene);
  return CAELIX_OK;
}

// ---------------------------------------------------------------------------
// Entities

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
    entities_.emplace_back();
    id = static_cast<uint32_t>(entities_.size() - 1);
  }
  Entity& entity = entities_[id];
  std::memcpy(entity.guid, desc.guid, sizeof(entity.guid));
  std::memcpy(entity.position, desc.position, sizeof(entity.position));
  std::memcpy(entity.rotation, desc.rotation, sizeof(entity.rotation));
  entity.groups.clear();
  entity.alive = true;
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

uint32_t Scene::CreateGroup(CaelixEntityId entity, const Location& loc, SlotRange rows) {
  uint32_t index;
  if (!free_groups_.empty()) {
    index = free_groups_.back();
    free_groups_.pop_back();
  } else {
    groups_.emplace_back();
    index = static_cast<uint32_t>(groups_.size() - 1);
  }
  Group& group = groups_[index];
  group.entity = entity;
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
  group.slot_rows = rows;
  group.brick_count = 0;
  group.full_rewrite = false;
  group.alive = true;
  // in_dirty_list keeps its value: the index may still be listed from its previous life.
  entities_[entity].groups.emplace(loc.group_key, index);
  ++groups_alive_;
  return index;
}

void Scene::ReleaseGroup(uint32_t group_index) {
  Group& group = groups_[group_index];
  // The GPU may still read the rows of the last flushed frame; the range is
  // reused only after this window's flush (see Flush).
  ranges_to_free_after_flush_.push_back(group.slot_rows);
  group.slot_rows = SlotRange{};
  group.slots.clear();
  group.free_ids.clear();
  group.dirty_ids.clear();
  group.brick_count = 0;
  group.full_rewrite = false;
  group.alive = false;
  free_groups_.push_back(group_index);
  --groups_alive_;
}

void Scene::MarkDirty(uint32_t group_index, uint16_t id) {
  Group& group = groups_[group_index];
  if (!group.full_rewrite) {
    group.dirty_ids.push_back(id);
  }
  if (!group.in_dirty_list) {
    group.in_dirty_list = true;
    dirty_groups_.push_back(group_index);
  }
}

uint32_t Scene::PlaceBrick(CaelixEntityId entity_id, const Location& loc) {
  Entity& entity = entities_[entity_id];
  const auto found = entity.groups.find(loc.group_key);
  uint32_t group_index = found != entity.groups.end() ? found->second : 0xFFFFFFFFu;

  // Replace an existing brick: same id, same record, unchanged row.
  if (found != entity.groups.end()) {
    Group& group = groups_[group_index];
    const uint16_t id = group.id_by_pos[loc.pos];
    if (id != Group::kNoId) {
      BrickSlotCpu& slot = group.slots[id];
      if (slot.state != kSlotResident) {
        // A known brick without a record (states for later versions).
        slot.record_index = records_.Allocate();
        slot.state = kSlotResident;
        ++bricks_resident_;
        MarkDirty(group_index, id);
      }
      return slot.record_index;
    }
  }

  // A new brick. GPU allocations first: when one throws, nothing has changed.
  const uint32_t record = records_.Allocate();
  SlotRange new_rows{};
  try {
    if (found == entity.groups.end()) {
      new_rows = slot_tables_.Allocate(SlotTablePool::kMinRows);
    } else {
      const Group& group = groups_[group_index];
      if (group.free_ids.empty() && group.slots.size() == group.slot_rows.rows) {
        new_rows = slot_tables_.Allocate(group.slot_rows.rows * 2);
      }
    }
  } catch (...) {
    records_.Free(record);
    throw;
  }

  if (found == entity.groups.end()) {
    group_index = CreateGroup(entity_id, loc, new_rows);
  } else if (new_rows.valid()) {
    // The range doubles. The old range is freed after this window's flush; every
    // row is restaged into the new range at that flush.
    Group& group = groups_[group_index];
    ranges_to_free_after_flush_.push_back(group.slot_rows);
    group.slot_rows = new_rows;
    group.full_rewrite = true;
    group.dirty_ids.clear();
  }

  Group& group = groups_[group_index];
  uint16_t id;
  if (!group.free_ids.empty()) {
    id = group.free_ids.back();
    group.free_ids.pop_back();
  } else {
    id = static_cast<uint16_t>(group.slots.size());
    group.slots.emplace_back();
  }
  group.id_by_pos[loc.pos] = id;
  group.slots[id] = BrickSlotCpu{loc.pos, kSlotResident, record};
  ++group.brick_count;
  ++bricks_resident_;
  ++bricks_known_;
  MarkDirty(group_index, id);
  return record;
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
  record[0] = pos & 0xFFFFu;  // bits 16..23 (coarse occupancy) stay 0 until a kernel fills them
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
    const uint32_t record = PlaceBrick(entity, loc);
    WriteRecord(span.data, loc.pos, block_bytes + i * brick_bytes);
    pending_.push_back(PendingCopy{span.offset, records_.PageBuffer(record),
                                   records_.PageOffset(record), layout_.record_bytes});
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
    }
  }
  return CAELIX_OK;
}

// ---------------------------------------------------------------------------
// Flush

bool Scene::StageRows(Group& group) {
  const VkBuffer table = slot_tables_.buffer();
  const VkDeviceSize range_base =
      static_cast<VkDeviceSize>(group.slot_rows.offset) * SlotTablePool::kRowBytes;
  if (group.full_rewrite) {
    const uint64_t rows = group.slots.size();
    if (rows > 0) {
      StagingRing::Span span;
      if (!staging_.TryAllocate(rows * SlotTablePool::kRowBytes, kRowAlignment,
                                StagingRing::Purpose::kRows, span)) {
        return false;
      }
      for (uint64_t id = 0; id < rows; ++id) {
        const SlotRow row = ToRow(group.slots[id]);
        std::memcpy(span.data + id * SlotTablePool::kRowBytes, &row, sizeof(row));
      }
      pending_.push_back(
          PendingCopy{span.offset, table, range_base, rows * SlotTablePool::kRowBytes});
    }
    group.full_rewrite = false;
    group.dirty_ids.clear();
    return true;
  }

  std::sort(group.dirty_ids.begin(), group.dirty_ids.end());
  group.dirty_ids.erase(std::unique(group.dirty_ids.begin(), group.dirty_ids.end()),
                        group.dirty_ids.end());
  const uint64_t rows = group.dirty_ids.size();
  if (rows > 0) {
    StagingRing::Span span;
    if (!staging_.TryAllocate(rows * SlotTablePool::kRowBytes, kRowAlignment,
                              StagingRing::Purpose::kRows, span)) {
      return false;
    }
    for (uint64_t k = 0; k < rows; ++k) {
      const uint16_t id = group.dirty_ids[k];
      const SlotRow row = ToRow(group.slots[id]);
      const VkDeviceSize src = span.offset + k * SlotTablePool::kRowBytes;
      std::memcpy(span.data + k * SlotTablePool::kRowBytes, &row, sizeof(row));
      pending_.push_back(PendingCopy{src, table,
                                     range_base + static_cast<VkDeviceSize>(id) *
                                                      SlotTablePool::kRowBytes,
                                     SlotTablePool::kRowBytes});
    }
  }
  group.dirty_ids.clear();
  return true;
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

  // 2. Stage the dirty slot table rows. A group is staged whole or not at all;
  //    when the ring is full, the rest stays dirty for the next flush.
  {
    std::vector<uint32_t> still_dirty;
    bool ring_full = false;
    for (const uint32_t group_index : dirty_groups_) {
      Group& group = groups_[group_index];
      if (!group.alive || (!group.full_rewrite && group.dirty_ids.empty())) {
        group.in_dirty_list = false;
        continue;
      }
      if (ring_full || !StageRows(group)) {
        ring_full = true;
        still_dirty.push_back(group_index);
        continue;
      }
      group.in_dirty_list = false;
    }
    if (ring_full) {
      CAELIX_LOGW("caelix_scene_flush: the staging ring is full; %zu groups keep dirty slot rows "
                  "for the next flush",
                  still_dirty.size());
    }
    dirty_groups_.swap(still_dirty);
  }

  std::optional<GpuBuffer> growth_source = slot_tables_.TakeGrowthSource();
  if (!pending_.empty() || growth_source.has_value()) {
    staging_.FlushHostWrites();

    // 3. Earlier commands (shader reads of reused records and rows included)
    //    complete before the copies write.
    MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // 4. The slot table grew: copy the old contents first, at the same offsets.
    if (growth_source.has_value()) {
      const VkBufferCopy region{0, 0, growth_source->size()};
      vkCmdCopyBuffer(cmd, growth_source->buffer(), slot_tables_.buffer(), 1, &region);
      MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      deferred_buffers_.push_back(DeferredBuffer{frame, std::move(*growth_source)});
    }

    // 5. Staging -> pools.
    RecordCopies(cmd);

    // 6. Later commands see the copies.
    MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
  }

  // 7. Close the window.
  staging_.CloseFrame(frame);
  pending_.clear();
  pending_bricks_ = 0;
  for (const SlotRange& range : ranges_to_free_after_flush_) {
    slot_tables_.Free(range);
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
  out.slot_row_offset_bytes =
      (static_cast<uint64_t>(group.slot_rows.offset) + id) * SlotTablePool::kRowBytes;
  return CAELIX_OK;
}

}  // namespace caelix
