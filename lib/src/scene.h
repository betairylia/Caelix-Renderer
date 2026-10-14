#pragma once
#include "caelix/caelix.h"
#include "device.h"
#include "gpu_buffer.h"
#include "hash_map.h"
#include "record_layout.h"
#include "record_pool.h"
#include "slot_table_pool.h"
#include "staging_ring.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace caelix {

// CPU copy of one slot table row.
struct BrickSlotCpu {
  uint32_t pos = 0;
  uint8_t state = kSlotFree;
  uint32_t record_index = RecordPool::kInvalidIndex;
};

// A render group: (1 << group_shift)^3 bricks of one entity with dense 16-bit
// brick ids. The ids index the group's slot table rows.
struct Group {
  static constexpr uint16_t kNoId = 0xFFFF;

  CaelixEntityId entity = CAELIX_INVALID_ENTITY;
  int32_t gx = 0, gy = 0, gz = 0;
  uint64_t key = 0;                  // packed group key in the entity's map
  std::vector<uint16_t> id_by_pos;   // bricks_per_group entries, kNoId = none
  std::vector<BrickSlotCpu> slots;   // by id, size = high water
  std::vector<uint16_t> free_ids;    // LIFO
  SlotRange slot_rows;               // capacity is a power of two >= slots.size()
  uint32_t brick_count = 0;          // ids in use
  bool alive = false;
  bool full_rewrite = false;         // all rows restaged at the next flush (the range moved)
  bool in_dirty_list = false;        // listed in Scene::dirty_groups_; survives free and reuse
  std::vector<uint16_t> dirty_ids;   // rows to restage (ignored when full_rewrite)
};

struct Entity {
  uint8_t guid[16] = {};
  float position[3] = {};
  float rotation[4] = {0, 0, 0, 1};
  ankerl::unordered_dense::map<uint64_t, uint32_t> groups;  // packed group key -> group index
  bool alive = false;
};

class Scene {
 public:
  static CaelixResult Create(Device& device, const CaelixSceneDesc* desc,
                             std::unique_ptr<Scene>& out);
  ~Scene();
  Scene(const Scene&) = delete;
  Scene& operator=(const Scene&) = delete;

  CaelixResult AddEntity(const CaelixEntityDesc& desc, CaelixEntityId& out_id);
  CaelixResult RemoveEntity(CaelixEntityId entity);
  CaelixResult SetEntityTransform(CaelixEntityId entity, const float position[3],
                                  const float rotation[4]);
  // `consumed` is updated brick by brick, so it stays correct when a later
  // brick throws.
  CaelixResult UploadBricks(CaelixEntityId entity, uint32_t count, const CaelixBrickKey* keys,
                            const void* blocks, uint32_t& consumed);
  CaelixResult RemoveBricks(CaelixEntityId entity, uint32_t count, const CaelixBrickKey* keys);
  CaelixResult Flush(const CaelixFrameDesc& desc);
  void Recycle(uint64_t safe_frame);
  void GetStats(CaelixSceneStats& out) const;
  CaelixResult QueryBrick(CaelixEntityId entity, CaelixBrickKey key, CaelixBrickInfo& out) const;

  bool EntityAlive(CaelixEntityId entity) const {
    return entity < entities_.size() && entities_[entity].alive;
  }

 private:
  struct Location {
    int32_t gx, gy, gz;
    uint64_t group_key;
    uint32_t pos;
  };
  struct PendingCopy {
    VkDeviceSize src_offset;
    VkBuffer dst;
    VkDeviceSize dst_offset;
    VkDeviceSize size;
  };
  struct DeferredBuffer {
    uint64_t frame;  // destroy when the safe frame is >= this
    GpuBuffer buffer;
  };

  explicit Scene(Device& device);

  static bool KeyInRange(const CaelixBrickKey& key);
  Location Locate(const CaelixBrickKey& key) const;
  // Finds or creates the slot of a brick and gives it a record. Returns the
  // record index. GPU allocations come first, so a VulkanError leaves the scene
  // unchanged.
  uint32_t PlaceBrick(CaelixEntityId entity, const Location& loc);
  uint32_t CreateGroup(CaelixEntityId entity, const Location& loc, SlotRange rows);
  void ReleaseGroup(uint32_t group_index);
  void MarkDirty(uint32_t group_index, uint16_t id);
  void WriteRecord(uint8_t* dst, uint32_t pos, const uint8_t* blocks) const;
  // Writes the dirty rows of one group into staging and appends their copies.
  // Returns false (and changes nothing) when the ring has no room.
  bool StageRows(Group& group);
  void RecordCopies(VkCommandBuffer cmd);

  Device& device_;
  RecordLayout layout_;
  uint32_t group_shift_ = 4;
  uint32_t bricks_per_group_ = 0;

  std::vector<Entity> entities_;
  std::vector<uint32_t> free_entities_;
  std::vector<Group> groups_;
  std::vector<uint32_t> free_groups_;
  std::vector<uint32_t> dirty_groups_;

  RecordPool records_;
  SlotTablePool slot_tables_;
  StagingRing staging_;

  std::vector<PendingCopy> pending_;
  std::vector<SlotRange> ranges_to_free_after_flush_;
  std::vector<DeferredBuffer> deferred_buffers_;

  uint64_t last_frame_ = 0;
  bool has_flushed_ = false;
  uint64_t safe_frame_ = 0;

  uint32_t entities_alive_ = 0;
  uint32_t groups_alive_ = 0;
  uint64_t bricks_resident_ = 0;
  uint64_t bricks_known_ = 0;
  uint64_t uploaded_bricks_total_ = 0;
  uint64_t pending_bricks_ = 0;
};

}  // namespace caelix
