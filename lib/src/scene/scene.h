#pragma once
#include "caelix/caelix.h"
#include "core/hash_map.h"
#include "rhi/alloc/as_pool.h"
#include "rhi/alloc/record_pool.h"
#include "rhi/alloc/row_pool.h"
#include "rhi/device.h"
#include "rhi/gpu_buffer.h"
#include "rhi/gpu_timer.h"
#include "rhi/pipeline.h"
#include "rhi/staging_ring.h"
#include "scene/gpu_structs.h"
#include "scene/material_table.h"
#include "scene/record_layout.h"
#include "scene/slot_table_pool.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace caelix {

// CPU copy of one slot table row.
struct BrickSlotCpu
{
    uint32_t pos = 0;
    uint8_t state = kSlotFree;
    uint32_t record_index = RecordPool::kInvalidIndex;
    // == Scene::finalize_epoch_ while the brick is queued in Scene::dirty_; dedupes
    // the queue. 0 = not queued (epochs start at 1).
    uint32_t finalize_epoch = 0;
};

// A render group: (1 << group_shift)^3 bricks of one entity with dense 16-bit
// brick ids. The ids index the group's slot table rows and AABB rows.
struct Group
{
    static constexpr uint16_t kNoId = kNoBrickId;

    CaelixEntityId entity = CAELIX_INVALID_ENTITY;
    int32_t gx = 0, gy = 0, gz = 0;
    uint64_t key = 0;                 // packed group key in the entity's map
    std::vector<uint16_t> id_by_pos;  // bricks_per_group entries, kNoId = none
    std::vector<BrickSlotCpu> slots;  // by id, size = high water
    std::vector<uint16_t> free_ids;   // LIFO
    SlotRange slot_rows;              // capacity is a power of two >= slots.size()
    RowRange aabb_rows;               // AABB rows, same capacity as slot_rows
    // The AABB range whose rows the GPU holds. It differs from aabb_rows after a
    // growth until the next flush copies the rows over (invalid = new group).
    RowRange aabb_gpu_rows;
    RowRange reverse_range;    // bricks_per_group u16 entries: position -> id
    uint32_t brick_count = 0;  // ids in use
    bool alive = false;
    bool full_rewrite = false;                  // all rows restaged at the next flush (the range moved)
    bool in_dirty_list = false;                 // listed in Scene::dirty_groups_; survives free and reuse
    bool info_dirty = false;                    // the GroupInfo row is restaged at the next flush
    bool reverse_fill = false;                  // the reverse range is new: filled with 0xFFFF first
    std::vector<uint16_t> dirty_ids;            // rows to restage (ignored when full_rewrite)
    std::vector<uint32_t> reverse_dirty_words;  // reverse table u32 words (pos >> 1) to restage
    std::vector<uint16_t> nan_ids;              // freed ids whose AABB row becomes NaN

    // Bottom-level acceleration structure over the AABB rows [0, high water).
    AsBlock blas;  // storage; invalid = never built
    VkAccelerationStructureKHR blas_handle = VK_NULL_HANDLE;
    VkDeviceAddress blas_address = 0;
    uint32_t blas_primitive_count = 0;  // high water of the last build
    // The BLAS is out of date: a brick of the group was queued for finalization or
    // removed, or the AABB range moved. Cleared when a build is recorded.
    bool blas_dirty = false;
    bool in_blas_list = false;  // listed in Scene::blas_pending_; survives free and reuse
    // == Scene::flush_serial_ while finalize entries of the group wait for a later
    // flush: the BLAS build waits too.
    uint64_t finalize_wait_serial = 0;
};

struct Entity
{
    uint8_t guid[16] = {};
    float position[3] = {};
    float rotation[4] = {0, 0, 0, 1};
    ankerl::unordered_dense::map<uint64_t, uint32_t> groups;  // packed group key -> group index
    bool alive = false;

    // Group grid: a dense array of group indices (kNoGroup = none) over the group
    // coordinates [grid_min, grid_min + grid_dims). Dims are powers of two and grow
    // geometrically; the grid is rebuilt (and fully restaged) when it grows.
    int32_t grid_min[3] = {};
    int32_t grid_dims[3] = {};   // 0 = no grid yet
    RowRange grid_range;         // in Scene::grids_
    std::vector<uint32_t> grid;  // CPU copy of the cells
    bool has_bounds = false;     // bounds of every group placed since the entity was added
    int32_t bounds_min[3] = {};
    int32_t bounds_max[3] = {};
    bool grid_fill = false;  // the grid range is new: filled, then every used cell restaged
    std::vector<uint32_t> grid_dirty_cells;
    bool info_dirty = false;     // the EntityInfo row is restaged at the next flush
    bool in_dirty_list = false;  // listed in Scene::dirty_entities_

    // Object-to-world rows (3x4, rotation | position) as staged in the EntityInfo
    // row: the current transform and the previous frame's transform.
    float cur_rows[3][4] = {};
    float prev_rows[3][4] = {};
    // position/rotation changed since the last flush: the next flush moves cur to
    // prev and computes cur.
    bool transform_changed = false;
};

class Scene
{
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
    void GetTlas(uint64_t& handle, uint64_t& address) const;
    // Materials (scene_materials.cpp). The changed part of the table is uploaded at
    // the next flush.
    CaelixResult SetMaterials(const CaelixMaterial* entries, uint32_t count);
    CaelixResult SetGlassPalette(const CaelixMaterial* entries, uint32_t count);
    CaelixResult GetMaterial(uint32_t code, CaelixMaterial& out) const;

    bool EntityAlive(CaelixEntityId entity) const
    {
        return entity < entities_.size() && entities_[entity].alive;
    }

    // What the renderer binds and reads. The addresses and the TLAS can change at
    // every flush (pool growth, TLAS rebuild into a new block): read them after the
    // flush of the frame that uses them.
    Device& device() const
    {
        return device_;
    }
    VkAccelerationStructureKHR tlas() const
    {
        return tlas_handle_;
    }
    VkDeviceAddress page_table_address() const
    {
        return page_table_.address();
    }
    VkDeviceAddress slot_tables_address() const
    {
        return slot_tables_.address();
    }
    VkDeviceAddress group_infos_address() const
    {
        return group_infos_.address();
    }
    VkDeviceAddress entity_infos_address() const
    {
        return entity_infos_.address();
    }
    // The GPU material table: 65536 CaelixMaterial entries by block code.
    VkDeviceAddress materials_address() const
    {
        return material_buffer_.address();
    }
    uint32_t records_per_page_shift() const;
    uint32_t record_bytes() const
    {
        return layout_.record_bytes;
    }
    uint32_t group_shift() const
    {
        return group_shift_;
    }
    uint32_t encoding() const
    {
        return encoding_;
    }
    // GPU time of the flush of `frame`, once that frame is safe. False when the
    // flush recorded no work, was not measured, or is not resolved yet.
    bool FlushMsForFrame(uint64_t frame, float& ms) const
    {
        return flush_timer_.MsForFrame(frame, ms);
    }

private:
    struct Location
    {
        int32_t gx, gy, gz;
        uint64_t group_key;
        uint32_t pos;
    };
    struct Placed
    {
        uint32_t record;
        uint32_t group_index;
        uint16_t id;
    };
    struct PendingCopy
    {
        VkDeviceSize src_offset;
        VkBuffer dst;
        VkDeviceSize dst_offset;
        VkDeviceSize size;
    };
    struct PendingFill
    {
        VkBuffer buffer;
        VkDeviceSize offset;
        VkDeviceSize size;
        uint32_t value;
    };
    struct DeferredBuffer
    {
        uint64_t frame;  // destroy when the safe frame is >= this
        GpuBuffer buffer;
    };
    struct RangeToFree
    {
        RowPool* pool;
        RowRange range;
    };
    struct DirtyBrick
    {
        uint32_t group_index;
        uint16_t id;
    };
    struct GridShape
    {
        int32_t min[3];
        int32_t dims[3];
        uint64_t cells;
    };
    struct RetiredBlas
    {
        AsBlock block;
        VkAccelerationStructureKHR handle;
    };

    explicit Scene(Device& device);

    static bool KeyInRange(const CaelixBrickKey& key);
    Location Locate(const CaelixBrickKey& key) const;
    // Finds or creates the slot of a brick and gives it a record. GPU allocations
    // come first, so a VulkanError leaves the scene unchanged.
    Placed PlaceBrick(CaelixEntityId entity, const Location& loc);
    uint32_t CreateGroup(CaelixEntityId entity, const Location& loc, SlotRange slot_rows,
                         RowRange aabb_rows, RowRange reverse_range);
    void ReleaseGroup(uint32_t group_index);
    void ListGroup(uint32_t group_index);
    void ListEntity(CaelixEntityId entity);
    void MarkDirty(uint32_t group_index, uint16_t id);
    void MarkReverseDirty(Group& group, uint32_t pos);
    void FreeAfterFlush(RowPool& pool, RowRange range);

    // Group grid of an entity.
    bool GridContains(const Entity& entity, int32_t gx, int32_t gy, int32_t gz) const;
    GridShape GrownGridShape(const Entity& entity, int32_t gx, int32_t gy, int32_t gz) const;
    // Rebuilds the CPU grid with a new shape and range (no GPU allocation).
    void RebuildGrid(Entity& entity, const GridShape& shape, RowRange range);
    uint32_t GridCell(const Entity& entity, int32_t gx, int32_t gy, int32_t gz) const;
    void SetGridCell(Entity& entity, int32_t gx, int32_t gy, int32_t gz, uint32_t value);

    // Finalize queue.
    void QueueFinalize(uint32_t group_index, uint16_t id);
    // Queues the resident face neighbours of `key`. `group_index` is the live group
    // that contains `key` (its neighbours inside it need no lookup), or 0xFFFFFFFF.
    void QueueNeighbors(CaelixEntityId entity, const CaelixBrickKey& key, uint32_t group_index);

    void WriteRecord(uint8_t* dst, uint32_t pos, const uint8_t* blocks) const;
    // AABB range moves and fills of a group (no staging space needed).
    void PrepareGroupAabbs(Group& group);
    // Writes the dirty rows of one group (slot rows, info row, reverse words, NaN
    // rows) into staging and appends their copies. Returns false (and changes
    // nothing) when the ring has no room.
    bool StageGroup(uint32_t group_index, Group& group);
    bool StageEntity(CaelixEntityId id, Entity& entity);
    bool StagePageTable();
    // Zero-fills the GPU table at the first flush, then stages the dirty chunks of
    // the material table that fit into the ring; the rest wait for a later flush.
    void StageMaterials();
    // Builds the dispatch list from dirty_. Returns the number of entries staged.
    uint32_t StageDirtyList();
    bool EnsureNanRow();
    void RecordCopies(VkCommandBuffer cmd);

    // Acceleration structures.
    void MarkBlasDirty(uint32_t group_index);
    // The rows of an entity's current transform.
    static void TransformRows(const float position[3], const float rotation[4], float rows[3][4]);
    VkAccelerationStructureBuildSizesInfoKHR BlasSizes(uint32_t primitive_count);
    VkAccelerationStructureBuildSizesInfoKHR TlasSizes(uint32_t instance_count) const;
    // Chooses the builds of this flush, allocates their storage and scratch, stages
    // the TLAS instances and fills blas_infos_ / tlas_info_. Returns false when
    // nothing is built. Sets `out_of_memory` (and builds nothing) when storage
    // cannot be allocated.
    bool PlanAccelerationStructures(uint64_t frame, bool& out_of_memory);
    void RecordAccelerationStructureBuilds(VkCommandBuffer cmd);
    void DestroyAccelerationStructures();

    Device& device_;
    RecordLayout layout_;
    uint32_t group_shift_ = 4;
    uint32_t bricks_per_group_ = 0;
    uint32_t encoding_ = 0;

    std::vector<Entity> entities_;
    std::vector<uint32_t> free_entities_;
    std::vector<Group> groups_;
    std::vector<uint32_t> free_groups_;
    std::vector<uint32_t> dirty_groups_;
    std::vector<CaelixEntityId> dirty_entities_;

    RecordPool records_;
    SlotTablePool slot_tables_;
    RowPool aabbs_;                // 24 B rows
    RowPool reverse_tables_;       // u16 rows
    RowPool grids_;                // u32 rows
    GrowableBuffer group_infos_;   // GroupInfo by group index
    GrowableBuffer entity_infos_;  // EntityInfo by entity id
    GrowableBuffer page_table_;    // uint64_t page device address by page
    GrowableBuffer dirty_buffer_;  // DirtyEntry list of the current flush
    uint32_t pages_staged_ = 0;
    StagingRing staging_;
    std::unique_ptr<MaterialTable> materials_;  // CPU copy of the material table
    GpuBuffer material_buffer_;                 // MaterialTable::kBytes, device local
    bool material_buffer_filled_ = false;       // the zero fill of the first flush was recorded
    ComputePipeline finalize_;
    GpuTimer flush_timer_;  // brackets the work that a flush records

    // Acceleration structures.
    BlockAllocator as_pool_;
    GpuBuffer scratch_;         // build scratch of one flush, bump-allocated
    GrowableBuffer instances_;  // VkAccelerationStructureInstanceKHR per TLAS instance
    AsBlock tlas_;
    VkAccelerationStructureKHR tlas_handle_ = VK_NULL_HANDLE;
    VkDeviceAddress tlas_address_ = 0;
    bool tlas_dirty_ = false;
    std::vector<uint32_t> blas_pending_;           // groups with blas_dirty (may hold stale entries)
    std::vector<RetiredBlas> retired_blas_;        // BLASes of freed groups; the current TLAS uses them
    std::vector<CaelixEntityId> motion_entities_;  // staged with prev != cur at the last flush
    ankerl::unordered_dense::map<uint32_t, VkAccelerationStructureBuildSizesInfoKHR> blas_sizes_;
    // The builds planned for the current flush.
    std::vector<VkAccelerationStructureGeometryKHR> blas_geometries_;
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> blas_infos_;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> blas_ranges_;
    bool tlas_planned_ = false;
    VkAccelerationStructureGeometryKHR tlas_geometry_{};
    VkAccelerationStructureBuildGeometryInfoKHR tlas_info_{};
    VkAccelerationStructureBuildRangeInfoKHR tlas_range_{};
    uint64_t flush_serial_ = 0;
    uint64_t blas_count_ = 0;
    uint64_t as_builds_total_ = 0;

    std::vector<PendingCopy> pending_;
    std::vector<PendingFill> fills_;
    std::vector<VkBufferCopy> aabb_moves_;  // inside aabbs_.buffer()
    std::vector<RangeToFree> ranges_to_free_after_flush_;
    std::vector<DeferredBuffer> deferred_buffers_;

    std::vector<DirtyBrick> dirty_;
    uint32_t finalize_epoch_ = 1;
    // The 24 B NaN row of the open flush window in the staging ring (copied to every
    // freed AABB row); valid only while nan_row_valid_.
    VkDeviceSize nan_row_offset_ = 0;
    bool nan_row_valid_ = false;

    uint64_t last_frame_ = 0;
    bool has_flushed_ = false;
    uint64_t safe_frame_ = 0;

    uint32_t entities_alive_ = 0;
    uint32_t groups_alive_ = 0;
    uint64_t bricks_resident_ = 0;
    uint64_t bricks_known_ = 0;
    uint64_t uploaded_bricks_total_ = 0;
    uint64_t pending_bricks_ = 0;
    uint64_t finalized_bricks_total_ = 0;
};

}  // namespace caelix
