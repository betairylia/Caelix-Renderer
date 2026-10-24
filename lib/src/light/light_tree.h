#pragma once
#include "caelix/caelix.h"
#include "rhi/alloc/row_pool.h"
#include "rhi/device.h"
#include "rhi/gpu_buffer.h"
#include "rhi/gpu_timer.h"
#include "rhi/pipeline.h"
#include "rhi/staging_ring.h"
#include "light/light_tree_layout.h"
#include "scene/staged_copy.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace caelix {

class MaterialTable;

// What the light tree reads from the scene at a flush.
struct LightSceneTables
{
    VkDeviceAddress page_table = 0;
    VkDeviceAddress slot_tables = 0;
    VkDeviceAddress reverse_tables = 0;
    VkDeviceAddress group_infos = 0;
    VkDeviceAddress entity_infos = 0;
    VkDeviceAddress grids = 0;
    VkDeviceAddress materials = 0;
    uint32_t records_per_page_shift = 0;
    uint32_t record_bytes = 0;
    uint32_t group_capacity = 0;   // group indices in use or free: [0, group_capacity)
    uint32_t entity_capacity = 0;  // entity ids: [0, entity_capacity)
};

// The light tree of a scene (docs/LIGHT_TREE.md): the light code set, the group light rows, the
// double-buffered light heap of the group blocks (brick light records), the two world-space BVHs
// over the light bricks (static and dynamic entities) with their gather and build, the descriptor,
// the readback of the results, the kernels and their GPU timer. Owned by Scene (null while
// disabled), which calls the hooks below when its entities and groups change, and the flush steps
// in this order:
//
//   Prepare            CPU work and every allocation (may throw; nothing is staged yet). Returns
//                      true when the set of light codes changed: the scene finalizes every
//                      resident brick again.
//   Stage              the staged group flags and entity motion words into the staging ring, all
//                      or nothing (false: retried at the next flush, nothing of the tree is
//                      recorded). Only called when the scene's own tables are consistent this flush.
//   PrepareTransfers   true when RecordTransfers has work (growth copies, fills, the descriptor and
//                      the code set through vkCmdUpdateBuffer): the flush must record then.
//   RecordTransfers    before the scene's staged copies (a transfer barrier follows).
//   Record             after the finalize kernel and its barrier: the group pass, then per tree the
//                      gather and the build; ends with a compute -> all commands barrier.
//   EndFlush           closes the flush window.
class LightTree
{
public:
    LightTree(Device& device, uint32_t group_shift, uint32_t encoding, float min_emission,
              uint64_t heap_bytes);
    ~LightTree();
    LightTree(const LightTree&) = delete;
    LightTree& operator=(const LightTree&) = delete;

    // Scene hooks. `motion` is CAELIX_ENTITY_STATIC or CAELIX_ENTITY_DYNAMIC: the tree that holds
    // the entity's lights.
    void OnEntityAdded(uint32_t entity, uint32_t motion);
    void OnEntityRemoved(uint32_t entity);
    void OnEntityMoved(uint32_t entity);
    void OnEntityMotion(uint32_t entity, uint32_t motion);
    void OnGroupCreated(uint32_t group);
    // `motion`: of the group's entity. Its tree is rebuilt (the group index may start a new life in
    // the other tree's entity before the next pass).
    void OnGroupReleased(uint32_t group, uint32_t motion);
    void OnGroupBricksRemoved(uint32_t group);
    void OnMaterialsChanged();
    void SetMinEmission(float min_emission);

    // Flush steps (see above).
    bool Prepare(const MaterialTable& materials, const LightSceneTables& tables);
    bool Stage(StagingRing& staging, std::vector<StagedCopy>& copies, bool finalize_ran);
    bool PrepareTransfers();
    void RecordTransfers(VkCommandBuffer cmd);
    void Record(VkCommandBuffer cmd, uint64_t frame, uint64_t safe_frame);
    void EndFlush(uint64_t frame);
    // Reads back the results of frames <= safe_frame; schedules heap and BVH growth and retries.
    void Recycle(uint64_t safe_frame);

    void GetStats(CaelixLightStats& out) const;
    VkDeviceAddress descriptor_address() const
    {
        return descriptor_buffer_.address();
    }
    VkDeviceAddress light_codes_address() const
    {
        return codes_buffer_.address();
    }
    VkDeviceAddress group_rows_address() const
    {
        return rows_.address();
    }

private:
    struct EntityState
    {
        uint32_t motion = CAELIX_ENTITY_STATIC;
        bool motion_dirty = false;  // the motion word is staged at the next Stage
    };
    struct GroupState
    {
        bool fresh = false;   // created since the last staged flags: copy nothing from the old block
        bool listed = false;  // in staged_groups_
        bool alive = false;
    };
    struct ReadbackSlot
    {
        GpuBuffer buffer;
        uint64_t frame = 0;
        uint64_t heap_epoch = 0;
        std::array<uint64_t, kLtTrees> tree_epoch{};
        bool pending = false;
    };
    struct Deferred
    {
        uint64_t frame;  // 0 = set at EndFlush
        GpuBuffer buffer;
    };
    struct Growth
    {
        GpuBuffer source;        // invalid = fill only
        VkBuffer destination;
        VkDeviceSize fill_from;  // zero-fill [fill_from, fill_to) after the copy
        VkDeviceSize fill_to;
    };

    EntityState& Entity(uint32_t entity);
    GroupState& Group(uint32_t group);
    void ListGroup(uint32_t group);
    void RebuildCodes(const MaterialTable& materials, bool& changed);
    void GrowHeap(uint64_t needed_units);
    // A growth copy of `source` into `destination`, then zeros up to `zero_tail_to` (0 = no fill).
    void TakeGrowth(std::optional<GpuBuffer> source, VkBuffer destination, VkDeviceSize zero_tail_to);
    // Keeps a replaced buffer alive until the frame of this flush is safe.
    void Defer(std::optional<GpuBuffer> old);
    LightTreeDescriptor BuildDescriptor() const;
    void RecordGroupPass(VkCommandBuffer cmd);
    void RecordTree(VkCommandBuffer cmd, uint32_t tree);

    Device& device_;
    uint32_t group_shift_ = 4;
    uint32_t encoding_ = 0;
    float min_emission_ = 20.0f;

    // CPU state mirrored from the scene.
    std::vector<EntityState> entities_;
    std::vector<GroupState> groups_;
    std::vector<uint32_t> staged_groups_;  // groups whose flags are staged at the next Stage
    bool materials_changed_ = true;
    bool pass_pending_ = false;
    std::array<bool, kLtTrees> tree_reason_{};  // the tree is rebuilt at the next recorded flush
    LightSceneTables tables_;
    uint32_t pass_group_count_ = 0;  // rows of the group pass recorded by this flush

    // Light code set: CPU copy and the GPU buffer (uploaded with vkCmdUpdateBuffer).
    std::vector<uint32_t> codes_;
    bool codes_upload_ = false;
    bool codes_valid_ = false;  // codes_ was computed at least once
    GpuBuffer codes_buffer_;

    // Group rows (zero-filled when new), staged flags (u32 per group, zero-filled when new; the size
    // kernel clears what it consumed), per-pass offsets (also the gather's), control, heap.
    GrowableBuffer rows_;
    GrowableBuffer staged_flags_;
    GrowableBuffer offsets_;
    GpuBuffer control_;
    std::array<GpuBuffer, 2> heaps_;
    uint64_t heap_units_ = 0;  // capacity of each heap buffer in 16-byte units
    uint64_t heap_epoch_ = 0;  // +1 at every growth
    uint64_t grow_request_units_ = 0;
    bool heap_limit_logged_ = false;  // GrowHeap at the heap limit logs its error once
    bool fills_pending_ = true;  // control, rows and staged flags of a new tree are zero-filled

    // Entity motion words, BVH leaves and nodes per tree (copied on growth, so the old tree stays
    // valid until its rebuild), the shared build scratch.
    GrowableBuffer entity_motion_;
    std::array<GrowableBuffer, kLtTrees> leaves_;
    std::array<GrowableBuffer, kLtTrees> nodes_;
    std::array<uint32_t, kLtTrees> leaf_capacity_{};
    std::array<uint32_t, kLtTrees> node_capacity_{};
    std::array<uint32_t, kLtTrees> leaf_request_{};  // grow the leaf capacity to this at Prepare
    std::array<uint32_t, kLtTrees> node_request_{};  // grow the node capacity to this at Prepare
    std::array<uint64_t, kLtTrees> tree_epoch_{};    // +1 at every capacity growth
    GrowableBuffer scratch_;

    // Descriptor.
    GpuBuffer descriptor_buffer_;
    LightTreeDescriptor descriptor_{};
    bool descriptor_upload_ = true;

    // Transfers and recording of the flush being built.
    std::vector<Growth> growth_;
    bool record_pass_ = false;
    std::array<bool, kLtTrees> record_tree_{};

    // Readback, stats, kernels.
    std::vector<ReadbackSlot> readback_;
    LightControl last_control_{};
    uint64_t last_control_frame_ = 0;
    std::vector<Deferred> deferred_;
    ComputePipeline size_kernel_;
    ComputePipeline scan_kernel_;
    ComputePipeline build_kernel_;
    ComputePipeline commit_kernel_;
    ComputePipeline bvh_begin_;
    ComputePipeline gather_count_;
    ComputePipeline gather_emit_;
    ComputePipeline bvh_scan_;
    ComputePipeline bvh_bounds_;
    ComputePipeline bvh_morton_;
    ComputePipeline sort_histogram_;
    ComputePipeline sort_scatter_;
    ComputePipeline bvh_karras_;
    ComputePipeline bvh_aggregate_;
    ComputePipeline bvh_depth_;
    ComputePipeline bvh_wide_;
    ComputePipeline bvh_finish_;
    GpuTimer timer_;
    uint64_t safe_frame_ = 0;
};

}  // namespace caelix
