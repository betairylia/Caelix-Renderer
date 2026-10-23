// The light tree: creation, the scene hooks, the light code set and the CPU half of the flush
// (Prepare, Stage). The recording half is light_tree_record.cpp.
#include "scene/light_tree.h"

#include "core/log.h"
#include "rhi/shader_loader.h"
#include "scene/material_table.h"

#include <algorithm>
#include <cstring>

namespace caelix {
namespace {

constexpr uint32_t kCodeWords = MaterialTable::kEntries / 32;  // one bit per material code
constexpr VkDeviceSize kGroupsInitial = 1024;
constexpr VkDeviceSize kEntitiesInitial = 64;
constexpr uint32_t kLeavesInitial = 1024;  // per tree; a multiple of 64
constexpr uint64_t kStageAlignment = 16;

// 4-wide nodes for a leaf capacity: the even-depth nodes of the binary radix tree, measured about
// leaves / 2 on Morton-ordered light bricks; a build that needs more sets the overflow flag and the
// node capacity grows to the leaf capacity (the bound: a binary tree over n leaves has n - 1
// internal nodes).
uint32_t NodeCapacityFor(uint32_t leaf_capacity)
{
    return leaf_capacity / 8 * 5 + 64;
}

ComputePipeline MakeKernel(VkDevice device, const char* name, uint32_t push_bytes)
{
    const std::vector<uint32_t> spirv = LoadShader(name);
    return ComputePipeline(device, spirv, push_bytes, name);
}

}  // namespace

LightTree::LightTree(Device& device, uint32_t group_shift, uint32_t encoding, float min_emission,
                     uint64_t heap_bytes)
    : device_(device), group_shift_(group_shift), encoding_(encoding), min_emission_(min_emission)
{
    const VmaAllocator allocator = device.allocator();
    const VkDevice vk_device = device.device();
    codes_.assign(kCodeWords, 0u);
    codes_buffer_ = GpuBuffer::DeviceLocal(allocator, vk_device, kCodeWords * sizeof(uint32_t));
    rows_ = GrowableBuffer(allocator, vk_device, kGroupsInitial * sizeof(GroupLightRow));
    staged_flags_ = GrowableBuffer(allocator, vk_device, kGroupsInitial * sizeof(uint32_t));
    offsets_ = GrowableBuffer(allocator, vk_device, kGroupsInitial * sizeof(uint32_t));
    control_ = GpuBuffer::DeviceLocal(allocator, vk_device, sizeof(LightControl));
    heap_units_ = (heap_bytes + kLtUnitBytes - 1) / kLtUnitBytes;
    for (GpuBuffer& heap : heaps_)
    {
        heap = GpuBuffer::DeviceLocal(allocator, vk_device, heap_units_ * kLtUnitBytes);
    }
    entity_motion_ = GrowableBuffer(allocator, vk_device, kEntitiesInitial * sizeof(uint32_t));
    for (uint32_t tree = 0; tree < kLtTrees; ++tree)
    {
        leaf_capacity_[tree] = kLeavesInitial;
        node_capacity_[tree] = NodeCapacityFor(kLeavesInitial);
        const VkDeviceSize leaf_bytes = VkDeviceSize{kLeavesInitial} * sizeof(LightBvhLeaf);
        const VkDeviceSize node_bytes = VkDeviceSize{node_capacity_[tree]} * sizeof(LightBvhNode);
        leaves_[tree] = GrowableBuffer(allocator, vk_device, leaf_bytes);
        nodes_[tree] = GrowableBuffer(allocator, vk_device, node_bytes);
    }
    scratch_ = GrowableBuffer(allocator, vk_device, LightBvhScratchBytes(kLeavesInitial));
    descriptor_buffer_ = GpuBuffer::DeviceLocal(allocator, vk_device, sizeof(LightTreeDescriptor));
    size_kernel_ = MakeKernel(vk_device, "caelix_light_tree_size", sizeof(LightGroupPush));
    scan_kernel_ = MakeKernel(vk_device, "caelix_light_tree_scan", sizeof(LightGroupPush));
    build_kernel_ = MakeKernel(vk_device, "caelix_light_tree_build", sizeof(LightGroupPush));
    commit_kernel_ = MakeKernel(vk_device, "caelix_light_tree_commit", sizeof(LightGroupPush));
    bvh_begin_ = MakeKernel(vk_device, "caelix_light_bvh_begin", sizeof(LightBvhPush));
    gather_count_ = MakeKernel(vk_device, "caelix_light_gather_count", sizeof(LightBvhPush));
    gather_emit_ = MakeKernel(vk_device, "caelix_light_gather_emit", sizeof(LightBvhPush));
    bvh_scan_ = MakeKernel(vk_device, "caelix_light_bvh_scan", sizeof(LightScanPush));
    bvh_bounds_ = MakeKernel(vk_device, "caelix_light_bvh_bounds", sizeof(LightBvhPush));
    bvh_morton_ = MakeKernel(vk_device, "caelix_light_bvh_morton", sizeof(LightBvhPush));
    sort_histogram_ = MakeKernel(vk_device, "caelix_light_bvh_sort_histogram", sizeof(LightSortPush));
    sort_scatter_ = MakeKernel(vk_device, "caelix_light_bvh_sort_scatter", sizeof(LightSortPush));
    bvh_karras_ = MakeKernel(vk_device, "caelix_light_bvh_karras", sizeof(LightBvhPush));
    bvh_aggregate_ = MakeKernel(vk_device, "caelix_light_bvh_aggregate", sizeof(LightBvhPush));
    bvh_depth_ = MakeKernel(vk_device, "caelix_light_bvh_depth", sizeof(LightBvhPush));
    bvh_wide_ = MakeKernel(vk_device, "caelix_light_bvh_wide", sizeof(LightBvhPush));
    bvh_finish_ = MakeKernel(vk_device, "caelix_light_bvh_finish", sizeof(LightBvhPush));
    timer_ = GpuTimer(device);
    CAELIX_LOGI("light tree created: min_emission %g, heap 2 x %llu bytes", static_cast<double>(min_emission),
                static_cast<unsigned long long>(heap_units_ * kLtUnitBytes));
}

LightTree::~LightTree() = default;

// ---------------------------------------------------------------------------
// Scene hooks

LightTree::EntityState& LightTree::Entity(uint32_t entity)
{
    if (entity >= entities_.size())
    {
        entities_.resize(static_cast<size_t>(entity) + 1);
    }
    return entities_[entity];
}

LightTree::GroupState& LightTree::Group(uint32_t group)
{
    if (group >= groups_.size())
    {
        groups_.resize(static_cast<size_t>(group) + 1);
    }
    return groups_[group];
}

void LightTree::ListGroup(uint32_t group)
{
    GroupState& state = Group(group);
    if (!state.listed)
    {
        state.listed = true;
        staged_groups_.push_back(group);
    }
}

void LightTree::OnEntityAdded(uint32_t entity, uint32_t motion)
{
    EntityState& state = Entity(entity);
    state.motion = motion;
    state.motion_dirty = true;
}

void LightTree::OnEntityRemoved(uint32_t entity)
{
    // Its groups are released (a group pass follows); its tree loses the leaves.
    tree_reason_[Entity(entity).motion] = true;
}

void LightTree::OnEntityMoved(uint32_t entity)
{
    tree_reason_[Entity(entity).motion] = true;
}

void LightTree::OnEntityMotion(uint32_t entity, uint32_t motion)
{
    EntityState& state = Entity(entity);
    if (state.motion != motion)
    {
        state.motion = motion;
        state.motion_dirty = true;
        tree_reason_[0] = true;
        tree_reason_[1] = true;
    }
}

void LightTree::OnGroupCreated(uint32_t group)
{
    GroupState& state = Group(group);
    state.alive = true;
    state.fresh = true;
    ListGroup(group);
}

void LightTree::OnGroupReleased(uint32_t group, uint32_t motion)
{
    Group(group).alive = false;
    ListGroup(group);
    tree_reason_[motion] = true;
}

void LightTree::OnGroupBricksRemoved(uint32_t group)
{
    ListGroup(group);
}

void LightTree::OnMaterialsChanged()
{
    materials_changed_ = true;
}

void LightTree::SetMinEmission(float min_emission)
{
    if (min_emission != min_emission_)
    {
        min_emission_ = min_emission;
        materials_changed_ = true;
    }
}

void LightTree::RebuildCodes(const MaterialTable& materials, bool& changed)
{
    std::vector<uint32_t> codes(kCodeWords, 0u);
    for (uint32_t code = 0; code < MaterialTable::kEntries; ++code)
    {
        const CaelixMaterial& m = materials.Entry(materials.CanonicalCode(code));
        if (LightLuminance(m.emission) >= min_emission_)
        {
            codes[code >> 5] |= 1u << (code & 31);
        }
    }
    changed = !codes_valid_ || codes != codes_;
    codes_ = std::move(codes);
    codes_valid_ = true;
}

// ---------------------------------------------------------------------------
// Flush: Prepare and Stage

bool LightTree::Prepare(const MaterialTable& materials, const LightSceneTables& tables)
{
    // Allocations first: a throw leaves the CPU state as it was.
    const VkDeviceSize groups = std::max<VkDeviceSize>(1, tables.group_capacity);
    rows_.Reserve(groups * sizeof(GroupLightRow));
    staged_flags_.Reserve(groups * sizeof(uint32_t));
    offsets_.Reserve(groups * sizeof(uint32_t));
    entity_motion_.Reserve(std::max<VkDeviceSize>(1, tables.entity_capacity) * sizeof(uint32_t));
    if (grow_request_units_ != 0)
    {
        GrowHeap(grow_request_units_);
        grow_request_units_ = 0;
    }
    for (uint32_t tree = 0; tree < kLtTrees; ++tree)
    {
        if (leaf_request_[tree] > leaf_capacity_[tree] || node_request_[tree] > node_capacity_[tree])
        {
            const uint32_t leaves = std::max(leaf_capacity_[tree], (leaf_request_[tree] + 63) / 64 * 64);
            const uint32_t nodes = std::max(
                {node_capacity_[tree], NodeCapacityFor(leaves), std::min(node_request_[tree], leaves)});
            leaves_[tree].Reserve(static_cast<VkDeviceSize>(leaves) * sizeof(LightBvhLeaf));
            nodes_[tree].Reserve(static_cast<VkDeviceSize>(nodes) * sizeof(LightBvhNode));
            CAELIX_LOGI("light tree: BVH %u grows from %u to %u leaves, %u to %u nodes", tree,
                        leaf_capacity_[tree], leaves, node_capacity_[tree], nodes);
            leaf_capacity_[tree] = leaves;
            node_capacity_[tree] = nodes;
            ++tree_epoch_[tree];
            tree_reason_[tree] = true;
        }
        leaf_request_[tree] = 0;
        node_request_[tree] = 0;
    }
    scratch_.Reserve(LightBvhScratchBytes(std::max(leaf_capacity_[0], leaf_capacity_[1])));
    bool has_free_slot = false;
    for (const ReadbackSlot& slot : readback_)
    {
        has_free_slot = has_free_slot || !slot.pending;
    }
    if (!has_free_slot)
    {
        ReadbackSlot slot;
        slot.buffer = GpuBuffer::HostReadback(device_.allocator(), device_.device(), sizeof(LightControl));
        readback_.push_back(std::move(slot));
    }

    bool requeue = false;
    if (materials_changed_)
    {
        bool changed = false;
        RebuildCodes(materials, changed);
        materials_changed_ = false;
        if (changed)
        {
            codes_upload_ = true;
            requeue = true;
        }
    }
    tables_ = tables;
    return requeue;
}

bool LightTree::Stage(StagingRing& staging, std::vector<StagedCopy>& copies, bool finalize_ran)
{
    record_pass_ = false;
    record_tree_ = {};
    if (finalize_ran || !staged_groups_.empty())
    {
        pass_pending_ = true;
    }

    // The staged flags: one dense range of group indices (zeros between the listed groups: the size
    // kernel of this flush consumes all of it).
    uint32_t flags_lo = 0;
    uint32_t flags_hi = 0;
    if (!staged_groups_.empty())
    {
        flags_lo = *std::min_element(staged_groups_.begin(), staged_groups_.end());
        flags_hi = *std::max_element(staged_groups_.begin(), staged_groups_.end()) + 1;
    }
    std::vector<uint32_t> dirty_entities;
    for (uint32_t entity = 0; entity < entities_.size(); ++entity)
    {
        if (entities_[entity].motion_dirty)
        {
            dirty_entities.push_back(entity);
        }
    }
    constexpr uint64_t kWord = sizeof(uint32_t);  // a staged flags word, an entity motion word
    const uint64_t flag_bytes = static_cast<uint64_t>(flags_hi - flags_lo) * kWord;
    const uint64_t motion_bytes = dirty_entities.size() * kWord;
    const bool pass = pass_pending_ && tables_.group_capacity > 0;
    const bool trees = tree_reason_[0] || tree_reason_[1] || pass;
    if (flag_bytes + motion_bytes == 0 && !trees)
    {
        return false;
    }
    const auto aligned = [](uint64_t bytes) {
        return (bytes + kStageAlignment - 1) / kStageAlignment * kStageAlignment;
    };
    const uint64_t total = aligned(flag_bytes) + aligned(motion_bytes);
    StagingRing::Span span;
    if (total > 0 && !staging.TryAllocate(total, kStageAlignment, StagingRing::Purpose::kRows, span))
    {
        CAELIX_LOGD("caelix_scene_flush: the staging ring is full; the light tree update waits for the next "
                    "flush");
        return false;
    }
    if (flag_bytes > 0)
    {
        uint32_t* flags = reinterpret_cast<uint32_t*>(span.data);
        std::fill(flags, flags + (flags_hi - flags_lo), 0u);
        for (const uint32_t group : staged_groups_)
        {
            GroupState& state = groups_[group];
            flags[group - flags_lo] =
                state.alive ? (kLtCpuRebuild | (state.fresh ? kLtCpuFresh : 0u)) : kLtCpuReleased;
            state.fresh = false;
            state.listed = false;
        }
        copies.push_back(StagedCopy{span.offset, staged_flags_.buffer(),
                                    static_cast<VkDeviceSize>(flags_lo) * kWord, flag_bytes});
        staged_groups_.clear();
    }
    const uint64_t motion_offset = aligned(flag_bytes);
    for (size_t i = 0; i < dirty_entities.size(); ++i)
    {
        const uint32_t entity = dirty_entities[i];
        EntityState& state = entities_[entity];
        std::memcpy(span.data + motion_offset + i * kWord, &state.motion, kWord);
        copies.push_back(StagedCopy{span.offset + motion_offset + i * kWord, entity_motion_.buffer(),
                                    static_cast<VkDeviceSize>(entity) * kWord, kWord});
        state.motion_dirty = false;
    }
    record_pass_ = pass;
    pass_group_count_ = tables_.group_capacity;
    pass_pending_ = false;
    for (uint32_t tree = 0; tree < kLtTrees; ++tree)
    {
        record_tree_[tree] = tree_reason_[tree] || record_pass_;
    }
    return true;
}

}  // namespace caelix
