// The light tree: the recording half of the flush (transfers, the group pass, the gathers and BVH
// builds), the heap growth, the readback of the results and the stats.
#include "scene/light_tree.h"

#include "core/log.h"
#include "scene/gpu_structs.h"
#include "scene/scene_internal.h"

#include <algorithm>
#include <cstring>

namespace caelix {
namespace {

// Compute writes -> compute (and transfer) reads and writes.
void ComputeBarrier(VkCommandBuffer cmd)
{
    MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                      VK_ACCESS_2_TRANSFER_READ_BIT);
}

constexpr uint32_t kThreads = 64;  // per workgroup of the per-item BVH kernels (one thread per item)
// Leaves per tree: the per-leaf kernels run in one dispatch of at most kMaxDispatchGroups workgroups.
// A multiple of 64, like every leaf capacity.
constexpr uint64_t kMaxLeaves = uint64_t{kMaxDispatchGroups} * kThreads;
static_assert(kMaxLeaves % 64 == 0);

uint32_t Groups(uint32_t items)
{
    return (items + kThreads - 1) / kThreads;
}

}  // namespace

void LightTree::GrowHeap(uint64_t needed_units)
{
    uint64_t units = std::max(heap_units_ * 2, needed_units + needed_units / 2);
    units = std::min(units, kLtMaxHeapUnits);
    if (units <= heap_units_)
    {
        if (!heap_limit_logged_)
        {
            CAELIX_LOGE("light tree: the heap needs %llu bytes, above the limit of %llu bytes per buffer",
                        static_cast<unsigned long long>(needed_units * kLtUnitBytes),
                        static_cast<unsigned long long>(kLtMaxHeapUnits * kLtUnitBytes));
            heap_limit_logged_ = true;
        }
        return;
    }
    std::array<GpuBuffer, 2> grown;
    for (GpuBuffer& heap : grown)
    {
        heap = GpuBuffer::DeviceLocal(device_.allocator(), device_.device(), units * kLtUnitBytes);
    }
    CAELIX_LOGI("light tree: the heap grows from 2 x %llu to 2 x %llu bytes (%llu needed)",
                static_cast<unsigned long long>(heap_units_ * kLtUnitBytes),
                static_cast<unsigned long long>(units * kLtUnitBytes),
                static_cast<unsigned long long>(needed_units * kLtUnitBytes));
    // The CPU does not know which buffer is current: both are copied, at the same offsets.
    for (size_t i = 0; i < heaps_.size(); ++i)
    {
        const VkBuffer destination = grown[i].buffer();
        growth_.push_back(Growth{std::move(heaps_[i]), destination, 0, 0});
        heaps_[i] = std::move(grown[i]);
    }
    heap_units_ = units;
    ++heap_epoch_;
}

void LightTree::TakeGrowth(std::optional<GpuBuffer> source, VkBuffer destination,
                           VkDeviceSize zero_tail_to)
{
    if (!source.has_value())
    {
        return;
    }
    const VkDeviceSize old_bytes = source->size();
    growth_.push_back(Growth{std::move(*source), destination, zero_tail_to > old_bytes ? old_bytes : 0,
                             zero_tail_to > old_bytes ? zero_tail_to : 0});
}

void LightTree::Defer(std::optional<GpuBuffer> old)
{
    if (old.has_value())
    {
        deferred_.push_back(Deferred{0, std::move(*old)});
    }
}

LightTreeDescriptor LightTree::BuildDescriptor() const
{
    LightTreeDescriptor d{};
    d.control = control_.address();
    d.heap[0] = heaps_[0].address();
    d.heap[1] = heaps_[1].address();
    d.group_rows = rows_.address();
    d.group_infos = tables_.group_infos;
    d.entity_infos = tables_.entity_infos;
    d.entity_motion = entity_motion_.address();
    d.grids = tables_.grids;
    for (uint32_t tree = 0; tree < kLtTrees; ++tree)
    {
        d.nodes[tree] = nodes_[tree].address();
        d.leaves[tree] = leaves_[tree].address();
    }
    d.light_codes = codes_buffer_.address();
    d.tree_headers = control_.address() + offsetof(LightControl, headers);
    d.group_shift = group_shift_;
    d.encoding = encoding_;
    d.group_count = tables_.group_capacity;
    return d;
}

bool LightTree::PrepareTransfers()
{
    if (fills_pending_)
    {
        // A new tree: the zero fills of RecordTransfers cover these buffers whole.
        Defer(rows_.TakeGrowthSource());
        Defer(staged_flags_.TakeGrowthSource());
        Defer(entity_motion_.TakeGrowthSource());
    }
    else
    {
        TakeGrowth(rows_.TakeGrowthSource(), rows_.buffer(), rows_.bytes());
        // The staged flags are all zero between flushes: the new buffer is zero-filled, nothing
        // copied.
        if (std::optional<GpuBuffer> old = staged_flags_.TakeGrowthSource(); old.has_value())
        {
            Defer(std::move(old));
            growth_.push_back(Growth{GpuBuffer{}, staged_flags_.buffer(), 0, staged_flags_.bytes()});
        }
        TakeGrowth(entity_motion_.TakeGrowthSource(), entity_motion_.buffer(), entity_motion_.bytes());
    }
    for (uint32_t tree = 0; tree < kLtTrees; ++tree)
    {
        TakeGrowth(leaves_[tree].TakeGrowthSource(), leaves_[tree].buffer(), 0);
        TakeGrowth(nodes_[tree].TakeGrowthSource(), nodes_[tree].buffer(), 0);
    }
    Defer(offsets_.TakeGrowthSource());
    Defer(scratch_.TakeGrowthSource());
    const LightTreeDescriptor descriptor = BuildDescriptor();
    if (std::memcmp(&descriptor, &descriptor_, sizeof(descriptor)) != 0)
    {
        descriptor_ = descriptor;
        descriptor_upload_ = true;
    }
    return fills_pending_ || descriptor_upload_ || codes_upload_ || !growth_.empty();
}

void LightTree::RecordTransfers(VkCommandBuffer cmd)
{
    // Growth copies (old contents at the same offsets) and the zero fills of new rows. Every
    // destination region is written once, so no barrier is needed between them.
    for (const Growth& growth : growth_)
    {
        if (growth.source.valid())
        {
            const VkBufferCopy region{0, 0, growth.source.size()};
            vkCmdCopyBuffer(cmd, growth.source.buffer(), growth.destination, 1, &region);
        }
        if (growth.fill_to > growth.fill_from)
        {
            vkCmdFillBuffer(cmd, growth.destination, growth.fill_from, growth.fill_to - growth.fill_from,
                            0);
        }
    }
    if (fills_pending_)
    {
        // Zero control: current 0, both trees empty after their first build (root 0 would be a
        // node: the headers start as empty trees).
        LightControl control{};
        for (LightTreeHeader& header : control.headers)
        {
            header.root = kLtNone;
        }
        vkCmdUpdateBuffer(cmd, control_.buffer(), 0, sizeof(control), &control);
        vkCmdFillBuffer(cmd, rows_.buffer(), 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, staged_flags_.buffer(), 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, entity_motion_.buffer(), 0, VK_WHOLE_SIZE, 0);
        fills_pending_ = false;
    }
    if (descriptor_upload_)
    {
        vkCmdUpdateBuffer(cmd, descriptor_buffer_.buffer(), 0, sizeof(descriptor_), &descriptor_);
        descriptor_upload_ = false;
    }
    if (codes_upload_)
    {
        vkCmdUpdateBuffer(cmd, codes_buffer_.buffer(), 0, codes_.size() * sizeof(uint32_t),
                          codes_.data());
        codes_upload_ = false;
    }
}

void LightTree::RecordGroupPass(VkCommandBuffer cmd)
{
    LightGroupPush push{};
    push.descriptor = descriptor_buffer_.address();
    push.page_table = tables_.page_table;
    push.slot_tables = tables_.slot_tables;
    push.reverse_tables = tables_.reverse_tables;
    push.materials = tables_.materials;
    push.offsets = offsets_.address();
    push.staged_flags = staged_flags_.address();
    push.records_per_page_shift = tables_.records_per_page_shift;
    push.record_bytes = tables_.record_bytes;
    push.group_count = pass_group_count_;
    push.capacity_units = static_cast<uint32_t>(heap_units_);
    const auto per_group = [&](const ComputePipeline& kernel) {
        for (uint32_t base = 0; base < pass_group_count_; base += kMaxDispatchGroups)
        {
            push.group_base = base;
            kernel.Dispatch(cmd, &push, sizeof(push),
                            std::min(kMaxDispatchGroups, pass_group_count_ - base));
        }
        push.group_base = 0;
        ComputeBarrier(cmd);
    };
    per_group(size_kernel_);
    scan_kernel_.Dispatch(cmd, &push, sizeof(push), 1);
    ComputeBarrier(cmd);
    per_group(build_kernel_);
    commit_kernel_.Dispatch(cmd, &push, sizeof(push), 1);
    ComputeBarrier(cmd);
}

void LightTree::RecordTree(VkCommandBuffer cmd, uint32_t tree)
{
    const uint64_t control = control_.address();
    const uint64_t header = control + kLtCtlTreeHeaders + uint64_t{tree} * kLtTreeHeaderBytes;
    const uint64_t state = control + kLtCtlTreeStates + uint64_t{tree} * kLtTreeStateBytes;
    const uint64_t gate = state + kLtStBuilding;
    const uint32_t capacity = leaf_capacity_[tree];
    const uint64_t scratch = scratch_.address();

    LightBvhPush push{};
    push.descriptor = descriptor_buffer_.address();
    push.scratch = scratch;
    push.offsets = offsets_.address();
    push.tree = tree;
    push.leaf_capacity = capacity;
    push.node_capacity = node_capacity_[tree];
    push.group_count = pass_group_count_;
    push.reason = tree_reason_[tree] ? 1u : 0u;
    push.pass_recorded = record_pass_ ? 1u : 0u;
    const auto dispatch = [&](const ComputePipeline& kernel, uint32_t workgroups) {
        kernel.Dispatch(cmd, &push, sizeof(push), workgroups);
        ComputeBarrier(cmd);
    };
    // One thread per group index, in dispatches of at most kMaxDispatchGroups workgroups.
    const auto per_group = [&](const ComputePipeline& kernel) {
        constexpr uint32_t kChunk = kMaxDispatchGroups * kThreads;
        for (uint32_t base = 0; base < pass_group_count_; base += kChunk)
        {
            push.group_base = base;
            kernel.Dispatch(cmd, &push, sizeof(push),
                            Groups(std::min(kChunk, pass_group_count_ - base)));
        }
        push.group_base = 0;
        ComputeBarrier(cmd);
    };
    const auto scan = [&](uint64_t data, uint32_t count, uint64_t total) {
        const LightScanPush scan_push{data, total, gate, count, 0};
        bvh_scan_.Dispatch(cmd, &scan_push, sizeof(scan_push), 1);
        ComputeBarrier(cmd);
    };

    // Gather: begin (the GPU decides whether this flush rebuilds the tree), counts, offsets, leaves.
    dispatch(bvh_begin_, 1);
    per_group(gather_count_);
    scan(offsets_.address(), pass_group_count_, state + kLtStLeavesFound);
    per_group(gather_emit_);

    // Build: bounds, Morton codes, radix sort, binary radix tree, aggregation, 4-wide collapse.
    dispatch(bvh_bounds_, 1);
    dispatch(bvh_morton_, Groups(capacity));
    // The scratch arrays (light_tree_layout.h): array X at kLtScratchX * capacity.
    const auto array = [&](uint32_t offset) { return scratch + uint64_t{offset} * capacity; };
    const uint32_t blocks = (capacity + kLtSortBlock - 1) / kLtSortBlock;
    const uint64_t histogram = array(kLtScratchHistogram);
    for (uint32_t pass = 0; pass < kLtSortPasses; ++pass)
    {
        const uint32_t in = pass & 1u;
        LightSortPush sort{};
        sort.keys_in = array(kLtScratchKeys + kLtScratchSortBufferStride * in);
        sort.values_in = array(kLtScratchValues + kLtScratchSortBufferStride * in);
        sort.keys_out = array(kLtScratchKeys + kLtScratchSortBufferStride * (1u - in));
        sort.values_out = array(kLtScratchValues + kLtScratchSortBufferStride * (1u - in));
        sort.histogram = histogram;
        sort.count_address = header + kLtHdrLeafCount;
        sort.gate_address = gate;
        sort.shift = kLtSortBits * pass;
        sort.block_count = blocks;
        sort_histogram_.Dispatch(cmd, &sort, sizeof(sort), blocks);
        ComputeBarrier(cmd);
        scan(histogram, kLtSortDigits * blocks, 0);
        sort_scatter_.Dispatch(cmd, &sort, sizeof(sort), blocks);
        ComputeBarrier(cmd);
    }
    dispatch(bvh_karras_, Groups(capacity));
    dispatch(bvh_aggregate_, Groups(capacity));
    dispatch(bvh_depth_, Groups(capacity));
    scan(array(kLtScratchWide), capacity, header + kLtHdrNodeCount);
    dispatch(bvh_wide_, Groups(capacity));
    dispatch(bvh_finish_, 1);
}

void LightTree::Record(VkCommandBuffer cmd, uint64_t frame, uint64_t safe_frame)
{
    if (!record_pass_ && !record_tree_[0] && !record_tree_[1])
    {
        return;
    }
    timer_.Begin(cmd, frame, safe_frame);
    if (record_pass_)
    {
        RecordGroupPass(cmd);
    }
    for (uint32_t tree = 0; tree < kLtTrees; ++tree)
    {
        if (record_tree_[tree])
        {
            RecordTree(cmd, tree);
            tree_reason_[tree] = false;
        }
    }

    // The results to a readback slot (Recycle reads them once the frame is safe).
    for (ReadbackSlot& slot : readback_)
    {
        if (!slot.pending)
        {
            const VkBufferCopy region{0, 0, sizeof(LightControl)};
            vkCmdCopyBuffer(cmd, control_.buffer(), slot.buffer.buffer(), 1, &region);
            MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
            slot.frame = frame;
            slot.heap_epoch = heap_epoch_;
            slot.tree_epoch = tree_epoch_;
            slot.pending = true;
            break;
        }
    }
    MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    timer_.End(cmd);
}

void LightTree::EndFlush(uint64_t frame)
{
    for (Growth& growth : growth_)
    {
        if (growth.source.valid())
        {
            deferred_.push_back(Deferred{frame, std::move(growth.source)});
        }
    }
    growth_.clear();
    for (Deferred& deferred : deferred_)
    {
        deferred.frame = deferred.frame == 0 ? frame : deferred.frame;
    }
    record_pass_ = false;
    record_tree_ = {};
}

void LightTree::Recycle(uint64_t safe_frame)
{
    safe_frame_ = std::max(safe_frame_, safe_frame);
    timer_.Resolve(safe_frame_);
    std::erase_if(deferred_, [this](const Deferred& deferred) {
        return deferred.frame != 0 && deferred.frame <= safe_frame_;
    });
    for (ReadbackSlot& slot : readback_)
    {
        if (!slot.pending || slot.frame > safe_frame_)
        {
            continue;
        }
        slot.pending = false;
        slot.buffer.InvalidateHostReads(0, sizeof(LightControl));
        LightControl control{};
        std::memcpy(&control, slot.buffer.mapped(), sizeof(control));
        if (slot.frame >= last_control_frame_)
        {
            last_control_ = control;
            last_control_frame_ = slot.frame;
        }
        if (control.pass_result == kLtPassFailed)
        {
            // Retry: the flags are still set on the GPU and the rows unchanged, whatever the heap
            // has done since. Grow first unless the heap has already grown since that pass.
            pass_pending_ = true;
            if (slot.heap_epoch == heap_epoch_)
            {
                CAELIX_LOGD("light tree: the group pass of frame %llu needs %llu heap bytes of %llu: it is "
                            "retried after growth",
                            static_cast<unsigned long long>(slot.frame),
                            static_cast<unsigned long long>(uint64_t{control.needed_units} * kLtUnitBytes),
                            static_cast<unsigned long long>(heap_units_ * kLtUnitBytes));
                grow_request_units_ = std::max<uint64_t>(
                    grow_request_units_, std::max<uint64_t>(control.needed_units, heap_units_ + 1));
            }
        }
        else if (control.pass_result == kLtPassOk && slot.heap_epoch == heap_epoch_ &&
                 uint64_t{control.needed_units} * 4 > heap_units_ * 3)
        {
            grow_request_units_ = std::max<uint64_t>(grow_request_units_, control.needed_units);
        }
        for (uint32_t tree = 0; tree < kLtTrees; ++tree)
        {
            if (slot.tree_epoch[tree] != tree_epoch_[tree])
            {
                continue;  // about a capacity that has grown since
            }
            const LightTreeState& state = control.states[tree];
            const uint64_t found = state.leaves_found;
            if (found * 4 > uint64_t{leaf_capacity_[tree]} * 3 || state.leaves_dropped != 0)
            {
                const uint64_t wanted =
                    std::max<uint64_t>(uint64_t{leaf_capacity_[tree]} * 3 / 2, found + found / 2);
                const uint32_t request = static_cast<uint32_t>(std::min<uint64_t>(wanted, kMaxLeaves));
                leaf_request_[tree] = std::max(leaf_request_[tree], request);
            }
            if (state.node_overflow != 0)
            {
                node_request_[tree] = leaf_capacity_[tree];
            }
        }
    }
}

void LightTree::GetStats(CaelixLightStats& out) const
{
    out = CaelixLightStats{};
    out.enabled = 1;
    out.light_groups = last_control_.light_groups;
    out.light_bricks = last_control_.light_bricks;
    out.light_faces = last_control_.light_faces;
    out.bricks_dropped = last_control_.bricks_dropped;
    out.heap_bytes_used = uint64_t{last_control_.needed_units} * kLtUnitBytes;
    out.heap_bytes_capacity = heap_units_ * kLtUnitBytes;
    out.passes = last_control_.passes;
    out.failed_passes = last_control_.failed_passes;
    out.records_recomputed = last_control_.records_recomputed;
    out.records_copied = last_control_.records_copied;
    out.gpu_ms = timer_.latest_ms();
    for (uint32_t tree = 0; tree < kLtTrees; ++tree)
    {
        out.tree_leaves[tree] = last_control_.headers[tree].leaf_count;
        out.tree_nodes[tree] = last_control_.headers[tree].node_count;
        out.tree_builds[tree] = last_control_.states[tree].builds;
        out.tree_leaves_dropped[tree] = last_control_.states[tree].leaves_dropped;
    }
}

}  // namespace caelix
