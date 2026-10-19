// The per-frame flush: staging of the tables, copies, fills and the finalize
// dispatch.
#include "scene/scene.h"

#include "core/log.h"
#include "scene/scene_internal.h"

#include <algorithm>
#include <cstring>
#include <optional>

namespace caelix {
namespace {

// Staging alignment of table rows. vkCmdCopyBuffer itself has no offset
// alignment rule.
constexpr uint64_t kRowAlignment = 16;

// vkCmdCopyBuffer regions per call.
constexpr size_t kMaxRegionsPerCopy = 4096;

SlotRow ToRow(const BrickSlotCpu& slot)
{
    SlotRow row{};
    row.pos_state = (slot.pos & 0x00FFFFFFu) | (static_cast<uint32_t>(slot.state) << 24);
    row.summary = 0;
    row.record_index = slot.record_index;
    row.last_used = 0;
    return row;
}

// Transfer writes (copies and fills) -> later transfer reads and writes.
void TransferBarrier(VkCommandBuffer cmd)
{
    MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                  VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

template <typename T>
void SortUnique(std::vector<T>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

// Consecutive u32 words [first, last] of a table, restaged with one copy region.
struct WordRun
{
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

void BuildRuns(const std::vector<uint32_t>& words, std::vector<WordRun>& runs)
{
    runs.clear();
    if (words.empty())
    {
        return;
    }
    WordRun run{words[0], words[0]};
    for (size_t i = 1; i < words.size(); ++i)
    {
        if (words[i] - run.last <= kMaxGapWords + 1)
        {
            run.last = words[i];
        }
        else
        {
            runs.push_back(run);
            run = WordRun{words[i], words[i]};
        }
    }
    runs.push_back(run);
}

uint64_t RunWords(const std::vector<WordRun>& runs)
{
    uint64_t words = 0;
    for (const WordRun& run : runs)
    {
        words += static_cast<uint64_t>(run.last) - run.first + 1;
    }
    return words;
}

}  // namespace

// ---------------------------------------------------------------------------
// Flush

void Scene::FreeAfterFlush(RowPool& pool, RowRange range)
{
    if (range.valid())
    {
        ranges_to_free_after_flush_.push_back(RangeToFree{&pool, range});
    }
}

void Scene::PrepareGroupAabbs(Group& group)
{
    if (group.reverse_fill)
    {
        // A new reverse range: every position "none" before the used words are staged.
        fills_.push_back(PendingFill{reverse_tables_.buffer(),
                                     reverse_tables_.ByteOffset(group.reverse_range.offset),
                                     reverse_tables_.ByteOffset(group.reverse_range.rows),
                                     0xFFFFFFFFu});
        group.reverse_fill = false;
    }
    if (group.aabb_rows == group.aabb_gpu_rows)
    {
        return;
    }
    // The AABB range is new or grew: copy the rows the GPU holds, and make every
    // other row inactive (NaN), so free ids never show up as primitives.
    uint32_t kept = 0;
    if (group.aabb_gpu_rows.valid())
    {
        kept = group.aabb_gpu_rows.rows;
        aabb_moves_.push_back(VkBufferCopy{aabbs_.ByteOffset(group.aabb_gpu_rows.offset),
                                           aabbs_.ByteOffset(group.aabb_rows.offset),
                                           aabbs_.ByteOffset(kept)});
    }
    fills_.push_back(PendingFill{aabbs_.buffer(), aabbs_.ByteOffset(group.aabb_rows.offset + kept),
                                 aabbs_.ByteOffset(group.aabb_rows.rows - kept), kQuietNaNBits});
    group.aabb_gpu_rows = group.aabb_rows;
}

bool Scene::EnsureNanRow()
{
    if (nan_row_valid_)
    {
        return true;
    }
    StagingRing::Span span;
    if (!staging_.TryAllocate(kAabbRowBytes, 8, StagingRing::Purpose::kRows, span))
    {
        return false;
    }
    const uint32_t nan_row[6] = {kQuietNaNBits, kQuietNaNBits, kQuietNaNBits,
                                 kQuietNaNBits, kQuietNaNBits, kQuietNaNBits};
    std::memcpy(span.data, nan_row, sizeof(nan_row));
    nan_row_offset_ = span.offset;
    nan_row_valid_ = true;
    return true;
}

bool Scene::StageGroup(uint32_t group_index, Group& group)
{
    if (!group.full_rewrite)
    {
        SortUnique(group.dirty_ids);
    }
    SortUnique(group.reverse_dirty_words);
    std::vector<WordRun> reverse_runs;
    BuildRuns(group.reverse_dirty_words, reverse_runs);
    if (!group.nan_ids.empty() && !EnsureNanRow())
    {
        return false;
    }
    const uint64_t slot_rows = group.full_rewrite ? group.slots.size() : group.dirty_ids.size();
    const uint64_t slot_bytes = slot_rows * kSlotRowBytes;
    const uint64_t info_bytes = group.info_dirty ? sizeof(GroupInfo) : 0;
    const uint64_t reverse_bytes = RunWords(reverse_runs) * sizeof(uint32_t);
    const uint64_t total = slot_bytes + info_bytes + reverse_bytes;
    StagingRing::Span span;
    if (total > 0 &&
        !staging_.TryAllocate(total, kRowAlignment, StagingRing::Purpose::kRows, span))
    {
        return false;
    }

    // Slot rows.
    const VkBuffer table = slot_tables_.buffer();
    const VkDeviceSize range_base = slot_tables_.ByteOffset(group.slot_rows.offset);
    for (uint64_t k = 0; k < slot_rows; ++k)
    {
        const uint16_t id = group.full_rewrite ? static_cast<uint16_t>(k) : group.dirty_ids[k];
        const SlotRow row = ToRow(group.slots[id]);
        std::memcpy(span.data + k * kSlotRowBytes, &row, sizeof(row));
        if (!group.full_rewrite)
        {
            pending_.push_back(PendingCopy{span.offset + k * kSlotRowBytes, table,
                                           range_base + static_cast<VkDeviceSize>(id) * kSlotRowBytes,
                                           kSlotRowBytes});
        }
    }
    if (group.full_rewrite && slot_rows > 0)
    {
        pending_.push_back(PendingCopy{span.offset, table, range_base, slot_bytes});
    }

    // GroupInfo row.
    uint64_t cursor = slot_bytes;
    if (group.info_dirty)
    {
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
    for (const WordRun& run : reverse_runs)
    {
        const uint64_t run_begin = cursor;
        for (uint32_t word = run.first; word <= run.last; ++word)
        {
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
    for (const uint16_t id : group.nan_ids)
    {
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

bool Scene::StageEntity(CaelixEntityId id, Entity& entity)
{
    SortUnique(entity.grid_dirty_cells);
    std::vector<WordRun> cell_runs;
    BuildRuns(entity.grid_dirty_cells, cell_runs);
    const uint64_t info_bytes = entity.info_dirty ? sizeof(EntityInfo) : 0;
    const uint64_t cell_bytes = RunWords(cell_runs) * sizeof(uint32_t);
    const uint64_t total = info_bytes + cell_bytes;
    if (total == 0)
    {
        return true;
    }
    StagingRing::Span span;
    if (!staging_.TryAllocate(total, kRowAlignment, StagingRing::Purpose::kRows, span))
    {
        return false;
    }
    uint64_t cursor = 0;
    if (entity.info_dirty)
    {
        EntityInfo info{};
        for (int a = 0; a < 3; ++a)
        {
            info.grid_min[a] = entity.grid_min[a];
            info.grid_dims[a] = entity.grid_dims[a];
        }
        info.grid_base = entity.grid_range.offset;
        std::memcpy(info.cur_rows, entity.cur_rows, sizeof(info.cur_rows));
        std::memcpy(info.prev_rows, entity.prev_rows, sizeof(info.prev_rows));
        std::memcpy(span.data, &info, sizeof(info));
        pending_.push_back(PendingCopy{span.offset, entity_infos_.buffer(),
                                       static_cast<VkDeviceSize>(id) * sizeof(EntityInfo),
                                       sizeof(EntityInfo)});
        cursor += sizeof(EntityInfo);
    }
    for (const WordRun& run : cell_runs)
    {
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

bool Scene::StagePageTable()
{
    const uint32_t pages = records_.page_count();
    if (pages_staged_ == pages)
    {
        return true;
    }
    const uint64_t count = pages - pages_staged_;
    StagingRing::Span span;
    if (!staging_.TryAllocate(count * sizeof(uint64_t), 8, StagingRing::Purpose::kRows, span))
    {
        return false;
    }
    for (uint32_t page = pages_staged_; page < pages; ++page)
    {
        const uint64_t address = records_.PageAddress(page);
        std::memcpy(span.data + (page - pages_staged_) * sizeof(uint64_t), &address, sizeof(address));
    }
    pending_.push_back(PendingCopy{span.offset, page_table_.buffer(),
                                   static_cast<VkDeviceSize>(pages_staged_) * sizeof(uint64_t),
                                   count * sizeof(uint64_t)});
    pages_staged_ = pages;
    return true;
}

void Scene::StageMaterials()
{
    if (!material_buffer_filled_)
    {
        // Until the chunks arrive, the kernels read zeros (black, no emission), not
        // uninitialized memory. The fill is recorded before the copies of this flush.
        fills_.push_back(PendingFill{material_buffer_.buffer(), 0, material_buffer_.size(), 0});
        material_buffer_filled_ = true;
    }
    const uint64_t dirty = materials_->dirty_chunks();
    for (uint32_t chunk = 0; chunk < MaterialTable::kChunks; ++chunk)
    {
        if ((dirty & (1ull << chunk)) == 0)
        {
            continue;
        }
        StagingRing::Span span;
        if (!staging_.TryAllocate(MaterialTable::kChunkBytes, kRowAlignment,
                                  StagingRing::Purpose::kRows, span))
        {
            CAELIX_LOGD("caelix_scene_flush: the staging ring is full; material table chunks wait for "
                        "the next flush");
            return;
        }
        const VkDeviceSize offset = static_cast<VkDeviceSize>(chunk) * MaterialTable::kChunkBytes;
        std::memcpy(span.data,
                    reinterpret_cast<const uint8_t*>(materials_->data()) + offset,
                    MaterialTable::kChunkBytes);
        pending_.push_back(
            PendingCopy{span.offset, material_buffer_.buffer(), offset, MaterialTable::kChunkBytes});
        materials_->MarkStaged(chunk);
    }
}

uint32_t Scene::StageDirtyList()
{
    if (dirty_.empty())
    {
        return 0;
    }
    // GPU allocation first, so a throw leaves the queue untouched.
    dirty_buffer_.Reserve(dirty_.size() * sizeof(DirtyEntry));

    // Drop entries whose brick was removed (or whose slot was reused and queued
    // again) and duplicates: a valid entry still carries the current epoch.
    uint32_t next_epoch = finalize_epoch_ + 1;
    if (next_epoch == 0)
    {
        next_epoch = 1;
    }
    std::vector<DirtyBrick> valid;
    valid.reserve(dirty_.size());
    for (const DirtyBrick& brick : dirty_)
    {
        if (brick.group_index >= groups_.size())
        {
            continue;
        }
        Group& group = groups_[brick.group_index];
        if (!group.alive || brick.id >= group.slots.size())
        {
            continue;
        }
        BrickSlotCpu& slot = group.slots[brick.id];
        if (slot.state != kSlotResident || slot.finalize_epoch != finalize_epoch_)
        {
            continue;
        }
        slot.finalize_epoch = next_epoch;  // seen: later duplicates are dropped
        valid.push_back(brick);
    }

    // As many entries as the ring holds; the rest waits for the next flush.
    uint64_t fit = valid.size();
    StagingRing::Span span;
    while (fit > 0 && !staging_.TryAllocate(fit * sizeof(DirtyEntry), kRowAlignment,
                                            StagingRing::Purpose::kRows, span))
    {
        fit /= 2;
    }
    for (uint64_t k = 0; k < fit; ++k)
    {
        BrickSlotCpu& slot = groups_[valid[k].group_index].slots[valid[k].id];
        const DirtyEntry entry{valid[k].group_index, valid[k].id, slot.record_index, 0};
        std::memcpy(span.data + k * sizeof(DirtyEntry), &entry, sizeof(entry));
        slot.finalize_epoch = 0;  // finalized at this flush
    }
    if (fit > 0)
    {
        pending_.push_back(
            PendingCopy{span.offset, dirty_buffer_.buffer(), 0, fit * sizeof(DirtyEntry)});
    }
    if (fit < valid.size())
    {
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

void Scene::RecordCopies(VkCommandBuffer cmd)
{
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
    for (size_t i = 0; i < pending_.size(); ++i)
    {
        if (kept > 0 && pending_[kept - 1].dst == pending_[i].dst &&
            pending_[kept - 1].dst_offset == pending_[i].dst_offset)
        {
            pending_[kept - 1] = pending_[i];
        }
        else
        {
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
    while (i < pending_.size())
    {
        const VkBuffer dst = pending_[i].dst;
        regions.clear();
        for (; i < pending_.size() && pending_[i].dst == dst; ++i)
        {
            const PendingCopy& copy = pending_[i];
            if (!regions.empty())
            {
                VkBufferCopy& last = regions.back();
                if (last.dstOffset + last.size == copy.dst_offset &&
                    last.srcOffset + last.size == copy.src_offset)
                {
                    last.size += copy.size;
                    continue;
                }
                if (regions.size() == kMaxRegionsPerCopy)
                {
                    vkCmdCopyBuffer(cmd, staging, dst, static_cast<uint32_t>(regions.size()),
                                    regions.data());
                    regions.clear();
                }
            }
            regions.push_back(VkBufferCopy{copy.src_offset, copy.dst_offset, copy.size});
        }
        if (!regions.empty())
        {
            vkCmdCopyBuffer(cmd, staging, dst, static_cast<uint32_t>(regions.size()), regions.data());
        }
    }
}

void Scene::Recycle(uint64_t safe_frame)
{
    safe_frame_ = std::max(safe_frame_, safe_frame);
    staging_.Recycle(safe_frame_);
    as_pool_.Recycle(safe_frame_);
    flush_timer_.Resolve(safe_frame_);
    std::erase_if(deferred_buffers_,
                  [this](const DeferredBuffer& deferred) { return deferred.frame <= safe_frame_; });
}

CaelixResult Scene::Flush(const CaelixFrameDesc& desc)
{
    if (desc.safe_frame_number >= desc.frame_number)
    {
        CAELIX_LOGE("caelix_scene_flush: safe_frame_number %llu must be below frame_number %llu",
                    static_cast<unsigned long long>(desc.safe_frame_number),
                    static_cast<unsigned long long>(desc.frame_number));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (has_flushed_ && desc.frame_number <= last_frame_)
    {
        CAELIX_LOGE("caelix_scene_flush: frame_number %llu is not above the previous flush's %llu",
                    static_cast<unsigned long long>(desc.frame_number),
                    static_cast<unsigned long long>(last_frame_));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    const VkCommandBuffer cmd = HandleFromU64<VkCommandBuffer>(desc.command_buffer);
    const uint64_t frame = desc.frame_number;

    // 1. Release what the GPU has finished with.
    Recycle(desc.safe_frame_number);
    ++flush_serial_;
    // TODO: VibeReview: Curious Q. Why is this here?
    // Entities staged with prev != cur at the last flush: prev = cur from now on.
    for (const CaelixEntityId id : motion_entities_)
    {
        Entity& entity = entities_[id];
        if (entity.alive)
        {
            std::memcpy(entity.prev_rows, entity.cur_rows, sizeof(entity.prev_rows));
            entity.info_dirty = true;
            ListEntity(id);
        }
    }
    motion_entities_.clear();

    // 2. GPU-side table work that needs no staging space: AABB range moves and
    //    NaN fills, reverse table and group grid fills of new ranges.
    for (const uint32_t group_index : dirty_groups_)
    {
        Group& group = groups_[group_index];
        if (group.alive)
        {
            PrepareGroupAabbs(group);
        }
    }
    for (const CaelixEntityId id : dirty_entities_)
    {
        Entity& entity = entities_[id];
        if (entity.alive && entity.transform_changed)
        {
            // The transform of the last flush becomes the previous frame's.
            std::memcpy(entity.prev_rows, entity.cur_rows, sizeof(entity.prev_rows));
            TransformRows(entity.position, entity.rotation, entity.cur_rows);
            entity.transform_changed = false;
            entity.info_dirty = true;
        }
        if (entity.alive && entity.grid_fill)
        {
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
        for (const uint32_t group_index : dirty_groups_)
        {
            Group& group = groups_[group_index];
            if (!group.alive)
            {
                group.in_dirty_list = false;
                continue;
            }
            if (ring_full || !StageGroup(group_index, group))
            {
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
        for (const CaelixEntityId id : dirty_entities_)
        {
            Entity& entity = entities_[id];
            if (!entity.alive)
            {
                entity.in_dirty_list = false;
                continue;
            }
            if (ring_full || !StageEntity(id, entity))
            {
                ring_full = true;
                still_dirty.push_back(id);
                continue;
            }
            entity.in_dirty_list = false;
            if (std::memcmp(entity.prev_rows, entity.cur_rows, sizeof(entity.prev_rows)) != 0)
            {
                motion_entities_.push_back(id);
            }
        }
        dirty_entities_.swap(still_dirty);
    }
    if (ring_full)
    {
        CAELIX_LOGW("caelix_scene_flush: the staging ring is full; %zu groups and %zu entities keep "
                    "dirty table rows for the next flush, and finalization waits",
                    dirty_groups_.size(), dirty_entities_.size());
    }

    // 4. The finalize list. Groups with entries that wait for a later flush do not
    //    build their BLAS in this one.
    const uint32_t dispatch_count = ring_full ? 0 : StageDirtyList();
    for (const DirtyBrick& brick : dirty_)
    {
        if (brick.group_index < groups_.size())
        {
            groups_[brick.group_index].finalize_wait_serial = flush_serial_;
        }
    }

    // 4b. Acceleration structure builds after the kernel (needs ring space for the
    //     TLAS instances).
    bool as_out_of_memory = false;
    const bool as_planned = PlanAccelerationStructures(frame, as_out_of_memory);

    // 4c. The material table: lowest priority, it takes the ring space that is left.
    StageMaterials();

    // 5. Pool growth since the last flush: the old contents are copied first, at
    //    the same offsets. The dirty list buffer is rewritten every flush and needs
    //    no copy.
    struct GrowthCopy
    {
        GpuBuffer source;
        VkBuffer destination;
    };
    std::vector<GrowthCopy> growth;
    const auto take_growth = [&](std::optional<GpuBuffer> source, VkBuffer destination) {
        if (source.has_value())
        {
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
    // The dirty list and the instance array are rewritten every time they are used
    // and need no copy.
    if (std::optional<GpuBuffer> old = dirty_buffer_.TakeGrowthSource(); old.has_value())
    {
        deferred_buffers_.push_back(DeferredBuffer{frame, std::move(*old)});
    }
    if (std::optional<GpuBuffer> old = instances_.TakeGrowthSource(); old.has_value())
    {
        deferred_buffers_.push_back(DeferredBuffer{frame, std::move(*old)});
    }

    if (!pending_.empty() || !growth.empty() || !fills_.empty() || !aabb_moves_.empty() ||
        dispatch_count > 0 || as_planned)
    {
        staging_.FlushHostWrites();
        flush_timer_.Begin(cmd, frame, safe_frame_);

        // 6. Earlier commands complete before this flush writes: shader reads of reused
        //    records and rows, and traversal of acceleration structures and builds of
        //    earlier frames (in-place rebuilds and the reused scratch buffer).
        MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);

        // 7. Growth copies.
        if (!growth.empty())
        {
            for (GrowthCopy& copy : growth)
            {
                const VkBufferCopy region{0, 0, copy.source.size()};
                vkCmdCopyBuffer(cmd, copy.source.buffer(), copy.destination, 1, &region);
                deferred_buffers_.push_back(DeferredBuffer{frame, std::move(copy.source)});
            }
            TransferBarrier(cmd);
        }

        // 8. AABB range moves and fills of new ranges. They come before the staged
        //    copies, which may write NaN rows or table words into the same ranges.
        if (!fills_.empty() || !aabb_moves_.empty())
        {
            for (size_t i = 0; i < aabb_moves_.size(); i += kMaxRegionsPerCopy)
            {
                const size_t n = std::min(kMaxRegionsPerCopy, aabb_moves_.size() - i);
                vkCmdCopyBuffer(cmd, aabbs_.buffer(), aabbs_.buffer(), static_cast<uint32_t>(n),
                                aabb_moves_.data() + i);
            }
            for (const PendingFill& fill : fills_)
            {
                vkCmdFillBuffer(cmd, fill.buffer, fill.offset, fill.size, fill.value);
            }
            TransferBarrier(cmd);
        }

        // 9. Staging -> pools and tables.
        RecordCopies(cmd);

        // 10. Finalize the dirty bricks, then later commands see everything.
        if (dispatch_count > 0)
        {
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
            push.records_per_page_shift = records_per_page_shift();
            push.record_bytes = layout_.record_bytes;
            push.group_shift = group_shift_;
            push.encoding = encoding_;
            for (uint32_t base = 0; base < dispatch_count; base += kMaxDispatchGroups)
            {
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
        }
        else
        {
            MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        }

        // 11. BLAS builds over the AABB rows, then the TLAS over the instances. The
        //     barrier above makes the copies and the kernel's writes visible to them.
        if (as_planned)
        {
            RecordAccelerationStructureBuilds(cmd);
        }
        flush_timer_.End(cmd);
    }

    // 12. Close the window.
    staging_.CloseFrame(frame);
    pending_.clear();
    fills_.clear();
    aabb_moves_.clear();
    nan_row_valid_ = false;
    pending_bricks_ = 0;
    for (const RangeToFree& item : ranges_to_free_after_flush_)
    {
        item.pool->Free(item.range);
    }
    ranges_to_free_after_flush_.clear();
    last_frame_ = frame;
    has_flushed_ = true;
    return as_out_of_memory ? CAELIX_ERROR_OUT_OF_MEMORY : CAELIX_OK;
}

}  // namespace caelix
