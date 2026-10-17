// Brick upload and removal, slot placement and the finalize queue.
#include "scene/scene.h"

#include "core/log.h"

#include <cstring>
#include <new>
#include <optional>

namespace caelix {
namespace {

// Staging alignment of records. Records are 8-byte multiples, so consecutive
// records stay contiguous in the ring and their copies merge. vkCmdCopyBuffer
// itself has no offset alignment rule.
constexpr uint64_t kRecordAlignment = 8;

// The group index is the instance's 24-bit instanceCustomIndex.
constexpr size_t kMaxGroups = size_t{1} << 24;

// Neighbour directions in face order: +X, -X, +Y, -Y, +Z, -Z.
constexpr int32_t kDirections[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

}  // namespace

// ---------------------------------------------------------------------------
// Slots

void Scene::MarkDirty(uint32_t group_index, uint16_t id)
{
    Group& group = groups_[group_index];
    if (!group.full_rewrite)
    {
        group.dirty_ids.push_back(id);
    }
    ListGroup(group_index);
}

void Scene::MarkReverseDirty(Group& group, uint32_t pos)
{
    group.reverse_dirty_words.push_back(pos >> 1);
}

Scene::Placed Scene::PlaceBrick(CaelixEntityId entity_id, const Location& loc)
{
    Entity& entity = entities_[entity_id];
    const auto found = entity.groups.find(loc.group_key);
    uint32_t group_index = found != entity.groups.end() ? found->second : 0xFFFFFFFFu;
    const uint32_t pages_before = records_.page_count();

    // Replace an existing brick: same id, same record, unchanged row.
    if (found != entity.groups.end())
    {
        Group& group = groups_[group_index];
        const uint16_t id = group.id_by_pos[loc.pos];
        if (id != Group::kNoId)
        {
            BrickSlotCpu& slot = group.slots[id];
            if (slot.state != kSlotResident)
            {
                // A known brick without a record (states for later versions).
                const uint32_t record = records_.Allocate();
                try
                {
                    page_table_.Reserve(static_cast<VkDeviceSize>(records_.page_count()) * sizeof(uint64_t));
                }
                catch (...)
                {
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
    try
    {
        if (records_.page_count() != pages_before)
        {
            page_table_.Reserve(static_cast<VkDeviceSize>(records_.page_count()) * sizeof(uint64_t));
        }
        if (found == entity.groups.end())
        {
            new_slots = slot_tables_.Allocate(kSlotMinRows);
            new_aabbs = aabbs_.Allocate(kSlotMinRows);
            new_reverse = reverse_tables_.Allocate(bricks_per_group_);
            if (free_groups_.empty())
            {
                if (groups_.size() >= kMaxGroups)
                {
                    CAELIX_LOGE("the scene has %zu render groups, the maximum (the group index is the 24-bit "
                                "TLAS instance custom index)",
                                groups_.size());
                    throw std::bad_alloc();
                }
                group_infos_.Reserve((groups_.size() + 1) * sizeof(GroupInfo));
            }
            if (!GridContains(entity, loc.gx, loc.gy, loc.gz))
            {
                grid_shape = GrownGridShape(entity, loc.gx, loc.gy, loc.gz);
                new_grid = grids_.Allocate(static_cast<uint32_t>(grid_shape->cells));
            }
        }
        else
        {
            const Group& group = groups_[group_index];
            if (group.free_ids.empty() && group.slots.size() == group.slot_rows.rows)
            {
                new_slots = slot_tables_.Allocate(group.slot_rows.rows * 2);
                new_aabbs = aabbs_.Allocate(group.slot_rows.rows * 2);
            }
        }
    }
    catch (...)
    {
        slot_tables_.Free(new_slots);
        aabbs_.Free(new_aabbs);
        reverse_tables_.Free(new_reverse);
        grids_.Free(new_grid);
        records_.Free(record);
        throw;
    }

    if (found == entity.groups.end())
    {
        if (grid_shape.has_value())
        {
            FreeAfterFlush(grids_, entity.grid_range);
            RebuildGrid(entity, *grid_shape, new_grid);
            ListEntity(entity_id);
        }
        group_index = CreateGroup(entity_id, loc, new_slots, new_aabbs, new_reverse);
    }
    else if (new_slots.valid())
    {
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
    if (!group.free_ids.empty())
    {
        id = group.free_ids.back();
        group.free_ids.pop_back();
    }
    else
    {
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

void Scene::QueueFinalize(uint32_t group_index, uint16_t id)
{
    BrickSlotCpu& slot = groups_[group_index].slots[id];
    if (slot.finalize_epoch == finalize_epoch_)
    {
        return;
    }
    slot.finalize_epoch = finalize_epoch_;
    dirty_.push_back(DirtyBrick{group_index, id});
    // The finalize kernel rewrites the brick's AABB row.
    MarkBlasDirty(group_index);
}

void Scene::QueueNeighbors(CaelixEntityId entity_id, const CaelixBrickKey& key,
                           uint32_t group_index)
{
    const Entity& entity = entities_[entity_id];
    const int32_t edge = 1 << group_shift_;
    const int32_t local[3] = {key.x & (edge - 1), key.y & (edge - 1), key.z & (edge - 1)};
    const int32_t stride[3] = {1, edge, edge * edge};
    const Location own = Locate(key);
    for (int d = 0; d < 6; ++d)
    {
        const int axis = d >> 1;
        const int32_t step = (d & 1) != 0 ? -1 : 1;
        uint32_t neighbour_group = 0xFFFFFFFFu;
        uint32_t pos = 0;
        const int32_t next = local[axis] + step;
        if (group_index != 0xFFFFFFFFu && next >= 0 && next < edge)
        {
            // Same group: no lookup.
            neighbour_group = group_index;
            pos = static_cast<uint32_t>(static_cast<int32_t>(own.pos) + step * stride[axis]);
        }
        else
        {
            const CaelixBrickKey neighbour{key.x + kDirections[d][0], key.y + kDirections[d][1],
                                           key.z + kDirections[d][2]};
            if (!KeyInRange(neighbour))
            {
                continue;
            }
            const Location loc = Locate(neighbour);
            const auto found = entity.groups.find(loc.group_key);
            if (found == entity.groups.end())
            {
                continue;
            }
            neighbour_group = found->second;
            pos = loc.pos;
        }
        const Group& group = groups_[neighbour_group];
        const uint16_t id = group.id_by_pos[pos];
        if (id != Group::kNoId && group.slots[id].state == kSlotResident)
        {
            QueueFinalize(neighbour_group, id);
        }
    }
}

// ---------------------------------------------------------------------------
// Upload and removal

void Scene::WriteRecord(uint8_t* dst, uint32_t pos, const uint8_t* blocks) const
{
    // Build the record in local memory, then copy it in one go: the staging memory
    // may be write-combined.
    uint32_t record[2 + 16 + 256];
    static_assert(sizeof(record) == 1096);
    uint16_t values[RecordLayout::kBlocksPerBrick];
    std::memcpy(values, blocks, sizeof(values));  // the host pointer may be unaligned
    record[0] = pos & 0xFFFFu;                    // the finalize kernel fills bits 16..23 and words 1..17
    for (uint32_t w = 1; w < layout_.payload_first_word(); ++w)
    {
        record[w] = 0;
    }
    uint32_t* payload = record + layout_.payload_first_word();
    // Voxel index i = x + 8y + 64z. For even x, i/2 = (x >> 1) + 4y + 32z, and the
    // pair (i, i + 1) shares one word: even x high, odd x low.
    for (uint32_t i = 0; i < RecordLayout::kBlocksPerBrick; i += 2)
    {
        payload[i / 2] = (static_cast<uint32_t>(values[i]) << 16) | values[i + 1];
    }
    std::memcpy(dst, record, layout_.record_bytes);
}

CaelixResult Scene::UploadBricks(CaelixEntityId entity, uint32_t count, const CaelixBrickKey* keys,
                                 const void* blocks, uint32_t& consumed)
{
    consumed = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
        if (!KeyInRange(keys[i]))
        {
            CAELIX_LOGE("caelix_scene_upload_bricks: key %u (%d, %d, %d) is out of range", i, keys[i].x,
                        keys[i].y, keys[i].z);
            return CAELIX_ERROR_INVALID_ARGUMENT;
        }
    }
    const uint8_t* block_bytes = static_cast<const uint8_t*>(blocks);
    const size_t brick_bytes = static_cast<size_t>(RecordLayout::kBlocksPerBrick) * layout_.block_bytes;
    for (uint32_t i = 0; i < count; ++i)
    {
        StagingRing::Span span;
        if (!staging_.TryAllocate(layout_.record_bytes, kRecordAlignment,
                                  StagingRing::Purpose::kRecords, span))
        {
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
                                 const CaelixBrickKey* keys)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        if (!KeyInRange(keys[i]))
        {
            CAELIX_LOGE("caelix_scene_remove_bricks: key %u (%d, %d, %d) is out of range", i, keys[i].x,
                        keys[i].y, keys[i].z);
            return CAELIX_ERROR_INVALID_ARGUMENT;
        }
    }
    Entity& entity = entities_[entity_id];
    for (uint32_t i = 0; i < count; ++i)
    {
        const Location loc = Locate(keys[i]);
        const auto found = entity.groups.find(loc.group_key);
        if (found == entity.groups.end())
        {
            continue;
        }
        const uint32_t group_index = found->second;
        Group& group = groups_[group_index];
        const uint16_t id = group.id_by_pos[loc.pos];
        if (id == Group::kNoId)
        {
            continue;
        }
        BrickSlotCpu& slot = group.slots[id];
        if (slot.state == kSlotResident)
        {
            // Reusable at once; see RecordPool for why that is safe on one queue.
            records_.Free(slot.record_index);
            --bricks_resident_;
        }
        slot = BrickSlotCpu{loc.pos, kSlotFree, RecordPool::kInvalidIndex};
        group.id_by_pos[loc.pos] = Group::kNoId;
        group.free_ids.push_back(id);
        --group.brick_count;
        --bricks_known_;
        if (group.brick_count == 0)
        {
            entity.groups.erase(found);
            ReleaseGroup(group_index);
        }
        else
        {
            MarkDirty(group_index, id);
            MarkReverseDirty(group, loc.pos);
            group.nan_ids.push_back(id);  // the freed id's AABB row becomes inactive
            MarkBlasDirty(group_index);
        }
        // The neighbours lose a neighbour: they are finalized again.
        QueueNeighbors(entity_id, keys[i], group.brick_count > 0 ? group_index : 0xFFFFFFFFu);
    }
    return CAELIX_OK;
}

// ---------------------------------------------------------------------------
// Acceleration structure dirty tracking

void Scene::MarkBlasDirty(uint32_t group_index)
{
    Group& group = groups_[group_index];
    group.blas_dirty = true;
    if (!group.in_blas_list)
    {
        group.in_blas_list = true;
        blas_pending_.push_back(group_index);
    }
}

}  // namespace caelix
