// Shared code of the light tree's group pass kernels (caelix_light_tree_size, _scan, _build,
// _commit; docs/LIGHT_TREE.md section 3): the push constants, the shared-memory scan of a group's
// light bricks (a bitmap over the Morton positions of the group's bricks with prefix popcounts:
// every light brick gets its Morton rank without a sort), and the plan of a brick's light record
// (recompute or copy from the old block).
#ifndef CAELIX_LIGHT_TREE_GROUP_HLSL
#define CAELIX_LIGHT_TREE_GROUP_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_light_tree_layout.hlsl"
#include "common/caelix_workgroup_scan.hlsl"
#include "scene/caelix_scene_lookup.hlsl"

// Push constants of the four group pass kernels (LightGroupPush). 80 bytes.
struct LightGroupPush
{
    uint64_t descriptor;      // light tree descriptor (layout: common/caelix_light_tree_layout.hlsl)
    uint64_t page_table;      // record page addresses
    uint64_t slot_tables;     // slot row pool base
    uint64_t reverse_tables;  // reverse table pool base
    uint64_t materials;       // CaelixMaterial by block code
    uint64_t offsets;         // u32 per group: block size (size kernel), then new offset (scan kernel)
    uint64_t staged_flags;    // u32 per group: the CPU's kLtCpu* flags of this flush, 0 = nothing new
    uint records_per_page_shift;
    uint record_bytes;
    uint group_base;      // size / build: first group index of this dispatch
    uint group_count;     // rows of the pass
    uint capacity_units;  // capacity of each heap buffer in units
    uint pad;
};

CAELIX_PUSH_CONSTANTS(LightGroupPush, g_push);

static const uint kLtThreads = 256;
// Bitmap words for group_shift 5 (32768 brick positions); smaller shifts use less.
static const uint kLtBitmapWordsMax = 1024;

groupshared uint gs_bits[kLtBitmapWordsMax];
groupshared uint gs_prefix[kLtBitmapWordsMax];  // exclusive popcount prefix per word

SceneLookup LtLookup(uint group_shift)
{
    SceneLookup s;
    s.page_table = g_push.page_table;
    s.slot_tables = g_push.slot_tables;
    s.reverse_tables = g_push.reverse_tables;
    s.group_infos = LtDescriptorU64(g_push.descriptor, kLtDescGroupInfos);
    s.entity_infos = LtDescriptorU64(g_push.descriptor, kLtDescEntityInfos);
    s.group_grids = LtDescriptorU64(g_push.descriptor, kLtDescGrids);
    s.records_per_page_shift = g_push.records_per_page_shift;
    s.record_bytes = g_push.record_bytes;
    s.group_shift = group_shift;
    return s;
}

uint LtBitmapWords(uint s)
{
    return max(1u, (1u << (3 * s)) >> 5);
}

// Exclusive popcount prefix of the bitmap's words (at most 1024 words: 4 per thread).
void LtScanBitmap(uint words, uint t)
{
    uint counts[4];
    CAELIX_UNROLL
    for (uint i = 0; i < 4; ++i)
    {
        const uint w = 4 * t + i;
        counts[i] = w < words ? countbits(gs_bits[w]) : 0u;
    }
    uint total;
    WorkgroupExclusiveSum4(counts, t, kLtThreads, total);
    CAELIX_UNROLL
    for (uint j = 0; j < 4; ++j)
    {
        const uint w = 4 * t + j;
        if (w < words)
        {
            gs_prefix[w] = counts[j];
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

// Set bits of the bitmap (after LtScanBitmap).
uint LtBitmapCount(uint words)
{
    return gs_prefix[words - 1] + countbits(gs_bits[words - 1]);
}

// Rank of the set bit m among the set bits.
uint LtRank(uint m)
{
    const uint w = m >> 5;
    return gs_prefix[w] + countbits(gs_bits[w] & ((1u << (m & 31)) - 1u));
}

bool LtBitSet(uint m)
{
    return ((gs_bits[m >> 5] >> (m & 31)) & 1u) != 0;
}

// Record word 1 of a dense id (light face count and stale bit, LtLightFaceCount), 0 when the id is
// not resident. Also returns the brick's group-local position and record address.
uint LtBrickWord1(SceneLookup s, uint slot_base, uint id, out uint pos, out uint64_t record)
{
    const uint4 row = CAELIX_LOAD_U32x4(s.slot_tables + (uint64_t(slot_base) + id) * kSlotRowBytes);
    pos = SlotRowPos(row.x);
    record = 0;
    if (SlotRowState(row.x) != kSlotStateResident || row.z == kInvalidRecord)
    {
        return 0;
    }
    record = SceneRecordAddress(s, row.z);
    return CAELIX_LOAD_U32(record + kRecordBoundsWord * kRecordWordBytes);
}

// The result of LtScanGroup.
struct LtGroupScan
{
    uint bricks;   // light bricks kept (at most kLtMaxLightBricksPerGroup)
    uint dropped;  // light bricks above the limit
};

// Scans the dense ids [0, high_water) of one group: sets the bitmap of the Morton positions of its
// light bricks and its prefix popcounts in shared memory (keeping the lowest
// kLtMaxLightBricksPerGroup positions). Called by every thread of the workgroup.
LtGroupScan LtScanGroup(SceneLookup s, uint slot_base, uint high_water, uint t)
{
    const uint shift = s.group_shift;
    const uint words = LtBitmapWords(shift);
    for (uint i = t; i < words; i += kLtThreads)
    {
        gs_bits[i] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint id = t; id < high_water; id += kLtThreads)
    {
        uint pos;
        uint64_t record;
        if (LtLightFaceCount(LtBrickWord1(s, slot_base, id, pos, record)) != 0)
        {
            const uint m = LtMorton(BrickOfPos(pos, shift));
            InterlockedOr(gs_bits[m >> 5], 1u << (m & 31));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    LtScanBitmap(words, t);
    LtGroupScan scan;
    const uint found = LtBitmapCount(words);
    scan.dropped = found > kLtMaxLightBricksPerGroup ? found - kLtMaxLightBricksPerGroup : 0u;
    if (scan.dropped != 0)
    {
        // group_shift 5 only: keep the lowest Morton positions.
        for (uint w = t; w < words; w += kLtThreads)
        {
            const uint before = gs_prefix[w];
            uint word = gs_bits[w];
            if (before >= kLtMaxLightBricksPerGroup)
            {
                word = 0;
            }
            else
            {
                const uint keep = kLtMaxLightBricksPerGroup - before;
                while (countbits(word) > keep)
                {
                    word &= ~(1u << firstbithigh(word));
                }
            }
            gs_bits[w] = word;
        }
        GroupMemoryBarrierWithGroupSync();
        LtScanBitmap(words, t);
    }
    scan.bricks = LtBitmapCount(words);
    return scan;
}

// Size of a light block in units: header, index entries, records.
uint LtBlockUnits(uint bricks, uint record_units)
{
    return bricks == 0 ? 0u
                       : kLtBlockIndexOffset / kLtUnitBytes + bricks * (kLtIndexEntryBytes / kLtUnitBytes) +
                             record_units;
}

// The group's block before this pass (in the current heap buffer); valid = false when it has none
// or the group index starts a new life (fresh): then every record is recomputed.
struct LtOldBlock
{
    uint64_t address;
    uint record_count;
    bool valid;
};

LtOldBlock LtOpenOldBlock(uint64_t heap, uint offset_units, uint units, bool fresh)
{
    LtOldBlock b;
    b.address = heap + uint64_t(offset_units) * kLtUnitBytes;
    b.valid = units != 0 && !fresh;
    b.record_count = b.valid ? CAELIX_LOAD_U32(b.address) : 0u;  // LightBlockHeader::record_count
    return b;
}

// Units of a brick light record with `entries` face entries.
uint LtRecordUnits(uint entries)
{
    return (kLtRecordHeaderBytes + kLtFaceEntryBytes * entries + kLtUnitBytes - 1) / kLtUnitBytes;
}

// How a kept light brick gets its record: recomputed (stale bit, no old block, a fresh group, or no
// old entry at its position) with room for the finalize kernel's face count, or copied from the old
// block (old_record = its offset in units from the old block's start).
struct LtBrickPlan
{
    bool recompute;
    uint units;
    uint old_record;
};

LtBrickPlan LtPlanBrick(uint word1, uint m, LtOldBlock old, uint shift)
{
    LtBrickPlan plan;
    plan.recompute = true;
    plan.old_record = 0;
    plan.units = LtRecordUnits(LtLightFaceCount(word1));
    if ((word1 & kLtRecordStaleBit) == 0 && old.valid)
    {
        const uint entry = LtFindIndexEntry(old.address, old.record_count, m, shift);
        if (entry != kLtNone)
        {
            plan.recompute = false;
            const uint64_t index_entry =
                old.address + kLtBlockIndexOffset + uint64_t(entry) * kLtIndexEntryBytes;
            plan.old_record = CAELIX_LOAD_U32(index_entry + kLtIndexRecordOffset);
            const uint64_t old_record = old.address + uint64_t(plan.old_record) * kLtUnitBytes;
            plan.units = LtRecordUnits(CAELIX_LOAD_U32(old_record + kLtRecordEntryCount));
        }
    }
    return plan;
}

#endif  // CAELIX_LIGHT_TREE_GROUP_HLSL
