// Light tree group pass, kernel 3 of 4 (docs/LIGHT_TREE.md section 3): one workgroup (256 threads)
// per group index writes the group's block into the target heap buffer (1 - current) at
// offsets[group], and its GroupLightRow. Returns at once unless the scan kernel decided kLtPassOk.
//   released -> the row is cleared (generation + 1)
//   flagged  -> the block is rebuilt (generation + 1):
//     1. the light brick scan (shared with the size kernel) gives every kept light brick its Morton
//        rank (= its index entry) and its plan: its brick light record is recomputed (the stale
//        bit, no old block, a fresh group, or no old entry at its position) or copied from the old
//        block. The stale bit of every resident brick of the group is cleared;
//     2. one wave per light brick writes its index entry (bvh_leaf = none until the gather) and its
//        record: a recomputed record resolves the six neighbour records, finds the open faces of
//        its light voxels (two passes over the voxels: counts, then entries) and writes them; a
//        copied record is copied as it is;
//     3. the row: offset, units, counts, generation + 1, flags cleared.
//   otherwise -> the block is copied to its new offset, the row gets the new offset.
// A recomputed record holds at most as many entries as the finalize kernel counted (record word 1):
// the two can differ only while a neighbour's change waits for its finalization, and that
// finalization marks the record stale again.
#include "common/caelix_material.hlsl"
#include "scene/caelix_light_tree_group.hlsl"

static const uint kModeCopy = 0;
static const uint kModeRebuild = 1;
static const uint kModeReleased = 2;

groupshared uint gs_mode;
groupshared uint gs_fresh;
groupshared uint4 gs_row;  // block_offset, block_units, generation, gpu_flags
groupshared uint gs_pass_result;
groupshared uint gs_current;
groupshared int3 gs_coord;
groupshared uint gs_entity;
groupshared uint gs_slot_base;
groupshared uint gs_reverse_base;
groupshared uint gs_high_water;
// The plan of every kept light brick, by rank:
// dense id | kLtPlanRecompute | record units << kLtPlanUnitsShift.
groupshared uint gs_plan[kLtMaxLightBricksPerGroup];
groupshared uint gs_chunk[kLtThreads];  // record units before rank kLtRanksPerThread * t
groupshared uint gs_faces_written;
groupshared uint gs_recomputed;
groupshared uint gs_copied;

static const uint kLtRanksPerThread = kLtMaxLightBricksPerGroup / kLtThreads;  // 16
static const uint kLtPlanIdMask = 0xFFFFu;
static const uint kLtPlanRecompute = 1u << 16;
static const uint kLtPlanUnitsShift = 17;
static const uint kLtPlanUnitsMask = 0x3FFu;  // LtRecordUnits(3072 face entries) = 772

uint LtPlanUnits(uint plan)
{
    return (plan >> kLtPlanUnitsShift) & kLtPlanUnitsMask;
}

uint LtRecordBlock(uint64_t record, uint3 v)
{
    const uint64_t word = record + uint64_t(RecordPayloadWord(v.x, v.y, v.z)) * kRecordWordBytes;
    return PayloadPairBlock(CAELIX_LOAD_U32(word), v.x);
}

// The block at brick-local p (one coordinate may be -1 or 8: a face neighbour in another brick).
uint LtNeighbourBlock(uint64_t own, uint64_t nbr[6], int3 p)
{
    uint64_t record = own;
    if (p.x > 7)
    {
        record = nbr[0];
    }
    else if (p.x < 0)
    {
        record = nbr[1];
    }
    else if (p.y > 7)
    {
        record = nbr[2];
    }
    else if (p.y < 0)
    {
        record = nbr[3];
    }
    else if (p.z > 7)
    {
        record = nbr[4];
    }
    else if (p.z < 0)
    {
        record = nbr[5];
    }
    return record != 0 ? LtRecordBlock(record, uint3(p & 7)) : 0u;
}

// Open-face mask (bit f) of voxel vi of the brick, 0 when it is not a light voxel.
uint LtOpenFaces(uint64_t own, uint64_t nbr[6], uint vi, uint64_t light_codes, uint encoding, out uint value)
{
    const int3 p = int3(vi & 7, (vi >> 3) & 7, vi >> 6);
    value = LtRecordBlock(own, uint3(p));
    const uint code = BlockMaterialCode(value, encoding);
    if (((CAELIX_LOAD_U32(light_codes + uint64_t(code >> 5) * 4) >> (code & 31)) & 1u) == 0)
    {
        return 0;
    }
    uint open = 0;
    for (uint f = 0; f < 6; ++f)
    {
        if (!BlockIsOpaque(LtNeighbourBlock(own, nbr, p + FaceDirection(f)), encoding))
        {
            open |= 1u << f;
        }
    }
    return open;
}

// One light brick (rank r), run by one whole wave: its index entry and its record.
void LtBuildBrick(SceneLookup lookup, uint64_t block, uint records_units_base, uint r, LtOldBlock old,
                  uint64_t light_codes, uint encoding)
{
    const uint lane = WaveGetLaneIndex();
    const uint lanes = WaveGetLaneCount();
    const uint plan = gs_plan[r];
    const uint id = plan & kLtPlanIdMask;
    uint record_units = records_units_base + gs_chunk[r / kLtRanksPerThread];
    for (uint j = (r / kLtRanksPerThread) * kLtRanksPerThread; j < r; ++j)
    {
        record_units += LtPlanUnits(gs_plan[j]);
    }
    const uint64_t record_address = block + uint64_t(record_units) * kLtUnitBytes;

    uint pos;
    uint64_t own;
    const uint word1 = LtBrickWord1(lookup, gs_slot_base, id, pos, own);
    const uint3 brick = BrickOfPos(pos, lookup.group_shift);
    if (WaveIsFirstLane())
    {
        // pos, dense_id, record_offset, bvh_leaf
        CAELIX_STORE_U32x4(block + kLtBlockIndexOffset + uint64_t(r) * kLtIndexEntryBytes,
                           uint4(pos, id, record_units, kLtNone));
    }

    if ((plan & kLtPlanRecompute) == 0)
    {
        // Copy the old record as it is.
        const uint old_entry =
            LtFindIndexEntry(old.address, old.record_count, LtMorton(brick), lookup.group_shift);
        const uint64_t old_index_entry =
            old.address + kLtBlockIndexOffset + uint64_t(old_entry) * kLtIndexEntryBytes;
        const uint old_record = CAELIX_LOAD_U32(old_index_entry + kLtIndexRecordOffset);
        const uint64_t from = old.address + uint64_t(old_record) * kLtUnitBytes;
        const uint units = LtPlanUnits(plan);
        for (uint u = lane; u < units; u += lanes)
        {
            CAELIX_STORE_U32x4(record_address + uint64_t(u) * kLtUnitBytes,
                               CAELIX_LOAD_U32x4(from + uint64_t(u) * kLtUnitBytes));
        }
        if (WaveIsFirstLane())
        {
            InterlockedAdd(gs_faces_written, CAELIX_LOAD_U32(from + kLtRecordEntryCount));
        }
        return;
    }

    SceneBrickPlace place;
    place.coord = gs_coord;
    place.entity = gs_entity;
    place.slot_base = gs_slot_base;
    place.reverse_base = gs_reverse_base;
    place.pos = pos;
    uint64_t mine = 0;
    if (lane < 6)
    {
        mine = SceneResolveNeighbor(lookup, place, lane);
    }
    const uint2 mine_words = uint2(uint(mine), uint(mine >> 32));
    uint64_t nbr[6];
    CAELIX_UNROLL
    for (uint d = 0; d < 6; ++d)
    {
        const uint2 w = WaveReadLaneAt(mine_words, d);
        nbr[d] = uint64_t(w.x) | (uint64_t(w.y) << 32);
    }

    // Pass 1: open faces per family. Lane l owns voxels [l * k, (l + 1) * k), so the entries of a
    // family come out in voxel order.
    const uint k = kBrickVoxels / lanes;
    const uint first_voxel = lane * k;
    uint counts[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    for (uint i = 0; i < k; ++i)
    {
        uint value;
        const uint open = LtOpenFaces(own, nbr, first_voxel + i, light_codes, encoding, value);
        CAELIX_UNROLL
        for (uint f = 0; f < 6; ++f)
        {
            counts[f] += (open >> f) & 1u;
        }
    }
    uint family_base[6];
    uint family_count[6];
    uint cursor[6];
    uint left = LtLightFaceCount(word1);
    uint written = 0;
    CAELIX_UNROLL
    for (uint f1 = 0; f1 < 6; ++f1)
    {
        const uint found = WaveActiveSum(counts[f1]);
        cursor[f1] = WavePrefixSum(counts[f1]);
        family_count[f1] = min(found, left);
        family_base[f1] = written;
        written += family_count[f1];
        left -= family_count[f1];
    }

    // Pass 2: entries, face sums, the tight box.
    const uint64_t entries = record_address + kLtRecordHeaderBytes;
    const uint64_t materials = g_push.materials;
    float sums[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint3 lo = uint3(8, 8, 8);
    uint3 hi = uint3(0, 0, 0);
    for (uint i2 = 0; i2 < k; ++i2)
    {
        const uint vi = first_voxel + i2;
        uint value;
        const uint open = LtOpenFaces(own, nbr, vi, light_codes, encoding, value);
        if (open == 0)
        {
            continue;
        }
        const uint code = BlockMaterialCode(value, encoding);
        const float weight = LtLuminance(
            asfloat(CAELIX_LOAD_U32x4(materials + uint64_t(code) * kMaterialBytes + kMaterialEmission).xyz));
        bool kept = false;
        CAELIX_UNROLL
        for (uint f = 0; f < 6; ++f)
        {
            if (((open >> f) & 1u) != 0)
            {
                if (cursor[f] < family_count[f])
                {
                    CAELIX_STORE_U32(entries + uint64_t(family_base[f] + cursor[f]) * kLtFaceEntryBytes,
                                     vi | (value << kLtFaceEntryBlockShift));
                    sums[f] += weight;
                    kept = true;
                }
                ++cursor[f];
            }
        }
        if (kept)
        {
            const uint3 p = uint3(vi & 7, (vi >> 3) & 7, vi >> 6);
            lo = min(lo, p);
            hi = max(hi, p + 1);
        }
    }
    float face[6];
    CAELIX_UNROLL
    for (uint f2 = 0; f2 < 6; ++f2)
    {
        face[f2] = WaveActiveSum(sums[f2]);
    }
    lo = WaveActiveMin(lo);
    hi = WaveActiveMax(hi);
    if (WaveIsFirstLane())
    {
        const uint3 origin = brick * kBrickEdge;
        const bool empty = written == 0;
        const uint box_lo = LtPackVoxel(origin + (empty ? uint3(0, 0, 0) : lo));
        const uint box_hi = LtPackVoxel(origin + (empty ? uint3(0, 0, 0) : hi));
        const uint b = kLtFamilyCountBits;
        // face[0..3]; face[4..5], box_lo, box_hi; pos, entry_count, counts_lo, counts_hi
        const uint counts_lo = family_count[0] | (family_count[1] << b) | (family_count[2] << (2 * b));
        const uint counts_hi = family_count[3] | (family_count[4] << b) | (family_count[5] << (2 * b));
        CAELIX_STORE_U32x4(record_address + kLtRecordFace,
                           asuint(float4(face[0], face[1], face[2], face[3])));
        CAELIX_STORE_U32x4(record_address + kLtRecordFace + 16,
                           uint4(asuint(face[4]), asuint(face[5]), box_lo, box_hi));
        CAELIX_STORE_U32x4(record_address + kLtRecordPos, uint4(pos, written, counts_lo, counts_hi));
        // Zero padding up to the end of the record's units (records are copied whole).
        const uint capacity = (LtPlanUnits(plan) * kLtUnitBytes - kLtRecordHeaderBytes) / kLtFaceEntryBytes;
        for (uint e = written; e < capacity; ++e)
        {
            CAELIX_STORE_U32(entries + uint64_t(e) * kLtFaceEntryBytes, 0u);
        }
        InterlockedAdd(gs_faces_written, written);
    }
}

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID)
{
    const uint group = g_push.group_base + group_id.x;
    if (group >= g_push.group_count)
    {
        return;  // uniform for the whole workgroup
    }
    const uint t = thread_id.x;
    const uint64_t descriptor = g_push.descriptor;
    const uint64_t control = LtDescriptorU64(descriptor, kLtDescControl);
    const uint64_t row = LtDescriptorU64(descriptor, kLtDescGroupRows) + uint64_t(group) * kLtGroupRowBytes;
    const uint shift = LtDescriptorU32(descriptor, kLtDescGroupShift);
    const SceneLookup lookup = LtLookup(shift);
    if (t == 0)
    {
        gs_pass_result = CAELIX_LOAD_U32(control + kLtCtlPassResult);
        gs_current = CAELIX_LOAD_U32(control + kLtCtlCurrent) & 1u;
        gs_row = CAELIX_LOAD_U32x4(row);
        const uint cpu_flags = CAELIX_LOAD_U32(row + kLtRowCpuFlags);  // merged by the size kernel
        gs_mode = (cpu_flags & kLtCpuReleased) != 0                                          ? kModeReleased
                  : ((cpu_flags & kLtCpuRebuild) != 0 || (gs_row.w & kLtGpuLightChanged) != 0) ? kModeRebuild
                                                                                              : kModeCopy;
        gs_fresh = cpu_flags & kLtCpuFresh;
        const uint64_t info = lookup.group_infos + uint64_t(group) * kGroupInfoBytes;
        const uint4 w0 = CAELIX_LOAD_U32x4(info);  // coord, entity (kGroupInfoCoord = 0)
        const uint4 w1 = CAELIX_LOAD_U32x4(info + kGroupInfoSlotRowBase);
        gs_coord = int3(w0.xyz);
        gs_entity = w0.w;
        gs_slot_base = w1.x;
        gs_reverse_base = w1.y;
        gs_high_water = w1.w;
        gs_faces_written = 0;
        gs_recomputed = 0;
        gs_copied = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (gs_pass_result != kLtPassOk)
    {
        return;
    }
    const uint mode = gs_mode;
    const uint4 old_row = gs_row;
    const uint new_offset = CAELIX_LOAD_U32(g_push.offsets + uint64_t(group) * 4);
    const uint64_t source = LtDescriptorU64(descriptor, gs_current != 0 ? kLtDescHeap1 : kLtDescHeap0);
    const uint64_t target = LtDescriptorU64(descriptor, gs_current != 0 ? kLtDescHeap0 : kLtDescHeap1);

    if (mode == kModeCopy)
    {
        // Unchanged: copy the block (the gather rewrites its index entries' bvh_leaf).
        const uint64_t from = source + uint64_t(old_row.x) * kLtUnitBytes;
        const uint64_t to = target + uint64_t(new_offset) * kLtUnitBytes;
        for (uint u = t; u < old_row.y; u += kLtThreads)
        {
            const uint64_t unit = uint64_t(u) * kLtUnitBytes;
            CAELIX_STORE_U32x4(to + unit, CAELIX_LOAD_U32x4(from + unit));
        }
        if (t == 0)
        {
            CAELIX_STORE_U32(row + kLtRowBlockOffset, old_row.y != 0 ? new_offset : 0u);
        }
        return;
    }
    if (mode == kModeReleased)
    {
        if (t == 0)
        {
            // block_offset, block_units, generation, gpu_flags; cpu_flags, bricks, light_faces, pad
            CAELIX_STORE_U32x4(row, uint4(0, 0, old_row.z + 1u, 0));
            CAELIX_STORE_U32x4(row + kLtRowCpuFlags, uint4(0, 0, 0, 0));
        }
        return;
    }

    // Rebuild.
    const LtGroupScan scan = LtScanGroup(lookup, gs_slot_base, gs_high_water, t);
    const LtOldBlock old = LtOpenOldBlock(source, old_row.x, old_row.y, gs_fresh != 0);
    // The plans by rank. The stale bit of every resident brick of the group is cleared.
    uint recomputed = 0;
    uint copied = 0;
    for (uint id = t; id < gs_high_water; id += kLtThreads)
    {
        uint pos;
        uint64_t record;
        const uint word1 = LtBrickWord1(lookup, gs_slot_base, id, pos, record);
        const uint m = LtMorton(BrickOfPos(pos, shift));
        if (LtLightFaceCount(word1) != 0 && LtBitSet(m))
        {
            const LtBrickPlan plan = LtPlanBrick(word1, m, old, shift);
            gs_plan[LtRank(m)] =
                id | (plan.recompute ? kLtPlanRecompute : 0u) | (plan.units << kLtPlanUnitsShift);
            recomputed += plan.recompute ? 1u : 0u;
            copied += plan.recompute ? 0u : 1u;
        }
        if ((word1 & kLtRecordStaleBit) != 0)
        {
            CAELIX_STORE_U32(record + kRecordBoundsWord * kRecordWordBytes, word1 & ~kLtRecordStaleBit);
        }
    }
    InterlockedAdd(gs_recomputed, recomputed);
    InterlockedAdd(gs_copied, copied);
    GroupMemoryBarrierWithGroupSync();
    // Record units before every chunk of kLtRanksPerThread ranks.
    uint chunk = 0;
    for (uint j = 0; j < kLtRanksPerThread; ++j)
    {
        const uint r = t * kLtRanksPerThread + j;
        chunk += r < scan.bricks ? LtPlanUnits(gs_plan[r]) : 0u;
    }
    uint records_units;
    gs_chunk[t] = WorkgroupExclusiveSum(chunk, t, kLtThreads, records_units);
    GroupMemoryBarrierWithGroupSync();

    const uint units = LtBlockUnits(scan.bricks, records_units);
    const uint64_t block = target + uint64_t(new_offset) * kLtUnitBytes;
    const uint records_offset = kLtBlockIndexOffset + scan.bricks * kLtIndexEntryBytes;
    if (scan.bricks != 0)
    {
        if (t == 0)
        {
            CAELIX_STORE_U32x4(block, uint4(scan.bricks, 0u, 0u, 0u));  // LightBlockHeader
        }
        const uint64_t light_codes = LtDescriptorU64(descriptor, kLtDescLightCodes);
        const uint encoding = LtDescriptorU32(descriptor, kLtDescEncoding);
        const uint lanes = WaveGetLaneCount();
        const uint waves = kLtThreads / lanes;
        for (uint r = t / lanes; r < scan.bricks; r += waves)
        {
            LtBuildBrick(lookup, block, records_offset / kLtUnitBytes, r, old, light_codes, encoding);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0)
    {
        // block_offset, block_units, generation, gpu_flags; cpu_flags, bricks, light_faces, pad
        CAELIX_STORE_U32x4(row, uint4(units != 0 ? new_offset : 0u, units, old_row.z + 1u, 0u));
        const uint bricks = scan.bricks | (min(scan.dropped, kLtRowBricksMask) << kLtRowDroppedShift);
        CAELIX_STORE_U32x4(row + kLtRowCpuFlags, uint4(0u, bricks, gs_faces_written, 0u));
        if (gs_recomputed != 0)
        {
            CAELIX_ATOMIC_ADD_U32(control + kLtCtlRecordsRecomputed, gs_recomputed);
        }
        if (gs_copied != 0)
        {
            CAELIX_ATOMIC_ADD_U32(control + kLtCtlRecordsCopied, gs_copied);
        }
    }
}
