// Record lookups of the scene kernels (finalize, light tree build): record addresses through the
// page table, and the record of the brick next to a brick, also across group borders of the same
// entity (through the entity's group grid). Every table comes in as a parameter.
#ifndef CAELIX_SCENE_LOOKUP_HLSL
#define CAELIX_SCENE_LOOKUP_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"

struct SceneLookup
{
    uint64_t page_table;      // uint64_t[page_count]: record page device addresses
    uint64_t slot_tables;     // slot row pool base
    uint64_t reverse_tables;  // u16 pool base: position -> id, 0xFFFF none
    uint64_t group_infos;     // GroupInfo[group_capacity]
    uint64_t entity_infos;    // EntityInfo[entity_capacity]
    uint64_t group_grids;     // u32 pool base: group index or 0xFFFFFFFF
    uint records_per_page_shift;
    uint record_bytes;
    uint group_shift;
};

// The brick whose neighbours are looked up: its group and group-local position.
struct SceneBrickPlace
{
    int3 coord;  // group coordinate
    uint entity;
    uint slot_base;
    uint reverse_base;
    uint pos;  // group-local brick position
};

uint64_t SceneRecordAddress(SceneLookup s, uint record_index)
{
    const uint page = record_index >> s.records_per_page_shift;
    const uint slot = record_index & ((1u << s.records_per_page_shift) - 1u);
    const uint64_t page_address = CAELIX_LOAD_U64(s.page_table + uint64_t(page) * kPageTableEntryBytes);
    return page_address + uint64_t(slot) * s.record_bytes;
}

// GroupInfo slot_row_base and reverse_base.
uint2 SceneGroupBases(SceneLookup s, uint group_index)
{
    const uint64_t info = s.group_infos + uint64_t(group_index) * kGroupInfoBytes;
    const uint4 w1 = CAELIX_LOAD_U32x4(info + kGroupInfoSlotRowBase);
    return w1.xy;
}

// The record address of the brick next to `place` in direction d (+X,-X,+Y,-Y,+Z,-Z), or 0 when
// there is none (absent bricks are air).
uint64_t SceneResolveNeighbor(SceneLookup s, SceneBrickPlace place, uint d)
{
    const uint shift = s.group_shift;
    const int mask = int((1u << shift) - 1u);
    // BrickOfPos in signed arithmetic (the neighbour may be outside the group).
    const int3 own = int3(int(place.pos) & mask, int(place.pos >> shift) & mask,
                          int(place.pos >> (2 * shift)) & mask);
    const int3 delta = FaceDirection(d);
    int3 n = own + delta;
    uint slot_base = place.slot_base;
    uint reverse_base = place.reverse_base;
    CAELIX_BRANCH
    if (any(n < 0) || any(n > mask))
    {
        // Another group of the same entity: look it up in the entity's group grid.
        const int3 coord = place.coord + delta;
        const uint64_t entity_address = s.entity_infos + uint64_t(place.entity) * kEntityInfoBytes;
        const uint4 e0 = CAELIX_LOAD_U32x4(entity_address);  // grid_min, grid_base (kEntityInfoGridMin = 0)
        const uint4 e1 = CAELIX_LOAD_U32x4(entity_address + kEntityInfoGridDims);
        const int3 rel = coord - int3(e0.xyz);
        const int3 dims = int3(e1.xyz);
        if (any(rel < 0) || any(rel >= dims))
        {
            return 0;
        }
        const uint cell = uint(rel.x) + uint(dims.x) * (uint(rel.y) + uint(dims.y) * uint(rel.z));
        const uint group = CAELIX_LOAD_U32(s.group_grids + (uint64_t(e0.w) + cell) * 4);
        if (group == kNoGroup)
        {
            return 0;
        }
        const uint2 bases = SceneGroupBases(s, group);
        slot_base = bases.x;
        reverse_base = bases.y;
        n &= mask;
    }
    const uint npos = uint(n.x) | (uint(n.y) << shift) | (uint(n.z) << (2 * shift));
    const uint64_t entry = uint64_t(reverse_base) + npos;
    const uint pair = CAELIX_LOAD_U32(s.reverse_tables + (entry >> 1) * 4);
    const uint id = (entry & 1) != 0 ? (pair >> 16) : (pair & 0xFFFFu);
    if (id == kNoBrickId)
    {
        return 0;
    }
    const uint4 row = CAELIX_LOAD_U32x4(s.slot_tables + (uint64_t(slot_base) + id) * kSlotRowBytes);
    if (SlotRowState(row.x) != kSlotStateResident || row.z == kInvalidRecord)
    {
        return 0;
    }
    return SceneRecordAddress(s, row.z);
}

#endif  // CAELIX_SCENE_LOOKUP_HLSL
