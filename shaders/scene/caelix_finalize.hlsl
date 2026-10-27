// Finalize kernel: turns the raw record of a brick into a renderer record.
//
// One workgroup per dirty brick, 64 threads, thread t owns the voxel row
// y = t & 7, z = t >> 3 (8 voxels along x). For each brick it computes:
//   - per voxel: alive = not all six neighbours opaque. Opaque voxels keep their
//     raw value. Transparent voxels get face bits: bit d (+X,-X,+Y,-Y,+Z,-Z) is
//     set when neighbour d is not opaque and has a different id. The payload is
//     rewritten (only the words that change) with these face bits.
//   - occupied = alive && payload value != 0, packed into micro occupancy words,
//     the coarse byte and the tight bounds of the header,
//   - the brick's AABB row (entity-local block coordinates; NaN when empty).
//
// Lossless rule: raw(v) = opaque(v) ? v : (v & id_mask) recovers the uploaded
// value from a finalized one (uploaded transparent values carry no face bits),
// so the kernel can run again on finalized records and neighbours and gives the
// same result. Dead voxels keep their (re-encoded) value too.
//
// Neighbour records are read while other workgroups of the same dispatch may
// rewrite their payload words. That is benign: only face bits of transparent
// voxels change, and raw() masks them off, so every read gives the same raw value.
//
// Light tree (push light_codes != 0): a light voxel (its material code is in the light code set)
// counts its open faces (neighbour not opaque, the same neighbour values as above). The brick's total
// goes into record word 1 at kLtRecordFaceCountShift (the tight bounds use bits 0..17). When the old
// count (word 1 as the kernel found it: an upload that replaces a resident brick keeps word 1) or the
// new count is non-zero, kLtRecordStaleBit and kLtGpuLightChanged in the group's
// GroupLightRow::gpu_flags are set: the light tree's group pass recomputes the brick's light record
// (and clears the stale bit) when it rebuilds the group's block. A set stale bit is kept.
// light_codes == 0 skips all of it (word 1 bits 18..30 = 0).
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_light_tree_layout.hlsl"
#include "scene/caelix_scene_lookup.hlsl"

struct FinalizePush
{
    uint64_t dirty_list;      // DirtyEntry[dirty_count]
    uint64_t page_table;      // uint64_t[page_count]: record page device addresses
    uint64_t slot_tables;     // slot row pool base
    uint64_t reverse_tables;  // u16 pool base: position -> id, 0xFFFF none
    uint64_t group_infos;     // GroupInfo[group_capacity]
    uint64_t entity_infos;    // EntityInfo[entity_capacity]
    uint64_t group_grids;     // u32 pool base: group index or 0xFFFFFFFF
    uint64_t aabb_pool;       // AABB row pool base
    uint dirty_count;
    uint records_per_page_shift;
    uint record_bytes;
    uint group_shift;
    uint encoding;  // kEncodingPacked or kEncodingBlockId
    uint pad;
    uint64_t light_codes;  // 65536-bit light code set; 0 = no light tree
    uint64_t light_group_rows;  // GroupLightRow by group index (light tree only)
};

CAELIX_PUSH_CONSTANTS(FinalizePush, g_push);

groupshared uint64_t gs_record;  // own record address
groupshared uint64_t gs_nbr[6];  // neighbour record addresses (+X,-X,+Y,-Y,+Z,-Z), 0 = absent
groupshared int3 gs_coord;       // own group coordinate
groupshared uint gs_entity;
groupshared uint gs_slot_base;
groupshared uint gs_reverse_base;
groupshared uint gs_aabb_base;
groupshared uint gs_id;
groupshared uint gs_group;         // own group index
groupshared uint gs_pos;           // group-local brick position
groupshared uint gs_old_word1;     // record word 1 before this kernel
groupshared uint gs_light_faces;   // open faces of light voxels in the brick
groupshared uint gs_payload[256];  // own payload as uploaded/last finalized
groupshared uint gs_occ[16];       // micro occupancy words
groupshared int gs_min[3];
groupshared int gs_max[3];

SceneLookup Lookup()
{
    SceneLookup s;
    s.page_table = g_push.page_table;
    s.slot_tables = g_push.slot_tables;
    s.reverse_tables = g_push.reverse_tables;
    s.group_infos = g_push.group_infos;
    s.entity_infos = g_push.entity_infos;
    s.group_grids = g_push.group_grids;
    s.records_per_page_shift = g_push.records_per_page_shift;
    s.record_bytes = g_push.record_bytes;
    s.group_shift = g_push.group_shift;
    return s;
}

// The 4 payload words of row (y, z) of a record (two 8-byte loads: records are 8-byte aligned);
// record 0 = absent = air.
uint4 LoadRow(uint64_t record, uint y, uint z)
{
    if (record == 0)
    {
        return uint4(0, 0, 0, 0);
    }
    const uint64_t address = record + uint64_t(RecordPayloadWord(0, y, z)) * kRecordWordBytes;
    return uint4(CAELIX_LOAD_U32x2(address), CAELIX_LOAD_U32x2(address + 8));
}

uint4 SharedRow(uint y, uint z)
{
    const uint base = 4 * y + 32 * z;
    return uint4(gs_payload[base], gs_payload[base + 1], gs_payload[base + 2], gs_payload[base + 3]);
}

uint BlockOf(uint4 row, uint x)
{
    return PayloadPairBlock(row[x >> 1], x);
}

[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID)
{
    if (group_id.x >= g_push.dirty_count)
    {
        return;  // uniform for the whole workgroup
    }
    const uint t = thread_id.x;
    const uint y = t & 7;
    const uint z = t >> 3;

    const bool packed = g_push.encoding == kEncodingPacked;
    const uint opaque_mask = packed ? kPackedOpaqueMask : kBlockIdOpaqueMask;
    const uint id_mask = packed ? kPackedIdMask : kBlockIdIdMask;
    const uint face_shift = packed ? kPackedFaceShift : kBlockIdFaceShift;

    // 1. Thread 0 resolves the dirty entry; threads 0..15 and 0..2 reset the accumulators.
    if (t == 0)
    {
        /*
            uint4 entry: Dirty entry struct
                .x  Render group index
                .y  Dense brick ID (in group) to fill, pre-assigned by CPU
                .z  Record pool (brick voxel data) index, essentially a shortcut?
                .w  Reserved
        */
        const uint4 entry = CAELIX_LOAD_U32x4(g_push.dirty_list + uint64_t(group_id.x) * kDirtyEntryBytes);
        const uint64_t info = g_push.group_infos + uint64_t(entry.x) * kGroupInfoBytes;
        const uint4 w0 = CAELIX_LOAD_U32x4(info);  // coord, entity (kGroupInfoCoord = 0)
        const uint4 w1 = CAELIX_LOAD_U32x4(info + kGroupInfoSlotRowBase);
        gs_coord = int3(w0.xyz);
        gs_entity = w0.w;
        gs_slot_base = w1.x;
        gs_reverse_base = w1.y;
        gs_aabb_base = w1.z;
        gs_id = entry.y;
        const uint pos_state =
            CAELIX_LOAD_U32(g_push.slot_tables + (uint64_t(w1.x) + entry.y) * kSlotRowBytes);
        gs_pos = SlotRowPos(pos_state);
        gs_group = entry.x;
        gs_record = SceneRecordAddress(Lookup(), entry.z);
        gs_old_word1 = CAELIX_LOAD_U32(gs_record + kRecordBoundsWord * kRecordWordBytes);
        gs_light_faces = 0;
    }
    if (t < 16)
    {
        gs_occ[t] = 0;
    }
    if (t < 3)
    {
        gs_min[t] = 8;
        gs_max[t] = -1;
    }
    GroupMemoryBarrierWithGroupSync();

    // 2. Threads 0..5 resolve the six neighbour records; every thread loads its row.
    if (t < 6)
    {
        SceneBrickPlace place;
        place.coord = gs_coord;
        place.entity = gs_entity;
        place.slot_base = gs_slot_base;
        place.reverse_base = gs_reverse_base;
        place.pos = gs_pos;
        gs_nbr[t] = SceneResolveNeighbor(Lookup(), place, t);
    }
    const uint64_t row_address = gs_record + uint64_t(RecordPayloadWord(0, y, z)) * kRecordWordBytes;
    const uint4 own = uint4(CAELIX_LOAD_U32x2(row_address), CAELIX_LOAD_U32x2(row_address + 8));
    {
        const uint base = 4 * y + 32 * z;
        gs_payload[base] = own.x;
        gs_payload[base + 1] = own.y;
        gs_payload[base + 2] = own.z;
        gs_payload[base + 3] = own.w;
    }
    GroupMemoryBarrierWithGroupSync();

    // 3. Neighbour rows: +Y, -Y, +Z, -Z, and the single blocks at +X / -X.
    const uint4 row_yp = y < 7 ? SharedRow(y + 1, z) : LoadRow(gs_nbr[2], 0, z);
    const uint4 row_yn = y > 0 ? SharedRow(y - 1, z) : LoadRow(gs_nbr[3], 7, z);
    const uint4 row_zp = z < 7 ? SharedRow(y, z + 1) : LoadRow(gs_nbr[4], y, 0);
    const uint4 row_zn = z > 0 ? SharedRow(y, z - 1) : LoadRow(gs_nbr[5], y, 7);
    const uint64_t xp = gs_nbr[0];
    const uint64_t xn = gs_nbr[1];
    const uint block_xp =
        xp != 0 ? PayloadPairBlock(
                      CAELIX_LOAD_U32(xp + uint64_t(RecordPayloadWord(0, y, z)) * kRecordWordBytes), 0)
                : 0u;
    const uint block_xn =
        xn != 0 ? PayloadPairBlock(
                      CAELIX_LOAD_U32(xn + uint64_t(RecordPayloadWord(7, y, z)) * kRecordWordBytes), 7)
                : 0u;

    // 4. Per voxel.
    uint out_words[4] = {0u, 0u, 0u, 0u};
    uint occupied = 0;  // bit x
    const bool lights = g_push.light_codes != 0;
    uint light_faces = 0;
    CAELIX_UNROLL
    for (uint x = 0; x < 8; ++x)
    {
        uint v = BlockOf(own, x);
        v = (v & opaque_mask) != 0 ? v : (v & id_mask);
        uint n[6];
        n[0] = x < 7 ? BlockOf(own, x + 1) : block_xp;
        n[1] = x > 0 ? BlockOf(own, x - 1) : block_xn;
        n[2] = BlockOf(row_yp, x);
        n[3] = BlockOf(row_yn, x);
        n[4] = BlockOf(row_zp, x);
        n[5] = BlockOf(row_zn, x);
        uint opaque_count = 0;
        uint faces = 0;
        CAELIX_UNROLL
        for (uint d = 0; d < 6; ++d)
        {
            const bool neighbour_opaque = (n[d] & opaque_mask) != 0;
            opaque_count += neighbour_opaque ? 1u : 0u;
            if (!neighbour_opaque && (n[d] & id_mask) != v)
            {
                faces |= 1u << d;
            }
        }
        CAELIX_BRANCH
        if (lights)
        {
            // v is the material code (opaque: the value, transparent: the id).
            const uint code_word = CAELIX_LOAD_U32(g_push.light_codes + uint64_t(v >> 5) * 4);
            if (((code_word >> (v & 31)) & 1u) != 0)
            {
                light_faces += 6 - opaque_count;
            }
        }
        const bool alive = opaque_count < 6;
        const uint result = (v & opaque_mask) != 0 ? v : (v | (faces << face_shift));

        const bool has_interface = (v & opaque_mask) != 0 || faces != 0;
        if (alive && result != 0 && has_interface)
        {
            occupied |= 1u << x;
        }
        out_words[x >> 1] |= (x & 1) != 0 ? (result << kPayloadOddShift) : result;
    }

    // Write back the payload words that changed.
    CAELIX_UNROLL
    for (uint w = 0; w < 4; ++w)
    {
        if (out_words[w] != own[w])
        {
            CAELIX_STORE_U32(row_address + w * kRecordWordBytes, out_words[w]);
        }
    }

    if (lights)
    {
        const uint wave_faces = WaveActiveSum(light_faces);
        if (WaveIsFirstLane() && wave_faces != 0)
        {
            InterlockedAdd(gs_light_faces, wave_faces);
        }
    }

    // 5. Occupancy and bounds. Voxels x 0..3 and 4..7 share one micro word each.
    if (occupied != 0)
    {
        const uint micro_base = ((y & 3) << 2) | ((z & 3) << 4);
        const uint coarse_yz = ((y >> 2) << 1) | ((z >> 2) << 2);
        const uint word = micro_base >> 5;
        const uint bit = micro_base & 31;
        const uint low = occupied & 0xFu;
        const uint high = occupied >> 4;
        if (low != 0)
        {
            InterlockedOr(gs_occ[coarse_yz * 2 + word], low << bit);
        }
        if (high != 0)
        {
            InterlockedOr(gs_occ[(coarse_yz | 1u) * 2 + word], high << bit);
        }
        InterlockedMin(gs_min[0], int(firstbitlow(occupied)));
        InterlockedMax(gs_max[0], int(firstbithigh(occupied)));
        InterlockedMin(gs_min[1], int(y));
        InterlockedMax(gs_max[1], int(y));
        InterlockedMin(gs_min[2], int(z));
        InterlockedMax(gs_max[2], int(z));
    }
    GroupMemoryBarrierWithGroupSync();

    // 6. Header words and the AABB row.
    if (t < 16)
    {
        CAELIX_STORE_U32(gs_record + (kRecordInfoWords + t) * kRecordWordBytes, gs_occ[t]);
    }
    if (t == 0)
    {
        uint coarse = 0;
        CAELIX_UNROLL
        for (uint c = 0; c < 8; ++c)
        {
            if ((gs_occ[2 * c] | gs_occ[2 * c + 1]) != 0)
            {
                coarse |= 1u << c;
            }
        }
        const bool empty = gs_max[0] < 0;
        uint bounds = 0;
        if (!empty)
        {
            const uint b = kRecordBoundsAxisBits;
            const uint m = kRecordBoundsMaxShift;
            bounds = uint(gs_min[0]) | (uint(gs_min[1]) << b) | (uint(gs_min[2]) << (2 * b)) |
                     (uint(gs_max[0]) << m) | (uint(gs_max[1]) << (m + b)) | (uint(gs_max[2]) << (m + 2 * b));
        }
        CAELIX_STORE_U32(gs_record, (gs_pos & kRecordPosMask) | (coarse << kRecordCoarseShift));
        CAELIX_BRANCH
        if (lights)
        {
            // The light face count (at most 512 * 6 = 3072) and the stale bit (set until a group
            // pass recomputes the brick's light record).
            const uint old_count = LtLightFaceCount(gs_old_word1);
            const uint new_count = gs_light_faces;
            bounds |= (new_count << kLtRecordFaceCountShift) | (gs_old_word1 & kLtRecordStaleBit);
            if (old_count != 0 || new_count != 0)
            {
                bounds |= kLtRecordStaleBit;
                const uint64_t row = g_push.light_group_rows + uint64_t(gs_group) * kLtGroupRowBytes;
                CAELIX_ATOMIC_OR_U32(row + kLtRowGpuFlags, kLtGpuLightChanged);
            }
        }
        CAELIX_STORE_U32(gs_record + kRecordBoundsWord * kRecordWordBytes, bounds);

        const uint shift = g_push.group_shift;
        // BrickOfPos(gs_pos, shift), written out on the groupshared word (keeps the kernel's code).
        const uint mask = (1u << shift) - 1u;
        const int3 brick = int3(gs_pos & mask, (gs_pos >> shift) & mask, (gs_pos >> (2 * shift)) & mask);
        const int3 origin = (gs_coord * int(1u << shift) + brick) * int(kBrickEdge);
        uint3 lo = uint3(kQuietNaN, kQuietNaN, kQuietNaN);
        uint3 hi = uint3(kQuietNaN, kQuietNaN, kQuietNaN);
        if (!empty)
        {
            lo = asuint(float3(origin + int3(gs_min[0], gs_min[1], gs_min[2])));
            hi = asuint(float3(origin + int3(gs_max[0], gs_max[1], gs_max[2]) + 1));
        }
        // The 24-byte row (float3 min, float3 max) as three 8-byte stores.
        const uint64_t aabb = g_push.aabb_pool + (uint64_t(gs_aabb_base) + gs_id) * kAabbRowBytes;
        CAELIX_STORE_U32x2(aabb, lo.xy);
        CAELIX_STORE_U32x2(aabb + 8, uint2(lo.z, hi.x));
        CAELIX_STORE_U32x2(aabb + 16, hi.yz);
    }
}
