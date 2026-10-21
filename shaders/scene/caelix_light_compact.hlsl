// Light table compact kernel (docs/LIGHT_LIST.md section 1.2, entry layout docs/LIGHT_FACES.md):
// turns the light words of every render group into the 448-byte entries of the scene's light table
// (common/caelix_light.hlsl).
//
// One workgroup of 64 threads per group index in [group_base, group_base + dispatch size); thread t
// walks the dense ids t, t + 64, ... below the group's high water. An id whose light word is not 0
// and whose slot row is resident is read from its record: the micro occupancy gives the occupied
// voxels, the payload their material code. Every occupied voxel whose material is a list light adds
// its emission luminance to the total and, for each of its six faces that is open (on the brick
// border, or the neighbour voxel inside the brick is not occupied; a neighbour brick is not
// consulted), to that face's weight; the face masks are the list-light voxels ANDed with the open
// bits of each face (OpenWord), so a face mask and its weight always cover the same voxels. Each
// face also gets its extent along its axis (the lowest row of a +a face mask, the highest of a -a
// one, 0xFF when empty), which the sampler's candidate rule uses as the face plane. An
// entry with a total of 0 (the materials changed since the light word was written) is not stored.
// The entry takes the next index of the header's count (InterlockedAdd); an index at or above the
// capacity counts in `dropped` instead. The entry's total also goes to cdf[index], which
// caelix_light_scan turns into the prefix sum.
//
// The brick -> entry map (docs/LIGHT_MIS.md section 2): every dense id below the high water gets
// light_indices[aabb_row_base + id] = entry index + 1, or 0 when it has no entry (light word 0, not
// resident, a total of 0, dropped beyond the capacity). Writing every id, not only the ones with a
// light word, leaves no stale index of an earlier rebuild behind; ids above the high water are never
// hit by a ray. The path tracer reads the map to find the entry of a list light that a BSDF segment
// hit (render/caelix_light_sample.hlsl, LightListCanSample).
//
// Freed groups have high water 0 in their GroupInfo row (the scene restages it when it releases the
// group), so they contribute nothing. The CPU zeroes the header's first 16 bytes before the dispatch.
//
// Inputs: LightPush (lib/src/scene/gpu_structs.h); the light table is set 0 binding 0 (a storage
// buffer: atomics need a descriptor with this DXC), every other table is read through its address.
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_light.hlsl"

struct LightPush
{
    uint64_t group_infos;
    uint64_t slot_tables;
    uint64_t page_table;
    uint64_t light_words;
    uint64_t light_indices;
    uint64_t materials;
    uint group_base;
    uint group_count;
    uint max_bricks;
    uint records_per_page_shift;
    uint record_bytes;
    uint group_shift;
    uint encoding;
    float min_emission;
};

CAELIX_PUSH_CONSTANTS(LightPush, g_push);
CAELIX_BINDING(0, 0) RWByteAddressBuffer g_table;

uint64_t RecordAddress(uint record_index)
{
    const uint page = record_index >> g_push.records_per_page_shift;
    const uint slot = record_index & ((1u << g_push.records_per_page_shift) - 1u);
    const uint64_t page_address = CAELIX_LOAD_U64(g_push.page_table + uint64_t(page) * 8);
    return page_address + uint64_t(slot) * g_push.record_bytes;
}

// The voxels of slice z whose face `face` (+X 0, -X 1, +Y 2, -Y 3, +Z 4, -Z 5) is open: the face lies
// on the brick border or the neighbour voxel in that direction is not occupied. Slice words hold bit
// x + 8 (y & 3) in the low (y 0..3) or high (y 4..7) half, so a neighbour along x is one bit away
// (masked at the row ends), along y one byte (carried between the halves), along z one slice.
uint2 OpenWord(uint2 occupied[8], uint face, uint z)
{
    const uint2 o = occupied[z];
    uint2 neighbour;
    if (face == 0u)
    {
        neighbour = (o >> 1) & 0x7F7F7F7Fu;  // x + 1; x = 7 has none
    }
    else if (face == 1u)
    {
        neighbour = (o << 1) & 0xFEFEFEFEu;  // x - 1; x = 0 has none
    }
    else if (face == 2u)
    {
        neighbour = uint2((o.x >> 8) | (o.y << 24), o.y >> 8);  // y + 1; y = 7 has none
    }
    else if (face == 3u)
    {
        neighbour = uint2(o.x << 8, (o.y << 8) | (o.x >> 24));  // y - 1; y = 0 has none
    }
    else if (face == 4u)
    {
        neighbour = z < 7u ? occupied[z + 1u] : uint2(0u, 0u);
    }
    else
    {
        neighbour = z > 0u ? occupied[z - 1u] : uint2(0u, 0u);
    }
    return ~neighbour;
}

// Builds and stores the entry of one brick. Returns the entry index + 1, or 0 when the brick got no
// entry (not resident, a total of 0, dropped).
uint CompactBrick(uint group, uint id, int3 coord, uint entity, uint slot_row_base)
{
    const uint4 row = CAELIX_LOAD_U32x4(g_push.slot_tables + (uint64_t(slot_row_base) + id) * kSlotRowBytes);
    if (SlotRowState(row.x) != kSlotStateResident || row.z == kInvalidRecord)
    {
        return 0u;
    }
    const uint64_t record = RecordAddress(row.z);

    // Occupancy: micro words -> slices (voxel v = x | y << 3 | z << 6).
    uint2 occupied[8];
    CAELIX_UNROLL
    for (uint i = 0; i < 8; ++i)
    {
        occupied[i] = uint2(0u, 0u);
    }
    CAELIX_LOOP
    for (uint c = 0; c < 8; ++c)
    {
        const uint2 words = CAELIX_LOAD_U32x2(record + (kRecordInfoWords + 2 * c) * 4);
        CAELIX_LOOP
        for (uint half_index = 0; half_index < 2; ++half_index)
        {
            uint bits = half_index == 0 ? words.x : words.y;
            while (bits != 0u)
            {
                const uint b = firstbitlow(bits);
                bits &= bits - 1u;
                const uint micro = b + 32u * half_index;
                const uint x = (c & 1u) * 4u + (micro & 3u);
                const uint y = ((c >> 1) & 1u) * 4u + ((micro >> 2) & 3u);
                const uint z = (c >> 2) * 4u + (micro >> 4);
                const uint bit = 1u << (x + 8u * (y & 3u));
                if (y < 4u)
                {
                    occupied[z].x |= bit;
                }
                else
                {
                    occupied[z].y |= bit;
                }
            }
        }
    }

    // List lights: the voxel mask (ANDed with the open words when the entry is stored), the face
    // weights and the total.
    uint2 mask[8];
    float face_weight[6];
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        face_weight[f] = 0.0f;
    }
    float total = 0.0f;
    uint last_code = 0xFFFFFFFFu;
    float last_luminance = 0.0f;
    CAELIX_LOOP
    for (uint z = 0; z < 8; ++z)
    {
        mask[z] = uint2(0u, 0u);
        if (all(occupied[z] == 0u))
        {
            continue;
        }
        uint2 open[6];
        CAELIX_UNROLL
        for (uint f2 = 0; f2 < 6; ++f2)
        {
            open[f2] = OpenWord(occupied, f2, z);
        }
        CAELIX_LOOP
        for (uint half_index = 0; half_index < 2; ++half_index)
        {
            uint bits = half_index == 0 ? occupied[z].x : occupied[z].y;
            while (bits != 0u)
            {
                const uint b = firstbitlow(bits);
                bits &= bits - 1u;
                const uint x = b & 7u;
                const uint y = (b >> 3) + 4u * half_index;
                const uint word = CAELIX_LOAD_U32(record + uint64_t(RecordPayloadWord(x, y, z)) * 4);
                const uint value = (x & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu);
                const uint code = CaelixMaterialCode(value);
                if (code != last_code)
                {
                    last_code = code;
                    last_luminance = MaterialEmissionLuminance(code);
                }
                if (!(last_luminance >= g_push.min_emission))
                {
                    continue;
                }
                if (half_index == 0)
                {
                    mask[z].x |= 1u << b;
                }
                else
                {
                    mask[z].y |= 1u << b;
                }
                // The total takes every list-light voxel (the light word's rule). A voxel with no
                // open face cannot exist after the finalize kernel's interior removal; if one did,
                // it would only weigh the CDF (it is in no face mask).
                total += last_luminance;
                CAELIX_UNROLL
                for (uint f3 = 0; f3 < 6; ++f3)
                {
                    const uint open_half = half_index == 0 ? open[f3].x : open[f3].y;
                    if (((open_half >> b) & 1u) != 0u)
                    {
                        face_weight[f3] += last_luminance;
                    }
                }
            }
        }
    }
    if (!(total > 0.0f))
    {
        return 0u;
    }

    uint index;
    g_table.InterlockedAdd(kLightHeaderCount, 1u, index);
    if (index >= g_push.max_bricks)
    {
        uint ignored;
        g_table.InterlockedAdd(kLightHeaderDropped, 1u, ignored);
        return 0u;
    }
    const uint shift = g_push.group_shift;
    const uint pos_mask = (1u << shift) - 1u;
    const uint pos = SlotRowPos(row.x);
    const int3 brick = int3(pos & pos_mask, (pos >> shift) & pos_mask, (pos >> (2 * shift)) & pos_mask);
    const int3 origin = (coord * int(1u << shift) + brick) * 8;

    // 24 x 16 B of face masks (face-major, two slices per store), then the 64 bytes of the header
    // part: the six weights, total, group, id, entity, origin, the two extent words and the pad.
    // The extent of a face (docs/LIGHT_FACES.md section 1): the rows along the face's axis that hold
    // a voxel of its mask as 8 bits (x: the bytes of the slice words folded with OR; y: one bit per
    // nonzero byte, low half rows 0..3, high half 4..7; z: one bit per nonzero slice), then the
    // lowest row for a +a face, the highest for a -a face, 0xFF when the mask is empty.
    const uint address = kLightHeaderBytes + index * kLightEntryBytes;
    uint extent0 = 0u;
    uint extent1 = 0u;
    CAELIX_LOOP
    for (uint f4 = 0; f4 < 6; ++f4)
    {
        uint2 face_bits = uint2(0u, 0u);
        uint z_rows = 0u;
        CAELIX_UNROLL
        for (uint w = 0; w < 4; ++w)
        {
            const uint2 lo = mask[2 * w] & OpenWord(occupied, f4, 2 * w);
            const uint2 hi = mask[2 * w + 1] & OpenWord(occupied, f4, 2 * w + 1);
            g_table.Store4(address + f4 * kLightEntryFaceMaskBytes + w * 16, uint4(lo, hi));
            face_bits |= lo | hi;
            z_rows |= any(lo != 0u) ? 1u << (2 * w) : 0u;
            z_rows |= any(hi != 0u) ? 2u << (2 * w) : 0u;
        }
        uint rows;
        if (f4 < 2u)
        {
            const uint x_word = face_bits.x | face_bits.y;
            rows = (x_word | (x_word >> 8) | (x_word >> 16) | (x_word >> 24)) & 0xFFu;
        }
        else if (f4 < 4u)
        {
            rows = 0u;
            CAELIX_UNROLL
            for (uint r = 0; r < 4; ++r)
            {
                rows |= ((face_bits.x >> (8u * r)) & 0xFFu) != 0u ? 1u << r : 0u;
                rows |= ((face_bits.y >> (8u * r)) & 0xFFu) != 0u ? 16u << r : 0u;
            }
        }
        else
        {
            rows = z_rows;
        }
        const uint row = rows == 0u ? 0xFFu : ((f4 & 1u) == 0u ? firstbitlow(rows) : firstbithigh(rows));
        if (f4 < 4u)
        {
            extent0 |= row << (8u * f4);
        }
        else
        {
            extent1 |= row << (8u * (f4 - 4u));
        }
    }
    g_table.Store4(address + kLightEntryWeightsOffset,
                   asuint(float4(face_weight[0], face_weight[1], face_weight[2], face_weight[3])));
    g_table.Store4(address + kLightEntryWeightsOffset + 16,
                   uint4(asuint(face_weight[4]), asuint(face_weight[5]), asuint(total), group));
    g_table.Store4(address + kLightEntryTotalOffset + 8, uint4(id, entity, asuint(origin.x), asuint(origin.y)));
    g_table.Store4(address + kLightEntryOriginOffset + 8, uint4(asuint(origin.z), extent0, extent1, 0u));
    g_table.Store(kLightHeaderBytes + g_push.max_bricks * kLightEntryBytes + index * 4, asuint(total));
    return index + 1u;
}

[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID)
{
    const uint group = g_push.group_base + group_id.x;
    if (group >= g_push.group_count)
    {
        return;
    }
    g_blockEncoding = g_push.encoding;
    g_materialTable = g_push.materials;
    // GroupInfo: coord, entity, slot_row_base, reverse_base, aabb_row_base, high_water.
    const uint4 w0 = CAELIX_LOAD_U32x4(g_push.group_infos + uint64_t(group) * 32);
    const uint4 w1 = CAELIX_LOAD_U32x4(g_push.group_infos + uint64_t(group) * 32 + 16);
    const uint high_water = w1.w;
    CAELIX_LOOP
    for (uint id = thread_id.x; id < high_water; id += 64u)
    {
        const uint word = CAELIX_LOAD_U32(g_push.light_words + (uint64_t(w1.z) + id) * 4);
        const uint entry = word != 0u ? CompactBrick(group, id, int3(w0.xyz), w0.w, w1.x) : 0u;
        CAELIX_STORE_U32(g_push.light_indices + (uint64_t(w1.z) + id) * 4, entry);
    }
}
