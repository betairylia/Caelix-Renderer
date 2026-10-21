// Light table compact kernel (docs/LIGHT_LIST.md, section 1.2): turns the light words of every
// render group into LightEntry rows of the scene's light table (common/caelix_light.hlsl).
//
// One workgroup of 64 threads per group index in [group_base, group_base + dispatch size); thread t
// walks the dense ids t, t + 64, ... below the group's high water. An id whose light word is not 0
// and whose slot row is resident is read from its record: the micro occupancy gives the occupied
// voxels, the payload their material code, and every occupied voxel whose material is a list light
// sets its mask bit, adds its emission luminance to its slice's region weight and the total, and adds
// the unit directions of its open faces (a face on the brick border, or whose neighbour in the brick
// is not occupied) to the normal sum. An entry with a total of 0 (the materials changed since the
// light word was written) is not stored. The entry takes the next index of the header's count
// (InterlockedAdd); an index at or above the capacity counts in `dropped` instead. The entry's
// total also goes to cdf[index], which caelix_light_scan turns into the prefix sum.
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

// Whether voxel (x, y, z) (each 0..7) is set in the per-slice masks.
bool MaskBit(uint2 slices[8], int x, int y, int z)
{
    const uint2 s = slices[z];
    return ((y < 4 ? s.x : s.y) >> (uint(x) + 8u * (uint(y) & 3u)) & 1u) != 0u;
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

    // List lights.
    uint2 mask[8];
    float region[8];
    float total = 0.0f;
    float3 normal_sum = float3(0.0f, 0.0f, 0.0f);
    uint last_code = 0xFFFFFFFFu;
    float last_luminance = 0.0f;
    CAELIX_LOOP
    for (uint z = 0; z < 8; ++z)
    {
        mask[z] = uint2(0u, 0u);
        region[z] = 0.0f;
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
                region[z] += last_luminance;
                total += last_luminance;
                const int3 p = int3(x, y, z);
                const float3 open = float3(
                    (p.x == 7 || !MaskBit(occupied, p.x + 1, p.y, p.z) ? 1.0f : 0.0f) -
                        (p.x == 0 || !MaskBit(occupied, p.x - 1, p.y, p.z) ? 1.0f : 0.0f),
                    (p.y == 7 || !MaskBit(occupied, p.x, p.y + 1, p.z) ? 1.0f : 0.0f) -
                        (p.y == 0 || !MaskBit(occupied, p.x, p.y - 1, p.z) ? 1.0f : 0.0f),
                    (p.z == 7 || !MaskBit(occupied, p.x, p.y, p.z + 1) ? 1.0f : 0.0f) -
                        (p.z == 0 || !MaskBit(occupied, p.x, p.y, p.z - 1) ? 1.0f : 0.0f));
                normal_sum += open;
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
    const float normal_length = length(normal_sum);
    const uint mean_normal = normal_length > 1e-6f ? EncodeOctahedral(normal_sum / normal_length) : 0u;

    const uint address = kLightHeaderBytes + index * kLightEntryBytes;
    CAELIX_UNROLL
    for (uint w = 0; w < 4; ++w)
    {
        g_table.Store4(address + w * 16, uint4(mask[2 * w], mask[2 * w + 1]));
    }
    g_table.Store4(address + 64, asuint(float4(region[0], region[1], region[2], region[3])));
    g_table.Store4(address + 80, asuint(float4(region[4], region[5], region[6], region[7])));
    g_table.Store4(address + 96, uint4(asuint(total), group, id, entity));
    g_table.Store4(address + 112, uint4(asuint(origin), mean_normal));
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
