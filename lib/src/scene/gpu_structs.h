#pragma once
#include <cstdint>

namespace caelix {

// CPU mirrors of the tables that shaders/scene/caelix_finalize.hlsl reads. The
// HLSL side is shaders/common/caelix_record.hlsl; change both together.

// One brick to finalize: one workgroup each. 16 bytes.
struct DirtyEntry
{
    uint32_t group_index;
    uint32_t id;
    uint32_t record_index;
    uint32_t pad;
};
static_assert(sizeof(DirtyEntry) == 16);

// One render group, indexed by the scene-wide group index. 32 bytes.
struct GroupInfo
{
    int32_t coord[3];  // group coordinate = brick key >> group_shift
    uint32_t entity;
    uint32_t slot_row_base;  // first slot table row
    uint32_t reverse_base;   // first u16 entry of the reverse table
    uint32_t aabb_row_base;  // first AABB row
    uint32_t high_water;     // dense ids in [0, high_water) were handed out
};
static_assert(sizeof(GroupInfo) == 32);

// One entity, indexed by the entity id. 128 bytes. The finalize kernel reads the
// first 32 bytes; the transform rows are for the trace kernels.
struct EntityInfo
{
    int32_t grid_min[3];   // group coordinate of grid cell (0, 0, 0)
    uint32_t grid_base;    // first u32 cell of the group grid
    int32_t grid_dims[3];  // cells per axis, powers of two; 0 = no grid
    uint32_t pad;
    // Object-to-world rows (3x4, row-major: rotation | position) of the current
    // transform, then of the previous frame's transform. prev differs from cur for
    // exactly one flush after a transform change.
    float cur_rows[3][4];
    float prev_rows[3][4];
};
static_assert(sizeof(EntityInfo) == 128);

// Push constants of the finalize kernel. 104 bytes.
struct FinalizePush
{
    uint64_t dirty_list;
    uint64_t page_table;
    uint64_t slot_tables;
    uint64_t reverse_tables;
    uint64_t group_infos;
    uint64_t entity_infos;
    uint64_t group_grids;
    uint64_t aabb_pool;
    uint64_t light_words;  // u32 per AABB row: the light word (docs/LIGHT_LIST.md)
    uint64_t materials;    // the material table (CaelixMaterial by block code)
    uint32_t dirty_count;
    uint32_t records_per_page_shift;
    uint32_t record_bytes;
    uint32_t group_shift;
    uint32_t encoding;
    float min_emission;  // list light: luminance(emission) >= this
};
static_assert(sizeof(FinalizePush) == 104);
static_assert(sizeof(FinalizePush) <= 128);  // the guaranteed minimum maxPushConstantsSize

// The scene's light table (docs/LIGHT_LIST.md, shaders/common/caelix_light.hlsl): a 64-byte
// header, max_bricks LightEntry rows, then max_bricks floats of the CDF.
struct LightTableHeader
{
    uint32_t count;     // entries (the scan clamps it to max_bricks)
    uint32_t dropped;   // bricks with list lights that did not fit
    uint32_t capacity;  // max_bricks: the CDF starts at 64 + capacity * 128 (written by the scan)
    uint32_t pad;
    float total_weight;  // cdf[count - 1]; 0 when count is 0
    float pad2[11];
};
static_assert(sizeof(LightTableHeader) == 64);

// One brick with list lights. 128 bytes. Voxel v = x | y << 3 | z << 6; word w of the mask is the
// z = w slice.
struct LightEntry
{
    uint64_t mask[8];          // bit v of word (v >> 6): voxel v is alive and a list light
    float region_weight[8];    // luminance sum of the set bits of word w
    float total_weight;        // sum of region_weight
    uint32_t group;            // scene-wide group index
    uint32_t id;               // dense brick id
    uint32_t entity;
    int32_t origin[3];         // entity-local voxel origin of the brick (brick key * 8)
    uint32_t mean_normal;      // octahedral 2 x 16-bit snorm of the mean open-face direction, 0 = none
};
static_assert(sizeof(LightEntry) == 128);

constexpr uint64_t kLightTableHeaderBytes = sizeof(LightTableHeader);
// Bytes of a light table of `max_bricks` entries.
constexpr uint64_t LightTableBytes(uint32_t max_bricks)
{
    return kLightTableHeaderBytes + static_cast<uint64_t>(max_bricks) * (sizeof(LightEntry) + sizeof(float));
}

// Push constants of the light table kernels (caelix_light_compact, caelix_light_scan). 80 bytes.
// The table itself is set 0 binding 0 of the kernels (a storage buffer: the compact kernel counts
// with atomics, which need a descriptor in this DXC version); the rest is read through addresses.
struct LightPush
{
    uint64_t group_infos;
    uint64_t slot_tables;
    uint64_t page_table;
    uint64_t light_words;
    uint64_t light_indices;  // u32 per AABB row: entry index + 1, 0 = none (written by the compact kernel)
    uint64_t materials;
    uint32_t group_base;   // the first group index of this dispatch (dispatches split at 65535)
    uint32_t group_count;  // groups_.size(): group indices [0, group_count)
    uint32_t max_bricks;
    uint32_t records_per_page_shift;
    uint32_t record_bytes;
    uint32_t group_shift;
    uint32_t encoding;
    float min_emission;
};
static_assert(sizeof(LightPush) == 80);

// One AABB row: float3 min, float3 max (entity-local block coordinates). All NaN
// = inactive primitive.
constexpr uint32_t kAabbRowBytes = 24;
constexpr uint32_t kQuietNaNBits = 0x7FC00000u;
constexpr uint16_t kNoBrickId = 0xFFFF;
constexpr uint32_t kNoGroup = 0xFFFFFFFFu;
// Workgroups per vkCmdDispatch (maxComputeWorkGroupCount[0] is at least 65535).
constexpr uint32_t kMaxDispatchGroups = 65535;

}  // namespace caelix
