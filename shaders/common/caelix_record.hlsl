// Shared GPU data layouts. Mirrors lib/src/scene/record_layout.h and lib/src/scene/gpu_structs.h,
// which hold the same constants by name next to the structs and their static_asserts
// (caelix_frame_layout_test compares the files); change both sides together.
//
// The named field offsets and bit fields are macros, the older constants static consts: DXC emits
// the constant of a static const where it is declared, so a new static const in this header
// (included by every scene and render kernel) would reorder the constants of kernels that never use
// it. A macro expands to the literal at its use and keeps their SPIR-V byte-identical.
#ifndef CAELIX_RECORD_HLSL
#define CAELIX_RECORD_HLSL

#include "common/caelix_platform.hlsl"

// ---------------------------------------------------------------------------
// Brick record (16-bit blocks): 274 words = 1096 bytes, 8-byte aligned.
//   word 0       bits 0..15  group-local brick position
//                bits 16..23 coarse occupancy (2x2x2 bits, bit i = coarse cell
//                            (x>>2) | (y>>2)<<1 | (z>>2)<<2 has an occupied voxel)
//   word 1       bits 0..17: tight bounds of the occupied voxels, inclusive:
//                minX | minY<<3 | minZ<<6 | maxX<<9 | maxY<<12 | maxZ<<15 (0 when empty);
//                bits 18..29: light face count, bit 30: light record stale (light tree only,
//                common/caelix_light_tree_layout.hlsl). Readers of the bounds mask bits 0..17.
//                GPU-owned: an upload that replaces a resident brick does not overwrite it.
//   words 2..17  micro occupancy: coarse cell c owns words 2+2c and 3+2c; the voxel
//                micro = (x&3) | (y&3)<<2 | (z&3)<<4 is bit (micro & 31) of word
//                2 + 2c + (micro >> 5)
//   words 18..   payload, two blocks per word: the block at (x, y, z) is in word
//                18 + (x>>1) + 4*y + 32*z, EVEN x in the LOW 16 bits.
// Page table: one uint64_t record page address per page.
#define kPageTableEntryBytes 8

static const uint kRecordInfoWords = 2;
static const uint kRecordOccupancyWords = 16;
static const uint kRecordPayloadWord = 18;
static const uint kRecordWords = 274;
#define kRecordWordBytes 4
#define kRecordPosMask 0xFFFFu   // word 0
#define kRecordCoarseShift 16    // word 0
#define kRecordCoarseMask 0xFFu  // word 0, after the shift
#define kRecordBoundsWord 1
#define kRecordBoundsAxisBits 3  // per bound coordinate
#define kRecordBoundsAxisMask 7u
#define kRecordBoundsMaxShift 9  // the max corner after the min corner
#define kBrickEdge 8             // voxels per brick edge
#define kBrickVoxels 512         // RecordLayout::kBlocksPerBrick

uint RecordPayloadWord(uint x, uint y, uint z)
{
    return kRecordPayloadWord + (x >> 1) + 4 * y + 32 * z;
}

// The block at x of a payload word (even x in the low half, odd x in the high half).
#define kPayloadOddShift 16
uint PayloadPairBlock(uint word, uint x)
{
    return (x & 1) != 0 ? (word >> kPayloadOddShift) : (word & 0xFFFFu);
}

// The brick coordinate in its group of a group-local brick position
// (pos = bx | by << shift | bz << 2 shift).
uint3 BrickOfPos(uint pos, uint group_shift)
{
    const uint mask = (1u << group_shift) - 1u;
    return uint3(pos & mask, (pos >> group_shift) & mask, (pos >> (2 * group_shift)) & mask);
}

// Face directions in face order: +X, -X, +Y, -Y, +Z, -Z.
int3 FaceDirection(uint f)
{
    const int s = (f & 1) != 0 ? -1 : 1;
    return f < 2 ? int3(s, 0, 0) : (f < 4 ? int3(0, s, 0) : int3(0, 0, s));
}

// ---------------------------------------------------------------------------
// Slot table row: 16 bytes per dense brick id.
//   x pos_state     bits 0..23 group-local position, bits 24..31 state
//   y summary
//   z record_index  0xFFFFFFFF when not resident
//   w last_used
static const uint kSlotRowBytes = 16;
static const uint kSlotStateResident = 1;
static const uint kInvalidRecord = 0xFFFFFFFFu;
#define kSlotRowPosMask 0x00FFFFFFu
#define kSlotRowStateShift 24

uint SlotRowPos(uint pos_state)
{
    return pos_state & kSlotRowPosMask;
}
uint SlotRowState(uint pos_state)
{
    return pos_state >> kSlotRowStateShift;
}

// ---------------------------------------------------------------------------
// AABB row: 24 bytes per dense brick id, float3 min then float3 max, in
// entity-local block coordinates. All six floats NaN = inactive primitive.
static const uint kAabbRowBytes = 24;
static const uint kQuietNaN = 0x7FC00000u;

// Reverse table: one 16-bit brick id per group-local position, 0xFFFF = none.
static const uint kNoBrickId = 0xFFFFu;
// Group grid cell: a scene-wide group index, 0xFFFFFFFF = none.
static const uint kNoGroup = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Tables of the finalize kernel. The structs document the layout; the kernel
// reads them word by word through buffer device addresses.

// One brick to finalize. 16 bytes.
struct DirtyEntry
{
    uint group_index;
    uint id;
    uint record_index;
    uint pad;
};

#define kDirtyEntryBytes 16

// One render group, indexed by the scene-wide group index. 32 bytes.
struct GroupInfo
{
    int3 coord;  // group coordinate (brick key >> group_shift)
    uint entity;
    uint slot_row_base;  // first slot table row of the group
    uint reverse_base;   // first u16 entry of the group's reverse table
    uint aabb_row_base;  // first AABB row of the group
    uint high_water;     // dense ids in [0, high_water) were handed out
};
// Kernels load it as two uint4: (coord, entity) at kGroupInfoCoord and (slot_row_base,
// reverse_base, aabb_row_base, high_water) at kGroupInfoSlotRowBase.
#define kGroupInfoCoord 0
#define kGroupInfoEntity 12
#define kGroupInfoSlotRowBase 16
#define kGroupInfoReverseBase 20
#define kGroupInfoAabbRowBase 24
#define kGroupInfoHighWater 28
#define kGroupInfoBytes 32

// One entity, indexed by the entity id. 128 bytes.
struct EntityInfo
{
    int3 grid_min;   // group coordinate of grid cell (0, 0, 0)
    uint grid_base;  // first u32 cell of the entity's group grid
    int3 grid_dims;  // cells per axis (powers of two; 0 = no grid)
    uint pad;
    // Object-to-world rows (rotation | position) of the current transform and of
    // the previous frame's transform (equal to cur except for one flush after a
    // transform change).
    float4 cur_row0;
    float4 cur_row1;
    float4 cur_row2;
    float4 prev_row0;
    float4 prev_row1;
    float4 prev_row2;
};
// (grid_min, grid_base) at kEntityInfoGridMin, (grid_dims, life) at kEntityInfoGridDims, the
// rows at kEntityInfoCurRow0 .. 2 and kEntityInfoPrevRow0 .. 2.
#define kEntityInfoGridMin 0
#define kEntityInfoGridBase 12
#define kEntityInfoGridDims 16
#define kEntityInfoLife 28  // u32: +1 every time the entity id is handed out again
#define kEntityInfoCurRow0 32
#define kEntityInfoCurRow1 48
#define kEntityInfoCurRow2 64
#define kEntityInfoPrevRow0 80
#define kEntityInfoPrevRow1 96
#define kEntityInfoPrevRow2 112
static const uint kEntityInfoBytes = 128;

// ---------------------------------------------------------------------------
// Record scatter (scene/caelix_record_scatter.hlsl): copies the records that
// caelix_scene_upload_bricks wrote into the staging ring (ReBAR path) or its device-local
// bounce buffer (bounce path) to their record slots. One entry, 16 bytes, loaded as a uint4:
//   x   source offset / kRecordScatterSourceUnit (records are 8-byte aligned in the ring)
//   y   flags: kScatterKeepBoundsWord
//   zw  the record's device address (low word in z)
#define kRecordScatterEntryBytes 16
#define kRecordScatterSourceUnit 8
// The destination's word 1 (kRecordBoundsWord) stays: the finalize kernel compares the old light
// state there with the new content of a replaced record.
#define kScatterKeepBoundsWord 1u
// Kernel shape: records and threads per workgroup. The largest record in 8-byte words sizes the
// kernel's per-thread word array.
#define kScatterRecordsPerGroup 16
#define kScatterThreadsPerGroup 256
#define kScatterMaxRecordWords8 (kRecordWords * kRecordWordBytes / kRecordScatterSourceUnit)

// Block encodings (FinalizePush::encoding).
static const uint kEncodingPacked = 0;
static const uint kEncodingBlockId = 1;
#define kPackedOpaqueMask 0xC000u  // opaque when any bit is set
#define kPackedIdMask 0xFFu        // transparent id
#define kPackedFaceShift 8u        // face bits of a transparent block
#define kBlockIdOpaqueMask 0x8000u
#define kBlockIdIdMask 0x1FFu
#define kBlockIdFaceShift 9u

// Encoding-parameterized forms of IsOpaque / CaelixMaterialCode (common/caelix_block.hlsl and
// common/caelix_material.hlsl read the encoding from a kernel global).
bool BlockIsOpaque(uint v, uint encoding)
{
    return (v & (encoding == kEncodingPacked ? kPackedOpaqueMask : kBlockIdOpaqueMask)) != 0;
}

// The material code of a raw block value: opaque values as they are, transparent ones without
// their face bits (the transparent id).
uint BlockMaterialCode(uint v, uint encoding)
{
    const uint id_mask = encoding == kEncodingPacked ? kPackedIdMask : kBlockIdIdMask;
    return BlockIsOpaque(v, encoding) ? v : (v & id_mask);
}

#endif  // CAELIX_RECORD_HLSL
