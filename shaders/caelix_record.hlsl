// Shared GPU data layouts. Mirrors lib/src/record_layout.h, lib/src/slot_table_pool.h
// and lib/src/finalize.h; change both sides together.
#ifndef CAELIX_RECORD_HLSL
#define CAELIX_RECORD_HLSL

#include "caelix_platform.hlsl"

// ---------------------------------------------------------------------------
// Brick record (16-bit blocks): 274 words = 1096 bytes, 8-byte aligned.
//   word 0       bits 0..15  group-local brick position
//                bits 16..23 coarse occupancy (2x2x2 bits, bit i = coarse cell
//                            (x>>2) | (y>>2)<<1 | (z>>2)<<2 has an occupied voxel)
//   word 1       tight bounds of the occupied voxels, inclusive:
//                minX | minY<<3 | minZ<<6 | maxX<<9 | maxY<<12 | maxZ<<15 (0 when empty)
//   words 2..17  micro occupancy: coarse cell c owns words 2+2c and 3+2c; the voxel
//                micro = (x&3) | (y&3)<<2 | (z&3)<<4 is bit (micro & 31) of word
//                2 + 2c + (micro >> 5)
//   words 18..   payload, two blocks per word: the block at (x, y, z) is in word
//                18 + (x>>1) + 4*y + 32*z, EVEN x in the HIGH 16 bits.
static const uint kRecordInfoWords = 2;
static const uint kRecordOccupancyWords = 16;
static const uint kRecordPayloadWord = 18;
static const uint kRecordWords = 274;

uint RecordPayloadWord(uint x, uint y, uint z) {
  return kRecordPayloadWord + (x >> 1) + 4 * y + 32 * z;
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

uint SlotRowPos(uint pos_state) { return pos_state & 0x00FFFFFFu; }
uint SlotRowState(uint pos_state) { return pos_state >> 24; }

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
struct DirtyEntry {
  uint group_index;
  uint id;
  uint record_index;
  uint pad;
};

// One render group, indexed by the scene-wide group index. 32 bytes.
struct GroupInfo {
  int3 coord;          // group coordinate (brick key >> group_shift)
  uint entity;
  uint slot_row_base;  // first slot table row of the group
  uint reverse_base;   // first u16 entry of the group's reverse table
  uint aabb_row_base;  // first AABB row of the group
  uint high_water;     // dense ids in [0, high_water) were handed out
};

// One entity, indexed by the entity id. 32 bytes.
struct EntityInfo {
  int3 grid_min;   // group coordinate of grid cell (0, 0, 0)
  uint grid_base;  // first u32 cell of the entity's group grid
  int3 grid_dims;  // cells per axis (powers of two; 0 = no grid)
  uint pad;
};

// Block encodings (FinalizePush::encoding).
static const uint kEncodingPacked = 0;   // opaque 0xC000, id mask 0xFF, faces at bit 8
static const uint kEncodingBlockId = 1;  // opaque 0x8000, id mask 0x1FF, faces at bit 9

#endif  // CAELIX_RECORD_HLSL
