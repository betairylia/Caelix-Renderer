#pragma once
// GPU layout of the per-voxel colour experiment (CAELIX_RENDER_FLAG_VOXEL_COLOR, experiment branch
// sss-voxel-color-expr, docs/VOXEL_COLOR_EXPERIMENT.md). Mirrors shaders/common/caelix_voxel_color.hlsl:
// every constant here has the same name and value there (caelix_frame_layout_test compares the two files).
#include <cstdint>

namespace caelix {

// One slot of the renderer's voxel hash table (a storage buffer of uint words, cleared to 0 every frame
// that runs the pass): word kVoxelSlotFingerprint the key's fingerprint (0 = empty), kVoxelSlotCount the
// pixel count, kVoxelSlotSumLo + c / kVoxelSlotSumHi + c the low / high word of the 64-bit fixed-point sum
// of colour channel c (r, g, b).
constexpr uint32_t kVoxelSlotFingerprint = 0;
constexpr uint32_t kVoxelSlotCount = 1;
constexpr uint32_t kVoxelSlotSumLo = 2;
constexpr uint32_t kVoxelSlotSumHi = 5;
constexpr uint32_t kVoxelSlotWords = 8;
constexpr uint32_t kVoxelSlotBytes = kVoxelSlotWords * 4;
// The slot image's value of a pixel without a slot (a miss, no coverage, or a full probe window).
constexpr uint32_t kVoxelSlotInvalid = 0xFFFFFFFFu;
// Linear probes of one insert.
constexpr uint32_t kVoxelProbes = 64;
// Seeds of the two key hashes: the start slot and the fingerprint.
constexpr uint32_t kVoxelSlotSeed = 0x2545F491u;
constexpr uint32_t kVoxelFingerprintSeed = 0x9E3779B9u;

// C++ only: the table has max(kVoxelMinSlots, bit_ceil(pass pixels) / 2) slots (a power of two; the frame
// constants carry the slot count - 1 as voxel_slot_mask).
constexpr uint32_t kVoxelMinSlots = 65536;

}  // namespace caelix
