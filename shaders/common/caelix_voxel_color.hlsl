// GPU layout and helpers of the per-voxel colour experiment (CAELIX_RENDER_FLAG_VOXEL_COLOR, experiment
// branch sss-voxel-color-expr, docs/VOXEL_COLOR_EXPERIMENT.md), shared by the accumulate kernel
// (render/caelix_voxel_accumulate.hlsl) and the encode kernel (render/caelix_upscale_encode.hlsl). Mirrors
// lib/src/render/voxel_color_layout.h: every constant here has the same name and value there
// (caelix_frame_layout_test compares the two files).
//
// The voxel hash table: fc.voxel_slot_mask + 1 slots (a power of two) of kVoxelSlotWords uint words in a
// storage buffer, cleared to 0 before the accumulate pass of every frame that runs it. Key = (entity,
// voxel x, y, z) of the camera ray's first hit; two independent 32-bit hashes of the 4 words give the start
// slot and the fingerprint (forced non-zero: 0 marks an empty slot). Accepted risk: two voxels with the same
// fingerprint in one probe chain share a slot and merge (expected well below 0.01 merges per frame).
// Colour sums are 64-bit fixed point (kVoxelFixedScale steps per unit) in two words per channel, added with
// a carry detected from the low word's atomic add (exact: every low add is atomic and returns the value it
// added to). No float <-> uint64_t conversion anywhere (DXC 1.8: float(uint64_t) drops the high word).
#ifndef CAELIX_VOXEL_COLOR_HLSL
#define CAELIX_VOXEL_COLOR_HLSL

#include "common/caelix_platform.hlsl"

// Slot words: fingerprint (0 = empty), pixel count, sum low words (r, g, b), sum high words (r, g, b).
static const uint kVoxelSlotFingerprint = 0;
static const uint kVoxelSlotCount = 1;
static const uint kVoxelSlotSumLo = 2;
static const uint kVoxelSlotSumHi = 5;
static const uint kVoxelSlotWords = 8;
static const uint kVoxelSlotBytes = kVoxelSlotWords * 4;
// The slot image's value of a pixel without a slot (a miss, no coverage, or a full probe window).
static const uint kVoxelSlotInvalid = 0xFFFFFFFFu;
// Linear probes of one insert.
static const uint kVoxelProbes = 64;
// Seeds of the two key hashes: the start slot and the fingerprint.
static const uint kVoxelSlotSeed = 0x2545F491u;
static const uint kVoxelFingerprintSeed = 0x9E3779B9u;

// Fixed point of the colour sums: a channel value is clamped to [0, kVoxelMaxValue] (the FP16 range of the
// source images) and stored as round(value * kVoxelFixedScale); 65504 * 65536 < 2^32.
static const float kVoxelFixedScale = 65536.0f;
static const float kVoxelMaxValue = 65504.0f;

// One round of the lowbias32 finaliser (HashU32 of render/caelix_shading.hlsl; a bijection of uint).
uint VoxelHashMix(uint value)
{
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    value ^= value >> 16;
    return value;
}

// A 32-bit hash of the key words (entity, x, y, z) under `seed`: one finaliser round per word.
uint VoxelKeyHash(uint4 key, uint seed)
{
    uint h = VoxelHashMix(seed ^ key.x);
    h = VoxelHashMix(h ^ key.y);
    h = VoxelHashMix(h ^ key.z);
    return VoxelHashMix(h ^ key.w);
}

// The fixed-point value of one colour channel.
uint VoxelFixed(float value)
{
    return uint(round(min(max(value, 0.0f), kVoxelMaxValue) * kVoxelFixedScale));
}

// The mean colour of a slot from its pixel count (>= 1) and its sum words.
float3 VoxelSlotMean(uint count, uint3 sum_lo, uint3 sum_hi)
{
    const float3 sum = float3(sum_hi) * 4294967296.0f + float3(sum_lo);
    return sum / (kVoxelFixedScale * float(count));
}

#endif  // CAELIX_VOXEL_COLOR_HLSL
