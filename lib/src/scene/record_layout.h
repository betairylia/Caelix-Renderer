#pragma once
#include <cstdint>

namespace caelix {

// The GPU record of one brick (8x8x8 blocks). One layout per scene, computed
// from the block size. It mirrors the brick record of Unity's
// CaelixBrickTrace.hlsl, so the trace kernels can be ported unchanged.
//
// Words (uint32), for block_bytes = 2:
//   word 0        bits 0..15  group-local brick position
//                             pos = bx | by << shift | bz << 2*shift
//                 bits 16..23 coarse occupancy (2x2x2 bits). 0 until a later
//                             kernel fills it.
//   word 1        tight bounds. 0 until a later kernel fills it.
//   words 2..17   micro occupancy (4^3 bits per 64-bit pair). 0 until a later
//                 kernel fills it.
//   words 18..273 raw blocks, two per word. The block at voxel (x, y, z) sits in
//                 word 18 + (x >> 1) + 4*y + 32*z; the block at EVEN x is in the
//                 HIGH 16 bits, the block at odd x in the LOW 16 bits.
//
// Lossless rule: the payload always holds the raw block values that the host
// uploaded. Later kernels add bits in the header words, never in the payload.
struct RecordLayout
{
    uint32_t block_bytes = 2;
    uint32_t info_words = 2;
    uint32_t occupancy_words = 16;
    uint32_t payload_words = 0;  // 512 * block_bytes / 4
    uint32_t record_words = 0;   // info + occupancy + payload
    uint32_t record_bytes = 0;

    static constexpr uint32_t kBlocksPerBrick = 512;

    static RecordLayout ForBlockBytes(uint32_t block_bytes)
    {
        RecordLayout layout;
        layout.block_bytes = block_bytes;
        layout.payload_words = kBlocksPerBrick * block_bytes / 4;
        layout.record_words = layout.info_words + layout.occupancy_words + layout.payload_words;
        layout.record_bytes = layout.record_words * 4;
        return layout;
    }

    uint32_t payload_first_word() const
    {
        return info_words + occupancy_words;
    }
};

}  // namespace caelix
