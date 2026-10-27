#pragma once
#include <cstdint>

namespace caelix {

// Constants of the record and of the block encodings, mirrored by name in
// shaders/common/caelix_record.hlsl (caelix_frame_layout_test compares the files).
constexpr uint32_t kRecordInfoWords = 2;        // words 0 and 1
constexpr uint32_t kRecordOccupancyWords = 16;  // words 2..17
constexpr uint32_t kRecordWordBytes = sizeof(uint32_t);
constexpr uint32_t kRecordPosMask = 0xFFFFu;   // word 0: group-local brick position
constexpr uint32_t kRecordCoarseShift = 16;    // word 0: coarse occupancy byte
constexpr uint32_t kRecordCoarseMask = 0xFFu;
// Word 1 is GPU-owned: the finalize kernel reads its light state before it rewrites it, so an
// upload that replaces a resident brick's content does not copy it (Scene::StageRecordScatter,
// kScatterKeepBoundsWord).
constexpr uint32_t kRecordBoundsWord = 1;
constexpr uint32_t kRecordBoundsAxisBits = 3;  // per bound coordinate
constexpr uint32_t kRecordBoundsAxisMask = (1u << kRecordBoundsAxisBits) - 1u;
constexpr uint32_t kRecordBoundsMaxShift = 9;  // the max corner after the min corner
constexpr uint32_t kPayloadOddShift = 16;      // the block at odd x of a payload word
constexpr uint32_t kBrickEdge = 8;             // voxels per brick edge
constexpr uint32_t kBrickVoxels = 512;
constexpr uint32_t kPageTableEntryBytes = sizeof(uint64_t);

// Block encodings (CaelixSceneDesc::encoding).
constexpr uint32_t kEncodingPacked = 0;
constexpr uint32_t kEncodingBlockId = 1;
constexpr uint32_t kPackedOpaqueMask = 0xC000u;  // opaque when any bit is set
constexpr uint32_t kPackedIdMask = 0xFFu;        // transparent id
constexpr uint32_t kPackedFaceShift = 8;         // face bits of a transparent block
constexpr uint32_t kBlockIdOpaqueMask = 0x8000u;
constexpr uint32_t kBlockIdIdMask = 0x1FFu;
constexpr uint32_t kBlockIdFaceShift = 9;

constexpr bool BlockIsOpaque(uint32_t v, uint32_t encoding)
{
    return (v & (encoding == kEncodingPacked ? kPackedOpaqueMask : kBlockIdOpaqueMask)) != 0;
}

// The material code of a raw block value: opaque values as they are, transparent ones without
// their face bits (the transparent id).
constexpr uint32_t BlockMaterialCode(uint32_t v, uint32_t encoding)
{
    const uint32_t id_mask = encoding == kEncodingPacked ? kPackedIdMask : kBlockIdIdMask;
    return BlockIsOpaque(v, encoding) ? v : (v & id_mask);
}

// The GPU record of one brick (8x8x8 blocks). One layout per scene, computed
// from the block size. It follows the brick record of Unity's
// CaelixBrickTrace.hlsl, except for the pair order (even x is low here).
//
// Words (uint32), for block_bytes = 2:
//   word 0        bits 0..15  group-local brick position
//                             pos = bx | by << shift | bz << 2*shift
//                 bits 16..23 coarse occupancy (2x2x2 bits). 0 until a later
//                             kernel fills it.
//   word 1        bits 0..17 tight bounds, bits 18..29 light face count, bit 30
//                 light record stale (light/light_tree_layout.h). 0 until a later
//                 kernel fills it; readers of the bounds mask bits 0..17.
//   words 2..17   micro occupancy (4^3 bits per 64-bit pair). 0 until a later
//                 kernel fills it.
//   words 18..273 raw blocks, two per word. The block at voxel (x, y, z) sits in
//                 word 18 + (x >> 1) + 4*y + 32*z; the block at EVEN x is in the
//                 LOW 16 bits, the block at odd x in the HIGH 16 bits (the
//                 little-endian image of a uint16_t[512] array indexed x + 8y + 64z).
//
// Lossless rule: the payload always holds the raw block values that the host
// uploaded. Later kernels add bits in the header words, never in the payload.
struct RecordLayout
{
    uint32_t block_bytes = 2;
    uint32_t info_words = kRecordInfoWords;
    uint32_t occupancy_words = kRecordOccupancyWords;
    uint32_t payload_words = 0;  // 512 * block_bytes / 4
    uint32_t record_words = 0;   // info + occupancy + payload
    uint32_t record_bytes = 0;

    static constexpr uint32_t kBlocksPerBrick = kBrickVoxels;

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
