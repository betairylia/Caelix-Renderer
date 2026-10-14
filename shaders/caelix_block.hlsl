// Block value decoding for the trace kernels. The encoding is a scene setting
// (CaelixSceneDesc::encoding), so it is a run-time value here, not a compile-time
// switch as in Unity's CaelixUtils.hlsl (CAELIX_PACKED_SCENE_COLOR).
//   packed   (0): opaque when (v & 0xC000) != 0, transparent id v & 0xFF,  face bits (v >> 8) & 63
//   block-ID (1): opaque when (v & 0x8000) != 0, transparent id v & 0x1FF, face bits (v >> 9) & 63
#ifndef CAELIX_BLOCK_HLSL
#define CAELIX_BLOCK_HLSL

#include "caelix_record.hlsl"

// Set by the kernel from CaelixFrameConstants::encoding before the first trace.
static uint g_blockEncoding;

inline bool IsOpaque(int blk) {
  return g_blockEncoding == kEncodingPacked ? (blk & 0xC000) != 0 : (blk & 0x8000) != 0;
}

inline int GetFaceBits(int blk) {
  return g_blockEncoding == kEncodingPacked ? ((blk >> 8) & 63) : ((blk >> 9) & 63);
}

// The id of a transparent block (face bits removed).
inline uint CaelixTransparentId(uint blk) {
  return g_blockEncoding == kEncodingPacked ? (blk & 0xFFu) : (blk & 0x1FFu);
}

#endif  // CAELIX_BLOCK_HLSL
