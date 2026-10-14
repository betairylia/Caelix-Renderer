// Per-frame constants of the trace kernels. Mirrors lib/src/frame_constants.h;
// change both together. The kernel's only push constant is the device address of
// one CaelixFrameConstants (256 bytes on the CPU side; the tail is padding and is
// not declared here). Every field sits at an offset that is a multiple of its
// size, so std430 and scalar layout rules give the same offsets.
#ifndef CAELIX_FRAME_HLSL
#define CAELIX_FRAME_HLSL

#include "caelix_platform.hlsl"

struct CaelixFrameConstants {
  float4 position;          //   0  camera position, w unused
  float4 right;             //  16  camera basis (+X right, +Y up, +Z forward), w unused
  float4 up;                //  32
  float4 forward;           //  48
  uint2 size;               //  64  target width, height in pixels
  float tan_half_fov_y;     //  72
  float aspect;             //  76
  float near_z;             //  80
  float far_z;              //  84
  uint64_t page_table;      //  88  uint64_t record page address per page
  uint64_t slot_tables;     //  96  slot row pool base (16-byte rows)
  uint64_t group_infos;     // 104  GroupInfo by group index (32 bytes)
  uint64_t entity_infos;    // 112  EntityInfo by entity id (128 bytes)
  uint records_per_page_shift;  // 120
  uint record_bytes;        // 124
  uint group_shift;         // 128
  uint encoding;            // 132  kEncodingPacked or kEncodingBlockId
  uint debug_view;          // 136  kView*
  uint flags;               // 140  kRenderFlag*
  float4 sun_dir;           // 144  normalized, toward the sun, w unused
  float4 sky_color;         // 160  w unused
};

struct CaelixTracePush {
  uint64_t frame_constants;  // device address of a CaelixFrameConstants
};

// CaelixDebugView
static const uint kViewAlbedo = 0;
static const uint kViewNormal = 1;
static const uint kViewDepth = 2;
static const uint kViewGroup = 3;
static const uint kViewBrick = 4;
static const uint kViewSteps = 5;

// CaelixRenderOptions::flags
static const uint kRenderFlagShading = 1u << 0;

#endif  // CAELIX_FRAME_HLSL
