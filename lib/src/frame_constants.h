#pragma once
#include <cstddef>
#include <cstdint>

namespace caelix {

// CPU mirror of CaelixFrameConstants in shaders/caelix_frame.hlsl; change both
// together. One per render call, in a host-visible buffer; its device address is
// the trace kernel's only push constant. 256 bytes. Every field sits at an offset
// that is a multiple of its size (float4 at 16, uint64_t and uint2 at 8), so
// std430 and scalar layout rules give the same offsets.
struct FrameConstants {
  float position[4];        //   0  camera position, w unused
  float right[4];           //  16  camera basis, w unused
  float up[4];              //  32
  float forward[4];         //  48
  uint32_t size[2];         //  64  target width, height in pixels
  float tan_half_fov_y;     //  72
  float aspect;             //  76
  float near_z;             //  80
  float far_z;              //  84
  uint64_t page_table;      //  88  uint64_t record page address per page
  uint64_t slot_tables;     //  96  slot row pool base
  uint64_t group_infos;     // 104  GroupInfo by group index
  uint64_t entity_infos;    // 112  EntityInfo by entity id
  uint32_t records_per_page_shift;  // 120
  uint32_t record_bytes;    // 124
  uint32_t group_shift;     // 128
  uint32_t encoding;        // 132  0 packed, 1 block-ID
  uint32_t debug_view;      // 136  CaelixDebugView
  uint32_t flags;           // 140  CAELIX_RENDER_FLAG_*
  float sun_dir[4];         // 144  normalized, toward the sun, w unused
  float sky_color[4];       // 160  w unused
  uint32_t pad[20];         // 176
};
static_assert(sizeof(FrameConstants) == 256);
static_assert(offsetof(FrameConstants, size) == 64);
static_assert(offsetof(FrameConstants, page_table) == 88);
static_assert(offsetof(FrameConstants, records_per_page_shift) == 120);
static_assert(offsetof(FrameConstants, sun_dir) == 144);
static_assert(offsetof(FrameConstants, sky_color) == 160);

// Push constants of the trace kernel: the address of the frame's FrameConstants.
struct TracePush {
  uint64_t frame_constants;
};
static_assert(sizeof(TracePush) == 8);

}  // namespace caelix
