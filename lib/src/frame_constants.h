#pragma once
#include <cstddef>
#include <cstdint>

namespace caelix {

// CPU mirror of CaelixFrameConstants in shaders/caelix_frame.hlsl; change both
// together. One per render call, in a host-visible buffer; its device address is
// the only push constant of the trace and composite kernels. 512 bytes. Every field sits at an offset
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
  float projection[4][4];   // 176  ROWS of CaelixCamera::projection: clip[i] = dot(row i, (view, 1))
  float prev_position[4];   // 240  camera of the previous render call, w unused
  float prev_right[4];      // 256
  float prev_up[4];         // 272
  float prev_forward[4];    // 288
  float prev_tan_half_fov_y;  // 304
  float prev_aspect;        // 308
  uint32_t target_mask;     // 312  kTarget* bits: present targets, normal encoding
  float miss_depth;         // 316
  float miss_view_z;        // 320
  uint32_t has_prev_camera; // 324  0 on the first render call of the renderer
  uint32_t has_tlas;        // 328  0 before the scene's first TLAS build: write the miss values
  uint32_t mode;            // 332  CaelixRenderMode
  uint32_t bounce_count;    // 336  stochastic bounces, 0..CAELIX_MAX_BOUNCES
  uint32_t frame_seed;      // 340  CaelixRenderOptions::frame_seed
  uint32_t accum_count;     // 344  reference mode: frames accumulated before this one (0 = restart)
  float exposure;           // 348  composite multiply (0 already replaced by 1)
  float sun_radiance[4];    // 352  w unused
  float sky_zenith[4];      // 368  w unused
  float sky_horizon[4];     // 384  w unused
  float sky_ground[4];      // 400  w unused
  float sun_cos_radius;     // 416  cos(sun angular radius)
  uint32_t output_srgb;     // 420  1 = the colour target is UNORM: the composite encodes linear -> sRGB
  float t_max;              // 424  kTMax: bounce ray length and the "no hit" hit distance
  float sun_solid_angle;    // 428  2 pi (1 - sun_cos_radius): solid angle of the sun cone (NEE pdf 1 / it)
  uint32_t pad[20];         // 432
};
static_assert(sizeof(FrameConstants) == 512);
static_assert(offsetof(FrameConstants, size) == 64);
static_assert(offsetof(FrameConstants, page_table) == 88);
static_assert(offsetof(FrameConstants, records_per_page_shift) == 120);
static_assert(offsetof(FrameConstants, sun_dir) == 144);
static_assert(offsetof(FrameConstants, sky_color) == 160);
static_assert(offsetof(FrameConstants, projection) == 176);
static_assert(offsetof(FrameConstants, prev_position) == 240);
static_assert(offsetof(FrameConstants, prev_forward) == 288);
static_assert(offsetof(FrameConstants, prev_tan_half_fov_y) == 304);
static_assert(offsetof(FrameConstants, target_mask) == 312);
static_assert(offsetof(FrameConstants, has_tlas) == 328);
static_assert(offsetof(FrameConstants, mode) == 332);
static_assert(offsetof(FrameConstants, exposure) == 348);
static_assert(offsetof(FrameConstants, sun_radiance) == 352);
static_assert(offsetof(FrameConstants, sky_ground) == 400);
static_assert(offsetof(FrameConstants, sun_cos_radius) == 416);
static_assert(offsetof(FrameConstants, t_max) == 424);
static_assert(offsetof(FrameConstants, sun_solid_angle) == 428);

// Hit distance written for "no hit" (a bounce that reaches the sky) and the length of bounce rays.
// Unity's K_T_MAX; it also fits the FP16 radiance targets (max 65504).
constexpr float kTMax = 1e4f;

// FrameConstants::target_mask: one bit per present target (bit i = descriptor
// binding i + 1), plus the encoding of the normal target.
constexpr uint32_t kTargetColor = 1u << 0;
constexpr uint32_t kTargetAlbedo = 1u << 1;
constexpr uint32_t kTargetNormal = 1u << 2;
constexpr uint32_t kTargetDepth = 1u << 3;
constexpr uint32_t kTargetMotion = 1u << 4;
constexpr uint32_t kTargetViewZ = 1u << 5;
constexpr uint32_t kTargetNormalUnorm = 1u << 16;  // the normal target stores n * 0.5 + 0.5

// Push constants of the trace and composite kernels: the address of the frame's FrameConstants.
struct TracePush {
  uint64_t frame_constants;
};
static_assert(sizeof(TracePush) == 8);

}  // namespace caelix
