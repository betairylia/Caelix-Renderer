#pragma once
#include <cstddef>
#include <cstdint>

namespace caelix {

// CPU mirror of CaelixFrameConstants in shaders/common/caelix_frame.hlsl; change both
// together. One per render call, in the frame slot's host-visible buffer, which the five
// frame-constant kernels (caelix_path_trace, caelix_composite, caelix_cross_resolve, caelix_sky_sh,
// caelix_upscale_encode) read as a uniform buffer: the frame set (set kFrameSetIndex, binding
// kFrameConstantsBinding, shaders/common/caelix_frame_ubo.hlsl). 560 bytes.
// Layout rule: the uniform block has DXC's default constant buffer layout, which gives these C
// offsets only while every field sits at an offset that is a multiple of its size (float4 at 16,
// uint64_t and uint2 at 8), no vector crosses a 16-byte row and every array element is 16 bytes (an
// array of scalars or 2-vectors would get a 16-byte stride). The test caelix_frame_layout_test
// (tests/frame_layout_test.cpp) reads the compiled kernels and fails when a member offset differs.
struct FrameConstants
{
    float position[4];                  //   0  camera position, w unused
    float right[4];                     //  16  camera basis, w unused
    float up[4];                        //  32
    float forward[4];                   //  48
    uint32_t size[2];                   //  64  render width, height in pixels (the targets; with an upscaler the render size)
    float tan_half_fov_y;               //  72
    float aspect;                       //  76
    float near_z;                       //  80
    float far_z;                        //  84
    uint64_t page_table;                //  88  uint64_t record page address per page
    uint64_t slot_tables;               //  96  slot row pool base
    uint64_t group_infos;               // 104  GroupInfo by group index
    uint64_t entity_infos;              // 112  EntityInfo by entity id
    uint32_t records_per_page_shift;    // 120
    uint32_t record_bytes;              // 124
    uint32_t group_shift;               // 128
    uint32_t encoding;                  // 132  0 packed, 1 block-ID
    uint32_t debug_view;                // 136  CaelixDebugView
    uint32_t flags;                     // 140  CAELIX_RENDER_FLAG_*
    float sun_dir[4];                   // 144  normalized, toward the sun, w unused
    float sky_color[4];                 // 160  w unused
    float projection[4][4];             // 176  ROWS of CaelixCamera::projection: clip[i] = dot(row i, (view, 1))
    float prev_position[4];             // 240  camera of the previous render call, w unused
    float prev_right[4];                // 256
    float prev_up[4];                   // 272
    float prev_forward[4];              // 288
    float prev_tan_half_fov_y;          // 304
    float prev_aspect;                  // 308
    uint32_t target_mask;               // 312  kTarget* bits: present targets, normal encoding
    float miss_depth;                   // 316
    float miss_view_z;                  // 320
    uint32_t has_prev_camera;           // 324  0 on the first render call of the renderer
    uint32_t has_tlas;                  // 328  0 before the scene's first TLAS build: write the miss values
    uint32_t mode;                      // 332  CaelixRenderMode
    uint32_t bounce_count;              // 336  stochastic bounces, 0..CAELIX_MAX_BOUNCES
    uint32_t frame_seed;                // 340  CaelixRenderOptions::frame_seed
    uint32_t accum_count;               // 344  reference mode: frames accumulated before this one (0 = restart)
    float exposure;                     // 348  composite multiply (0 already replaced by 1)
    float sun_radiance[4];              // 352  w unused
    float sky_zenith[4];                // 368  w unused
    float sky_horizon[4];               // 384  w unused
    float sky_ground[4];                // 400  w unused
    float sun_cos_radius;               // 416  cos(sun angular radius)
    uint32_t output_srgb;               // 420  1 = the colour target is UNORM: the composite encodes linear -> sRGB
    float t_max;                        // 424  kTMax: bounce ray length and the "no hit" hit distance
    float sun_solid_angle;              // 428  2 pi (1 - sun_cos_radius): solid angle of the sun cone (NEE pdf 1 / it)
    uint32_t denoised;                  // 432  1 = NRD ran this frame: the composite reads its outputs (validation view)
    uint32_t bounce_count_transparent;  // 436  delta chain interfaces, 0..CAELIX_MAX_BOUNCES_TRANSPARENT
    uint64_t materials;                 // 440  CaelixMaterial (48 B) by block code, 65536 entries
    float hit_dist_params[4];           // 448  REBLUR hit distance normalization A, B, C (w unused), = ReblurSettings
    uint32_t cross_resolve;             // 464  1 = the composite writes the resolve image, the cross resolve the target
    uint32_t pad_468;                   // 468
    float jitter[2];                    // 472  CaelixCamera::jitter (pixels, y down)
    float fog_color[4];                 // 480  rgb = fog radiance, w = fog density per unit (0 = no fog)
    uint64_t sky_sh;                    // 496  address of 9 float4 (rgb, 0): irradiance SH coefficients (sky_cubemap = 1)
    float emission_scale;               // 504  multiplies every material's emission (0 already replaced by 1)
    uint32_t sky_cubemap;               // 508  1 = the kernels sample the host cubemap, 0 = the procedural sky
    float sky_max_lod;                  // 512  mip_levels - 1 of the cubemap
    float sky_sh_lod;                   // 516  the mip the SH kernel samples (face about 32 px)
    uint32_t output_size[2];            // 520  the colour target's size (= size without an upscaler)
    uint32_t upscale_src_size[2];       // 528  size of the encode kernel's source (upscaler output, or size on the fallback)
    uint32_t upscaler;                  // 536  0 none, 1 DLSS SR, 2 DLSS RR (the trace writes the RR guides)
    uint32_t upscale;                   // 540  1 = the composite / cross resolve write upscale_in instead of the colour target
    uint64_t light_tree;                // 544  light tree descriptor (docs/LIGHT_TREE.md), 0 = no tree
    uint32_t light_nee_flags;           // 552  kLightNee*: the light NEE of this frame (0 = off)
    uint32_t pad_556;                   // 556
};
static_assert(sizeof(FrameConstants) == 560);
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
static_assert(offsetof(FrameConstants, denoised) == 432);
static_assert(offsetof(FrameConstants, bounce_count_transparent) == 436);
static_assert(offsetof(FrameConstants, materials) == 440);
static_assert(offsetof(FrameConstants, hit_dist_params) == 448);
static_assert(offsetof(FrameConstants, cross_resolve) == 464);
static_assert(offsetof(FrameConstants, jitter) == 472);
static_assert(offsetof(FrameConstants, fog_color) == 480);
static_assert(offsetof(FrameConstants, sky_sh) == 496);
static_assert(offsetof(FrameConstants, emission_scale) == 504);
static_assert(offsetof(FrameConstants, sky_cubemap) == 508);
static_assert(offsetof(FrameConstants, sky_max_lod) == 512);
static_assert(offsetof(FrameConstants, sky_sh_lod) == 516);
static_assert(offsetof(FrameConstants, output_size) == 520);
static_assert(offsetof(FrameConstants, upscale_src_size) == 528);
static_assert(offsetof(FrameConstants, upscaler) == 536);
static_assert(offsetof(FrameConstants, upscale) == 540);
static_assert(offsetof(FrameConstants, light_tree) == 544);
static_assert(offsetof(FrameConstants, light_nee_flags) == 552);
static_assert(offsetof(FrameConstants, pad_556) == 556);

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

// FrameConstants::light_nee_flags (docs/LIGHT_NEE.md; the macros of shaders/common/caelix_frame.hlsl):
// kLightNeeOn = every diffuse NEE vertex draws a lamp point from the light tree and chooses between it
// and the sun candidate (the trace kernel caelix_path_trace; 0 binds caelix_path_trace_no_light_nee);
// kLightNeeMis = the balance heuristic with the BSDF path (else the exclusive rule).
constexpr uint32_t kLightNeeOn = 1u << 0;
constexpr uint32_t kLightNeeMis = 1u << 1;

// The frame set: one descriptor set layout with one binding, the frame slot's FrameConstants as a
// uniform buffer (range sizeof(FrameConstants), offset 0). Bound as set kFrameSetIndex by the five
// frame-constant kernels (each keeps its own set 0); they have no push constants.
constexpr uint32_t kFrameSetIndex = 1;
constexpr uint32_t kFrameConstantsBinding = 0;

}  // namespace caelix
