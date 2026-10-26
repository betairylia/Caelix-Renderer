// Per-frame constants of the trace, composite, cross resolve, sky SH, upscale encode and voxel accumulate
// kernels.
// Mirrors lib/src/render/frame_constants.h; change both together. 592 bytes, read as the uniform
// buffer g_frame (common/caelix_frame_ubo.hlsl: set 1, binding 0).
// Layout rule: the uniform block gets DXC's default constant buffer layout, which gives the C
// offsets only while every field sits at an offset that is a multiple of its size, no vector
// crosses a 16-byte row and every array element is 16 bytes (an array of scalars or 2-vectors
// would get a 16-byte stride). The test caelix_frame_layout_test (tests/frame_layout_test.cpp)
// reads the uniform block of the five compiled kernels and fails on any member offset or array
// stride that differs from the C layout: run ctest after every change of this struct.
#ifndef CAELIX_FRAME_HLSL
#define CAELIX_FRAME_HLSL

#include "common/caelix_platform.hlsl"

struct CaelixFrameConstants
{
    float4 position;                //   0  camera position, w unused
    float4 right;                   //  16  camera basis (+X right, +Y up, +Z forward), w unused
    float4 up;                      //  32
    float4 forward;                 //  48
    uint2 size;                     //  64  render width, height in pixels (with an upscaler the render size)
    float tan_half_fov_y;           //  72
    float aspect;                   //  76
    float near_z;                   //  80
    float far_z;                    //  84
    uint64_t page_table;            //  88  uint64_t record page address per page
    uint64_t slot_tables;           //  96  slot row pool base (16-byte rows)
    uint64_t group_infos;           // 104  GroupInfo by group index (32 bytes)
    uint64_t entity_infos;          // 112  EntityInfo by entity id (128 bytes)
    uint records_per_page_shift;    // 120
    uint record_bytes;              // 124
    uint group_shift;               // 128
    uint encoding;                  // 132  kEncodingPacked or kEncodingBlockId
    uint debug_view;                // 136  kView*
    uint flags;                     // 140  kRenderFlag*
    float4 sun_dir;                 // 144  normalized, toward the sun, w unused
    float4 sky_color;               // 160  w unused
    float4 projection[4];           // 176  ROWS of the host projection: clip[i] = dot(projection[i], float4(view, 1))
    float4 prev_position;           // 240  camera of the previous render call, w unused
    float4 prev_right;              // 256
    float4 prev_up;                 // 272
    float4 prev_forward;            // 288
    float prev_tan_half_fov_y;      // 304
    float prev_aspect;              // 308
    uint target_mask;               // 312  kTarget* bits
    float miss_depth;               // 316
    float miss_view_z;              // 320
    uint has_prev_camera;           // 324  0 on the first render call of the renderer
    uint has_tlas;                  // 328  0 before the scene's first TLAS build
    uint mode;                      // 332  kMode*
    uint bounce_count;              // 336  stochastic bounces, 0..14
    uint frame_seed;                // 340
    uint accum_count;               // 344  reference mode: frames accumulated before this one (0 = restart)
    float exposure;                 // 348  composite multiply
    float4 sun_radiance;            // 352  w unused
    float4 sky_zenith;              // 368  w unused
    float4 sky_horizon;             // 384  w unused
    float4 sky_ground;              // 400  w unused
    float sun_cos_radius;           // 416  cos(sun angular radius)
    uint output_srgb;               // 420  1 = the colour target is UNORM: encode linear -> sRGB
    float t_max;                    // 424  bounce ray length and the "no hit" hit distance (1e4)
    float sun_solid_angle;          // 428  2 pi (1 - sun_cos_radius): solid angle of the sun cone
    uint denoised;                  // 432  1 = NRD ran this frame (composite: validation view)
    uint bounce_count_transparent;  // 436  delta chain interfaces, 0..15
    uint64_t materials;             // 440  CaelixMaterial (64 B) by block code (common/caelix_material.hlsl)
    float4 hit_dist_params;         // 448  REBLUR hit distance normalization A, B, C (w unused)
    uint cross_resolve;             // 464  1 = composite -> resolve image -> caelix_cross_resolve -> colour
    uint pad_468;                   // 468
    float2 jitter;                  // 472  CaelixCamera::jitter: sub-pixel offset of the primary sample (pixels, y down)
    float4 fog_color;               // 480  rgb = fog radiance, w = fog density per unit (0 = no fog)
    uint64_t sky_sh;                // 496  address of 9 float4 (rgb, 0): irradiance SH coefficients, valid when sky_cubemap = 1
    float emission_scale;           // 504  multiplies every material's emission (>= 0, never 0)
    uint sky_cubemap;               // 508  1 = sample the host cubemap (g_sky), 0 = procedural sky
    float sky_max_lod;              // 512  mip_levels - 1 of the cubemap
    float sky_sh_lod;               // 516  the mip the SH kernel samples (face about 32 px)
    uint2 output_size;              // 520  the colour target's size (= size without an upscaler)
    uint2 upscale_src_size;         // 528  size of the encode kernel's source image
    uint upscaler;                  // 536  kUpscaler*: 2 = the trace writes the RR guide images
    uint upscale;                   // 540  1 = composite / cross resolve write g_int_upscale_in (linear)
    uint64_t light_tree;            // 544  light tree descriptor (light/caelix_light_tree.hlsl), 0 = no tree
    uint light_nee_flags;           // 552  kLightNee*: the light NEE of this frame (0 = off)
    uint voxel_slot_mask;           // 556  voxel hash slots - 1 (common/caelix_voxel_color.hlsl; voxel_color = 1 only)
    uint64_t reverse_tables;        // 560  u16 reverse table pool base (group-local position -> dense id, 0xFFFF none)
    uint64_t group_grids;           // 568  u32 group grid pool base (group index or 0xFFFFFFFF)
    uint64_t subsurface_table;      // 576  (sss_albedo, sss_probability) rows (16 B) by block code (render/caelix_subsurface.hlsl)
    uint pad_584;                   // 584
    uint voxel_color;               // 588  1 = per-voxel colour pass (experiment, docs/VOXEL_COLOR_EXPERIMENT.md) -> 592
};

// CaelixDebugView
static const uint kViewAlbedo = 0;
static const uint kViewNormal = 1;
static const uint kViewDepth = 2;
static const uint kViewGroup = 3;
static const uint kViewBrick = 4;
static const uint kViewSteps = 5;
static const uint kViewNrdValidation = 6;  // path trace mode: NRD's validation layer
static const uint kViewNrdDiff = 7;        // path trace mode: denoised diffuse radiance
static const uint kViewMotion = 8;         // path trace mode: the motion vector of the primary hit
static const uint kViewMaterial = 9;       // primary mode: the material table's albedo, tinted by kind
static const uint kViewNrdSpec = 10;       // path trace mode: denoised specular radiance

// CaelixUpscaler
static const uint kUpscalerNone = 0;
static const uint kUpscalerDlssSr = 1;
static const uint kUpscalerDlssRr = 2;

// CaelixRenderMode
static const uint kModePrimary = 0;
static const uint kModePathTrace = 1;
static const uint kModeReference = 2;

// CaelixRenderOptions::flags
static const uint kRenderFlagShading = 1u << 0;
static const uint kRenderFlagSkyOnMiss = 1u << 1;
static const uint kRenderFlagSunNee = 1u << 2;
static const uint kRenderFlagDeltaCheckerboard = 1u << 3;
static const uint kRenderFlagFlipY = 1u << 4;
static const uint kRenderFlagSubsurface = 1u << 5;
static const uint kRenderFlagSssEntryJitter = 1u << 6;

// CaelixFrameConstants::light_nee_flags (lib/src/render/frame_constants.h). Macros: a static const here
// would add constants to the four other frame-constant kernels, which never read them.
#define kLightNeeOn (1u << 0)
#define kLightNeeMis (1u << 1)

// CaelixFrameConstants::target_mask: bit i = the target at descriptor binding i + 1.
static const uint kTargetColor = 1u << 0;
static const uint kTargetAlbedo = 1u << 1;
static const uint kTargetNormal = 1u << 2;
static const uint kTargetDepth = 1u << 3;
static const uint kTargetMotion = 1u << 4;
static const uint kTargetViewZ = 1u << 5;
static const uint kTargetNormalUnorm = 1u << 16;  // the normal target stores n * 0.5 + 0.5

// The sample position of pixel `pixel` in pixels (row 0 = the top row, y down): the pixel centre
// plus the camera jitter (CaelixCamera::jitter; 0 for a host without jitter).
float2 CaelixSamplePosition(CaelixFrameConstants fc, uint2 pixel)
{
    return float2(float(pixel.x) + 0.5f + fc.jitter.x, float(pixel.y) + 0.5f + fc.jitter.y);
}

// The pixel of a HOST target that receives the result of pixel `pixel`: the same pixel, or with
// kRenderFlagFlipY the mirrored row (size.y - 1 - y). Internal targets are never flipped.
uint2 CaelixOutputPixel(CaelixFrameConstants fc, uint2 pixel)
{
    return (fc.flags & kRenderFlagFlipY) != 0u ? uint2(pixel.x, fc.size.y - 1u - pixel.y) : pixel;
}

// The primary ray direction of pixel `pixel` (row 0 = the top row), CaelixCamera's pinhole mapping
// at the jittered sample position.
float3 CaelixPrimaryDirection(CaelixFrameConstants fc, uint2 pixel)
{
    const float2 sample_position = CaelixSamplePosition(fc, pixel);
    const float2 ndc = float2(sample_position.x / float(fc.size.x) * 2.0f - 1.0f,
                              1.0f - sample_position.y / float(fc.size.y) * 2.0f);
    return normalize(fc.right.xyz * (ndc.x * fc.tan_half_fov_y * fc.aspect) +
                     fc.up.xyz * (ndc.y * fc.tan_half_fov_y) + fc.forward.xyz);
}

// The camera ray direction through the centre of OUTPUT pixel `pixel` (row 0 = the top row; the colour
// target's size output_size), without the jitter: the view that an upscaler's output shows.
float3 CaelixOutputCentreDirection(CaelixFrameConstants fc, uint2 pixel)
{
    const float2 ndc = float2((float(pixel.x) + 0.5f) / float(fc.output_size.x) * 2.0f - 1.0f,
                              1.0f - (float(pixel.y) + 0.5f) / float(fc.output_size.y) * 2.0f);
    return normalize(fc.right.xyz * (ndc.x * fc.tan_half_fov_y * fc.aspect) +
                     fc.up.xyz * (ndc.y * fc.tan_half_fov_y) + fc.forward.xyz);
}

#endif  // CAELIX_FRAME_HLSL
