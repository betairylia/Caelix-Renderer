#pragma once
#include "caelix/caelix.h"
#include "render/nrd_executor.h"
#include "render/render_kernels.h"
#include "render/upscaler.h"
#include "rhi/device.h"
#include "rhi/gpu_buffer.h"
#include "rhi/gpu_image.h"
#include "rhi/gpu_timer.h"
#include "rhi/pipeline.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace caelix {

class Scene;
class NgxRuntime;  // render/dlss_upscaler.h (CAELIX_DLSS builds)

// The light cell layout (render/caelix_light_sample.hlsl): kLightCellBytes per cell, then the
// usage counters (live, active, inserted, failed, pushed, push_dropped: six uints and 8 bytes of
// padding) at the buffer's tail.
constexpr VkDeviceSize kLightCellBytes = 832;
constexpr VkDeviceSize kLightCellCounterBytes = 32;
// Readback slots of the light cell counters (slot = frame_number % kLightCounterSlots).
constexpr uint32_t kLightCounterSlots = 8;

// Traces one scene into the host's targets (caelix_renderer_*).
//
// Several renderers (one per view) may share one scene. The immutable objects are
// per device (RenderKernels, borrowed): the descriptor set layouts, the compute
// pipelines, the sky sampler, the fallback sky cube and the NRD pipelines. Each
// renderer owns its per-view state: frame slots, cached views, internal targets, the
// NRD instance, the sky SH buffer, timers and the previous camera.
//
// Two kernels. The trace kernel (shaders/render/caelix_path_trace.hlsl) reads the
// scene's tables through buffer device addresses in the frame constants; the
// TLAS and the images are its only descriptors (set 0: binding 0 acceleration
// structure, bindings 1..6 the host targets colour, albedo, normal, depth,
// motion, view_z, bindings 7..18 the internal targets of the lighting modes,
// binding 19 the sky cubemap (sampled image), binding 20 its sampler; every
// binding PARTIALLY_BOUND, so absent targets and a missing TLAS are simply
// not written). In the lighting modes the composite kernel
// (shaders/render/caelix_composite.hlsl, its own set: binding 0 the host colour target,
// bindings 1..8 internal targets, 9 the upscale input, 10 the sky cubemap, 11 its sampler)
// writes the colour target. Upscaler (CaelixRenderOptions::upscaler, path trace mode with
// a colour target; renderer_upscale.cpp, docs/DLSS.md): the trace and the composite run at
// the render size, the composite writes the upscale input (internal, linear) instead of the
// colour target, the Upscaler (DlssUpscaler) writes upscale_out_ at the output size, and the
// encode kernel (shaders/render/caelix_upscale_encode.hlsl, a fourth set: binding 0 the
// host colour target, 1 the source, 2 the internal albedo) writes the colour target; after
// an NGX error the encode kernel reads the upscale input instead (nearest upscale). The
// upscaler and upscale_out_ are created at the first upscaled render and recreated (the
// old ones retired until safe) when any UpscalerDesc field changes. Sky cubemap
// (CaelixRenderOptions::sky, lighting modes): before the trace,
// the sky SH kernel (shaders/render/caelix_sky_sh.hlsl, one workgroup, a third set:
// binding 0 the cubemap, 1 the sampler) projects it onto 9 irradiance SH coefficients
// in sky_sh_buffer_ (one per renderer: frames execute in queue order, the lighting
// modes' first barrier orders the previous frame's reads before the write, a barrier
// after the dispatch orders the write before the trace); its time is part of the
// trace timestamps. The sampler is the device's (RenderKernels: linear, mipmapped,
// clamp to edge). Without a host cubemap every sky binding holds the device's fallback
// sky (a 1 x 1 cube in GENERAL, never sampled because sky_cubemap is 0). In the path
// trace mode with denoising, NRD (NrdExecutor, REBLUR_DIFFUSE_SPECULAR) runs between
// the two and the composite reads its outputs instead of the raw radiance; the
// executor is created at the first denoised render and recreated (history lost)
// when the size changes, the old one is destroyed once its last frame is safe.
// Per-frame resources
// live in a ring of slots (four descriptor sets + 720-byte constants buffer); a
// slot is reused only when its frame is safe. The camera of each successful
// render call is kept as the previous camera of the next one (motion targets). A
// renderer renders at most once per frame number (last_frame_).
//
// ReSTIR GI (CaelixGiOptions::method = CAELIX_GI_RESTIR, lighting modes with bounces;
// docs/GI_EXPERIMENTS.md): the trace kernel stores one initial reservoir per pixel, then
// caelix_restir_gi_temporal, up to two caelix_restir_gi_spatial passes and
// caelix_restir_gi_shade (all with the trace set and TracePush; gi_pass = the spatial pass)
// run between the trace and NRD / the composite, and the shade pass writes the diffuse
// radiance target. Four reservoir buffers (48 bytes per pixel each): initial, prev (the
// previous frame's final reservoirs), temp and out (ping-pong); after a ReSTIR frame the
// final one becomes prev. Created at the first ReSTIR render, recreated (old ones retired
// until safe) when the render size changes, kept while another method runs.
//
// Light NEE (CAELIX_GI_FLAG_LIGHT_NEE, lighting modes; docs/LIGHT_LIST.md): the renderer owns the
// light cells (CaelixGiOptions::light_cell_capacity x 832 bytes + 32 bytes of usage counters, trace
// set binding 21; the counters are reset before the fill and read back after the trace), created and
// zero-filled at the first render with the flag, recreated (the old buffer retired until safe) when
// the capacity changes, kept while the flag is off. When the scene has a light table, the light cell
// fill (caelix_light_cells, trace set, TracePush) runs before the sky SH kernel and the trace, with
// its own timer (CaelixGiStats::light_fill_ms); a compute -> compute barrier orders its writes before
// the trace, which reads the reservoirs and inserts new cells. frame_counter_ (the render count) is
// the cells' clock. With CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL the fill merges each reservoir's previous
// fill (FrameConstants::light_temporal_frames > 0); the history is dropped (0 for one fill) when the
// scene rebuilt its light table since the previous fill (light_cell_rebuilds_) or after a reset
// (light_cell_reset_).
//
// ReSTIR DI (CAELIX_GI_FLAG_RESTIR_DI with LIGHT_NEE, lighting modes with bounces and a light
// table; docs/RESTIR_DI.md): the trace kernel stores one initial DI reservoir per pixel, then
// caelix_restir_di_temporal, up to two caelix_restir_di_spatial passes and caelix_restir_di_shade run
// after the ReSTIR GI passes (when those run) and before NRD / the composite; the shade pass adds
// the direct light of the list lights to the diffuse radiance target. Four DI reservoir buffers
// (48 bytes per pixel each), rotated, created, retired and resized as the GI buffers (di_*). The DI
// history is also dropped when the scene's light table was rebuilt since this renderer's previous
// render (CaelixLightStats::rebuilds changed: entry indices may have moved). gi_timer_ brackets
// the GI and the DI passes.
//
// Internal targets (lighting modes only): created at the first lighting render,
// recreated when the host target size changes (the old images are destroyed once
// the frame that last used them is safe), transitioned to GENERAL once and kept
// there. Coverage is albedo.a.
class Renderer
{
public:
    // `kernels` must outlive the renderer. `ngx` returns the device's NGX runtime (created on
    // the first call; null when NGX does not start); unused without CAELIX_DLSS.
    static CaelixResult Create(Device& device, RenderKernels& kernels, Scene& scene,
                               std::function<NgxRuntime*()> ngx, std::unique_ptr<Renderer>& out);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    CaelixResult Render(const CaelixFrameDesc& frame, const CaelixRenderTargets& targets,
                        const CaelixCamera& camera, const CaelixRenderOptions& options);
    void ReleaseTargetViews();
    void GetTimings(CaelixGpuTimings& out) const;
    void GetNrdInfo(CaelixNrdInfo& out) const;
    void GetUpscalerInfo(CaelixUpscalerInfo& out) const;
    // caelix_renderer_set_gi_options: validates (INVALID_ARGUMENT, logged) and stores the options.
    CaelixResult SetGiOptions(const CaelixGiOptions& options);
    void GetGiStats(CaelixGiStats& out) const;

private:
    static constexpr uint32_t kInitialSlots = 3;
    static constexpr uint32_t kMaxSlots = 8;
    static constexpr size_t kMaxCachedViews = 16;
    // Internal targets of the lighting modes, in trace binding order (binding 7 + index;
    // from kInternalAccum on they are not in the trace set).
    enum Internal : uint32_t
    {
        kInternalViewZ,             // R32F     view z (miss 1e30); NRD IN_VIEWZ
        kInternalNormalRoughness,   // RGB10A2  NRD_FrontEnd_PackNormalAndRoughness; NRD IN_NORMAL_ROUGHNESS
        kInternalMotion,            // RGBA16F  motion px (xy), previous view z - view z (z); NRD IN_MV, DLSS MVs
        kInternalAlbedo,            // RGBA8    material albedo, a = coverage
        kInternalDeterministic,     // RGBA16F  primary emission
        kInternalDiff,              // RGBA16F  REBLUR-packed diffuse radiance + norm hit distance (IN_DIFF_*)
        kInternalSpec,              // RGBA16F  REBLUR-packed specular radiance + norm hit distance (IN_SPEC_*)
        kInternalDepth,             // R32F     device depth of the host projection (miss = miss_depth); DLSS depth
        kInternalRrDiffuseAlbedo,   // RGBA8    DLSS RR diffuse albedo (sky 0.5), written when upscaler = RR
        kInternalRrSpecularAlbedo,  // RGBA8    DLSS RR specular albedo (EnvBRDFApprox2; sky 0)
        kInternalRrNormalRoughness, // RGBA16F  DLSS RR world normal + linear roughness (sky 0)
        kInternalRrHitDistance,     // R16F     DLSS RR specular hit distance (sky 65504)
        kInternalAccum,             // RGBA32F  reference accumulation
        kInternalDiffOut,           // RGBA16F  NRD OUT_DIFF_RADIANCE_HITDIST
        kInternalSpecOut,           // RGBA16F  NRD OUT_SPEC_RADIANCE_HITDIST
        kInternalValidation,        // RGBA8    NRD OUT_VALIDATION
        kInternalResolve,           // RGBA16F  composite output before the cross resolve (linear)
        kInternalUpscaleIn,         // RGBA16F  the upscaler's colour input (composite or cross resolve, linear)
        kInternalCount
    };
    static_assert(kTraceInternalCount == kInternalAccum,
                  "the trace set binds the internal targets up to kInternalAccum");

    struct FrameSlot
    {
        uint32_t index = 0;  // position in slots_ (the NRD executor's frame slot)
        uint64_t frame = 0;  // last frame that used the slot (0 = never)
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;            // trace
        VkDescriptorSet composite_set = VK_NULL_HANDLE;  // composite
        VkDescriptorSet sky_set = VK_NULL_HANDLE;        // sky SH kernel
        VkDescriptorSet encode_set = VK_NULL_HANDLE;     // upscale encode kernel
        GpuBuffer constants;                             // host-visible, mapped, one FrameConstants
    };
    // A target's 2D storage view, or a sky cubemap's cube sampled view (cube = true, all
    // mips). The key is image + format + cube + mip_levels.
    struct CachedView
    {
        VkImage image = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        bool cube = false;
        uint32_t mip_levels = 1;
        VkImageView view = VK_NULL_HANDLE;
        uint64_t last_frame = 0;
    };
    struct FormatSupport
    {
        VkFormat format;
        bool supported;
    };
    using InternalImages = std::array<GpuImage, kInternalCount>;
    struct RetiredImages
    {
        uint64_t frame = 0;  // the last frame that used them
        InternalImages images;
    };
    struct RetiredDenoiser
    {
        uint64_t frame = 0;  // the last frame that used it
        std::unique_ptr<NrdExecutor> executor;
    };
    // ReSTIR GI reservoir buffers, indexed through gi_index_.
    enum GiBuffer : uint32_t
    {
        kGiInitial,  // the trace kernel's initial samples
        kGiPrev,     // the previous ReSTIR frame's final reservoirs
        kGiTemp,     // the temporal output, spatial ping-pong
        kGiOut,      // spatial ping-pong
        kGiBufferCount
    };
    using GiBuffers = std::array<GpuBuffer, kGiBufferCount>;
    struct RetiredGiBuffers
    {
        uint64_t frame = 0;  // the last frame that used them
        GiBuffers buffers;
    };
    struct RetiredLightCells
    {
        uint64_t frame = 0;  // the last frame that used it
        GpuBuffer buffer;
        GpuBuffer counter_readback;
    };
    struct RetiredUpscaler
    {
        uint64_t frame = 0;  // the last frame that used them
        std::unique_ptr<Upscaler> upscaler;
        GpuImage output;
    };
    // The newest caelix_upscaler_query answer of this renderer (render calls re-query only when
    // the key changes).
    struct UpscaleQuery
    {
        bool valid = false;
        uint32_t upscaler = 0;
        uint32_t quality = 0;
        uint32_t output_width = 0;
        uint32_t output_height = 0;
        CaelixUpscalerQueryResult result{};
    };

    Renderer(Device& device, RenderKernels& kernels, Scene& scene, std::function<NgxRuntime*()> ngx);

    void AddSlot();
    // A slot whose frame is safe (adds one while there are fewer than kMaxSlots),
    // or null.
    FrameSlot* AcquireSlot(uint64_t frame, uint64_t safe_frame);
    // The cached view of `target` (created on a miss); evicts the least recently
    // used view when the cache is over its size and that view's frame is safe.
    VkImageView ViewFor(const CaelixTargetImage& target, uint64_t frame, uint64_t safe_frame);
    // The cached cube view (sampled usage, every mip, 6 layers) of the host's sky cubemap.
    VkImageView SkyViewFor(const CaelixSkyCubemap& sky, uint64_t frame, uint64_t safe_frame);
    // The cache behind both: the view of (image, format, cube, mip_levels), created on a miss.
    VkImageView CachedViewFor(VkImage image, VkFormat format, bool cube, uint32_t mip_levels,
                              uint64_t frame, uint64_t safe_frame);
    bool FormatSupported(VkFormat format);
    // Creates the internal targets when they are missing or have another size; the
    // old ones are retired until `frame` is safe. Returns true when it created them.
    bool EnsureInternalTargets(uint32_t width, uint32_t height, uint64_t frame);
    // Creates the ReSTIR GI reservoir buffers when they are missing or of another size; the old
    // ones are retired until `frame` is safe and the history is dropped. Returns true when it
    // created them.
    bool EnsureGiBuffers(uint32_t width, uint32_t height, uint64_t frame);
    // The same for the ReSTIR DI reservoir buffers (48 bytes per pixel).
    bool EnsureDiBuffers(uint32_t width, uint32_t height, uint64_t frame);
    // Creates the light cells when they are missing or of another capacity; the old buffer is
    // retired until `frame` is safe. A new buffer is zero-filled by the next render
    // (light_cells_need_clear_).
    void EnsureLightCells(uint32_t capacity, uint64_t frame);
    // The device address of reservoir buffer `which` (GiBuffer) under the current rotation.
    uint64_t GiAddress(GiBuffer which) const
    {
        return gi_buffers_[gi_index_[which]].address();
    }
    // The device address of DI reservoir buffer `which` (same roles) under the DI rotation.
    uint64_t DiAddress(GiBuffer which) const
    {
        return di_buffers_[di_index_[which]].address();
    }
    // Destroys retired internal targets, reservoir buffers, NRD executors and upscalers whose
    // frame is <= safe_frame.
    void FreeRetired(uint64_t safe_frame);
    // The render size of `upscaler` at `quality` for an output_width x output_height colour target
    // (renderer_upscale.cpp): CAELIX_OK and the size, or CAELIX_ERROR_UNSUPPORTED (logged once per
    // query key) when the build, NGX or the GPU cannot run it.
    CaelixResult QueryUpscaler(uint32_t upscaler, uint32_t quality, uint32_t output_width,
                               uint32_t output_height, uint32_t& render_width, uint32_t& render_height);
    // The upscaler for `desc`, created (the old one and upscale_out_ retired) when missing or of
    // another desc; the feature creation is recorded into `cmd`. Null when it failed (NGX error:
    // upscaler_failed_, logged, `error` set on the call that failed; later calls with the same desc
    // return null without an error). Throws VulkanError when upscale_out_ cannot be allocated.
    Upscaler* EnsureUpscaler(const UpscalerDesc& desc, VkCommandBuffer cmd, uint64_t frame, bool& error);
    // The NRD executor for a width x height target, created (the old one retired)
    // when missing or of another size. Null after an NRD error (nrd_failed_, logged,
    // `error` set); throws VulkanError when its GPU objects cannot be created.
    NrdExecutor* EnsureDenoiser(uint32_t width, uint32_t height, bool& error);
    // Retires nrd_ (destroyed once its last frame is safe) and forgets its history.
    void RetireDenoiser();
    // nrd::CommonSettings and nrd::ReblurSettings of one frame.
    bool ConfigureDenoiser(NrdExecutor& executor, const CaelixCamera& camera,
                           const CaelixRenderOptions& options, uint32_t width, uint32_t height,
                           bool restart);

    Device& device_;
    RenderKernels& kernels_;   // the device's layouts, pipelines, sky sampler, fallback sky, NRD kernels
    Scene& scene_;
    GpuBuffer sky_sh_buffer_;  // 9 float4: the irradiance SH of the host cubemap
    std::vector<std::unique_ptr<FrameSlot>> slots_;
    uint32_t next_slot_ = 0;
    std::vector<CachedView> views_;
    std::vector<FormatSupport> formats_;
    GpuTimer timer_;            // trace dispatch
    GpuTimer composite_timer_;  // composite dispatch
    GpuTimer denoise_timer_;    // NRD dispatches
    GpuTimer upscale_timer_;    // the upscaler's evaluate and the encode dispatch
    CaelixCamera prev_camera_{};
    bool has_prev_camera_ = false;
    uint64_t last_frame_ = 0;    // frame number of the newest recorded render call
    bool has_rendered_ = false;  // last_frame_ is valid
    InternalImages internal_;
    uint64_t internal_last_frame_ = 0;  // the last frame that used internal_
    bool internal_needs_init_ = false;  // internal_ still in VK_IMAGE_LAYOUT_UNDEFINED
    std::vector<RetiredImages> retired_;
    uint32_t last_mode_ = UINT32_MAX;  // mode of the previous render call
    uint32_t accum_count_ = 0;         // reference frames accumulated since the last restart
    // NRD (path trace mode with denoising).
    std::unique_ptr<NrdExecutor> nrd_;
    std::vector<RetiredDenoiser> retired_nrd_;
    uint64_t nrd_last_frame_ = 0;     // the last frame that used nrd_
    bool nrd_failed_ = false;         // an NRD error happened: never denoise again
    bool nrd_clear_ = false;          // nrd_ is new: its next frame is CLEAR_AND_RESTART
    bool nrd_history_valid_ = false;  // the previous render call denoised with nrd_
    uint32_t nrd_frame_index_ = 0;    // CommonSettings::frameIndex
    std::chrono::steady_clock::time_point nrd_last_time_{};
    // Upscaler (path trace mode with a colour target and CaelixRenderOptions::upscaler).
#if CAELIX_DLSS
    std::function<NgxRuntime*()> ngx_;  // the device's NGX runtime (CaelixDevice, created lazily)
#endif
    UpscaleQuery upscale_query_;
    std::unique_ptr<Upscaler> upscaler_;  // null when none or after a failed creation
    GpuImage upscale_out_;                // RGBA16F, output size: the upscaler's output
    bool upscale_out_needs_init_ = false;  // upscale_out_ still in VK_IMAGE_LAYOUT_UNDEFINED
    UpscalerDesc upscaler_desc_{};         // the desc of upscaler_ (also after a failed creation)
    bool has_upscaler_desc_ = false;       // upscaler_desc_ is valid (an upscaled render happened)
    bool upscaler_failed_ = false;         // NGX error with upscaler_desc_: the fallback until the desc changes
    bool upscaler_new_ = false;            // upscaler_ is new: its first evaluate resets the history
    uint64_t upscaler_last_frame_ = 0;     // the last frame that used upscaler_ / upscale_out_
    std::vector<RetiredUpscaler> retired_upscalers_;
    std::chrono::steady_clock::time_point upscale_last_time_{};
    bool upscale_unsupported_logged_ = false;  // the build has no upscaler: logged once
    // ReSTIR GI (lighting modes with CaelixGiOptions::method = CAELIX_GI_RESTIR).
    CaelixGiOptions gi_options_{};       // zero = the path tracer
    bool gi_history_valid_ = false;      // the buffer at gi_index_[kGiPrev] holds the previous frame's reservoirs
    GiBuffers gi_buffers_;               // created at the first ReSTIR render
    std::array<uint32_t, kGiBufferCount> gi_index_{kGiInitial, kGiPrev, kGiTemp, kGiOut};  // role -> buffer
    uint32_t gi_width_ = 0;              // size of gi_buffers_
    uint32_t gi_height_ = 0;
    uint64_t gi_reservoir_bytes_ = 0;    // bytes of gi_buffers_
    uint64_t gi_last_frame_ = 0;         // the last frame that used gi_buffers_
    std::vector<RetiredGiBuffers> retired_gi_;
    GpuTimer gi_timer_;                  // temporal + spatial + shade dispatches (GI and DI)
    // ReSTIR DI (CAELIX_GI_FLAG_RESTIR_DI): the same roles as the GI buffers, 48 bytes per pixel.
    bool di_history_valid_ = false;      // the buffer at di_index_[kGiPrev] holds the previous frame's DI reservoirs
    GiBuffers di_buffers_;               // created at the first ReSTIR DI render
    std::array<uint32_t, kGiBufferCount> di_index_{kGiInitial, kGiPrev, kGiTemp, kGiOut};
    uint32_t di_width_ = 0;
    uint32_t di_height_ = 0;
    uint64_t di_reservoir_bytes_ = 0;
    uint64_t di_last_frame_ = 0;         // the last frame that used di_buffers_
    uint64_t di_light_rebuilds_ = 0;     // CaelixLightStats::rebuilds at the previous ReSTIR DI render
    std::vector<RetiredGiBuffers> retired_di_;
    // Light NEE (CAELIX_GI_FLAG_LIGHT_NEE).
    GpuBuffer light_cells_;                  // capacity x 832 bytes + 32 bytes of usage counters
                                             // (render/caelix_light_sample.hlsl)
    uint32_t light_cell_capacity_ = 0;       // cells of light_cells_
    // The usage counters of the light cells, copied after the trace of each light NEE frame into
    // slot frame_number % kLightCounterSlots (32 bytes each; persistently mapped), and the frame of
    // each slot (0 = none). Created and retired with light_cells_.
    GpuBuffer light_counter_readback_;
    std::array<uint64_t, kLightCounterSlots> light_counter_frames_{};
    bool light_cells_need_clear_ = false;    // light_cells_ is new: the next render zero-fills it
    uint64_t light_cells_last_frame_ = 0;    // the last frame that used light_cells_
    bool light_cell_reset_ = false;          // CaelixGiOptions::reset: the next fill drops the cell history
    uint64_t light_cell_rebuilds_ = 0;       // CaelixLightStats::rebuilds at the previous light cell fill
    std::vector<RetiredLightCells> retired_light_cells_;
    GpuTimer light_timer_;                   // the light cell fill
    uint32_t frame_counter_ = 0;             // render calls of this renderer (the cells' clock)
};

}  // namespace caelix
