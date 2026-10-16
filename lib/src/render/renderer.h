#pragma once
#include "caelix/caelix.h"
#include "render/nrd_executor.h"
#include "rhi/device.h"
#include "rhi/gpu_buffer.h"
#include "rhi/gpu_image.h"
#include "rhi/gpu_timer.h"
#include "rhi/pipeline.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace caelix {

class Scene;

// Traces one scene into the host's targets (caelix_renderer_*).
//
// Two kernels. The trace kernel (shaders/render/caelix_path_trace.hlsl) reads the
// scene's tables through buffer device addresses in the frame constants; the
// TLAS and the images are its only descriptors (set 0: binding 0 acceleration
// structure, bindings 1..6 the host targets colour, albedo, normal, depth,
// motion, view_z, bindings 7..13 the internal targets of the lighting modes,
// binding 14 the sky cubemap (sampled image), binding 15 its sampler; every
// binding PARTIALLY_BOUND, so absent targets and a missing TLAS are simply
// not written). In the lighting modes the composite kernel
// (shaders/render/caelix_composite.hlsl, its own set: binding 0 the host colour target,
// bindings 1..8 internal targets, 9 the sky cubemap, 10 its sampler) writes the colour
// target. Sky cubemap (CaelixRenderOptions::sky, lighting modes): before the trace,
// the sky SH kernel (shaders/render/caelix_sky_sh.hlsl, one workgroup, a third set:
// binding 0 the cubemap, 1 the sampler) projects it onto 9 irradiance SH coefficients
// in sky_sh_buffer_ (one per renderer: frames execute in queue order, the lighting
// modes' first barrier orders the previous frame's reads before the write, a barrier
// after the dispatch orders the write before the trace); its time is part of the
// trace timestamps. The sampler is the renderer's own (linear, mipmapped, clamp to
// edge). Without a host cubemap every sky binding holds fallback_sky_ (a 1 x 1 cube
// in GENERAL, never sampled because sky_cubemap is 0). In the path trace
// mode with denoising, NRD (NrdExecutor, REBLUR_DIFFUSE_SPECULAR) runs between
// the two and the composite reads its outputs instead of the raw radiance; the
// executor is created at the first denoised render and recreated (history lost)
// when the size changes, the old one is destroyed once its last frame is safe.
// Per-frame resources
// live in a ring of slots (three descriptor sets + 528-byte constants buffer); a
// slot is reused only when its frame is safe. The camera of each successful
// render call is kept as the previous camera of the next one (motion targets).
//
// Internal targets (lighting modes only): created at the first lighting render,
// recreated when the host target size changes (the old images are destroyed once
// the frame that last used them is safe), transitioned to GENERAL once and kept
// there. Coverage is albedo.a.
class Renderer {
 public:
  static CaelixResult Create(Device& device, Scene& scene, std::unique_ptr<Renderer>& out);
  ~Renderer();
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  CaelixResult Render(const CaelixFrameDesc& frame, const CaelixRenderTargets& targets,
                      const CaelixCamera& camera, const CaelixRenderOptions& options);
  void ReleaseTargetViews();
  void GetTimings(CaelixGpuTimings& out) const;
  void GetNrdInfo(CaelixNrdInfo& out) const;

 private:
  static constexpr uint32_t kInitialSlots = 3;
  static constexpr uint32_t kMaxSlots = 8;
  static constexpr size_t kMaxCachedViews = 16;
  static constexpr uint32_t kTargetCount = 6;  // descriptor bindings 1..6
  // Internal targets of the lighting modes, in trace binding order (binding 7 + index;
  // kInternalAccum is read by the composite only).
  enum Internal : uint32_t {
    kInternalViewZ,            // R32F       view z (miss 1e30); NRD IN_VIEWZ
    kInternalNormalRoughness,  // RGB10A2    NRD_FrontEnd_PackNormalAndRoughness; NRD IN_NORMAL_ROUGHNESS
    kInternalMotion,           // RGBA16F    motion px (xy), previous view z - view z (z); NRD IN_MV
    kInternalAlbedo,           // RGBA8      material albedo, a = coverage
    kInternalDeterministic,    // RGBA16F    primary emission
    kInternalDiff,             // RGBA16F    REBLUR-packed diffuse radiance + norm hit distance (IN_DIFF_*)
    kInternalSpec,             // RGBA16F    REBLUR-packed specular radiance + norm hit distance (IN_SPEC_*)
    kInternalAccum,            // RGBA32F    reference accumulation
    kInternalDiffOut,          // RGBA16F    NRD OUT_DIFF_RADIANCE_HITDIST
    kInternalSpecOut,          // RGBA16F    NRD OUT_SPEC_RADIANCE_HITDIST
    kInternalValidation,       // RGBA8      NRD OUT_VALIDATION
    kInternalResolve,          // RGBA16F    composite output before the cross resolve (linear)
    kInternalCount
  };
  static constexpr uint32_t kTraceInternalCount = kInternalAccum;  // bindings 7..13
  // Trace set: binding 14 the sky cubemap (sampled image), 15 its sampler.
  static constexpr uint32_t kTraceSkyBinding = 1 + kTargetCount + kTraceInternalCount;
  static constexpr uint32_t kTraceBindingCount = kTraceSkyBinding + 2;
  // Composite set (also bound by the cross resolve kernel): binding 0 the host colour
  // target, bindings 1..8 internal targets (8 = the resolve image), binding 9 the sky
  // cubemap (sampled image), 10 its sampler.
  static constexpr uint32_t kCompositeStorageCount = 9;  // bindings 0..8
  static constexpr uint32_t kCompositeSkyBinding = kCompositeStorageCount;
  static constexpr uint32_t kCompositeBindingCount = kCompositeSkyBinding + 2;

  struct FrameSlot {
    uint32_t index = 0;  // position in slots_ (the NRD executor's frame slot)
    uint64_t frame = 0;  // last frame that used the slot (0 = never)
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;            // trace
    VkDescriptorSet composite_set = VK_NULL_HANDLE;  // composite
    VkDescriptorSet sky_set = VK_NULL_HANDLE;        // sky SH kernel
    GpuBuffer constants;  // host-visible, mapped, one FrameConstants
  };
  // A target's 2D storage view, or a sky cubemap's cube sampled view (cube = true, all
  // mips). The key is image + format + cube + mip_levels.
  struct CachedView {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool cube = false;
    uint32_t mip_levels = 1;
    VkImageView view = VK_NULL_HANDLE;
    uint64_t last_frame = 0;
  };
  struct FormatSupport {
    VkFormat format;
    bool supported;
  };
  using InternalImages = std::array<GpuImage, kInternalCount>;
  struct RetiredImages {
    uint64_t frame = 0;  // the last frame that used them
    InternalImages images;
  };
  struct RetiredDenoiser {
    uint64_t frame = 0;  // the last frame that used it
    std::unique_ptr<NrdExecutor> executor;
  };

  Renderer(Device& device, Scene& scene);

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
  // Destroys retired internal targets and NRD executors whose frame is <= safe_frame.
  void FreeRetired(uint64_t safe_frame);
  // The NRD executor for a width x height target, created (the old one retired)
  // when missing or of another size. Null after an NRD error (nrd_failed_, logged,
  // `error` set); throws VulkanError when its GPU objects cannot be created.
  NrdExecutor* EnsureDenoiser(uint32_t width, uint32_t height, bool& error);
  // nrd::CommonSettings and nrd::ReblurSettings of one frame.
  bool ConfigureDenoiser(NrdExecutor& executor, const CaelixCamera& camera,
                         const CaelixRenderOptions& options, uint32_t width, uint32_t height,
                         bool restart);

  Device& device_;
  Scene& scene_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout composite_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout sky_layout_ = VK_NULL_HANDLE;  // sky SH kernel: cubemap, sampler
  ComputePipeline trace_;
  ComputePipeline composite_;
  ComputePipeline cross_resolve_;  // shaders/render/caelix_cross_resolve.hlsl, composite set
  ComputePipeline sky_sh_;         // shaders/render/caelix_sky_sh.hlsl, sky set
  VkSampler sky_sampler_ = VK_NULL_HANDLE;  // linear, mipmapped, clamp to edge
  GpuImage fallback_sky_;                   // 1 x 1 RGBA8 cube bound without a host cubemap
  bool fallback_sky_needs_init_ = true;     // fallback_sky_ still in VK_IMAGE_LAYOUT_UNDEFINED
  GpuBuffer sky_sh_buffer_;                 // 9 float4: the irradiance SH of the host cubemap
  std::vector<std::unique_ptr<FrameSlot>> slots_;
  uint32_t next_slot_ = 0;
  std::vector<CachedView> views_;
  std::vector<FormatSupport> formats_;
  GpuTimer timer_;            // trace dispatch
  GpuTimer composite_timer_;  // composite dispatch
  GpuTimer denoise_timer_;    // NRD dispatches
  CaelixCamera prev_camera_{};
  bool has_prev_camera_ = false;
  InternalImages internal_;
  uint64_t internal_last_frame_ = 0;  // the last frame that used internal_
  bool internal_needs_init_ = false;  // internal_ still in VK_IMAGE_LAYOUT_UNDEFINED
  std::vector<RetiredImages> retired_;
  uint32_t last_mode_ = UINT32_MAX;   // mode of the previous render call
  uint32_t accum_count_ = 0;          // reference frames accumulated since the last restart
  // NRD (path trace mode with denoising).
  std::unique_ptr<NrdExecutor> nrd_;
  std::vector<RetiredDenoiser> retired_nrd_;
  uint64_t nrd_last_frame_ = 0;       // the last frame that used nrd_
  bool nrd_failed_ = false;           // an NRD error happened: never denoise again
  bool nrd_clear_ = false;            // nrd_ is new: its next frame is CLEAR_AND_RESTART
  bool nrd_history_valid_ = false;    // the previous render call denoised with nrd_
  uint32_t nrd_frame_index_ = 0;      // CommonSettings::frameIndex
  std::chrono::steady_clock::time_point nrd_last_time_{};
};

}  // namespace caelix
