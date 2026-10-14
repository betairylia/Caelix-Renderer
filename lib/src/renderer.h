#pragma once
#include "caelix/caelix.h"
#include "device.h"
#include "gpu_buffer.h"
#include "gpu_image.h"
#include "gpu_timer.h"
#include "pipeline.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace caelix {

class Scene;

// Traces one scene into the host's targets (caelix_renderer_*).
//
// Two kernels. The trace kernel (shaders/caelix_path_trace.hlsl) reads the
// scene's tables through buffer device addresses in the frame constants; the
// TLAS and the images are its only descriptors (set 0: binding 0 acceleration
// structure, bindings 1..6 the host targets colour, albedo, normal, depth,
// motion, view_z, bindings 7..13 the internal targets of the lighting modes;
// every binding PARTIALLY_BOUND, so absent targets and a missing TLAS are simply
// not written). In the lighting modes the composite kernel
// (shaders/caelix_composite.hlsl, its own set: binding 0 the host colour target,
// bindings 1..5 internal targets) writes the colour target. Per-frame resources
// live in a ring of slots (two descriptor sets + 512-byte constants buffer); a
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

 private:
  static constexpr uint32_t kInitialSlots = 3;
  static constexpr uint32_t kMaxSlots = 8;
  static constexpr size_t kMaxCachedViews = 16;
  static constexpr uint32_t kTargetCount = 6;  // descriptor bindings 1..6
  // Internal targets of the lighting modes, in trace binding order (binding 7 + index;
  // kInternalAccum is read by the composite only).
  enum Internal : uint32_t {
    kInternalViewZ,            // R32F       view z (miss 1e30)
    kInternalNormalRoughness,  // RGBA16F    world normal, roughness 1
    kInternalMotion,           // RGBA16F    motion px (xy), previous view z - view z (z)
    kInternalAlbedo,           // RGBA8      material albedo, a = coverage
    kInternalDeterministic,    // RGBA16F    primary emission
    kInternalDiff,             // RGBA16F    diffuse incident radiance, a = hit distance
    kInternalSpec,             // RGBA16F    specular incident radiance, a = hit distance
    kInternalAccum,            // RGBA32F    reference accumulation
    kInternalCount
  };
  static constexpr uint32_t kTraceInternalCount = kInternalAccum;  // bindings 7..13
  static constexpr uint32_t kCompositeBindingCount = 6;

  struct FrameSlot {
    uint64_t frame = 0;  // last frame that used the slot (0 = never)
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;            // trace
    VkDescriptorSet composite_set = VK_NULL_HANDLE;  // composite
    GpuBuffer constants;  // host-visible, mapped, one FrameConstants
  };
  struct CachedView {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
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

  Renderer(Device& device, Scene& scene);

  void AddSlot();
  // A slot whose frame is safe (adds one while there are fewer than kMaxSlots),
  // or null.
  FrameSlot* AcquireSlot(uint64_t frame, uint64_t safe_frame);
  // The cached view of `target` (created on a miss); evicts the least recently
  // used view when the cache is over its size and that view's frame is safe.
  VkImageView ViewFor(const CaelixTargetImage& target, uint64_t frame, uint64_t safe_frame);
  bool FormatSupported(VkFormat format);
  // Creates the internal targets when they are missing or have another size; the
  // old ones are retired until `frame` is safe. Returns true when it created them.
  bool EnsureInternalTargets(uint32_t width, uint32_t height, uint64_t frame);
  // Destroys retired internal targets whose frame is <= safe_frame.
  void FreeRetired(uint64_t safe_frame);

  Device& device_;
  Scene& scene_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout composite_layout_ = VK_NULL_HANDLE;
  ComputePipeline trace_;
  ComputePipeline composite_;
  std::vector<std::unique_ptr<FrameSlot>> slots_;
  uint32_t next_slot_ = 0;
  std::vector<CachedView> views_;
  std::vector<FormatSupport> formats_;
  GpuTimer timer_;            // trace dispatch
  GpuTimer composite_timer_;  // composite dispatch
  CaelixCamera prev_camera_{};
  bool has_prev_camera_ = false;
  InternalImages internal_;
  uint64_t internal_last_frame_ = 0;  // the last frame that used internal_
  bool internal_needs_init_ = false;  // internal_ still in VK_IMAGE_LAYOUT_UNDEFINED
  std::vector<RetiredImages> retired_;
  uint32_t last_mode_ = UINT32_MAX;   // mode of the previous render call
  uint32_t accum_count_ = 0;          // reference frames accumulated since the last restart
};

}  // namespace caelix
