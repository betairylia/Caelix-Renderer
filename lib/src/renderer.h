#pragma once
#include "caelix/caelix.h"
#include "device.h"
#include "gpu_buffer.h"
#include "gpu_timer.h"
#include "pipeline.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace caelix {

class Scene;

// Traces one scene into the host's targets (caelix_renderer_*).
//
// The trace kernel (shaders/caelix_trace_primary.hlsl) reads the scene's tables
// through buffer device addresses in the frame constants; the TLAS and the
// targets are the only descriptors (set 0: binding 0 acceleration structure,
// bindings 1..6 storage images colour, albedo, normal, depth, motion, view_z;
// every binding PARTIALLY_BOUND, so absent targets and a missing TLAS are simply
// not written). Per-frame resources live in a ring of slots (descriptor set +
// 512-byte constants buffer); a slot is reused only when its frame is safe. The
// camera of each successful render call is kept as the previous camera of the
// next one (motion target).
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

  struct FrameSlot {
    uint64_t frame = 0;  // last frame that used the slot (0 = never)
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
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

  Renderer(Device& device, Scene& scene);

  void AddSlot();
  // A slot whose frame is safe (adds one while there are fewer than kMaxSlots),
  // or null.
  FrameSlot* AcquireSlot(uint64_t frame, uint64_t safe_frame);
  // The cached view of `target` (created on a miss); evicts the least recently
  // used view when the cache is over its size and that view's frame is safe.
  VkImageView ViewFor(const CaelixTargetImage& target, uint64_t frame, uint64_t safe_frame);
  bool FormatSupported(VkFormat format);

  Device& device_;
  Scene& scene_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  ComputePipeline trace_;
  std::vector<std::unique_ptr<FrameSlot>> slots_;
  uint32_t next_slot_ = 0;
  std::vector<CachedView> views_;
  std::vector<FormatSupport> formats_;
  GpuTimer timer_;
  CaelixCamera prev_camera_{};
  bool has_prev_camera_ = false;
};

}  // namespace caelix
