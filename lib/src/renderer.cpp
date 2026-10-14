#include "renderer.h"

#include "frame_constants.h"
#include "log.h"
#include "scene.h"
#include "shader_loader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <span>

namespace caelix {
namespace {

// [numthreads(8, 8, 1)] in caelix_path_trace.hlsl and caelix_composite.hlsl.
constexpr uint32_t kWorkgroupSize = 8;
constexpr uint32_t kDebugViewCount = 8;  // CAELIX_VIEW_ALBEDO .. CAELIX_VIEW_NRD_DIFF
constexpr uint32_t kModeCount = 3;
constexpr float kDefaultSunAngularRadius = 0.00873f;  // 0.5 degrees

const VkImageSubresourceRange kColorRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

// The renderer's one NRD denoiser (REBLUR_DIFFUSE_SPECULAR).
constexpr nrd::Identifier kReblurIdentifier = 1;
// CommonSettings::denoisingRange: pixels with a larger view z are not denoised.
// The worlds are at most 8192 units across (the 8K citadel), so 1e6 keeps every
// hit in range, and the miss view z (1e30) far out of it.
constexpr float kDenoisingRange = 1e6f;

// REBLUR's hit distance normalization (A, B, C), NRD's defaults (3, 0.1, 20): the
// trace kernel's REBLUR_FrontEnd_GetNormHitDist and ReblurSettings use the same.
nrd::ReblurHitDistanceParameters HitDistanceParameters() { return nrd::ReblurHitDistanceParameters{}; }

// NRD's world-to-view matrix of `camera`: column-major, column vectors
// (view = M * world), rows right, up, forward (the camera's view space, +z forward,
// left-handed like the projection), translation -dot(axis, position).
void WorldToView(const CaelixCamera& camera, float m[16]) {
  const float* axes[3] = {camera.right, camera.up, camera.forward};
  for (uint32_t row = 0; row < 3; ++row) {
    const float* axis = axes[row];
    m[0 * 4 + row] = axis[0];
    m[1 * 4 + row] = axis[1];
    m[2 * 4 + row] = axis[2];
    m[3 * 4 + row] = -(axis[0] * camera.position[0] + axis[1] * camera.position[1] +
                       axis[2] * camera.position[2]);
  }
  m[3] = 0.0f;
  m[7] = 0.0f;
  m[11] = 0.0f;
  m[15] = 1.0f;
}

// The targets of CaelixRenderTargets in descriptor binding order (binding = index + 1).
// `formats` lists the accepted formats; all VK_FORMAT_UNDEFINED = any storage format.
struct TargetSpec {
  const char* name;
  uint32_t bit;
  VkFormat formats[2];
};
constexpr TargetSpec kTargetSpecs[] = {
    {"colour", kTargetColor, {VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED}},
    {"albedo", kTargetAlbedo, {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT}},
    {"normal", kTargetNormal, {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM}},
    {"depth", kTargetDepth, {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SFLOAT}},
    {"motion", kTargetMotion, {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32G32_SFLOAT}},
    {"view_z", kTargetViewZ, {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SFLOAT}},
};

// The internal targets in Renderer::Internal order. The formats match the
// image_format qualifiers of the kernels.
struct InternalSpec {
  const char* name;
  VkFormat format;
};
constexpr InternalSpec kInternalSpecs[] = {
    {"view_z", VK_FORMAT_R32_SFLOAT},
    {"normal_roughness", VK_FORMAT_A2B10G10R10_UNORM_PACK32},
    {"motion", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"albedo", VK_FORMAT_R8G8B8A8_UNORM},
    {"deterministic", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"diff_in", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"spec_in", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"accum", VK_FORMAT_R32G32B32A32_SFLOAT},
    {"diff_out", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"spec_out", VK_FORMAT_R16G16B16A16_SFLOAT},
    {"validation", VK_FORMAT_R8G8B8A8_UNORM},
};

bool FormatAccepted(const TargetSpec& spec, VkFormat format) {
  return spec.formats[0] == VK_FORMAT_UNDEFINED || format == spec.formats[0] ||
         format == spec.formats[1];
}

// UNORM colour formats: the composite writes sRGB-encoded values into them.
bool IsUnormFormat(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16G16B16A16_UNORM:
      return true;
    default:
      return false;
  }
}

// The barriers of one vkCmdPipelineBarrier2: image barriers (any layout -> GENERAL)
// and at most one global memory barrier.
class Barriers {
 public:
  void Image(VkImage image, VkImageLayout old_layout, VkPipelineStageFlags2 src_stage,
             VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2& barrier = images_[image_count_++];
    barrier = VkImageMemoryBarrier2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = kColorRange;
  }
  void Memory(VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
              VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    memory_ = VkMemoryBarrier2{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory_.srcStageMask = src_stage;
    memory_.srcAccessMask = src_access;
    memory_.dstStageMask = dst_stage;
    memory_.dstAccessMask = dst_access;
    has_memory_ = true;
  }
  void Record(VkCommandBuffer cmd) const {
    if (image_count_ == 0 && !has_memory_) {
      return;
    }
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = has_memory_ ? 1u : 0u;
    dependency.pMemoryBarriers = &memory_;
    dependency.imageMemoryBarrierCount = image_count_;
    dependency.pImageMemoryBarriers = images_;
    vkCmdPipelineBarrier2(cmd, &dependency);
  }

 private:
  VkImageMemoryBarrier2 images_[std::size(kTargetSpecs) + std::size(kInternalSpecs)] = {};
  uint32_t image_count_ = 0;
  VkMemoryBarrier2 memory_{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  bool has_memory_ = false;
};

void Copy3(const float in[3], float out[4]) {
  out[0] = in[0];
  out[1] = in[1];
  out[2] = in[2];
  out[3] = 0.0f;
}

// A descriptor set layout of `count` bindings, every one PARTIALLY_BOUND, compute
// stage. `types[i]` is the type of binding i.
VkDescriptorSetLayout CreateSetLayout(VkDevice device, uint32_t count,
                                      const VkDescriptorType* types) {
  VkDescriptorSetLayoutBinding bindings[16];
  VkDescriptorBindingFlags binding_flags[16];
  for (uint32_t i = 0; i < count; ++i) {
    bindings[i] = {i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    binding_flags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
  }
  VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
  flags_info.bindingCount = count;
  flags_info.pBindingFlags = binding_flags;
  VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layout_info.pNext = &flags_info;
  layout_info.bindingCount = count;
  layout_info.pBindings = bindings;
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  const VkResult result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &layout);
  if (result != VK_SUCCESS) {
    CAELIX_LOGE("vkCreateDescriptorSetLayout failed: %d", static_cast<int>(result));
    throw VulkanError("vkCreateDescriptorSetLayout failed", result);
  }
  return layout;
}

}  // namespace

Renderer::Renderer(Device& device, Scene& scene) : device_(device), scene_(scene) {}

CaelixResult Renderer::Create(Device& device, Scene& scene, std::unique_ptr<Renderer>& out) {
  out.reset();
  std::unique_ptr<Renderer> renderer(new Renderer(device, scene));
  const VkDevice vk_device = device.device();
  static_assert(std::size(kTargetSpecs) == kTargetCount);
  static_assert(std::size(kInternalSpecs) == kInternalCount);

  // Trace set: binding 0 the TLAS, bindings 1..6 the host targets, 7..13 the
  // internal targets. Every binding is PARTIALLY_BOUND (descriptorBindingPartiallyBound,
  // a device requirement): a render call writes only the TLAS when there is one,
  // the targets that are present and the internal targets in the lighting modes,
  // and the kernel touches only those.
  constexpr uint32_t kTraceBindingCount = 1 + kTargetCount + kTraceInternalCount;
  VkDescriptorType trace_types[kTraceBindingCount];
  std::fill(trace_types, trace_types + kTraceBindingCount, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
  trace_types[0] = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  renderer->set_layout_ = CreateSetLayout(vk_device, kTraceBindingCount, trace_types);
  // Composite set: binding 0 the host colour target, 1..5 internal targets.
  VkDescriptorType composite_types[kCompositeBindingCount];
  std::fill(composite_types, composite_types + kCompositeBindingCount,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
  renderer->composite_layout_ = CreateSetLayout(vk_device, kCompositeBindingCount, composite_types);

  const std::vector<uint32_t> trace_spirv = LoadShader("caelix_path_trace");
  renderer->trace_ = ComputePipeline(vk_device, trace_spirv, sizeof(TracePush), "caelix_path_trace",
                                     renderer->set_layout_);
  const std::vector<uint32_t> composite_spirv = LoadShader("caelix_composite");
  renderer->composite_ = ComputePipeline(vk_device, composite_spirv, sizeof(TracePush),
                                         "caelix_composite", renderer->composite_layout_);
  for (uint32_t i = 0; i < kInitialSlots; ++i) {
    renderer->AddSlot();
  }
  renderer->timer_ = GpuTimer(device);
  renderer->composite_timer_ = GpuTimer(device);
  renderer->denoise_timer_ = GpuTimer(device);
  CAELIX_LOGI("renderer created: kernels caelix_path_trace and caelix_composite, %u frame slots",
              kInitialSlots);
  out = std::move(renderer);
  return CAELIX_OK;
}

Renderer::~Renderer() {
  const VkDevice device = device_.device();
  ReleaseTargetViews();
  for (const std::unique_ptr<FrameSlot>& slot : slots_) {
    if (slot->pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device, slot->pool, nullptr);  // frees the sets too
    }
  }
  slots_.clear();
  trace_ = ComputePipeline();
  composite_ = ComputePipeline();
  if (set_layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);
  }
  if (composite_layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, composite_layout_, nullptr);
  }
  // The device is idle (caelix_renderer_destroy contract): internal_ and retired_
  // are destroyed with the members.
}

void Renderer::AddSlot() {
  const VkDevice device = device_.device();
  auto slot = std::make_unique<FrameSlot>();
  slot->index = static_cast<uint32_t>(slots_.size());
  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
       kTargetCount + kTraceInternalCount + kCompositeBindingCount},
  };
  VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.maxSets = 2;
  pool_info.poolSizeCount = 2;
  pool_info.pPoolSizes = sizes;
  VkResult result = vkCreateDescriptorPool(device, &pool_info, nullptr, &slot->pool);
  if (result != VK_SUCCESS) {
    slot->pool = VK_NULL_HANDLE;
    CAELIX_LOGE("vkCreateDescriptorPool failed: %d", static_cast<int>(result));
    throw VulkanError("vkCreateDescriptorPool failed", result);
  }
  // Pushed now so that the destructor destroys the pool if a later call throws.
  slots_.push_back(std::move(slot));
  FrameSlot& added = *slots_.back();
  const VkDescriptorSetLayout layouts[2] = {set_layout_, composite_layout_};
  VkDescriptorSet sets[2] = {};
  VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  set_info.descriptorPool = added.pool;
  set_info.descriptorSetCount = 2;
  set_info.pSetLayouts = layouts;
  result = vkAllocateDescriptorSets(device, &set_info, sets);
  if (result != VK_SUCCESS) {
    CAELIX_LOGE("vkAllocateDescriptorSets failed: %d", static_cast<int>(result));
    throw VulkanError("vkAllocateDescriptorSets failed", result);
  }
  added.set = sets[0];
  added.composite_set = sets[1];
  VmaAllocationCreateInfo allocation_info{};
  allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
  allocation_info.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  added.constants = GpuBuffer(device_.allocator(), device, sizeof(FrameConstants),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              allocation_info, true);
}

Renderer::FrameSlot* Renderer::AcquireSlot(uint64_t frame, uint64_t safe_frame) {
  const uint32_t count = static_cast<uint32_t>(slots_.size());
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t index = (next_slot_ + i) % count;
    FrameSlot& slot = *slots_[index];
    if (slot.frame <= safe_frame) {
      next_slot_ = (index + 1) % count;
      slot.frame = frame;
      return &slot;
    }
  }
  if (count >= kMaxSlots) {
    return nullptr;
  }
  AddSlot();
  CAELIX_LOGI("renderer: %zu frame slots (more frames in flight)", slots_.size());
  FrameSlot& slot = *slots_.back();
  slot.frame = frame;
  next_slot_ = 0;
  return &slot;
}

bool Renderer::FormatSupported(VkFormat format) {
  for (const FormatSupport& known : formats_) {
    if (known.format == format) {
      return known.supported;
    }
  }
  VkFormatProperties3 properties3{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
  VkFormatProperties2 properties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
  properties.pNext = &properties3;
  vkGetPhysicalDeviceFormatProperties2(device_.physical_device(), format, &properties);
  const VkFormatFeatureFlags2 needed =
      VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
  const bool supported = (properties3.optimalTilingFeatures & needed) == needed;
  formats_.push_back(FormatSupport{format, supported});
  if (!supported) {
    CAELIX_LOGE("caelix_renderer_render: VkFormat %d does not support storage image writes "
                "without a format qualifier (optimal tiling)",
                static_cast<int>(format));
  }
  return supported;
}

VkImageView Renderer::ViewFor(const CaelixTargetImage& target, uint64_t frame,
                              uint64_t safe_frame) {
  const VkImage image = HandleFromU64<VkImage>(target.image);
  const VkFormat format = static_cast<VkFormat>(target.format);
  for (CachedView& cached : views_) {
    if (cached.image == image && cached.format == format) {
      cached.last_frame = frame;
      return cached.view;
    }
  }
  VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  view_info.subresourceRange = kColorRange;
  // The view is used as a storage image only (the image may have other usages
  // whose format support differs, for example sRGB attachments).
  VkImageViewUsageCreateInfo usage_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
  usage_info.usage = VK_IMAGE_USAGE_STORAGE_BIT;
  view_info.pNext = &usage_info;
  VkImageView view = VK_NULL_HANDLE;
  const VkResult result = vkCreateImageView(device_.device(), &view_info, nullptr, &view);
  if (result != VK_SUCCESS) {
    CAELIX_LOGE("vkCreateImageView failed: %d", static_cast<int>(result));
    throw VulkanError("vkCreateImageView failed", result);
  }
  views_.push_back(CachedView{image, format, view, frame});
  CAELIX_LOGD("renderer: image view created for image 0x%llx (%zu cached)",
              static_cast<unsigned long long>(target.image), views_.size());

  // Over the cap: destroy the least recently used view whose last frame is safe.
  // When every other view is still in flight, the cache stays larger for now.
  if (views_.size() > kMaxCachedViews) {
    size_t oldest = views_.size();
    for (size_t i = 0; i + 1 < views_.size(); ++i) {  // never the view just created
      if (views_[i].last_frame <= safe_frame &&
          (oldest == views_.size() || views_[i].last_frame < views_[oldest].last_frame)) {
        oldest = i;
      }
    }
    if (oldest < views_.size()) {
      vkDestroyImageView(device_.device(), views_[oldest].view, nullptr);
      views_.erase(views_.begin() + static_cast<std::ptrdiff_t>(oldest));
    }
  }
  return view;
}

void Renderer::ReleaseTargetViews() {
  for (const CachedView& cached : views_) {
    vkDestroyImageView(device_.device(), cached.view, nullptr);
  }
  views_.clear();
}

bool Renderer::EnsureInternalTargets(uint32_t width, uint32_t height, uint64_t frame) {
  GpuImage& first = internal_[0];
  if (first.valid() && first.width() == width && first.height() == height) {
    return false;
  }
  if (first.valid()) {
    RetiredImages retired;
    retired.frame = internal_last_frame_;
    retired.images = std::move(internal_);
    retired_.push_back(std::move(retired));
  }
  // On a throw the images created so far are destroyed with `fresh`, and the next
  // lighting render tries again.
  InternalImages fresh;
  for (uint32_t i = 0; i < kInternalCount; ++i) {
    fresh[i] = GpuImage(device_.allocator(), device_.device(), width, height,
                        kInternalSpecs[i].format,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        kInternalSpecs[i].name);
  }
  internal_ = std::move(fresh);
  internal_last_frame_ = frame;
  internal_needs_init_ = true;
  CAELIX_LOGI("renderer: internal targets created at %ux%u", width, height);
  return true;
}

void Renderer::FreeRetired(uint64_t safe_frame) {
  std::erase_if(retired_, [safe_frame](const RetiredImages& retired) {
    return retired.frame <= safe_frame;
  });
  std::erase_if(retired_nrd_, [safe_frame](const RetiredDenoiser& retired) {
    return retired.frame <= safe_frame;
  });
}

CaelixResult Renderer::Render(const CaelixFrameDesc& frame, const CaelixRenderTargets& targets,
                              const CaelixCamera& camera, const CaelixRenderOptions& options) {
  constexpr const char* kFunction = "caelix_renderer_render";
  if (frame.command_buffer == 0) {
    CAELIX_LOGE("%s: the frame's command buffer is null", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (frame.safe_frame_number >= frame.frame_number) {
    CAELIX_LOGE("%s: safe_frame_number %llu must be below frame_number %llu", kFunction,
                static_cast<unsigned long long>(frame.safe_frame_number),
                static_cast<unsigned long long>(frame.frame_number));
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (options.debug_view >= kDebugViewCount) {
    CAELIX_LOGE("%s: unknown debug view %u", kFunction, options.debug_view);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (options.mode >= kModeCount) {
    CAELIX_LOGE("%s: unknown mode %u", kFunction, options.mode);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (options.bounce_count > CAELIX_MAX_BOUNCES) {
    CAELIX_LOGE("%s: bounce_count %u is above the maximum %u", kFunction, options.bounce_count,
                CAELIX_MAX_BOUNCES);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  const bool lighting = options.mode != CAELIX_MODE_PRIMARY;

  // The present targets (image != 0), in binding order.
  const CaelixTargetImage* const all[kTargetCount] = {&targets.color,  &targets.albedo,
                                                      &targets.normal, &targets.depth,
                                                      &targets.motion, &targets.view_z};
  uint32_t present[kTargetCount];  // indices into `all`
  uint32_t present_count = 0;
  uint32_t mask = 0;
  for (uint32_t i = 0; i < kTargetCount; ++i) {
    const CaelixTargetImage& target = *all[i];
    const TargetSpec& spec = kTargetSpecs[i];
    if (target.image == 0) {
      continue;
    }
    if (target.width == 0 || target.height == 0 || target.format == VK_FORMAT_UNDEFINED) {
      CAELIX_LOGE("%s: the %s target needs a format and a nonzero size", kFunction, spec.name);
      return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (present_count > 0) {
      const CaelixTargetImage& first = *all[present[0]];
      if (target.width != first.width || target.height != first.height) {
        CAELIX_LOGE("%s: the %s target is %ux%u, the %s target %ux%u; every target must have the "
                    "same size",
                    kFunction, spec.name, target.width, target.height,
                    kTargetSpecs[present[0]].name, first.width, first.height);
        return CAELIX_ERROR_INVALID_ARGUMENT;
      }
    }
    for (uint32_t j = 0; j < present_count; ++j) {
      if (all[present[j]]->image == target.image) {
        CAELIX_LOGE("%s: the %s and %s targets are the same image", kFunction,
                    kTargetSpecs[present[j]].name, spec.name);
        return CAELIX_ERROR_INVALID_ARGUMENT;
      }
    }
    if (!FormatAccepted(spec, static_cast<VkFormat>(target.format))) {
      CAELIX_LOGE("%s: VkFormat %u is not an accepted format of the %s target (see "
                  "CaelixRenderTargets)",
                  kFunction, target.format, spec.name);
      return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    present[present_count++] = i;
    mask |= spec.bit;
  }
  if (present_count == 0) {
    CAELIX_LOGE("%s: no target is present (every target image is 0)", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  for (uint32_t p = 0; p < present_count; ++p) {
    if (!FormatSupported(static_cast<VkFormat>(all[present[p]]->format))) {
      return CAELIX_ERROR_UNSUPPORTED;
    }
  }
  if (targets.normal.image != 0 && targets.normal.format == VK_FORMAT_R8G8B8A8_UNORM) {
    mask |= kTargetNormalUnorm;
  }
  const uint32_t width = all[present[0]]->width;
  const uint32_t height = all[present[0]]->height;
  const bool has_color = targets.color.image != 0;
  const bool composite = lighting && has_color;

  const VkCommandBuffer cmd = HandleFromU64<VkCommandBuffer>(frame.command_buffer);
  const uint64_t safe = frame.safe_frame_number;
  timer_.Resolve(safe);
  composite_timer_.Resolve(safe);
  denoise_timer_.Resolve(safe);
  FreeRetired(safe);

  // Internal targets of the lighting modes and the NRD executor (before anything is
  // recorded: the allocations can throw).
  bool restart_accumulation = options.reset_accumulation != 0 || options.mode != last_mode_;
  if (lighting && EnsureInternalTargets(width, height, frame.frame_number)) {
    restart_accumulation = true;
  }
  bool nrd_error = false;
  NrdExecutor* denoiser = nullptr;
  if (options.mode == CAELIX_MODE_PATH_TRACE && options.denoise != 0 && composite && !nrd_failed_) {
    denoiser = EnsureDenoiser(width, height, nrd_error);
  }

  FrameSlot* slot = AcquireSlot(frame.frame_number, safe);
  if (slot == nullptr) {
    CAELIX_LOGE("%s: more than %u frames in flight (safe frame %llu, frame %llu)", kFunction,
                kMaxSlots, static_cast<unsigned long long>(safe),
                static_cast<unsigned long long>(frame.frame_number));
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }

  // NRD settings of this frame (an NRD error here: no denoising, raw output).
  if (denoiser != nullptr) {
    const bool restart = options.reset_accumulation != 0 || !nrd_history_valid_;
    if (!ConfigureDenoiser(*denoiser, camera, options, width, height, restart)) {
      nrd_failed_ = true;
      nrd_error = true;
      denoiser = nullptr;
    }
  }

  // Descriptors: the slot's previous frame has completed, so its sets are free.
  // Only the TLAS (when the scene has one), the present targets and, in the
  // lighting modes, the internal targets are written; the other bindings keep
  // stale or no descriptors, which PARTIALLY_BOUND allows because the kernels do
  // not touch them (has_tlas, target_mask, mode). The composite set is written
  // after the NRD dispatches are recorded: it reads NRD's outputs only when they
  // were.
  const VkAccelerationStructureKHR tlas = scene_.tlas();
  VkWriteDescriptorSetAccelerationStructureKHR as_write{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  as_write.accelerationStructureCount = 1;
  as_write.pAccelerationStructures = &tlas;
  constexpr uint32_t kMaxWrites =
      1 + kTargetCount + kTraceInternalCount + kCompositeBindingCount;
  VkDescriptorImageInfo image_infos[kMaxWrites];
  VkWriteDescriptorSet writes[kMaxWrites];
  uint32_t write_count = 0;
  const auto write_image = [&](VkDescriptorSet set, uint32_t binding, VkImageView view) {
    image_infos[write_count] = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet& write = writes[write_count];
    write = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &image_infos[write_count];
    ++write_count;
  };
  if (tlas != VK_NULL_HANDLE) {
    VkWriteDescriptorSet& write = writes[write_count++];
    write = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.pNext = &as_write;
    write.dstSet = slot->set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  }
  VkImage images[kTargetCount];
  VkImageLayout layouts[kTargetCount];
  VkImageView color_view = VK_NULL_HANDLE;
  for (uint32_t p = 0; p < present_count; ++p) {
    const CaelixTargetImage& target = *all[present[p]];
    images[p] = HandleFromU64<VkImage>(target.image);
    layouts[p] = static_cast<VkImageLayout>(target.layout);
    const VkImageView view = ViewFor(target, frame.frame_number, safe);
    if (present[p] == 0) {
      color_view = view;
    }
    write_image(slot->set, 1 + present[p], view);
  }
  if (lighting) {
    for (uint32_t i = 0; i < kTraceInternalCount; ++i) {
      write_image(slot->set, 1 + kTargetCount + i, internal_[i].view());
    }
  }
  vkUpdateDescriptorSets(device_.device(), write_count, writes, 0, nullptr);

  // Frame constants (written to the slot's buffer below, before the submit).
  FrameConstants constants{};
  Copy3(camera.position, constants.position);
  Copy3(camera.right, constants.right);
  Copy3(camera.up, constants.up);
  Copy3(camera.forward, constants.forward);
  constants.size[0] = width;
  constants.size[1] = height;
  constants.tan_half_fov_y = camera.tan_half_fov_y;
  constants.aspect = camera.aspect;
  constants.near_z = camera.near_z;
  constants.far_z = camera.far_z;
  constants.page_table = scene_.page_table_address();
  constants.slot_tables = scene_.slot_tables_address();
  constants.group_infos = scene_.group_infos_address();
  constants.entity_infos = scene_.entity_infos_address();
  constants.records_per_page_shift = scene_.records_per_page_shift();
  constants.record_bytes = scene_.record_bytes();
  constants.group_shift = scene_.group_shift();
  constants.encoding = scene_.encoding();
  constants.debug_view = options.debug_view;
  constants.flags = options.flags;
  float sun[3] = {options.sun_dir[0], options.sun_dir[1], options.sun_dir[2]};
  const float length = std::sqrt(sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]);
  if (length > 0.0f && std::isfinite(length)) {
    sun[0] /= length;
    sun[1] /= length;
    sun[2] /= length;
  } else {
    sun[0] = 0.0f;
    sun[1] = 1.0f;
    sun[2] = 0.0f;
  }
  Copy3(sun, constants.sun_dir);
  Copy3(options.sky_color, constants.sky_color);
  // CaelixCamera::projection is column-major; the constants hold its rows.
  for (uint32_t row = 0; row < 4; ++row) {
    for (uint32_t column = 0; column < 4; ++column) {
      constants.projection[row][column] = camera.projection[column * 4 + row];
    }
  }
  if (has_prev_camera_) {
    Copy3(prev_camera_.position, constants.prev_position);
    Copy3(prev_camera_.right, constants.prev_right);
    Copy3(prev_camera_.up, constants.prev_up);
    Copy3(prev_camera_.forward, constants.prev_forward);
    constants.prev_tan_half_fov_y = prev_camera_.tan_half_fov_y;
    constants.prev_aspect = prev_camera_.aspect;
    constants.has_prev_camera = 1;
  }
  constants.target_mask = mask;
  constants.miss_depth = options.miss_depth;
  constants.miss_view_z = options.miss_view_z;
  constants.has_tlas = tlas != VK_NULL_HANDLE ? 1u : 0u;
  // Lighting.
  const uint32_t accum_count = restart_accumulation ? 0u : accum_count_;
  constants.mode = options.mode;
  constants.bounce_count = options.bounce_count;
  constants.frame_seed = options.frame_seed;
  constants.accum_count = accum_count;
  constants.exposure = options.exposure != 0.0f ? options.exposure : 1.0f;
  Copy3(options.sun_radiance, constants.sun_radiance);
  Copy3(options.sky_zenith, constants.sky_zenith);
  Copy3(options.sky_horizon, constants.sky_horizon);
  Copy3(options.sky_ground, constants.sky_ground);
  const float sun_radius =
      options.sun_angular_radius > 0.0f ? options.sun_angular_radius : kDefaultSunAngularRadius;
  constants.sun_cos_radius = std::cos(sun_radius);
  // 2 pi (1 - cos r), written as 4 pi sin^2(r / 2) in double: 1 - cos r of a float cancels for
  // a small disk (0.5 degrees: 1 - cos r = 3.8e-5, float spacing near 1 is 6e-8).
  const double half_sin = std::sin(0.5 * static_cast<double>(sun_radius));
  constants.sun_solid_angle = static_cast<float>(4.0 * 3.14159265358979323846 * half_sin * half_sin);
  constants.output_srgb = IsUnormFormat(static_cast<VkFormat>(targets.color.format)) ? 1u : 0u;
  constants.t_max = kTMax;
  // The kernels' REBLUR_FrontEnd_GetNormHitDist uses the same A, B, C as the denoiser.
  const nrd::ReblurHitDistanceParameters hit_distance = HitDistanceParameters();
  constants.hit_dist_params[0] = hit_distance.A;
  constants.hit_dist_params[1] = hit_distance.B;
  constants.hit_dist_params[2] = hit_distance.C;

  // Any earlier use of the host targets (the host's acquire chains into
  // ALL_COMMANDS) -> the kernels' writes. Lighting modes: the internal targets'
  // first use (UNDEFINED -> GENERAL), else a memory barrier that orders the
  // previous frame's kernel reads and writes before this frame's.
  {
    Barriers before;
    for (uint32_t p = 0; p < present_count; ++p) {
      before.Image(images[p], layouts[p], VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                   VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    if (lighting) {
      if (internal_needs_init_) {
        for (uint32_t i = 0; i < kInternalCount; ++i) {
          before.Image(internal_[i].image(), VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
      }
      before.Memory(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    before.Record(cmd);
  }
  const TracePush push{slot->constants.address()};

  // Trace.
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trace_.pipeline());
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trace_.layout(), 0, 1, &slot->set,
                          0, nullptr);
  vkCmdPushConstants(cmd, trace_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  const uint32_t groups_x = (width + kWorkgroupSize - 1) / kWorkgroupSize;
  const uint32_t groups_y = (height + kWorkgroupSize - 1) / kWorkgroupSize;
  timer_.Begin(cmd, frame.frame_number, safe);
  vkCmdDispatch(cmd, groups_x, groups_y, 1);
  timer_.End(cmd);

  // NRD (path trace mode with denoising). Its first dispatch waits for the trace's
  // writes, its last barrier orders its outputs before the composite.
  bool denoised = false;
  if (denoiser != nullptr) {
    NrdExecutor::Bindings bindings;
    bindings.view_z = internal_[kInternalViewZ].view();
    bindings.normal_roughness = internal_[kInternalNormalRoughness].view();
    bindings.motion = internal_[kInternalMotion].view();
    bindings.diff_in = internal_[kInternalDiff].view();
    bindings.spec_in = internal_[kInternalSpec].view();
    bindings.diff_out = internal_[kInternalDiffOut].view();
    bindings.spec_out = internal_[kInternalSpecOut].view();
    bindings.validation = internal_[kInternalValidation].view();
    const nrd::Identifier identifier = kReblurIdentifier;
    const bool timed = denoise_timer_.Begin(cmd, frame.frame_number, safe);
    denoised = denoiser->Denoise(cmd, slot->index, bindings, std::span(&identifier, 1));
    if (timed) {
      denoise_timer_.End(cmd);
    }
    if (!denoised) {
      nrd_failed_ = true;
      nrd_error = true;
    }
    nrd_last_frame_ = frame.frame_number;
  }
  nrd_history_valid_ = denoised;
  constants.denoised = denoised ? 1u : 0u;
  std::memcpy(slot->constants.mapped(), &constants, sizeof(constants));
  slot->constants.FlushHostWrites(0, sizeof(constants));

  // Composite (lighting modes with a colour target). Bindings 3 and 4 hold NRD's
  // outputs when this frame was denoised, else the raw radiance.
  if (composite) {
    write_count = 0;
    write_image(slot->composite_set, 0, color_view);
    const uint32_t composite_internals[] = {
        kInternalAlbedo, kInternalDeterministic, denoised ? kInternalDiffOut : kInternalDiff,
        denoised ? kInternalSpecOut : kInternalSpec, kInternalAccum, kInternalValidation};
    static_assert(std::size(composite_internals) + 1 == kCompositeBindingCount);
    for (uint32_t i = 0; i < std::size(composite_internals); ++i) {
      write_image(slot->composite_set, 1 + i, internal_[composite_internals[i]].view());
    }
    vkUpdateDescriptorSets(device_.device(), write_count, writes, 0, nullptr);

    Barriers between;
    between.Memory(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    between.Record(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, composite_.pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, composite_.layout(), 0, 1,
                            &slot->composite_set, 0, nullptr);
    vkCmdPushConstants(cmd, composite_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                       &push);
    composite_timer_.Begin(cmd, frame.frame_number, safe);
    vkCmdDispatch(cmd, groups_x, groups_y, 1);
    composite_timer_.End(cmd);
  }

  // The kernels' writes -> all later commands (the host's use of its targets).
  {
    Barriers after;
    for (uint32_t p = 0; p < present_count; ++p) {
      after.Image(images[p], VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    }
    after.Record(cmd);
  }

  prev_camera_ = camera;
  has_prev_camera_ = true;
  last_mode_ = options.mode;
  if (lighting) {
    internal_needs_init_ = false;
    internal_last_frame_ = frame.frame_number;
    accum_count_ =
        (options.mode == CAELIX_MODE_REFERENCE && composite) ? accum_count + 1 : accum_count;
  }
  // An NRD error is reported once, by the call in which it happened; the frame is
  // recorded (raw output) all the same.
  return nrd_error ? CAELIX_ERROR_VULKAN : CAELIX_OK;
}

NrdExecutor* Renderer::EnsureDenoiser(uint32_t width, uint32_t height, bool& error) {
  if (nrd_ != nullptr && nrd_->width() == width && nrd_->height() == height) {
    return nrd_.get();
  }
  if (nrd_ != nullptr) {
    retired_nrd_.push_back(RetiredDenoiser{nrd_last_frame_, std::move(nrd_)});
  }
  const nrd::DenoiserDesc denoiser{kReblurIdentifier, nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR};
  try {
    nrd_ = std::make_unique<NrdExecutor>(device_, width, height, std::span(&denoiser, 1));
  } catch (const NrdError& e) {
    CAELIX_LOGE("caelix_renderer_render: NRD: %s; denoising is disabled for this renderer",
                e.what());
    nrd_failed_ = true;
    error = true;
    return nullptr;
  }
  nrd_clear_ = true;
  nrd_history_valid_ = false;
  nrd_frame_index_ = 0;
  nrd_last_time_ = {};
  return nrd_.get();
}

bool Renderer::ConfigureDenoiser(NrdExecutor& executor, const CaelixCamera& camera,
                                 const CaelixRenderOptions& options, uint32_t width,
                                 uint32_t height, bool restart) {
  const CaelixCamera& prev = has_prev_camera_ ? prev_camera_ : camera;
  nrd::CommonSettings common{};
  WorldToView(camera, common.worldToViewMatrix);
  WorldToView(prev, common.worldToViewMatrixPrev);
  std::memcpy(common.viewToClipMatrix, camera.projection, sizeof(common.viewToClipMatrix));
  std::memcpy(common.viewToClipMatrixPrev, prev.projection, sizeof(common.viewToClipMatrixPrev));
  // The trace writes motion in pixels (xy) and viewZprev - viewZ (z).
  common.motionVectorScale[0] = 1.0f / static_cast<float>(width);
  common.motionVectorScale[1] = 1.0f / static_cast<float>(height);
  common.motionVectorScale[2] = 1.0f;
  common.isMotionVectorInWorldSpace = false;
  common.cameraJitter[0] = common.cameraJitter[1] = 0.0f;
  common.cameraJitterPrev[0] = common.cameraJitterPrev[1] = 0.0f;
  common.resourceSize[0] = common.resourceSizePrev[0] = common.rectSize[0] =
      common.rectSizePrev[0] = static_cast<uint16_t>(width);
  common.resourceSize[1] = common.resourceSizePrev[1] = common.rectSize[1] =
      common.rectSizePrev[1] = static_cast<uint16_t>(height);
  common.denoisingRange = kDenoisingRange;
  common.disocclusionThreshold = 0.01f;
  common.frameIndex = nrd_frame_index_;
  common.accumulationMode = nrd_clear_ ? nrd::AccumulationMode::CLEAR_AND_RESTART
                            : restart  ? nrd::AccumulationMode::RESTART
                                       : nrd::AccumulationMode::CONTINUE;
  common.enableValidation = options.debug_view == CAELIX_VIEW_NRD_VALIDATION;
  const auto now = std::chrono::steady_clock::now();
  if (nrd_last_time_ != std::chrono::steady_clock::time_point{}) {
    common.timeDeltaBetweenFrames =
        std::chrono::duration<float, std::milli>(now - nrd_last_time_).count();
  }  // else 0: NRD measures the time itself
  nrd_last_time_ = now;
  if (!executor.SetCommonSettings(common)) {
    return false;
  }
  nrd::ReblurSettings reblur{};
  reblur.hitDistanceParameters = HitDistanceParameters();
  reblur.enableAntiFirefly = true;
  reblur.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
  reblur.maxAccumulatedFrameNum = 30;
  if (!executor.SetReblurSettings(kReblurIdentifier, reblur)) {
    return false;
  }
  nrd_clear_ = false;
  ++nrd_frame_index_;
  return true;
}

void Renderer::GetNrdInfo(CaelixNrdInfo& out) const {
  out = CaelixNrdInfo{};
  const nrd::LibraryDesc& library = NrdExecutor::lib();
  out.version_major = library.versionMajor;
  out.version_minor = library.versionMinor;
  out.version_build = library.versionBuild;
  out.failed = nrd_failed_ ? 1u : 0u;
  if (nrd_ == nullptr) {
    return;
  }
  out.width = nrd_->width();
  out.height = nrd_->height();
  out.pipeline_count = nrd_->desc().pipelinesNum;
  out.permanent_texture_count = nrd_->permanent_count();
  out.transient_texture_count = nrd_->transient_count();
  out.permanent_bytes = nrd_->permanent_bytes();
  out.transient_bytes = nrd_->transient_bytes();
  out.dispatch_count = nrd_->last_dispatch_count();
}

void Renderer::GetTimings(CaelixGpuTimings& out) const {
  out = CaelixGpuTimings{};
  out.frame_number = timer_.latest_frame();
  if (out.frame_number == 0) {
    return;
  }
  out.trace_ms = timer_.latest_ms();
  float flush_ms = 0.0f;
  if (scene_.FlushMsForFrame(out.frame_number, flush_ms)) {
    out.flush_ms = flush_ms;
  }
  float composite_ms = 0.0f;
  if (composite_timer_.MsForFrame(out.frame_number, composite_ms)) {
    out.composite_ms = composite_ms;
  }
  float denoise_ms = 0.0f;
  if (denoise_timer_.MsForFrame(out.frame_number, denoise_ms)) {
    out.denoise_ms = denoise_ms;
  }
}

}  // namespace caelix
