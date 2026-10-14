#include "renderer.h"

#include "frame_constants.h"
#include "log.h"
#include "scene.h"
#include "shader_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace caelix {
namespace {

constexpr uint32_t kWorkgroupSize = 8;  // [numthreads(8, 8, 1)] in caelix_trace_primary.hlsl
constexpr uint32_t kDebugViewCount = 6;

const VkImageSubresourceRange kColorRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

void ImageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout,
                  VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                  VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
  VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
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
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(cmd, &dependency);
}

void Copy3(const float in[3], float out[4]) {
  out[0] = in[0];
  out[1] = in[1];
  out[2] = in[2];
  out[3] = 0.0f;
}

}  // namespace

Renderer::Renderer(Device& device, Scene& scene) : device_(device), scene_(scene) {}

CaelixResult Renderer::Create(Device& device, Scene& scene, std::unique_ptr<Renderer>& out) {
  out.reset();
  std::unique_ptr<Renderer> renderer(new Renderer(device, scene));
  const VkDevice vk_device = device.device();

  const VkDescriptorSetLayoutBinding bindings[2] = {
      {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  };
  VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layout_info.bindingCount = 2;
  layout_info.pBindings = bindings;
  const VkResult result =
      vkCreateDescriptorSetLayout(vk_device, &layout_info, nullptr, &renderer->set_layout_);
  if (result != VK_SUCCESS) {
    renderer->set_layout_ = VK_NULL_HANDLE;
    CAELIX_LOGE("vkCreateDescriptorSetLayout failed: %d", static_cast<int>(result));
    throw VulkanError("vkCreateDescriptorSetLayout failed", result);
  }
  const std::vector<uint32_t> spirv = LoadShader("caelix_trace_primary");
  renderer->trace_ = ComputePipeline(vk_device, spirv, sizeof(TracePush), "caelix_trace_primary",
                                     renderer->set_layout_);
  for (uint32_t i = 0; i < kInitialSlots; ++i) {
    renderer->AddSlot();
  }
  renderer->timer_ = GpuTimer(device);
  CAELIX_LOGI("renderer created: trace kernel caelix_trace_primary, %u frame slots",
              kInitialSlots);
  out = std::move(renderer);
  return CAELIX_OK;
}

Renderer::~Renderer() {
  const VkDevice device = device_.device();
  ReleaseTargetViews();
  for (const std::unique_ptr<FrameSlot>& slot : slots_) {
    if (slot->pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device, slot->pool, nullptr);  // frees the set too
    }
  }
  slots_.clear();
  trace_ = ComputePipeline();
  if (set_layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);
  }
}

void Renderer::AddSlot() {
  const VkDevice device = device_.device();
  auto slot = std::make_unique<FrameSlot>();
  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
  };
  VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.maxSets = 1;
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
  VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  set_info.descriptorPool = added.pool;
  set_info.descriptorSetCount = 1;
  set_info.pSetLayouts = &set_layout_;
  result = vkAllocateDescriptorSets(device, &set_info, &added.set);
  if (result != VK_SUCCESS) {
    CAELIX_LOGE("vkAllocateDescriptorSets failed: %d", static_cast<int>(result));
    throw VulkanError("vkAllocateDescriptorSets failed", result);
  }
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

CaelixResult Renderer::Render(const CaelixFrameDesc& frame, const CaelixRenderTargets& targets,
                              const CaelixCamera& camera, const CaelixRenderOptions& options) {
  constexpr const char* kFunction = "caelix_renderer_render";
  const CaelixTargetImage& color = targets.color;
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
  if (color.image == 0 || color.width == 0 || color.height == 0 ||
      color.format == VK_FORMAT_UNDEFINED) {
    CAELIX_LOGE("%s: the colour target needs an image, a format and a nonzero size", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (options.debug_view >= kDebugViewCount) {
    CAELIX_LOGE("%s: unknown debug view %u", kFunction, options.debug_view);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (!FormatSupported(static_cast<VkFormat>(color.format))) {
    return CAELIX_ERROR_UNSUPPORTED;
  }

  const VkCommandBuffer cmd = HandleFromU64<VkCommandBuffer>(frame.command_buffer);
  const VkImage image = HandleFromU64<VkImage>(color.image);
  const uint64_t safe = frame.safe_frame_number;
  timer_.Resolve(safe);

  const VkAccelerationStructureKHR tlas = scene_.tlas();
  if (tlas == VK_NULL_HANDLE) {
    // No TLAS yet (no bricks were ever built): the image shows the sky.
    ImageBarrier(cmd, image, static_cast<VkImageLayout>(color.layout),
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    timer_.Begin(cmd, frame.frame_number, safe);
    VkClearColorValue sky{};
    sky.float32[0] = options.sky_color[0];
    sky.float32[1] = options.sky_color[1];
    sky.float32[2] = options.sky_color[2];
    sky.float32[3] = 1.0f;
    vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &sky, 1, &kColorRange);
    timer_.End(cmd);
    ImageBarrier(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    return CAELIX_OK;
  }

  FrameSlot* slot = AcquireSlot(frame.frame_number, safe);
  if (slot == nullptr) {
    CAELIX_LOGE("%s: more than %u frames in flight (safe frame %llu, frame %llu)", kFunction,
                kMaxSlots, static_cast<unsigned long long>(safe),
                static_cast<unsigned long long>(frame.frame_number));
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  const VkImageView view = ViewFor(color, frame.frame_number, safe);

  // Descriptors: the slot's previous frame has completed, so its set is free.
  VkWriteDescriptorSetAccelerationStructureKHR as_write{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  as_write.accelerationStructureCount = 1;
  as_write.pAccelerationStructures = &tlas;
  const VkDescriptorImageInfo image_info{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet writes[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET},
                                    {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
  writes[0].pNext = &as_write;
  writes[0].dstSet = slot->set;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  writes[1].dstSet = slot->set;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writes[1].pImageInfo = &image_info;
  vkUpdateDescriptorSets(device_.device(), 2, writes, 0, nullptr);

  // Frame constants.
  FrameConstants constants{};
  Copy3(camera.position, constants.position);
  Copy3(camera.right, constants.right);
  Copy3(camera.up, constants.up);
  Copy3(camera.forward, constants.forward);
  constants.size[0] = color.width;
  constants.size[1] = color.height;
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
  std::memcpy(slot->constants.mapped(), &constants, sizeof(constants));
  slot->constants.FlushHostWrites(0, sizeof(constants));

  // Any earlier use of the image (the host's acquire chains into ALL_COMMANDS)
  // -> the kernel's writes.
  ImageBarrier(cmd, image, static_cast<VkImageLayout>(color.layout),
               VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trace_.pipeline());
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trace_.layout(), 0, 1, &slot->set,
                          0, nullptr);
  const TracePush push{slot->constants.address()};
  vkCmdPushConstants(cmd, trace_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  timer_.Begin(cmd, frame.frame_number, safe);
  vkCmdDispatch(cmd, (color.width + kWorkgroupSize - 1) / kWorkgroupSize,
                (color.height + kWorkgroupSize - 1) / kWorkgroupSize, 1);
  timer_.End(cmd);
  ImageBarrier(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
               VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
  return CAELIX_OK;
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
}

}  // namespace caelix
