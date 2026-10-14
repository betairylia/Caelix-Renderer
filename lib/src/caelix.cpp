// The C ABI. Every function here is extern "C" and exception-free: internal
// code may throw, and each entry point converts exceptions to a CaelixResult.
#include "caelix/caelix.h"

#include "device.h"
#include "log.h"
#include "vk_common.h"

#include <memory>
#include <new>

// The opaque handle behind CaelixDevice*.
struct CaelixDevice {
  std::unique_ptr<caelix::Device> impl;
};

namespace {

CaelixResult ClearTarget(CaelixDevice* device, uint64_t command_buffer,
                         const CaelixTargetImage* target, const float rgba[4]) {
  if (device == nullptr || device->impl == nullptr || command_buffer == 0 ||
      target == nullptr || rgba == nullptr || target->image == 0) {
    CAELIX_LOGE("caelix_clear_target: null argument");
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  const VkCommandBuffer cmd = caelix::HandleFromU64<VkCommandBuffer>(command_buffer);
  const VkImage image = caelix::HandleFromU64<VkImage>(target->image);
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                                      VK_REMAINING_ARRAY_LAYERS};

  // Any layout -> TRANSFER_DST_OPTIMAL. The source scope is ALL_COMMANDS, so it
  // chains with whatever the host did before (a semaphore wait, an earlier pass).
  VkImageMemoryBarrier2 to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  to_transfer.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  to_transfer.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  to_transfer.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
  to_transfer.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  to_transfer.oldLayout = static_cast<VkImageLayout>(target->layout);
  to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_transfer.image = image;
  to_transfer.subresourceRange = range;

  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &to_transfer;
  vkCmdPipelineBarrier2(cmd, &dependency);

  VkClearColorValue color{};
  color.float32[0] = rgba[0];
  color.float32[1] = rgba[1];
  color.float32[2] = rgba[2];
  color.float32[3] = rgba[3];
  vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);

  // TRANSFER_DST_OPTIMAL -> GENERAL, the layout contract of every render call.
  // The destination scope is ALL_COMMANDS; the host adds its own barrier for present.
  VkImageMemoryBarrier2 to_general = to_transfer;
  to_general.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
  to_general.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  to_general.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  to_general.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  dependency.pImageMemoryBarriers = &to_general;
  vkCmdPipelineBarrier2(cmd, &dependency);
  return CAELIX_OK;
}

}  // namespace

extern "C" {

CAELIX_API void caelix_get_version(CaelixVersion* out) {
  if (out == nullptr) {
    return;
  }
  out->major = CAELIX_VERSION_MAJOR;
  out->minor = CAELIX_VERSION_MINOR;
  out->patch = CAELIX_VERSION_PATCH;
  out->abi = CAELIX_ABI_VERSION;
}

CAELIX_API const char* caelix_version_string(void) {
  return CAELIX_VERSION_STRING;
}

CAELIX_API void caelix_set_log_callback(CaelixLogCallback callback, void* user_data) {
  caelix::SetLogCallback(callback, user_data);
}

CAELIX_API void caelix_get_device_requirements(CaelixDeviceRequirements* out) {
  if (out == nullptr) {
    return;
  }
  *out = caelix::GetDeviceRequirements();
}

CAELIX_API CaelixResult caelix_device_adopt(const CaelixDeviceAdoptDesc* desc,
                                            CaelixDevice** out_device) {
  if (out_device == nullptr) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  *out_device = nullptr;
  if (desc == nullptr) {
    CAELIX_LOGE("caelix_device_adopt: desc is null");
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    std::unique_ptr<caelix::Device> impl;
    const CaelixResult result = caelix::Device::Adopt(*desc, impl);
    if (result != CAELIX_OK) {
      return result;
    }
    *out_device = new CaelixDevice{std::move(impl)};
    return CAELIX_OK;
  } catch (const std::bad_alloc&) {
    CAELIX_LOGE("caelix_device_adopt: out of memory");
    return CAELIX_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    CAELIX_LOGE("caelix_device_adopt: unexpected exception");
    return CAELIX_ERROR_VULKAN;
  }
}

CAELIX_API void caelix_device_destroy(CaelixDevice* device) {
  try {
    delete device;
  } catch (...) {
    CAELIX_LOGE("caelix_device_destroy: unexpected exception");
  }
}

CAELIX_API CaelixResult caelix_clear_target(CaelixDevice* device, uint64_t command_buffer,
                                            const CaelixTargetImage* target,
                                            const float rgba[4]) {
  try {
    return ClearTarget(device, command_buffer, target, rgba);
  } catch (const std::bad_alloc&) {
    CAELIX_LOGE("caelix_clear_target: out of memory");
    return CAELIX_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    CAELIX_LOGE("caelix_clear_target: unexpected exception");
    return CAELIX_ERROR_VULKAN;
  }
}

}  // extern "C"
