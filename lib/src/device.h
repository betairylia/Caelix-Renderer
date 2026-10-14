#pragma once
#include "caelix/caelix.h"
#include "vk_common.h"

#include <memory>

namespace caelix {

// The requirements the library publishes through caelix_get_device_requirements.
// Static data; the strings live for the lifetime of the DLL.
const CaelixDeviceRequirements& GetDeviceRequirements();

// A host-owned Vulkan device that the library has adopted. The library owns only
// what it creates on top of it (the VMA allocator); the instance, device and
// queue stay the host's.
class Device {
 public:
  // Loads Vulkan through the host's vkGetInstanceProcAddr, checks the physical
  // device against GetDeviceRequirements() and creates the allocator.
  static CaelixResult Adopt(const CaelixDeviceAdoptDesc& desc, std::unique_ptr<Device>& out);

  ~Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue graphics_queue() const { return graphics_queue_; }
  uint32_t graphics_queue_family() const { return graphics_queue_family_; }
  VmaAllocator allocator() const { return allocator_; }

 private:
  Device() = default;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue graphics_queue_ = VK_NULL_HANDLE;
  uint32_t graphics_queue_family_ = 0;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
};

}  // namespace caelix
