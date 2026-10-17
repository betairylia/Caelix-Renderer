#pragma once
#include "caelix/caelix.h"
#include "core/vk_common.h"

#include <memory>

namespace caelix {

// The requirements the library publishes through caelix_get_device_requirements.
// Static data; the strings live for the lifetime of the DLL.
const CaelixDeviceRequirements& GetDeviceRequirements();

// A host-owned Vulkan device that the library has adopted. The library owns only
// what it creates on top of it (the VMA allocator); the instance, device and
// queue stay the host's.
class Device
{
public:
    // Loads Vulkan through the host's vkGetInstanceProcAddr, checks the physical
    // device against GetDeviceRequirements() and creates the allocator.
    static CaelixResult Adopt(const CaelixDeviceAdoptDesc& desc, std::unique_ptr<Device>& out);

    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    VkInstance instance() const
    {
        return instance_;
    }
    VkPhysicalDevice physical_device() const
    {
        return physical_device_;
    }
    VkDevice device() const
    {
        return device_;
    }
    VkQueue graphics_queue() const
    {
        return graphics_queue_;
    }
    uint32_t graphics_queue_family() const
    {
        return graphics_queue_family_;
    }
    VmaAllocator allocator() const
    {
        return allocator_;
    }
    // VK_KHR_acceleration_structure limits of the physical device (queried at adoption).
    const VkPhysicalDeviceAccelerationStructurePropertiesKHR& as_properties() const
    {
        return as_properties_;
    }
    // Nanoseconds per timestamp tick (VkPhysicalDeviceLimits::timestampPeriod).
    float timestamp_period_ns() const
    {
        return timestamp_period_ns_;
    }
    // timestampValidBits of the graphics queue family; 0 = no timestamps.
    uint32_t timestamp_valid_bits() const
    {
        return timestamp_valid_bits_;
    }

private:
    Device() = default;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    uint32_t graphics_queue_family_ = 0;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties_{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    float timestamp_period_ns_ = 1.0f;
    uint32_t timestamp_valid_bits_ = 0;
};

}  // namespace caelix
