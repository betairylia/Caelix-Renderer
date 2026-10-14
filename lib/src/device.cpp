#include "device.h"

#include "log.h"

#include <cstring>
#include <vector>

namespace caelix {
namespace {

constexpr const char* kRequiredDeviceExtensions[] = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};

constexpr uint32_t kRequiredFeatureFlags =
    CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS | CAELIX_FEATURE_DESCRIPTOR_INDEXING |
    CAELIX_FEATURE_SYNCHRONIZATION2 | CAELIX_FEATURE_DYNAMIC_RENDERING |
    CAELIX_FEATURE_TIMELINE_SEMAPHORE | CAELIX_FEATURE_ACCELERATION_STRUCTURE |
    CAELIX_FEATURE_RAY_QUERY | CAELIX_FEATURE_SHADER_INT64 | CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT |
    CAELIX_FEATURE_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT | CAELIX_FEATURE_STORAGE_IMAGE_READ_WITHOUT_FORMAT;

const char* VkResultName(VkResult result) {
  switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    default: return "VkResult";
  }
}

CaelixResult FromVkResult(VkResult result) {
  if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
    return CAELIX_ERROR_OUT_OF_MEMORY;
  }
  return CAELIX_ERROR_VULKAN;
}

// Returns false (and logs every missing item) when the physical device lacks a requirement.
bool CheckRequirements(VkPhysicalDevice physical_device, const CaelixDeviceRequirements& req) {
  bool ok = true;

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device, &properties);
  if (properties.apiVersion < req.api_version) {
    CAELIX_LOGE("device requirement missing: Vulkan %u.%u (device reports %u.%u.%u)",
                VK_API_VERSION_MAJOR(req.api_version), VK_API_VERSION_MINOR(req.api_version),
                VK_API_VERSION_MAJOR(properties.apiVersion),
                VK_API_VERSION_MINOR(properties.apiVersion),
                VK_API_VERSION_PATCH(properties.apiVersion));
    ok = false;
  }

  uint32_t extension_count = 0;
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count,
                                       extensions.data());
  for (uint32_t i = 0; i < req.device_extension_count; ++i) {
    bool found = false;
    for (const VkExtensionProperties& extension : extensions) {
      if (std::strcmp(extension.extensionName, req.device_extensions[i]) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      CAELIX_LOGE("device requirement missing: extension %s", req.device_extensions[i]);
      ok = false;
    }
  }
  // The extension feature structs below may only be queried when the extensions exist.
  if (!ok) {
    return false;
  }

  VkPhysicalDeviceRayQueryFeaturesKHR ray_query{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  acceleration_structure.pNext = &ray_query;
  VkPhysicalDeviceVulkan13Features vk13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  vk13.pNext = &acceleration_structure;
  VkPhysicalDeviceVulkan12Features vk12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  vk12.pNext = &vk13;
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &vk12;
  vkGetPhysicalDeviceFeatures2(physical_device, &features);

  struct FeatureCheck {
    uint32_t flag;
    VkBool32 supported;
    const char* name;
  };
  const FeatureCheck checks[] = {
      {CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS, vk12.bufferDeviceAddress, "bufferDeviceAddress"},
      {CAELIX_FEATURE_DESCRIPTOR_INDEXING, vk12.runtimeDescriptorArray, "runtimeDescriptorArray"},
      {CAELIX_FEATURE_DESCRIPTOR_INDEXING, vk12.shaderStorageBufferArrayNonUniformIndexing,
       "shaderStorageBufferArrayNonUniformIndexing"},
      {CAELIX_FEATURE_DESCRIPTOR_INDEXING, vk12.descriptorBindingPartiallyBound,
       "descriptorBindingPartiallyBound"},
      {CAELIX_FEATURE_DESCRIPTOR_INDEXING, vk12.descriptorBindingVariableDescriptorCount,
       "descriptorBindingVariableDescriptorCount"},
      {CAELIX_FEATURE_SYNCHRONIZATION2, vk13.synchronization2, "synchronization2"},
      {CAELIX_FEATURE_DYNAMIC_RENDERING, vk13.dynamicRendering, "dynamicRendering"},
      {CAELIX_FEATURE_TIMELINE_SEMAPHORE, vk12.timelineSemaphore, "timelineSemaphore"},
      {CAELIX_FEATURE_ACCELERATION_STRUCTURE, acceleration_structure.accelerationStructure,
       "accelerationStructure"},
      {CAELIX_FEATURE_RAY_QUERY, ray_query.rayQuery, "rayQuery"},
      {CAELIX_FEATURE_SHADER_INT64, features.features.shaderInt64, "shaderInt64"},
      {CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT, vk12.scalarBlockLayout, "scalarBlockLayout"},
      {CAELIX_FEATURE_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT,
       features.features.shaderStorageImageWriteWithoutFormat,
       "shaderStorageImageWriteWithoutFormat"},
      {CAELIX_FEATURE_STORAGE_IMAGE_READ_WITHOUT_FORMAT,
       features.features.shaderStorageImageReadWithoutFormat,
       "shaderStorageImageReadWithoutFormat"},
  };
  for (const FeatureCheck& check : checks) {
    if ((req.feature_flags & check.flag) != 0 && check.supported != VK_TRUE) {
      CAELIX_LOGE("device requirement missing: feature %s", check.name);
      ok = false;
    }
  }

#if CAELIX_NRD_QUAD_INTRINSICS
  // NRD's shaders (REBLUR HistoryFix, Blur, PostBlur) use QuadReadAcrossX/Y in compute shaders
  // (SPIR-V capability GroupNonUniformQuad). A property of the device, nothing to enable. Their
  // SPIR-V declares no DerivativeGroupQuads execution mode, so VK_KHR_compute_shader_derivatives is
  // not needed.
  VkPhysicalDeviceVulkan11Properties vk11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
  VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties2.pNext = &vk11;
  vkGetPhysicalDeviceProperties2(physical_device, &properties2);
  if ((vk11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0 ||
      (vk11.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) == 0) {
    CAELIX_LOGE("device requirement missing: subgroup quad operations in compute shaders "
                "(subgroupSupportedStages 0x%x, subgroupSupportedOperations 0x%x); build the "
                "library with CAELIX_NRD_QUAD_INTRINSICS=OFF for such a device",
                vk11.subgroupSupportedStages, vk11.subgroupSupportedOperations);
    ok = false;
  }
#endif
  return ok;
}

}  // namespace

const CaelixDeviceRequirements& GetDeviceRequirements() {
  static const CaelixDeviceRequirements requirements = {
      VK_MAKE_API_VERSION(0, 1, 3, 0),
      static_cast<uint32_t>(sizeof(kRequiredDeviceExtensions) / sizeof(kRequiredDeviceExtensions[0])),
      kRequiredDeviceExtensions,
      kRequiredFeatureFlags,
  };
  return requirements;
}

CaelixResult Device::Adopt(const CaelixDeviceAdoptDesc& desc, std::unique_ptr<Device>& out) {
  out.reset();
  if (desc.get_instance_proc_addr == nullptr || desc.instance == 0 ||
      desc.physical_device == 0 || desc.device == 0 || desc.graphics_queue == 0) {
    CAELIX_LOGE("caelix_device_adopt: the desc has a null handle or function pointer");
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }

  std::unique_ptr<Device> device(new Device());
  device->instance_ = HandleFromU64<VkInstance>(desc.instance);
  device->physical_device_ = HandleFromU64<VkPhysicalDevice>(desc.physical_device);
  device->device_ = HandleFromU64<VkDevice>(desc.device);
  device->graphics_queue_ = HandleFromU64<VkQueue>(desc.graphics_queue);
  device->graphics_queue_family_ = desc.graphics_queue_family;

  // Load Vulkan through the host's loader, so the library sees the same layers
  // (validation included) as the host. volk's global function tables live inside
  // this DLL, so they do not conflict with the host's loader or its own tables.
  volkInitializeCustom(reinterpret_cast<PFN_vkGetInstanceProcAddr>(desc.get_instance_proc_addr));
  volkLoadInstanceOnly(device->instance_);
  volkLoadDevice(device->device_);

  const CaelixDeviceRequirements& req = GetDeviceRequirements();
  if (!CheckRequirements(device->physical_device_, req)) {
    return CAELIX_ERROR_UNSUPPORTED;
  }
  // Vulkan cannot report which features the host actually ENABLED on an adopted device, so a
  // feature the host forgot shows up as a validation error later, not here.
  CAELIX_LOGI("device requirements are supported by the physical device");

  VmaAllocatorCreateInfo allocator_info{};
  allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  allocator_info.physicalDevice = device->physical_device_;
  allocator_info.device = device->device_;
  allocator_info.instance = device->instance_;
  allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
  VmaVulkanFunctions functions{};
  VkResult result = vmaImportVulkanFunctionsFromVolk(&allocator_info, &functions);
  if (result != VK_SUCCESS) {
    CAELIX_LOGE("vmaImportVulkanFunctionsFromVolk failed: %s (%d)", VkResultName(result),
                static_cast<int>(result));
    return FromVkResult(result);
  }
  allocator_info.pVulkanFunctions = &functions;
  result = vmaCreateAllocator(&allocator_info, &device->allocator_);
  if (result != VK_SUCCESS) {
    CAELIX_LOGE("vmaCreateAllocator failed: %s (%d)", VkResultName(result),
                static_cast<int>(result));
    return FromVkResult(result);
  }

  VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceVulkan12Properties vk12_properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
  vk12_properties.pNext = &as_properties;
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &vk12_properties;
  vkGetPhysicalDeviceProperties2(device->physical_device_, &properties);
  as_properties.pNext = nullptr;
  device->as_properties_ = as_properties;
  device->timestamp_period_ns_ = properties.properties.limits.timestampPeriod;

  // Timestamps of the graphics queue (GPU timings). 0 valid bits = no timestamp support.
  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device->physical_device_, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device->physical_device_, &family_count,
                                           families.data());
  if (desc.graphics_queue_family < family_count) {
    device->timestamp_valid_bits_ = families[desc.graphics_queue_family].timestampValidBits;
  }
  if (device->timestamp_valid_bits_ == 0) {
    CAELIX_LOGW("queue family %u has no timestamp support: GPU timings are not available",
                desc.graphics_queue_family);
  }
  CAELIX_LOGI("adopted device: %s, driver %s %s (0x%08x), Vulkan %u.%u.%u, queue family %u",
              properties.properties.deviceName, vk12_properties.driverName,
              vk12_properties.driverInfo, properties.properties.driverVersion,
              VK_API_VERSION_MAJOR(properties.properties.apiVersion),
              VK_API_VERSION_MINOR(properties.properties.apiVersion),
              VK_API_VERSION_PATCH(properties.properties.apiVersion),
              device->graphics_queue_family_);
  CAELIX_LOGI("acceleration structure properties: scratch offset alignment %u, max instance count "
              "%llu, max primitive count %llu, max geometry count %llu",
              as_properties.minAccelerationStructureScratchOffsetAlignment,
              static_cast<unsigned long long>(as_properties.maxInstanceCount),
              static_cast<unsigned long long>(as_properties.maxPrimitiveCount),
              static_cast<unsigned long long>(as_properties.maxGeometryCount));

  out = std::move(device);
  return CAELIX_OK;
}

Device::~Device() {
  if (allocator_ != VK_NULL_HANDLE) {
    vmaDestroyAllocator(allocator_);
  }
}

}  // namespace caelix
