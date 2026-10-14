#pragma once
// The one include for Vulkan + VMA. Every translation unit that touches Vulkan or
// VMA includes this header, so all of them agree on the VMA configuration macros.
//
// Vulkan entry points come from volk (VK_NO_PROTOTYPES). VMA must not fetch
// functions by itself; Device fills VmaVulkanFunctions from volk instead.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <volk.h>  // must come before vk_mem_alloc.h: it enables vmaImportVulkanFunctionsFromVolk

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <cstdint>
#include <type_traits>

namespace caelix {

// Converts a raw handle value from the C ABI (uint64_t) back to a Vulkan handle type.
// Dispatchable handles are always pointers; non-dispatchable handles are pointers on
// 64-bit targets and uint64_t on 32-bit targets.
template <typename T>
T HandleFromU64(uint64_t value) {
  if constexpr (std::is_pointer_v<T>) {
    return reinterpret_cast<T>(static_cast<uintptr_t>(value));
  } else {
    return static_cast<T>(value);
  }
}

// The inverse of HandleFromU64: a Vulkan handle as the raw uint64_t of the C ABI.
template <typename T>
uint64_t HandleToU64(T handle) {
  if constexpr (std::is_pointer_v<T>) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
  } else {
    return static_cast<uint64_t>(handle);
  }
}

}  // namespace caelix
