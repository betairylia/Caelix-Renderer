# Third-party dependencies, fetched at configure time.
#
# Every dependency is pinned to a release TAG on purpose: a branch would make
# builds non-reproducible. volk tags its releases by Vulkan header version;
# pick the tag that matches the installed Vulkan SDK, or the closest tag below it. Update the tags together with the SDK.
include(FetchContent)

# volk: Vulkan meta-loader. Static library target volk::volk. The library loads
# Vulkan through the HOST's vkGetInstanceProcAddr (volkInitializeCustom).
# VOLK_PULL_IN_VULKAN adds the Vulkan SDK headers to volk's include directories.
set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
set(VOLK_HEADERS_ONLY OFF CACHE BOOL "" FORCE)
FetchContent_Declare(volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG        1.4.304
  GIT_SHALLOW    TRUE)

# VulkanMemoryAllocator: header-only interface target GPUOpen::VulkanMemoryAllocator.
FetchContent_Declare(VulkanMemoryAllocator
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG        v3.3.0
  GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(volk VulkanMemoryAllocator)
