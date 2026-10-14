# Third-party dependencies, fetched at configure time.
#
# Every dependency is pinned to a release TAG on purpose: a branch would make
# builds non-reproducible. volk and vk-bootstrap tag their releases by Vulkan
# header version; pick the tag that matches the installed Vulkan SDK, or the
# closest tag below it. Update the tags together with the SDK.
include(FetchContent)

# volk: Vulkan meta-loader. Static library target volk::volk.
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

# vk-bootstrap: instance/device selection helpers. There is no v1.4.304 tag;
# v1.3.302 is the closest release below SDK 1.4.304.
set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(vk-bootstrap
  GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
  GIT_TAG        v1.3.302
  GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(volk VulkanMemoryAllocator vk-bootstrap)
