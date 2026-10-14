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

# unordered_dense: header-only hash map, target unordered_dense::unordered_dense.
# SYSTEM marks its include directory as external, so /W4 on our targets does not
# report warnings from it.
FetchContent_Declare(unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
  GIT_TAG        v4.11.0
  GIT_SHALLOW    TRUE
  SYSTEM)

FetchContent_MakeAvailable(volk VulkanMemoryAllocator unordered_dense)

# Test-only dependencies. The library never links them.
if(CAELIX_BUILD_TESTS)
  # Catch2 v3: targets Catch2::Catch2 and Catch2::Catch2WithMain.
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.16.1
    GIT_SHALLOW    TRUE
    SYSTEM)
  # vk-bootstrap: instance and device creation for the headless tests. Target
  # vk-bootstrap::vk-bootstrap. It finds the Vulkan headers through the
  # Vulkan::Headers target that find_package(Vulkan) created.
  set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
  set(VK_BOOTSTRAP_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG        v1.3.302
    GIT_SHALLOW    TRUE
    SYSTEM)
  FetchContent_MakeAvailable(Catch2 vk_bootstrap)
endif()
