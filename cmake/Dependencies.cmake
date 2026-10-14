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

# NVIDIA Real-Time Denoisers (NRD): static library target `NRD` (C++17), with its
# SPIR-V shaders embedded (built by NRD's own ShaderMake step with the Vulkan SDK's
# DXC; SHADERMAKE_FIND_DXC OFF stops ShaderMake from downloading a second DXC). The
# library links it PRIVATE (lib/src/nrd_executor.*). The kernels include
# NRD.hlsli and the NRDConfig.hlsli that NRD's configure step writes next to it
# (<nrd_SOURCE_DIR>/Shaders, see cmake/Shaders.cmake). Normal encoding 2
# (R10G10B10A2, oct-packed) and roughness encoding 1 (linear) are NRD's defaults
# and are baked into both NRD's shaders and ours. NRD needs CMake 3.30. License:
# third_party/NOTICES.md.
set(NRD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
set(NRD_EMBEDS_SPIRV_SHADERS ON CACHE BOOL "" FORCE)
set(NRD_EMBEDS_DXIL_SHADERS OFF CACHE BOOL "" FORCE)
set(NRD_EMBEDS_DXBC_SHADERS OFF CACHE BOOL "" FORCE)
set(NRD_NORMAL_ENCODING "2" CACHE STRING "" FORCE)
set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "" FORCE)
set(SHADERMAKE_FIND_DXC OFF CACHE BOOL "" FORCE)
# NRD's REBLUR passes use quad intrinsics in compute shaders (better image quality).
# Device::Adopt then requires subgroup quad operations in the compute stage. OFF
# builds NRD without them, for a device that lacks that support.
option(CAELIX_NRD_QUAD_INTRINSICS "Build NRD with quad intrinsics (needs subgroup quad ops in compute)" ON)
set(NRD_SUPPORTS_QUAD_INTRINSICS ${CAELIX_NRD_QUAD_INTRINSICS} CACHE BOOL "" FORCE)
FetchContent_Declare(nrd
  GIT_REPOSITORY https://github.com/NVIDIA-RTX/NRD.git
  GIT_TAG        v4.18.0
  GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(volk VulkanMemoryAllocator unordered_dense nrd)
# The directory of NRD.hlsli and the generated NRDConfig.hlsli (DXC include path).
set(CAELIX_NRD_SHADER_INCLUDE_DIR ${nrd_SOURCE_DIR}/Shaders)

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
