# 3.30: NRD's own CMakeLists (fetched, cmake/Dependencies.cmake) requires it.
cmake_minimum_required(VERSION 3.30)
project(CaelixRenderer VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Read every source (ours and fetched) as UTF-8, independent of the system code page.
if(MSVC)
  add_compile_options(/utf-8)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)

# Builds the headless tests (tests/) and fetches their test-only dependencies
# (Catch2, vk-bootstrap). The library itself never depends on them.
option(CAELIX_BUILD_TESTS "Build the Caelix tests" ON)
# Builds the Unity native plugin (plugin/unity/, docs/UNITY_PLUGIN.md). Needs Unity's plugin API
# headers (UNITY_PLUGIN_API_DIR).
if(WIN32)
  option(CAELIX_BUILD_UNITY_PLUGIN "Build the Unity native plugin caelix_unity" ON)
else()
  option(CAELIX_BUILD_UNITY_PLUGIN "Build the Unity native plugin caelix_unity" OFF)
endif()
# NVIDIA DLSS Super Resolution and Ray Reconstruction through the NGX SDK (fetched, see
# cmake/Dependencies.cmake and docs/DLSS.md). OFF builds the library without any NGX code:
# the upscaler options then return CAELIX_ERROR_UNSUPPORTED. NGX is Windows x64 only here.
if(WIN32)
  option(CAELIX_DLSS "Build the DLSS upscaler (NVIDIA NGX SDK)" ON)
else()
  option(CAELIX_DLSS "Build the DLSS upscaler (NVIDIA NGX SDK)" OFF)
endif()
# The SDK's development DLLs (lib/Windows_x86_64/dev) carry an on-screen watermark and the debug
# overlay (CTRL+ALT+F12). For integration work only; they must never ship.
option(CAELIX_DLSS_DEV_DLL "Copy the NGX development DLLs (debug overlay) instead of the release DLLs" OFF)
option(CAELIX_SHADER_DEBUG_INFO "Compile the shaders with DXC -Zi (source + line info for Nsight; same code)" OFF)

find_package(Vulkan REQUIRED)   # headers only; loader goes through volk later
include(cmake/Dependencies.cmake)
include(cmake/Shaders.cmake)
add_subdirectory(lib)
if(CAELIX_BUILD_UNITY_PLUGIN)
  add_subdirectory(plugin/unity)
endif()
if(CAELIX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
