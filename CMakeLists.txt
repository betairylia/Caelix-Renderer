cmake_minimum_required(VERSION 3.28)
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

find_package(Vulkan REQUIRED)   # headers only; loader goes through volk later
include(cmake/Dependencies.cmake)
add_subdirectory(lib)
if(CAELIX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
