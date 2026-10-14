add_library(caelix_renderer SHARED
  src/caelix.cpp
  src/device.cpp
  src/device.h
  src/finalize.h
  src/gpu_buffer.cpp
  src/gpu_buffer.h
  src/hash_map.h
  src/log.cpp
  src/log.h
  src/pipeline.cpp
  src/pipeline.h
  src/record_layout.h
  src/record_pool.cpp
  src/record_pool.h
  src/row_pool.cpp
  src/row_pool.h
  src/scene.cpp
  src/scene.h
  src/shader_loader.cpp
  src/shader_loader.h
  src/shaders.h
  src/slot_table_pool.h
  src/staging_ring.cpp
  src/staging_ring.h
  src/vk_common.h
  src/vma.cpp)
set_target_properties(caelix_renderer PROPERTIES OUTPUT_NAME caelix_renderer)
target_include_directories(caelix_renderer
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  # The generated caelix_shaders_embedded.cpp includes "shaders.h".
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(caelix_renderer PRIVATE
  CAELIX_BUILD_SHARED
  # Vulkan entry points come from volk only. volk.h defines this too, but a TU
  # that includes <vulkan/vulkan.h> first would otherwise get prototypes.
  VK_NO_PROTOTYPES
  CAELIX_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  CAELIX_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  CAELIX_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  CAELIX_VERSION_STRING="${PROJECT_VERSION}")
# Strict warnings for our own code only; fetched dependencies keep their defaults.
if(MSVC)
  target_compile_options(caelix_renderer PRIVATE /W4)
else()
  target_compile_options(caelix_renderer PRIVATE -Wall -Wextra)
endif()
target_link_libraries(caelix_renderer PRIVATE
  volk::volk
  GPUOpen::VulkanMemoryAllocator
  unordered_dense::unordered_dense)

# Kernels (HLSL -> SPIR-V, embedded). See cmake/Shaders.cmake.
caelix_add_shaders(caelix_renderer CAELIX_SHADER_SPV
  SOURCES
    ${PROJECT_SOURCE_DIR}/shaders/caelix_platform.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/caelix_record.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/caelix_finalize.hlsl)
