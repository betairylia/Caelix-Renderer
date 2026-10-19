add_library(caelix_renderer SHARED
  # C ABI entry points (include/caelix/caelix.h).
  src/abi/caelix.cpp
  # Logging, Vulkan includes, shared containers.
  src/core/hash_map.h
  src/core/log.cpp
  src/core/log.h
  src/core/vk_common.h
  # Vulkan wrappers: device, buffers, images, pipelines, timers, shader loading.
  src/rhi/device.cpp
  src/rhi/device.h
  src/rhi/gpu_buffer.cpp
  src/rhi/gpu_buffer.h
  src/rhi/gpu_image.cpp
  src/rhi/gpu_image.h
  src/rhi/gpu_timer.cpp
  src/rhi/gpu_timer.h
  src/rhi/pipeline.cpp
  src/rhi/pipeline.h
  src/rhi/shader_loader.cpp
  src/rhi/shader_loader.h
  src/rhi/shaders.h
  src/rhi/staging_ring.cpp
  src/rhi/staging_ring.h
  src/rhi/vma.cpp
  # GPU memory pools.
  src/rhi/alloc/as_pool.cpp
  src/rhi/alloc/as_pool.h
  src/rhi/alloc/record_pool.cpp
  src/rhi/alloc/record_pool.h
  src/rhi/alloc/row_pool.cpp
  src/rhi/alloc/row_pool.h
  # The scene: entities, bricks, flush, acceleration structures.
  src/scene/gpu_structs.h
  src/scene/record_layout.h
  src/scene/scene.h
  src/scene/scene_accel.cpp
  src/scene/scene_bricks.cpp
  src/scene/scene_entities.cpp
  src/scene/scene_flush.cpp
  src/scene/scene_materials.cpp
  src/scene/material_table.cpp
  src/scene/material_table.h
  src/scene/glass_palette_default.inc
  src/scene/scene_internal.h
  src/scene/slot_table_pool.h
  # The renderer: per-device kernels, render targets, the pass sequence, NRD.
  src/render/frame_constants.h
  src/render/nrd_executor.cpp
  src/render/nrd_executor.h
  src/render/render_kernels.cpp
  src/render/render_kernels.h
  src/render/renderer.h
  src/render/renderer_denoise.cpp
  src/render/renderer_internal.h
  src/render/renderer_render.cpp
  src/render/renderer_targets.cpp)
set_target_properties(caelix_renderer PROPERTIES OUTPUT_NAME caelix_renderer)
target_include_directories(caelix_renderer
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  # Includes are relative to src/ ("rhi/device.h"); the generated
  # caelix_shaders_embedded.cpp includes "rhi/shaders.h".
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(caelix_renderer PRIVATE
  CAELIX_BUILD_SHARED
  # Vulkan entry points come from volk only. volk.h defines this too, but a TU
  # that includes <vulkan/vulkan.h> first would otherwise get prototypes.
  VK_NO_PROTOTYPES
  CAELIX_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  CAELIX_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  CAELIX_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  CAELIX_VERSION_STRING="${PROJECT_VERSION}"
  CAELIX_NRD_QUAD_INTRINSICS=$<BOOL:${CAELIX_NRD_QUAD_INTRINSICS}>)
# Strict warnings for our own code only; fetched dependencies keep their defaults.
if(MSVC)
  target_compile_options(caelix_renderer PRIVATE /W4)
else()
  target_compile_options(caelix_renderer PRIVATE -Wall -Wextra)
endif()
target_link_libraries(caelix_renderer PRIVATE
  volk::volk
  GPUOpen::VulkanMemoryAllocator
  unordered_dense::unordered_dense
  NRD)

# Kernels (HLSL -> SPIR-V, embedded). See cmake/Shaders.cmake.
caelix_add_shaders(caelix_renderer CAELIX_SHADER_SPV
  SOURCES
    ${PROJECT_SOURCE_DIR}/shaders/common/caelix_platform.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/common/caelix_record.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/common/caelix_block.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/common/caelix_frame.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/common/caelix_material.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/scene/caelix_finalize.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/render/caelix_brick_trace.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/render/caelix_scene_trace.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/render/caelix_shading.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/render/caelix_path_trace.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/render/caelix_composite_set.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/render/caelix_composite.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/render/caelix_cross_resolve.hlsl
    ${PROJECT_SOURCE_DIR}/shaders/render/caelix_sky_sh.hlsl)
