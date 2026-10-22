# Headless tests. They use the PUBLIC C ABI only.

# The ABI header must stay valid C99: this target compiles it as C.
add_library(caelix_abi_c99 STATIC abi_c99.c)
set_target_properties(caelix_abi_c99 PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(caelix_abi_c99 PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(MSVC)
  target_compile_options(caelix_abi_c99 PRIVATE /W4)
else()
  target_compile_options(caelix_abi_c99 PRIVATE -Wall -Wextra -pedantic)
endif()

# Shared Vulkan setup of the GPU tests: a Vulkan 1.3 device (vk-bootstrap,
# validation layers when installed), readback buffers, the frame loop; and the
# synthetic scenes and the one-entity scene rig (test_scenes.h).
add_library(caelix_test_device STATIC test_device.cpp test_device.h test_scenes.cpp test_scenes.h)
target_compile_definitions(caelix_test_device PUBLIC VK_NO_PROTOTYPES)
target_link_libraries(caelix_test_device PUBLIC
  caelix_renderer
  volk::volk
  vk-bootstrap::vk-bootstrap
  Catch2::Catch2)

function(caelix_add_gpu_test name)
  add_executable(${name} ${ARGN})
  if(MSVC)
    target_compile_options(${name} PRIVATE /W4)
  else()
    target_compile_options(${name} PRIVATE -Wall -Wextra)
  endif()
  target_link_libraries(${name} PRIVATE caelix_test_device Catch2::Catch2WithMain)
  add_test(NAME ${name} COMMAND ${name})
endfunction()
if(MSVC)
  target_compile_options(caelix_test_device PRIVATE /W4)
else()
  target_compile_options(caelix_test_device PRIVATE -Wall -Wextra)
endif()

# Scene smoke test: the whole upload -> flush -> GPU readback path.
caelix_add_gpu_test(caelix_scene_smoke scene_smoke.cpp)

# Finalize kernel against a CPU reference, by GPU readback.
caelix_add_gpu_test(caelix_finalize_test finalize_test.cpp)

# Acceleration structures (BLAS per group, TLAS): validation, property queries, stats.
caelix_add_gpu_test(caelix_as_test as_test.cpp)

# Primary ray trace kernel: rendered images read back and checked.
caelix_add_gpu_test(caelix_trace_test trace_test.cpp)

# NRD denoising through the renderer: executor creation and resize, denoised image
# against a reference, history reset, debug views.
caelix_add_gpu_test(caelix_nrd_test nrd_test.cpp)

# The material table through the C ABI: the packed defaults against the old shader decode and
# data/glass-palette.json, the glass palette, the block-ID table, argument checks.
caelix_add_gpu_test(caelix_material_test material_test.cpp)
target_compile_definitions(caelix_material_test PRIVATE
  CAELIX_GLASS_PALETTE_JSON="${PROJECT_SOURCE_DIR}/data/glass-palette.json")

# The DLSS upscalers (SR, RR) through the C ABI: the availability query, the jitter helpers, SR and
# RR smoke tests against an accumulated reference, argument checks, quality changes and the upscale
# cost at 1920x1080. The GPU cases skip themselves when the build (CAELIX_DLSS=OFF) or the machine
# has no DLSS.
caelix_add_gpu_test(caelix_dlss_test dlss_test.cpp)

# The frame constants' uniform block in the five compiled frame-constant kernels against the C layout of
# lib/src/render/frame_constants.h (no GPU: parses the .spv files of the build's shader folder). Fails
# when a member offset or an array stride of DXC's constant buffer layout differs from the C struct.
add_executable(caelix_frame_layout_test frame_layout_test.cpp)
if(MSVC)
  target_compile_options(caelix_frame_layout_test PRIVATE /W4)
else()
  target_compile_options(caelix_frame_layout_test PRIVATE -Wall -Wextra)
endif()
target_include_directories(caelix_frame_layout_test PRIVATE ${PROJECT_SOURCE_DIR}/lib/src)
target_compile_definitions(caelix_frame_layout_test PRIVATE
  CAELIX_SHADER_SPV_DIR="${CMAKE_BINARY_DIR}/shaders")
target_link_libraries(caelix_frame_layout_test PRIVATE Catch2::Catch2WithMain)
# The .spv files are outputs of the caelix_renderer build (cmake/Shaders.cmake).
add_dependencies(caelix_frame_layout_test caelix_renderer)
add_test(NAME caelix_frame_layout_test COMMAND caelix_frame_layout_test)
