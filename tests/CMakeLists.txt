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
add_library(caelix_test_device STATIC test_device.cpp test_device.h test_scenes.cpp test_scenes.h
  test_kernels.cpp test_kernels.h)
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

# The GPU layouts shared by C++ and HLSL, no GPU: the frame constants' uniform block in the five
# compiled frame-constant kernels against lib/src/render/frame_constants.h (fails when a member offset
# or an array stride of DXC's constant buffer layout differs from the C struct), the push constant
# blocks of the scene and light tree kernels against their C++ structs by member name, and every named
# constant of the HLSL layout headers against the C++ constant of the same name.
add_executable(caelix_frame_layout_test frame_layout_test.cpp)
if(MSVC)
  target_compile_options(caelix_frame_layout_test PRIVATE /W4)
else()
  target_compile_options(caelix_frame_layout_test PRIVATE -Wall -Wextra)
endif()
target_include_directories(caelix_frame_layout_test PRIVATE ${PROJECT_SOURCE_DIR}/lib/src
  ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(caelix_frame_layout_test PRIVATE
  CAELIX_SHADER_SPV_DIR="${CMAKE_BINARY_DIR}/shaders"
  CAELIX_SHADER_SOURCE_DIR="${PROJECT_SOURCE_DIR}/shaders")
target_link_libraries(caelix_frame_layout_test PRIVATE Catch2::Catch2WithMain)
# The .spv files are outputs of the caelix_renderer build (cmake/Shaders.cmake).
add_dependencies(caelix_frame_layout_test caelix_renderer)
add_test(NAME caelix_frame_layout_test COMMAND caelix_frame_layout_test)

# The light tree (docs/LIGHT_TREE.md) through the C ABI and a probe kernel that runs its device API
# (tests/shaders/caelix_light_tree_probe.hlsl, compiled, not embedded: the test loads it from
# ${CMAKE_BINARY_DIR}/shaders): light faces against a CPU reference, rows, leaves, records,
# normalisation, sample/pdf consistency, histogram, bound property, updates, record copies, heap
# growth. The hidden case [.bench] measures draws, the group pass and the BVH builds at about 10k,
# 54k and 1M light bricks.
caelix_compile_shaders(light_tree_probe_spv
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/caelix_light_tree_probe.hlsl)
add_custom_target(caelix_light_tree_probe_shaders DEPENDS ${light_tree_probe_spv})
caelix_add_gpu_test(caelix_light_tree_test light_tree_test.cpp)
add_dependencies(caelix_light_tree_test caelix_light_tree_probe_shaders)
# The library's layout headers (lib/src/light/light_tree_layout.h) are the test's view of the GPU
# memory; the probe and the library kernels load from the build's shader folder.
target_include_directories(caelix_light_tree_test PRIVATE ${PROJECT_SOURCE_DIR}/lib/src)
target_compile_definitions(caelix_light_tree_test PRIVATE
  CAELIX_TEST_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")

# The reusable GPU hash table (docs/HASH_TABLE.md) against a CPU model: the library's
# maintenance kernels (from ${CMAKE_BINARY_DIR}/shaders) and a probe kernel that runs the device API
# from many threads (tests/shaders/caelix_hash_probe.hlsl): contention on one key and on many keys,
# colliding home slots and equal key_lo, payload exhaustion, eviction, the key rebuild, the active list
# and the indirect dispatch, the free stack over random frames.
caelix_compile_shaders(hash_probe_spv
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/caelix_hash_probe.hlsl)
add_custom_target(caelix_hash_probe_shaders DEPENDS ${hash_probe_spv})
caelix_add_gpu_test(caelix_hash_table_test hash_table_test.cpp)
add_dependencies(caelix_hash_table_test caelix_hash_probe_shaders)
target_include_directories(caelix_hash_table_test PRIVATE ${PROJECT_SOURCE_DIR}/lib/src)
target_compile_definitions(caelix_hash_table_test PRIVATE
  CAELIX_TEST_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")

# The light tree functions of the light NEE and its MIS weight through a probe kernel (tests/shaders/
# caelix_light_nee_probe.hlsl) on a scene with lamps on a plain and a rotated entity: the box sampler's
# support (every light face that can reach a receiver of a box has a positive box pdf and target); both
# MIS sides see the same exact pdf and their weights sum to 1; the weight at degenerate inputs. (The image
# tests of the light NEE are in caelix_trace_test.)
caelix_compile_shaders(light_nee_probe_spv
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/caelix_light_nee_probe.hlsl)
add_custom_target(caelix_light_nee_probe_shaders DEPENDS ${light_nee_probe_spv})
caelix_add_gpu_test(caelix_light_nee_test light_nee_test.cpp)
add_dependencies(caelix_light_nee_test caelix_light_nee_probe_shaders)
target_compile_definitions(caelix_light_nee_test PRIVATE
  CAELIX_TEST_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")

# Upload benchmark, not a test (no add_test): the record upload of a synthetic FractalCave (one slab of
# 18,432 bricks re-uploaded per frame) with the flush GPU time from timestamps. --path auto|bar|bounce
# sets CAELIX_UPLOAD_PATH; --no-validation for timings. Its own main() runs one Catch2 test case (the
# helpers of test_device.h use Catch2 assertions).
add_executable(caelix_upload_bench upload_bench.cpp)
if(MSVC)
  target_compile_options(caelix_upload_bench PRIVATE /W4)
else()
  target_compile_options(caelix_upload_bench PRIVATE -Wall -Wextra)
endif()
target_link_libraries(caelix_upload_bench PRIVATE caelix_test_device Catch2::Catch2)
