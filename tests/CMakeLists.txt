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
# validation layers when installed), readback buffers, the frame loop.
add_library(caelix_test_device STATIC test_device.cpp test_device.h)
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
