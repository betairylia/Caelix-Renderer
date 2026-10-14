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

# Scene smoke test: a Vulkan 1.3 device (vk-bootstrap, validation layers when
# installed) and the whole upload -> flush -> GPU readback path.
add_executable(caelix_scene_smoke scene_smoke.cpp)
target_compile_definitions(caelix_scene_smoke PRIVATE VK_NO_PROTOTYPES)
if(MSVC)
  target_compile_options(caelix_scene_smoke PRIVATE /W4)
else()
  target_compile_options(caelix_scene_smoke PRIVATE -Wall -Wextra)
endif()
target_link_libraries(caelix_scene_smoke PRIVATE
  caelix_renderer
  volk::volk
  vk-bootstrap::vk-bootstrap
  Catch2::Catch2WithMain)
add_test(NAME caelix_scene_smoke COMMAND caelix_scene_smoke)
