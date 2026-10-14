add_library(caelix_renderer SHARED
  src/caelix.cpp
  src/device.cpp
  src/device.h
  src/log.cpp
  src/log.h
  src/vk_common.h
  src/vma.cpp)
set_target_properties(caelix_renderer PROPERTIES OUTPUT_NAME caelix_renderer)
target_include_directories(caelix_renderer PUBLIC ${PROJECT_SOURCE_DIR}/include)
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
  GPUOpen::VulkanMemoryAllocator)
