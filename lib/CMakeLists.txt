add_library(caelix_renderer SHARED src/caelix.cpp)
set_target_properties(caelix_renderer PROPERTIES OUTPUT_NAME caelix_renderer)
target_include_directories(caelix_renderer PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(caelix_renderer PRIVATE
  CAELIX_BUILD_SHARED
  CAELIX_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  CAELIX_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  CAELIX_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  CAELIX_VERSION_STRING="${PROJECT_VERSION}")
target_link_libraries(caelix_renderer PRIVATE
  volk::volk
  GPUOpen::VulkanMemoryAllocator
  vk-bootstrap::vk-bootstrap)
