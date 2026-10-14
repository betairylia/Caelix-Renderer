# caelix_unity: the Unity native plugin (docs/UNITY_PLUGIN.md). Links the renderer's C ABI only.
#
# Unity's plugin API headers are NOT copied into this repository (Unity Companion License, see
# docs/UNITY_PLUGIN.md); they are read from the Unity installation.
set(UNITY_PLUGIN_API_DIR "F:/Unity/6000.5.6f1/Editor/Data/PluginAPI" CACHE PATH
  "Directory of Unity's native plugin API headers (IUnityInterface.h, IUnityGraphicsVulkan.h)")
# When set, the build copies caelix_unity.dll, caelix_renderer.dll and their .pdb files there
# (for example <Caelix package>/Runtime/Plugins/x86_64). Unity locks loaded plugins: close the
# Editor before building.
set(CAELIX_UNITY_PLUGIN_DIR "" CACHE PATH "Copy the plugin DLLs to this directory after the build (empty = no copy)")

if(NOT EXISTS "${UNITY_PLUGIN_API_DIR}/IUnityGraphicsVulkan.h")
  message(FATAL_ERROR "UNITY_PLUGIN_API_DIR (${UNITY_PLUGIN_API_DIR}) has no IUnityGraphicsVulkan.h; "
    "set it to <Unity>/Editor/Data/PluginAPI or configure with -DCAELIX_BUILD_UNITY_PLUGIN=OFF")
endif()

add_library(caelix_unity SHARED
  include/caelix_unity.h
  plugin_state.h
  unity_plugin.cpp
  unity_bridge.cpp
  vulkan_intercept.cpp)
target_include_directories(caelix_unity
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# SYSTEM: Unity's headers are external code, our /W4 does not apply to them.
target_include_directories(caelix_unity SYSTEM PRIVATE "${UNITY_PLUGIN_API_DIR}")
target_compile_definitions(caelix_unity PRIVATE CAELIX_UNITY_BUILD VK_NO_PROTOTYPES)
if(MSVC)
  target_compile_options(caelix_unity PRIVATE /W4 /WX)
else()
  target_compile_options(caelix_unity PRIVATE -Wall -Wextra -Werror)
endif()
# Vulkan types only: the plugin loads the few functions it calls itself from the
# vkGetInstanceProcAddr that Unity hands to the init callback (no volk state shared with the library).
target_link_libraries(caelix_unity PRIVATE caelix_renderer Vulkan::Headers)

if(CAELIX_UNITY_PLUGIN_DIR)
  add_custom_command(TARGET caelix_unity POST_BUILD
    COMMAND ${CMAKE_COMMAND}
      -DDEST=${CAELIX_UNITY_PLUGIN_DIR}
      "-DFILES=$<TARGET_FILE:caelix_unity>;$<TARGET_FILE:caelix_renderer>;$<TARGET_PDB_FILE:caelix_unity>;$<TARGET_PDB_FILE:caelix_renderer>"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/CopyPlugin.cmake
    COMMENT "Copying the Unity plugin DLLs to ${CAELIX_UNITY_PLUGIN_DIR}"
    VERBATIM)
endif()
