// Shared state of the Unity plugin (unity_plugin.cpp, vulkan_intercept.cpp, unity_bridge.cpp).
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include "IUnityGraphics.h"
#include "IUnityGraphicsVulkan.h"
#include "IUnityInterface.h"
#include "IUnityLog.h"

#include <caelix/caelix.h>

#include <atomic>
#include <cstdint>

namespace caelix_unity {

enum class LogLevel { Info, Warning, Error };

// printf-style message to Unity's log (IUnityLog), prefixed with "[caelix_unity +<ms>]", the time
// since UnityPluginLoad. Without IUnityLog it goes to OutputDebugString. Any thread.
void Log(LogLevel level, const char* format, ...);

// "1.3.0" for a Vulkan api version. Returns `buffer`.
const char* FormatVersion(uint32_t version, char* buffer, size_t size);

struct State {
  // Set in UnityPluginLoad, read on every thread afterwards.
  IUnityInterfaces* interfaces = nullptr;
  IUnityGraphics* graphics = nullptr;
  IUnityGraphicsVulkanV2* vulkan = nullptr;
  IUnityLog* log = nullptr;
  int render_event_id = 0;
  int release_event_id = 0;

  // Status for C# (caelix_unity_status).
  std::atomic<uint32_t> unity_renderer{0};
  std::atomic<uint32_t> intercept_registered{0};
  std::atomic<uint32_t> intercept_called{0};
  std::atomic<uint32_t> instance_patched{0};
  std::atomic<uint32_t> device_patched{0};
  std::atomic<uint32_t> device_adopted{0};
  std::atomic<int32_t> last_error{0};
  std::atomic<uint64_t> last_consumed_sequence{0};
  std::atomic<uint64_t> last_unity_frame{0};
  std::atomic<uint64_t> last_safe_frame{0};
  std::atomic<uint64_t> last_plugin_frame{0};
  std::atomic<uint64_t> renders{0};
  std::atomic<uint64_t> failed_events{0};
  std::atomic<uint64_t> view_releases{0};
  // Version 2 status: the scene after the newest flush (CaelixUnityStatus).
  std::atomic<uint32_t> scene_encoding{0xFFFFFFFFu};
  std::atomic<uint32_t> scene_entities{0};
  std::atomic<uint32_t> scene_groups{0};
  std::atomic<uint32_t> scene_recreations{0};
  std::atomic<uint64_t> bricks_resident{0};
  std::atomic<uint64_t> bricks_pending_plugin{0};
  std::atomic<uint64_t> as_builds_pending{0};
  std::atomic<uint64_t> uploaded_total{0};
  std::atomic<uint64_t> removed_total{0};
  std::atomic<uint64_t> scene_packets{0};
  std::atomic<uint64_t> record_bytes{0};
  std::atomic<uint64_t> blas_bytes{0};
  std::atomic<float> flush_ms{0.0f};
  std::atomic<float> trace_ms{0.0f};
  std::atomic<float> composite_ms{0.0f};
  std::atomic<float> denoise_ms{0.0f};
  std::atomic<uint64_t> timings_frame{0};

  // Render thread only (device events and plugin events).
  CaelixDevice* device = nullptr;
  CaelixScene* scene = nullptr;
  CaelixRenderer* renderer = nullptr;
  VkDevice vk_device = VK_NULL_HANDLE;
  VkQueue vk_queue = VK_NULL_HANDLE;
  PFN_vkDeviceWaitIdle device_wait_idle = nullptr;
  PFN_vkQueueWaitIdle queue_wait_idle = nullptr;
};

State& GetState();

// vulkan_intercept.cpp: the UnityVulkanInitCallback given to InterceptInitialization.
PFN_vkGetInstanceProcAddr UNITY_INTERFACE_API InterceptVulkanInitialization(
    PFN_vkGetInstanceProcAddr get_instance_proc_addr, void* user_data);

// unity_bridge.cpp: plugin events and the render-thread objects.
void UNITY_INTERFACE_API OnPluginEvent(int event_id, void* data);
// Configures the two plugin events (IUnityGraphicsVulkanV2::ConfigureEvent).
void ConfigureEvents();
// Destroys renderer and scene (device idle) and forgets the frame mapping, the entity map and the
// pending uploads.
void DestroyRenderObjects();

}  // namespace caelix_unity
