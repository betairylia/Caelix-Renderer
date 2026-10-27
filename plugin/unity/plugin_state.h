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
#include <caelix_unity.h>

#include <atomic>
#include <cstdint>
#include <mutex>

namespace caelix_unity {

enum class LogLevel
{
    Info,
    Warning,
    Error
};

// printf-style message to Unity's log (IUnityLog), prefixed with "[caelix_unity +<ms>]", the time
// since UnityPluginLoad. Without IUnityLog it goes to OutputDebugString. Any thread.
void Log(LogLevel level, const char* format, ...);

// "1.3.0" for a Vulkan api version. Returns `buffer`.
const char* FormatVersion(uint32_t version, char* buffer, size_t size);

struct State
{
    // Set in UnityPluginLoad, read on every thread afterwards.
    IUnityInterfaces* interfaces = nullptr;
    IUnityGraphics* graphics = nullptr;
    IUnityGraphicsVulkanV2* vulkan = nullptr;
    IUnityLog* log = nullptr;
    // Plugin event ids (ReserveEventIDRange(3) in UnityPluginLoad): frame, render, release.
    int frame_event_id = 0;
    int render_event_id = 0;
    int release_event_id = 0;

    // Status for C# (caelix_unity_status, CaelixUnityStatus). Written on the render thread and in
    // the device events, read on any thread.
    std::atomic<uint32_t> unity_renderer{0};
    std::atomic<uint32_t> intercept_registered{0};
    std::atomic<uint32_t> intercept_called{0};
    std::atomic<uint32_t> instance_patched{0};
    std::atomic<uint32_t> device_patched{0};
    // 1 when vkCreateDevice enabled every device extension of the upscalers.
    std::atomic<uint32_t> upscaler_extensions{0};
    std::atomic<uint32_t> device_adopted{0};
    std::atomic<int32_t> last_error{0};
    // The packets the render thread is done with, one sequence per packet kind.
    std::atomic<uint64_t> last_consumed_frame_sequence{0};
    std::atomic<uint64_t> last_consumed_view_sequence{0};
    std::atomic<uint64_t> last_consumed_release_sequence{0};
    // Unity's frame numbers at the newest frame or render event, and the newest flushed frame.
    std::atomic<uint64_t> last_unity_frame{0};
    std::atomic<uint64_t> last_safe_frame{0};
    std::atomic<uint64_t> last_flushed_frame{0};
    std::atomic<uint64_t> frames{0};
    std::atomic<uint64_t> renders{0};
    std::atomic<uint64_t> releases{0};
    std::atomic<uint64_t> failed_events{0};
    // The scene after the newest flush.
    std::atomic<uint32_t> scene_encoding{0xFFFFFFFFu};
    std::atomic<uint32_t> scene_entities{0};
    std::atomic<uint32_t> scene_groups{0};
    std::atomic<uint32_t> scene_resets{0};
    std::atomic<uint32_t> live_views{0};
    std::atomic<uint64_t> bricks_resident{0};
    std::atomic<uint64_t> as_builds_pending{0};
    std::atomic<uint64_t> uploaded_total{0};
    std::atomic<uint64_t> removed_total{0};
    std::atomic<uint64_t> bricks_dropped{0};
    std::atomic<uint64_t> upload_bricks_free{0};
    std::atomic<uint64_t> upload_bricks_cap{0};
    std::atomic<uint64_t> record_bytes{0};
    std::atomic<uint64_t> blas_bytes{0};
    std::atomic<uint64_t> staging_bytes{0};
    // GPU timings of the view that asked for them (CAELIX_UNITY_VIEW_REPORT_TIMINGS).
    std::atomic<float> flush_ms{0.0f};
    std::atomic<float> trace_ms{0.0f};
    std::atomic<float> composite_ms{0.0f};
    std::atomic<float> denoise_ms{0.0f};
    std::atomic<float> upscale_ms{0.0f};
    std::atomic<uint64_t> timings_frame{0};
    std::atomic<uint32_t> timings_view{0};
    // The light tree after the newest flush (CaelixLightStats), and per view bits of the newest
    // render: bit v = view v ran the light NEE / passed a voxel id target.
    std::atomic<uint64_t> light_bricks{0};
    std::atomic<float> light_gpu_ms{0.0f};
    std::atomic<uint32_t> light_enabled{0};
    std::atomic<uint32_t> light_nee_running{0};
    std::atomic<uint32_t> voxel_id_views{0};
    // The newest upscaler query of each view (CaelixUnityStatus::upscalers). Written on the render
    // thread when a view's query key changes, read by caelix_unity_status; the mutex keeps the key
    // and the result of one entry together.
    std::mutex upscalers_mutex;
    CaelixUnityUpscalerStatus upscalers[CAELIX_UNITY_MAX_VIEWS] = {};

    // Render thread only (device events and plugin events).
    CaelixDevice* device = nullptr;
    CaelixScene* scene = nullptr;
    // The scene of the current session: created by the frame packet with CAELIX_UNITY_FRAME_RESET,
    // destroyed by the release event with CAELIX_UNITY_RELEASE_SCENE.
    // One renderer per view (CaelixUnityViewPacket::view), all over `scene`. Created on the first
    // render event of the view, destroyed by a release event (VIEW or SCENE).
    CaelixRenderer* renderers[CAELIX_UNITY_MAX_VIEWS] = {};
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
// Configures the three plugin events (IUnityGraphicsVulkanV2::ConfigureEvent).
void ConfigureEvents();
// Destroys every view's renderer and the scene (device idle) and forgets the render-thread state
// (the entity map, the newest flushed frame, the log counters).
void DestroyRenderObjects();

}  // namespace caelix_unity
