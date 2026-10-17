// Unity plugin entry points: UnityPluginLoad / UnityPluginUnload, logging, graphics device events
// (adopt Unity's Vulkan device at initialization, release everything at shutdown).
#include "plugin_state.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace caelix_unity {
namespace {

State g_state;
const std::chrono::steady_clock::time_point g_load_time = std::chrono::steady_clock::now();

// Routes the renderer library's log to Unity's. DEBUG messages are dropped (per-frame noise).
void LibraryLogCallback(CaelixLogLevel level, const char* message, void* /*user_data*/)
{
    switch (level)
    {
        case CAELIX_LOG_DEBUG: return;
        case CAELIX_LOG_INFO: Log(LogLevel::Info, "caelix: %s", message); return;
        case CAELIX_LOG_WARNING: Log(LogLevel::Warning, "caelix: %s", message); return;
        case CAELIX_LOG_ERROR: Log(LogLevel::Error, "caelix: %s", message); return;
    }
}

PFN_vkVoidFunction DeviceProc(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkDevice device,
                              const char* name)
{
    auto get_device_proc_addr =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
    return get_device_proc_addr != nullptr ? get_device_proc_addr(device, name) : nullptr;
}

void OnDeviceInitialize()
{
    State& state = GetState();
    const UnityGfxRenderer renderer = state.graphics->GetRenderer();
    state.unity_renderer.store(static_cast<uint32_t>(renderer));
    if (renderer != kUnityGfxRendererVulkan)
    {
        Log(LogLevel::Warning, "graphics device initialized with renderer %d, not Vulkan (21): the "
                               "plugin stays inactive",
            static_cast<int>(renderer));
        return;
    }
    if (state.device != nullptr)
    {
        return;  // already adopted (UnityPluginLoad's manual call and the real event)
    }
    const UnityVulkanInstance instance = state.vulkan->Instance();
    if (instance.device == VK_NULL_HANDLE)
    {
        Log(LogLevel::Info, "graphics device event: Unity's Vulkan device does not exist yet");
        return;
    }
    Log(LogLevel::Info, "graphics device initialized: VkInstance %p, VkDevice %p, queue family %u",
        static_cast<void*>(instance.instance), static_cast<void*>(instance.device),
        instance.queueFamilyIndex);

    // Proof that the extensions are enabled on Unity's device: vkGetDeviceProcAddr returns NULL
    // for a command of an extension that is not enabled.
    const bool has_as =
        DeviceProc(instance.getInstanceProcAddr, instance.instance, instance.device,
                   "vkCmdBuildAccelerationStructuresKHR") != nullptr;
    const bool has_dho =
        DeviceProc(instance.getInstanceProcAddr, instance.instance, instance.device,
                   "vkCreateDeferredOperationKHR") != nullptr;
    Log(has_as && has_dho ? LogLevel::Info : LogLevel::Error,
        "Unity's device: vkCmdBuildAccelerationStructuresKHR %s, vkCreateDeferredOperationKHR %s",
        has_as ? "present" : "MISSING", has_dho ? "present" : "MISSING");

    CaelixDeviceAdoptDesc desc{};
    desc.get_instance_proc_addr = reinterpret_cast<void*>(instance.getInstanceProcAddr);
    desc.instance = reinterpret_cast<uint64_t>(instance.instance);
    desc.physical_device = reinterpret_cast<uint64_t>(instance.physicalDevice);
    desc.device = reinterpret_cast<uint64_t>(instance.device);
    desc.graphics_queue = reinterpret_cast<uint64_t>(instance.graphicsQueue);
    desc.graphics_queue_family = instance.queueFamilyIndex;
    CaelixDevice* device = nullptr;
    const CaelixResult result = caelix_device_adopt(&desc, &device);
    if (result != CAELIX_OK)
    {
        state.last_error.store(static_cast<int32_t>(result));
        Log(LogLevel::Error, "caelix_device_adopt failed: CaelixResult %d", static_cast<int>(result));
        return;
    }
    state.device = device;
    state.vk_device = instance.device;
    state.vk_queue = instance.graphicsQueue;
    state.device_wait_idle = reinterpret_cast<PFN_vkDeviceWaitIdle>(
        DeviceProc(instance.getInstanceProcAddr, instance.instance, instance.device,
                   "vkDeviceWaitIdle"));
    state.queue_wait_idle = reinterpret_cast<PFN_vkQueueWaitIdle>(
        DeviceProc(instance.getInstanceProcAddr, instance.instance, instance.device,
                   "vkQueueWaitIdle"));
    state.device_adopted.store(1);
    Log(LogLevel::Info, "caelix_device_adopt OK");

    ConfigureEvents();
}

void OnDeviceShutdown()
{
    State& state = GetState();
    if (state.device == nullptr)
    {
        return;
    }
    // Unity guarantees an idle device at shutdown; wait anyway, the call is cheap then.
    if (state.device_wait_idle != nullptr)
    {
        state.device_wait_idle(state.vk_device);
    }
    DestroyRenderObjects();
    caelix_device_destroy(state.device);
    state.device = nullptr;
    state.vk_device = VK_NULL_HANDLE;
    state.vk_queue = VK_NULL_HANDLE;
    state.device_adopted.store(0);
    Log(LogLevel::Info, "graphics device shutdown: renderer, scene and device released");
}

void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType event_type)
{
    switch (event_type)
    {
        case kUnityGfxDeviceEventInitialize: OnDeviceInitialize(); break;
        case kUnityGfxDeviceEventShutdown: OnDeviceShutdown(); break;
        default: break;
    }
}

}  // namespace

State& GetState()
{
    return g_state;
}

const char* FormatVersion(uint32_t version, char* buffer, size_t size)
{
    std::snprintf(buffer, size, "%u.%u.%u", VK_API_VERSION_MAJOR(version),
                  VK_API_VERSION_MINOR(version), VK_API_VERSION_PATCH(version));
    return buffer;
}

void Log(LogLevel level, const char* format, ...)
{
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                g_load_time)
                          .count();
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "[caelix_unity +%.1f ms] ", ms);

    va_list args;
    va_start(args, format);
    va_list args_copy;
    va_copy(args_copy, args);
    const int length = std::vsnprintf(nullptr, 0, format, args);
    va_end(args);
    std::string message(prefix);
    if (length > 0)
    {
        const size_t offset = message.size();
        message.resize(offset + static_cast<size_t>(length) + 1);
        std::vsnprintf(message.data() + offset, static_cast<size_t>(length) + 1, format, args_copy);
        message.resize(offset + static_cast<size_t>(length));
    }
    va_end(args_copy);

    IUnityLog* log = g_state.log;
    if (log == nullptr)
    {
#if defined(_WIN32)
        message += "\n";
        OutputDebugStringA(message.c_str());
#endif
        return;
    }
    switch (level)
    {
        case LogLevel::Info: UNITY_LOG(log, message.c_str()); break;
        case LogLevel::Warning: UNITY_LOG_WARNING(log, message.c_str()); break;
        case LogLevel::Error: UNITY_LOG_ERROR(log, message.c_str()); break;
    }
}

}  // namespace caelix_unity

using caelix_unity::GetState;
using caelix_unity::Log;
using caelix_unity::LogLevel;

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginLoad(
    IUnityInterfaces* interfaces)
{
    caelix_unity::State& state = GetState();
    state.interfaces = interfaces;
    state.log = interfaces->Get<IUnityLog>();
    state.graphics = interfaces->Get<IUnityGraphics>();
    state.vulkan = interfaces->Get<IUnityGraphicsVulkanV2>();
    caelix_set_log_callback(&caelix_unity::LibraryLogCallback, nullptr);

    const UnityGfxRenderer renderer =
        state.graphics != nullptr ? state.graphics->GetRenderer() : static_cast<UnityGfxRenderer>(0);
    state.unity_renderer.store(static_cast<uint32_t>(renderer));
    Log(LogLevel::Info, "UnityPluginLoad: caelix_unity loaded, caelix_renderer %s, Unity renderer %d "
                        "(4 = null: no device yet, 21 = Vulkan)",
        caelix_version_string(), static_cast<int>(renderer));
    if (state.graphics == nullptr)
    {
        Log(LogLevel::Error, "IUnityGraphics is not available: the plugin stays inactive");
        return;
    }
    // Event ids are global across plugins (ConfigureEvent applies to an id), so reserve our own.
    state.frame_event_id = state.graphics->ReserveEventIDRange(3);
    state.render_event_id = state.frame_event_id + 1;
    state.release_event_id = state.frame_event_id + 2;
    Log(LogLevel::Info, "plugin event ids: frame %d, render %d, release %d", state.frame_event_id,
        state.render_event_id, state.release_event_id);

    // At preload time Unity has not created a graphics device yet and reports the null renderer
    // (4); only a known other renderer switches the plugin off. The init callback runs only when
    // Unity initializes Vulkan, so registering it is harmless otherwise.
    if (renderer != 0 && renderer != kUnityGfxRendererNull && renderer != kUnityGfxRendererVulkan)
    {
        Log(LogLevel::Warning, "Unity renderer %d is not Vulkan: the plugin stays inactive",
            static_cast<int>(renderer));
        return;
    }
    if (state.vulkan == nullptr)
    {
        Log(LogLevel::Error, "IUnityGraphicsVulkanV2 is not available: the plugin stays inactive");
        return;
    }
    const bool registered =
        state.vulkan->InterceptInitialization(&caelix_unity::InterceptVulkanInitialization, nullptr);
    state.intercept_registered.store(registered ? 1u : 0u);
    Log(registered ? LogLevel::Info : LogLevel::Error,
        "InterceptInitialization returned %s%s", registered ? "true" : "false",
        registered ? "" : " (the plugin loaded after Unity created its Vulkan device: enable "
                          "'Load on startup' on caelix_unity.dll)");

    state.graphics->RegisterDeviceEventCallback(&caelix_unity::OnGraphicsDeviceEvent);
    // A plugin loaded after the device exists gets no Initialize event: run it now. At preload the
    // device does not exist yet and the call returns early.
    caelix_unity::OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginUnload()
{
    caelix_unity::State& state = GetState();
    if (state.graphics != nullptr)
    {
        state.graphics->UnregisterDeviceEventCallback(&caelix_unity::OnGraphicsDeviceEvent);
    }
    Log(LogLevel::Info, "UnityPluginUnload");
    caelix_set_log_callback(nullptr, nullptr);
    state.log = nullptr;
}
