// Interception of Unity's Vulkan initialization (IUnityGraphicsVulkanV2::InterceptInitialization).
//
// Unity calls InterceptVulkanInitialization with the loader's vkGetInstanceProcAddr and then uses
// the function we return for every Vulkan entry point. We return HookGetInstanceProcAddr, which
// hands out two wrappers and passes everything else through:
//   vkCreateInstance: raises VkApplicationInfo::apiVersion to the library's api version (1.3) when
//     Unity asks for less, so the core 1.2 / 1.3 entry points that the library loads exist, and
//     appends the upscalers' instance extensions (caelix_upscaler_instance_extensions) that the
//     loader offers. A missing one is logged and skipped.
//   vkCreateDevice: appends the library's device extensions and enables its feature bits
//     (caelix_get_device_requirements), keeping everything Unity asked for, and appends the
//     upscalers' device extensions (caelix_upscaler_device_extensions) that the device offers.
//     Those are optional: a missing one is logged and skipped (NGX then does not start, the
//     upscaler query reports CAELIX_UPSCALER_NO_NGX); CaelixUnityStatus::upscaler_extensions is 1
//     only when every one was enabled.
//
// Features, without breaking Vulkan's valid-usage rules on Unity's own chain:
//   - A feature struct that Unity already chained is edited IN PLACE (the bits are OR-ed in) and
//     restored after vkCreateDevice returns. A struct type may appear only once in a chain.
//   - Missing structs are prepended (our structs first, then Unity's chain unchanged).
//   - Vulkan 1.2 / 1.3 bits go into VkPhysicalDeviceVulkan12Features / 13Features when Unity uses
//     those, else into the individual structs (VkPhysicalDeviceBufferDeviceAddressFeatures, ...).
//     A chain must not mix VkPhysicalDeviceVulkan12Features with the structs it replaces, and
//     adding VkPhysicalDeviceVulkan12Features would also oblige us to set the bits of every
//     promoted extension Unity enabled (draw indirect count, ...), so the individual structs are
//     the safe default.
//   - Core bits go into Unity's VkPhysicalDeviceFeatures2 when it has one; otherwise a new
//     VkPhysicalDeviceFeatures2 takes Unity's pEnabledFeatures (copied) plus our bits, and
//     pEnabledFeatures becomes NULL (the two must not be used together).
// When the physical device lacks a requirement, the wrapper logs it and creates Unity's device
// unchanged (caelix_device_adopt then reports CAELIX_ERROR_UNSUPPORTED). When vkCreateDevice fails
// with our additions, it logs the error and retries with Unity's create info. vkCreateInstance
// does the same when its additions fail.
#include "plugin_state.h"

#include <cstring>
#include <string>
#include <vector>

namespace caelix_unity {
namespace {

PFN_vkGetInstanceProcAddr g_next_gipa = nullptr;
VkInstance g_instance = VK_NULL_HANDLE;  // the newest instance Unity created

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL HookGetInstanceProcAddr(VkInstance instance,
                                                                 const char* name);

std::string JoinNames(const char* const* names, uint32_t count)
{
    std::string joined;
    for (uint32_t i = 0; i < count; ++i)
    {
        if (i > 0) joined += ", ";
        joined += names[i];
    }
    return joined.empty() ? std::string("(none)") : joined;
}

bool ContainsName(const std::vector<const char*>& names, const char* name)
{
    for (const char* existing : names)
    {
        if (std::strcmp(existing, name) == 0) return true;
    }
    return false;
}

bool Offers(const std::vector<VkExtensionProperties>& offered, const char* name)
{
    for (const VkExtensionProperties& extension : offered)
    {
        if (std::strcmp(extension.extensionName, name) == 0) return true;
    }
    return false;
}

void AppendName(std::string& list, const char* name)
{
    if (!list.empty()) list += ", ";
    list += name;
}

// The device extensions the physical device offers (empty when the query cannot be loaded).
std::vector<VkExtensionProperties> DeviceExtensions(VkPhysicalDevice physical_device)
{
    auto enumerate_extensions = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        g_next_gipa(g_instance, "vkEnumerateDeviceExtensionProperties"));
    std::vector<VkExtensionProperties> available;
    if (enumerate_extensions == nullptr) return available;
    uint32_t count = 0;
    enumerate_extensions(physical_device, nullptr, &count, nullptr);
    available.resize(count);
    enumerate_extensions(physical_device, nullptr, &count, available.data());
    available.resize(count);
    return available;
}

// ---------------------------------------------------------------------------------------------
// vkCreateInstance

VKAPI_ATTR VkResult VKAPI_CALL HookCreateInstance(const VkInstanceCreateInfo* create_info,
                                                  const VkAllocationCallbacks* allocator,
                                                  VkInstance* out_instance)
{
    auto next = reinterpret_cast<PFN_vkCreateInstance>(g_next_gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    if (next == nullptr)
    {
        Log(LogLevel::Error, "vkCreateInstance: the loader returned no vkCreateInstance");
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (create_info == nullptr)
    {
        return next(create_info, allocator, out_instance);
    }

    CaelixDeviceRequirements requirements{};
    caelix_get_device_requirements(&requirements);

    uint32_t loader_version = VK_API_VERSION_1_0;
    auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        g_next_gipa(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    if (enumerate_version != nullptr)
    {
        enumerate_version(&loader_version);
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    if (create_info->pApplicationInfo != nullptr)
    {
        app = *create_info->pApplicationInfo;
    }
    // apiVersion 0 means 1.0.
    const uint32_t unity_version = app.apiVersion == 0 ? VK_API_VERSION_1_0 : app.apiVersion;
    uint32_t our_version = unity_version;
    if (unity_version < requirements.api_version)
    {
        if (loader_version >= requirements.api_version)
        {
            our_version = requirements.api_version;
        }
        else
        {
            char loader_text[32];
            Log(LogLevel::Error, "vkCreateInstance: the Vulkan loader supports only %s; the instance "
                                 "stays at Unity's version",
                FormatVersion(loader_version, loader_text, sizeof(loader_text)));
        }
    }
    app.apiVersion = our_version;
    VkInstanceCreateInfo info = *create_info;
    info.pApplicationInfo = &app;

    // The upscalers' instance extensions that the loader offers and Unity did not enable.
    std::vector<const char*> extensions(create_info->ppEnabledExtensionNames,
                                        create_info->ppEnabledExtensionNames + create_info->enabledExtensionCount);
    std::string upscaler_added;
    std::string upscaler_missing;
    {
        uint32_t upscaler_count = 0;
        const char* const* upscaler_names = nullptr;
        caelix_upscaler_instance_extensions(&upscaler_count, &upscaler_names);
        std::vector<VkExtensionProperties> offered;
        auto enumerate_extensions = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
            g_next_gipa(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
        if (upscaler_count > 0 && enumerate_extensions != nullptr)
        {
            uint32_t count = 0;
            enumerate_extensions(nullptr, &count, nullptr);
            offered.resize(count);
            enumerate_extensions(nullptr, &count, offered.data());
            offered.resize(count);
        }
        for (uint32_t i = 0; i < upscaler_count; ++i)
        {
            const char* name = upscaler_names[i];
            if (ContainsName(extensions, name)) continue;
            if (Offers(offered, name))
            {
                extensions.push_back(name);
                AppendName(upscaler_added, name);
            }
            else
            {
                AppendName(upscaler_missing, name);
            }
        }
    }
    if (!upscaler_missing.empty())
    {
        Log(LogLevel::Warning, "vkCreateInstance: the loader lacks upscaler instance extensions (skipped): %s",
            upscaler_missing.c_str());
    }
    info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();

    char unity_text[32], our_text[32], loader_text[32];
    Log(LogLevel::Info, "vkCreateInstance: Unity requested Vulkan %s%s, the plugin requests %s "
                        "(loader %s); instance extensions: %s; added for the upscalers: %s",
        FormatVersion(unity_version, unity_text, sizeof(unity_text)),
        create_info->pApplicationInfo == nullptr ? " (no VkApplicationInfo)" : "",
        FormatVersion(our_version, our_text, sizeof(our_text)),
        FormatVersion(loader_version, loader_text, sizeof(loader_text)),
        JoinNames(create_info->ppEnabledExtensionNames, create_info->enabledExtensionCount).c_str(),
        upscaler_added.empty() ? "none" : upscaler_added.c_str());

    VkResult result = next(&info, allocator, out_instance);
    if (result != VK_SUCCESS && (our_version != unity_version || !upscaler_added.empty()))
    {
        Log(LogLevel::Error, "vkCreateInstance with Vulkan %s and the added extensions failed (VkResult %d); "
                             "retrying with Unity's create info",
            our_text, static_cast<int>(result));
        result = next(create_info, allocator, out_instance);
        our_version = unity_version;
    }
    if (result == VK_SUCCESS)
    {
        g_instance = *out_instance;
        GetState().instance_patched.store(our_version >= requirements.api_version ? 1u : 0u);
        Log(LogLevel::Info, "vkCreateInstance OK: VkInstance %p, Vulkan %s",
            static_cast<void*>(*out_instance), FormatVersion(our_version, our_text, sizeof(our_text)));
    }
    else
    {
        Log(LogLevel::Error, "vkCreateInstance failed: VkResult %d", static_cast<int>(result));
    }
    return result;
}

// ---------------------------------------------------------------------------------------------
// vkCreateDevice

// What the library needs, as Vulkan feature bits.
struct Wanted
{
    bool buffer_device_address = false;
    bool descriptor_indexing = false;
    bool timeline_semaphore = false;
    bool scalar_block_layout = false;
    bool synchronization2 = false;
    bool dynamic_rendering = false;
    bool acceleration_structure = false;
    bool ray_query = false;
    bool shader_int64 = false;
    bool storage_image_write_without_format = false;
    bool storage_image_read_without_format = false;
};

Wanted WantedFromFlags(uint32_t flags)
{
    Wanted w;
    w.buffer_device_address = (flags & CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS) != 0;
    w.descriptor_indexing = (flags & CAELIX_FEATURE_DESCRIPTOR_INDEXING) != 0;
    w.timeline_semaphore = (flags & CAELIX_FEATURE_TIMELINE_SEMAPHORE) != 0;
    w.scalar_block_layout = (flags & CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT) != 0;
    w.synchronization2 = (flags & CAELIX_FEATURE_SYNCHRONIZATION2) != 0;
    w.dynamic_rendering = (flags & CAELIX_FEATURE_DYNAMIC_RENDERING) != 0;
    w.acceleration_structure = (flags & CAELIX_FEATURE_ACCELERATION_STRUCTURE) != 0;
    w.ray_query = (flags & CAELIX_FEATURE_RAY_QUERY) != 0;
    w.shader_int64 = (flags & CAELIX_FEATURE_SHADER_INT64) != 0;
    w.storage_image_write_without_format =
        (flags & CAELIX_FEATURE_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT) != 0;
    w.storage_image_read_without_format =
        (flags & CAELIX_FEATURE_STORAGE_IMAGE_READ_WITHOUT_FORMAT) != 0;
    return w;
}

void Enable(VkBool32& bit, bool wanted)
{
    if (wanted) bit = VK_TRUE;
}

// Returns false (and logs every missing item) when the physical device lacks a requirement.
bool CheckSupport(VkPhysicalDevice physical_device, const CaelixDeviceRequirements& requirements,
                  const Wanted& w)
{
    auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        g_next_gipa(g_instance, "vkGetPhysicalDeviceProperties"));
    auto enumerate_extensions = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        g_next_gipa(g_instance, "vkEnumerateDeviceExtensionProperties"));
    auto get_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
        g_next_gipa(g_instance, "vkGetPhysicalDeviceFeatures2"));
    if (get_properties == nullptr || enumerate_extensions == nullptr || get_features2 == nullptr)
    {
        Log(LogLevel::Error, "vkCreateDevice: cannot load the physical device queries (instance %p)",
            static_cast<void*>(g_instance));
        return false;
    }
    bool ok = true;
    VkPhysicalDeviceProperties properties{};
    get_properties(physical_device, &properties);
    if (properties.apiVersion < requirements.api_version)
    {
        char text[32];
        Log(LogLevel::Error, "vkCreateDevice: %s supports Vulkan %s only", properties.deviceName,
            FormatVersion(properties.apiVersion, text, sizeof(text)));
        ok = false;
    }
    const std::vector<VkExtensionProperties> available = DeviceExtensions(physical_device);
    for (uint32_t i = 0; i < requirements.device_extension_count; ++i)
    {
        if (!Offers(available, requirements.device_extensions[i]))
        {
            Log(LogLevel::Error, "vkCreateDevice: %s lacks extension %s", properties.deviceName,
                requirements.device_extensions[i]);
            ok = false;
        }
    }
    if (!ok)
    {
        return false;  // the extension feature structs may only be queried when the extensions exist
    }

    VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    as.pNext = &rq;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    v13.pNext = &as;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.pNext = &v13;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &v12;
    get_features2(physical_device, &f2);

    struct Check
    {
        bool wanted;
        VkBool32 supported;
        const char* name;
    };
    const Check checks[] = {
        {w.buffer_device_address, v12.bufferDeviceAddress, "bufferDeviceAddress"},
        {w.descriptor_indexing, v12.runtimeDescriptorArray, "runtimeDescriptorArray"},
        {w.descriptor_indexing, v12.shaderStorageBufferArrayNonUniformIndexing,
         "shaderStorageBufferArrayNonUniformIndexing"},
        {w.descriptor_indexing, v12.descriptorBindingPartiallyBound, "descriptorBindingPartiallyBound"},
        {w.descriptor_indexing, v12.descriptorBindingVariableDescriptorCount,
         "descriptorBindingVariableDescriptorCount"},
        {w.timeline_semaphore, v12.timelineSemaphore, "timelineSemaphore"},
        {w.scalar_block_layout, v12.scalarBlockLayout, "scalarBlockLayout"},
        {w.synchronization2, v13.synchronization2, "synchronization2"},
        {w.dynamic_rendering, v13.dynamicRendering, "dynamicRendering"},
        {w.acceleration_structure, as.accelerationStructure, "accelerationStructure"},
        {w.ray_query, rq.rayQuery, "rayQuery"},
        {w.shader_int64, f2.features.shaderInt64, "shaderInt64"},
        {w.storage_image_write_without_format, f2.features.shaderStorageImageWriteWithoutFormat,
         "shaderStorageImageWriteWithoutFormat"},
        {w.storage_image_read_without_format, f2.features.shaderStorageImageReadWithoutFormat,
         "shaderStorageImageReadWithoutFormat"},
    };
    for (const Check& check : checks)
    {
        if (check.wanted && check.supported != VK_TRUE)
        {
            Log(LogLevel::Error, "vkCreateDevice: %s lacks feature %s", properties.deviceName, check.name);
            ok = false;
        }
    }
    return ok;
}

// The patched VkDeviceCreateInfo and everything it points to. Restores Unity's structs that it
// edited in place when destroyed.
class DevicePatch
{
public:
    DevicePatch() = default;
    DevicePatch(const DevicePatch&) = delete;
    DevicePatch& operator=(const DevicePatch&) = delete;
    ~DevicePatch()
    {
        for (auto it = saved_.rbegin(); it != saved_.rend(); ++it)
        {
            std::memcpy(it->target, it->bytes.data(), it->bytes.size());
        }
    }

    VkDeviceCreateInfo info{};
    std::vector<const char*> extensions;
    std::vector<std::string> added_extensions;
    std::string chain_report;    // the struct types of Unity's chain
    std::string feature_report;  // where each feature bit went

    // Saves Unity's struct before an in-place edit. T is the struct type.
    template <typename T>
    T* EditInPlace(const void* unity_struct)
    {
        Saved saved;
        saved.target = const_cast<void*>(unity_struct);
        saved.bytes.resize(sizeof(T));
        std::memcpy(saved.bytes.data(), unity_struct, sizeof(T));
        saved_.push_back(std::move(saved));
        return static_cast<T*>(const_cast<void*>(unity_struct));
    }

    // Our own structs (prepended). Fixed members, so the pointers stay valid.
    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceDescriptorIndexingFeatures di{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDeviceTimelineSemaphoreFeatures ts{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    VkPhysicalDeviceScalarBlockLayoutFeatures sbl{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES};
    VkPhysicalDeviceSynchronization2Features s2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    VkPhysicalDeviceDynamicRenderingFeatures dr{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};

    // Prepends one of our structs to the chain that starts at info.pNext.
    void Prepend(void* our_struct, const char* name)
    {
        auto* base = static_cast<VkBaseOutStructure*>(our_struct);
        base->pNext = static_cast<VkBaseOutStructure*>(const_cast<void*>(info.pNext));
        info.pNext = base;
        Report(name, "added");
    }

    void Report(const char* what, const char* how)
    {
        if (!feature_report.empty()) feature_report += "; ";
        feature_report += what;
        feature_report += " ";
        feature_report += how;
    }

private:
    struct Saved
    {
        void* target = nullptr;
        std::vector<unsigned char> bytes;
    };
    std::vector<Saved> saved_;
};

const VkBaseInStructure* FindInChain(const void* chain, VkStructureType type)
{
    for (auto* node = static_cast<const VkBaseInStructure*>(chain); node != nullptr;
         node = node->pNext)
    {
        if (node->sType == type) return node;
    }
    return nullptr;
}

// `optional_extensions`: the upscalers' device extensions that the device offers.
void BuildPatch(const VkDeviceCreateInfo& unity, const CaelixDeviceRequirements& requirements,
                const Wanted& w, const std::vector<const char*>& optional_extensions, DevicePatch& patch)
{
    patch.info = unity;

    // Extensions: Unity's, then ours that are missing, then the optional ones that are missing.
    patch.extensions.assign(unity.ppEnabledExtensionNames,
                            unity.ppEnabledExtensionNames + unity.enabledExtensionCount);
    auto add = [&patch](const char* name) {
        if (ContainsName(patch.extensions, name)) return;
        patch.extensions.push_back(name);
        patch.added_extensions.emplace_back(name);
    };
    for (uint32_t i = 0; i < requirements.device_extension_count; ++i)
    {
        add(requirements.device_extensions[i]);
    }
    for (const char* name : optional_extensions)
    {
        add(name);
    }
    patch.info.enabledExtensionCount = static_cast<uint32_t>(patch.extensions.size());
    patch.info.ppEnabledExtensionNames = patch.extensions.data();

    for (auto* node = static_cast<const VkBaseInStructure*>(unity.pNext); node != nullptr;
         node = node->pNext)
    {
        if (!patch.chain_report.empty()) patch.chain_report += ", ";
        patch.chain_report += std::to_string(static_cast<int>(node->sType));
    }
    if (patch.chain_report.empty()) patch.chain_report = "(empty)";

    const void* chain = unity.pNext;

    // Core features.
    if (const auto* found = FindInChain(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2))
    {
        auto* f2 = patch.EditInPlace<VkPhysicalDeviceFeatures2>(found);
        Enable(f2->features.shaderInt64, w.shader_int64);
        Enable(f2->features.shaderStorageImageWriteWithoutFormat, w.storage_image_write_without_format);
        Enable(f2->features.shaderStorageImageReadWithoutFormat, w.storage_image_read_without_format);
        patch.Report("core features", "OR-ed into Unity's VkPhysicalDeviceFeatures2");
    }
    else
    {
        if (unity.pEnabledFeatures != nullptr)
        {
            patch.features2.features = *unity.pEnabledFeatures;
        }
        Enable(patch.features2.features.shaderInt64, w.shader_int64);
        Enable(patch.features2.features.shaderStorageImageWriteWithoutFormat,
               w.storage_image_write_without_format);
        Enable(patch.features2.features.shaderStorageImageReadWithoutFormat,
               w.storage_image_read_without_format);
        patch.info.pEnabledFeatures = nullptr;
        patch.Prepend(&patch.features2, unity.pEnabledFeatures != nullptr
                                            ? "VkPhysicalDeviceFeatures2 (Unity's pEnabledFeatures + core bits)"
                                            : "VkPhysicalDeviceFeatures2 (core bits)");
    }

    // Vulkan 1.2 features.
    if (const auto* found = FindInChain(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES))
    {
        auto* v12 = patch.EditInPlace<VkPhysicalDeviceVulkan12Features>(found);
        Enable(v12->bufferDeviceAddress, w.buffer_device_address);
        Enable(v12->runtimeDescriptorArray, w.descriptor_indexing);
        Enable(v12->shaderStorageBufferArrayNonUniformIndexing, w.descriptor_indexing);
        Enable(v12->descriptorBindingPartiallyBound, w.descriptor_indexing);
        Enable(v12->descriptorBindingVariableDescriptorCount, w.descriptor_indexing);
        Enable(v12->timelineSemaphore, w.timeline_semaphore);
        Enable(v12->scalarBlockLayout, w.scalar_block_layout);
        patch.Report("1.2 features", "OR-ed into Unity's VkPhysicalDeviceVulkan12Features");
    }
    else
    {
        if (w.buffer_device_address)
        {
            if (const auto* f = FindInChain(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES))
            {
                patch.EditInPlace<VkPhysicalDeviceBufferDeviceAddressFeatures>(f)->bufferDeviceAddress = VK_TRUE;
                patch.Report("VkPhysicalDeviceBufferDeviceAddressFeatures", "edited");
            }
            else
            {
                patch.bda.bufferDeviceAddress = VK_TRUE;
                patch.Prepend(&patch.bda, "VkPhysicalDeviceBufferDeviceAddressFeatures");
            }
        }
        if (w.descriptor_indexing)
        {
            VkPhysicalDeviceDescriptorIndexingFeatures* di = &patch.di;
            if (const auto* f = FindInChain(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES))
            {
                di = patch.EditInPlace<VkPhysicalDeviceDescriptorIndexingFeatures>(f);
                patch.Report("VkPhysicalDeviceDescriptorIndexingFeatures", "edited");
            }
            di->runtimeDescriptorArray = VK_TRUE;
            di->shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
            di->descriptorBindingPartiallyBound = VK_TRUE;
            di->descriptorBindingVariableDescriptorCount = VK_TRUE;
            if (di == &patch.di) patch.Prepend(&patch.di, "VkPhysicalDeviceDescriptorIndexingFeatures");
        }
        if (w.timeline_semaphore)
        {
            if (const auto* f = FindInChain(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES))
            {
                patch.EditInPlace<VkPhysicalDeviceTimelineSemaphoreFeatures>(f)->timelineSemaphore = VK_TRUE;
                patch.Report("VkPhysicalDeviceTimelineSemaphoreFeatures", "edited");
            }
            else
            {
                patch.ts.timelineSemaphore = VK_TRUE;
                patch.Prepend(&patch.ts, "VkPhysicalDeviceTimelineSemaphoreFeatures");
            }
        }
        if (w.scalar_block_layout)
        {
            if (const auto* f = FindInChain(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES))
            {
                patch.EditInPlace<VkPhysicalDeviceScalarBlockLayoutFeatures>(f)->scalarBlockLayout = VK_TRUE;
                patch.Report("VkPhysicalDeviceScalarBlockLayoutFeatures", "edited");
            }
            else
            {
                patch.sbl.scalarBlockLayout = VK_TRUE;
                patch.Prepend(&patch.sbl, "VkPhysicalDeviceScalarBlockLayoutFeatures");
            }
        }
    }

    // Vulkan 1.3 features.
    if (const auto* found = FindInChain(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES))
    {
        auto* v13 = patch.EditInPlace<VkPhysicalDeviceVulkan13Features>(found);
        Enable(v13->synchronization2, w.synchronization2);
        Enable(v13->dynamicRendering, w.dynamic_rendering);
        patch.Report("1.3 features", "OR-ed into Unity's VkPhysicalDeviceVulkan13Features");
    }
    else
    {
        if (w.synchronization2)
        {
            if (const auto* f = FindInChain(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES))
            {
                patch.EditInPlace<VkPhysicalDeviceSynchronization2Features>(f)->synchronization2 = VK_TRUE;
                patch.Report("VkPhysicalDeviceSynchronization2Features", "edited");
            }
            else
            {
                patch.s2.synchronization2 = VK_TRUE;
                patch.Prepend(&patch.s2, "VkPhysicalDeviceSynchronization2Features");
            }
        }
        if (w.dynamic_rendering)
        {
            if (const auto* f = FindInChain(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES))
            {
                patch.EditInPlace<VkPhysicalDeviceDynamicRenderingFeatures>(f)->dynamicRendering = VK_TRUE;
                patch.Report("VkPhysicalDeviceDynamicRenderingFeatures", "edited");
            }
            else
            {
                patch.dr.dynamicRendering = VK_TRUE;
                patch.Prepend(&patch.dr, "VkPhysicalDeviceDynamicRenderingFeatures");
            }
        }
    }

    // Ray tracing.
    if (w.acceleration_structure)
    {
        if (const auto* f = FindInChain(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR))
        {
            patch.EditInPlace<VkPhysicalDeviceAccelerationStructureFeaturesKHR>(f)->accelerationStructure = VK_TRUE;
            patch.Report("VkPhysicalDeviceAccelerationStructureFeaturesKHR", "edited");
        }
        else
        {
            patch.as.accelerationStructure = VK_TRUE;
            patch.Prepend(&patch.as, "VkPhysicalDeviceAccelerationStructureFeaturesKHR");
        }
    }
    if (w.ray_query)
    {
        if (const auto* f = FindInChain(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR))
        {
            patch.EditInPlace<VkPhysicalDeviceRayQueryFeaturesKHR>(f)->rayQuery = VK_TRUE;
            patch.Report("VkPhysicalDeviceRayQueryFeaturesKHR", "edited");
        }
        else
        {
            patch.rq.rayQuery = VK_TRUE;
            patch.Prepend(&patch.rq, "VkPhysicalDeviceRayQueryFeaturesKHR");
        }
    }
}

std::string WantedNames(const Wanted& w)
{
    std::string names;
    auto add = [&names](bool on, const char* name) {
        if (!on) return;
        if (!names.empty()) names += ", ";
        names += name;
    };
    add(w.buffer_device_address, "bufferDeviceAddress");
    add(w.descriptor_indexing, "runtimeDescriptorArray, shaderStorageBufferArrayNonUniformIndexing, "
                               "descriptorBindingPartiallyBound, descriptorBindingVariableDescriptorCount");
    add(w.timeline_semaphore, "timelineSemaphore");
    add(w.scalar_block_layout, "scalarBlockLayout");
    add(w.synchronization2, "synchronization2");
    add(w.dynamic_rendering, "dynamicRendering");
    add(w.acceleration_structure, "accelerationStructure");
    add(w.ray_query, "rayQuery");
    add(w.shader_int64, "shaderInt64");
    add(w.storage_image_write_without_format, "shaderStorageImageWriteWithoutFormat");
    add(w.storage_image_read_without_format, "shaderStorageImageReadWithoutFormat");
    return names;
}

VKAPI_ATTR VkResult VKAPI_CALL HookCreateDevice(VkPhysicalDevice physical_device,
                                                const VkDeviceCreateInfo* create_info,
                                                const VkAllocationCallbacks* allocator,
                                                VkDevice* out_device)
{
    auto next = reinterpret_cast<PFN_vkCreateDevice>(g_next_gipa(g_instance, "vkCreateDevice"));
    if (next == nullptr)
    {
        Log(LogLevel::Error, "vkCreateDevice: the loader returned no vkCreateDevice (instance %p)",
            static_cast<void*>(g_instance));
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (create_info == nullptr)
    {
        return next(physical_device, create_info, allocator, out_device);
    }

    CaelixDeviceRequirements requirements{};
    caelix_get_device_requirements(&requirements);
    const Wanted wanted = WantedFromFlags(requirements.feature_flags);

    Log(LogLevel::Info, "vkCreateDevice: Unity enables %u device extensions: %s",
        create_info->enabledExtensionCount,
        JoinNames(create_info->ppEnabledExtensionNames, create_info->enabledExtensionCount).c_str());

    GetState().upscaler_extensions.store(0);
    if (!CheckSupport(physical_device, requirements, wanted))
    {
        Log(LogLevel::Error, "vkCreateDevice: the physical device lacks a Caelix requirement; Unity's "
                             "device is created unchanged and the plugin will not render");
        return next(physical_device, create_info, allocator, out_device);
    }

    // The upscalers' device extensions (optional): the ones the device offers are added.
    std::vector<const char*> upscaler_extensions;
    bool upscaler_complete = false;
    {
        uint32_t upscaler_count = 0;
        const char* const* upscaler_names = nullptr;
        const CaelixResult query = caelix_upscaler_device_extensions(
            reinterpret_cast<uint64_t>(g_instance), reinterpret_cast<uint64_t>(physical_device), &upscaler_count,
            &upscaler_names);
        if (query != CAELIX_OK)
        {
            Log(LogLevel::Warning, "vkCreateDevice: caelix_upscaler_device_extensions failed (CaelixResult %d); "
                                   "no upscaler extensions are added",
                static_cast<int>(query));
        }
        const std::vector<VkExtensionProperties> available = DeviceExtensions(physical_device);
        std::string missing;
        for (uint32_t i = 0; i < upscaler_count; ++i)
        {
            // Unity enables VK_EXT_buffer_device_address; the KHR extension must not be enabled with
            // it (VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-03328). The device is Vulkan 1.2+
            // with the core bufferDeviceAddress feature (a Caelix requirement), which NGX accepts
            // (measured: the DLSS test passes without the KHR extension), so it counts as satisfied.
            if (std::strcmp(upscaler_names[i], VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0 &&
                ContainsName(std::vector<const char*>(create_info->ppEnabledExtensionNames,
                                                      create_info->ppEnabledExtensionNames +
                                                          create_info->enabledExtensionCount),
                             VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))
            {
                Log(LogLevel::Info, "vkCreateDevice: %s is not added: Unity enables %s, and the core "
                                    "bufferDeviceAddress feature satisfies NGX",
                    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
                continue;
            }
            if (Offers(available, upscaler_names[i]))
                upscaler_extensions.push_back(upscaler_names[i]);
            else
                AppendName(missing, upscaler_names[i]);
        }
        upscaler_complete = query == CAELIX_OK && upscaler_count > 0 && missing.empty();
        if (!missing.empty())
        {
            Log(LogLevel::Warning, "vkCreateDevice: the device lacks upscaler extensions (skipped; the "
                                   "upscalers will not start): %s",
                missing.c_str());
        }
        Log(LogLevel::Info, "vkCreateDevice: upscaler device extensions: %s",
            JoinNames(upscaler_extensions.data(), static_cast<uint32_t>(upscaler_extensions.size())).c_str());
    }

    VkResult result;
    {
        DevicePatch patch;
        BuildPatch(*create_info, requirements, wanted, upscaler_extensions, patch);
        std::string added;
        for (const std::string& name : patch.added_extensions)
        {
            if (!added.empty()) added += ", ";
            added += name;
        }
        Log(LogLevel::Info, "vkCreateDevice: Unity's pNext chain (sType values): %s; "
                            "pEnabledFeatures %s",
            patch.chain_report.c_str(),
            create_info->pEnabledFeatures != nullptr ? "set" : "NULL");
        Log(LogLevel::Info, "vkCreateDevice: final extension list (%u, added: %s): %s",
            patch.info.enabledExtensionCount, added.empty() ? "none" : added.c_str(),
            JoinNames(patch.info.ppEnabledExtensionNames, patch.info.enabledExtensionCount).c_str());
        Log(LogLevel::Info, "vkCreateDevice: feature bits enabled for Caelix: %s; structs: %s",
            WantedNames(wanted).c_str(), patch.feature_report.c_str());
        result = next(physical_device, &patch.info, allocator, out_device);
    }  // restores Unity's structs
    if (result == VK_SUCCESS)
    {
        GetState().device_patched.store(1);
        GetState().upscaler_extensions.store(upscaler_complete ? 1u : 0u);
        Log(LogLevel::Info, "vkCreateDevice OK: VkDevice %p with the Caelix extensions and features (upscaler "
                            "extensions %s)",
            static_cast<void*>(*out_device), upscaler_complete ? "complete" : "incomplete");
        return result;
    }
    Log(LogLevel::Error, "vkCreateDevice with the Caelix additions failed (VkResult %d); retrying "
                         "with Unity's create info",
        static_cast<int>(result));
    return next(physical_device, create_info, allocator, out_device);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL HookGetInstanceProcAddr(VkInstance instance,
                                                                 const char* name)
{
    if (name == nullptr)
    {
        return nullptr;
    }
    if (std::strcmp(name, "vkGetInstanceProcAddr") == 0)
    {
        return reinterpret_cast<PFN_vkVoidFunction>(&HookGetInstanceProcAddr);
    }
    if (std::strcmp(name, "vkCreateInstance") == 0)
    {
        return reinterpret_cast<PFN_vkVoidFunction>(&HookCreateInstance);
    }
    if (std::strcmp(name, "vkCreateDevice") == 0)
    {
        return reinterpret_cast<PFN_vkVoidFunction>(&HookCreateDevice);
    }
    return g_next_gipa(instance, name);
}

}  // namespace

PFN_vkGetInstanceProcAddr UNITY_INTERFACE_API InterceptVulkanInitialization(
    PFN_vkGetInstanceProcAddr get_instance_proc_addr, void* /*user_data*/)
{
    g_next_gipa = get_instance_proc_addr;
    GetState().intercept_called.store(1);
    Log(LogLevel::Info, "Vulkan init callback: Unity is initializing Vulkan; the plugin wraps "
                        "vkCreateInstance and vkCreateDevice");
    return &HookGetInstanceProcAddr;
}

}  // namespace caelix_unity
