#include "test_device.h"

#include <cstdio>
#include <cstring>

namespace test {

std::atomic<int> g_validation_errors{0};
std::atomic<int> g_validation_warnings{0};
std::atomic<int> g_validation_ignored{0};
std::atomic<int> g_library_errors{0};
std::atomic<int> g_library_warnings{0};

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void* /*user_data*/)
{
    const char* message = data != nullptr && data->pMessage != nullptr ? data->pMessage : "";
    const char* id = data != nullptr && data->pMessageIdName != nullptr ? data->pMessageIdName : "";
    // Implicit layers of other software (Overwolf, the Epic Games overlay, whose 32- and
    // 64-bit manifests share one layer name) and the loader's notes about them are not this
    // project's problem.
    if (std::strstr(message, "VK_LAYER_OW_") != nullptr || std::strstr(id, "VK_LAYER_OW_") != nullptr ||
        std::strstr(message, "VK_LAYER_EOS_Overlay") != nullptr ||
        std::strstr(message, "uses API version 1.2") != nullptr)
    {
        ++g_validation_ignored;
        std::printf("[vulkan ignored] %s\n", message);
        return VK_FALSE;
    }
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0)
    {
        ++g_validation_errors;
        std::printf("[vulkan ERROR] %s: %s\n", id, message);
    }
    else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0)
    {
        ++g_validation_warnings;
        std::printf("[vulkan WARNING] %s: %s\n", id, message);
    }
    std::fflush(stdout);
    return VK_FALSE;
}

}  // namespace

void LibraryLog(CaelixLogLevel level, const char* message, void* /*user_data*/)
{
    static const char* const kNames[] = {"debug", "info", "warning", "ERROR"};
    if (level == CAELIX_LOG_ERROR)
    {
        ++g_library_errors;
    }
    else if (level == CAELIX_LOG_WARNING)
    {
        ++g_library_warnings;
    }
    if (level != CAELIX_LOG_DEBUG)
    {
        std::printf("[caelix %s] %s\n", kNames[level], message);
        std::fflush(stdout);
    }
}

void CreateGpu(Gpu& gpu, const char* app_name)
{
    REQUIRE(volkInitialize() == VK_SUCCESS);
    const vkb::SystemInfo system = vkb::SystemInfo::get_system_info(vkGetInstanceProcAddr).value();
    std::printf("validation layer available: %s, debug utils available: %s\n",
                system.validation_layers_available ? "yes" : "no",
                system.debug_utils_available ? "yes" : "no");

    vkb::InstanceBuilder instance_builder(vkGetInstanceProcAddr);
    instance_builder.set_app_name(app_name)
        .require_api_version(1, 3, 0)
        .set_headless(true)
        .request_validation_layers(true)
        .set_debug_messenger_severity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        .set_debug_callback(DebugCallback);
    // The upscalers' instance extensions (DLSS through NGX; none without CAELIX_DLSS), when the
    // instance supports them.
    uint32_t upscaler_instance_count = 0;
    const char* const* upscaler_instance_names = nullptr;
    caelix_upscaler_instance_extensions(&upscaler_instance_count, &upscaler_instance_names);
    for (uint32_t i = 0; i < upscaler_instance_count; ++i)
    {
        if (system.is_extension_available(upscaler_instance_names[i]))
        {
            instance_builder.enable_extension(upscaler_instance_names[i]);
        }
        else
        {
            std::printf("upscaler instance extension %s is not available (skipped)\n", upscaler_instance_names[i]);
        }
    }
    auto instance = instance_builder.build();
    if (!instance)
    {
        FAIL("vkb instance: " << instance.error().message());
    }
    gpu.instance = instance.value();
    volkLoadInstanceOnly(gpu.instance.instance);

    CaelixDeviceRequirements requirements{};
    caelix_get_device_requirements(&requirements);
    REQUIRE(requirements.api_version >= VK_MAKE_API_VERSION(0, 1, 3, 0));

    // Translate the library's feature bits to Vulkan feature structs.
    const uint32_t flags = requirements.feature_flags;
    const uint32_t known = CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS | CAELIX_FEATURE_DESCRIPTOR_INDEXING |
                           CAELIX_FEATURE_SYNCHRONIZATION2 | CAELIX_FEATURE_DYNAMIC_RENDERING |
                           CAELIX_FEATURE_TIMELINE_SEMAPHORE | CAELIX_FEATURE_ACCELERATION_STRUCTURE |
                           CAELIX_FEATURE_RAY_QUERY | CAELIX_FEATURE_SHADER_INT64 |
                           CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT |
                           CAELIX_FEATURE_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT |
                           CAELIX_FEATURE_STORAGE_IMAGE_READ_WITHOUT_FORMAT;
    REQUIRE((flags & ~known) == 0);  // a new bit needs a translation here
    VkPhysicalDeviceFeatures f10{};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    if (flags & CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS) f12.bufferDeviceAddress = VK_TRUE;
    if (flags & CAELIX_FEATURE_DESCRIPTOR_INDEXING)
    {
        f12.runtimeDescriptorArray = VK_TRUE;
        f12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
        f12.descriptorBindingPartiallyBound = VK_TRUE;
        f12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    }
    if (flags & CAELIX_FEATURE_SYNCHRONIZATION2) f13.synchronization2 = VK_TRUE;
    if (flags & CAELIX_FEATURE_DYNAMIC_RENDERING) f13.dynamicRendering = VK_TRUE;
    if (flags & CAELIX_FEATURE_TIMELINE_SEMAPHORE) f12.timelineSemaphore = VK_TRUE;
    if (flags & CAELIX_FEATURE_ACCELERATION_STRUCTURE) acceleration.accelerationStructure = VK_TRUE;
    if (flags & CAELIX_FEATURE_RAY_QUERY) ray_query.rayQuery = VK_TRUE;
    if (flags & CAELIX_FEATURE_SHADER_INT64) f10.shaderInt64 = VK_TRUE;
    if (flags & CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT) f12.scalarBlockLayout = VK_TRUE;
    if (flags & CAELIX_FEATURE_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT)
    {
        f10.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    }
    if (flags & CAELIX_FEATURE_STORAGE_IMAGE_READ_WITHOUT_FORMAT)
    {
        f10.shaderStorageImageReadWithoutFormat = VK_TRUE;
    }

    vkb::PhysicalDeviceSelector selector(gpu.instance);
    selector.set_minimum_version(1, 3)
        .require_present(false)
        .set_required_features(f10)
        .set_required_features_12(f12)
        .set_required_features_13(f13);
    for (uint32_t i = 0; i < requirements.device_extension_count; ++i)
    {
        selector.add_required_extension(requirements.device_extensions[i]);
    }
    if (flags & CAELIX_FEATURE_ACCELERATION_STRUCTURE) selector.add_required_extension_features(acceleration);
    if (flags & CAELIX_FEATURE_RAY_QUERY) selector.add_required_extension_features(ray_query);
    auto physical = selector.select();
    if (!physical)
    {
        FAIL("vkb physical device: " << physical.error().message());
    }
    vkb::PhysicalDevice physical_device = physical.value();
    std::printf("physical device: %s\n", physical_device.name.c_str());
    // The upscalers' device extensions (DLSS through NGX), when the physical device has them.
    uint32_t upscaler_device_count = 0;
    const char* const* upscaler_device_names = nullptr;
    REQUIRE(caelix_upscaler_device_extensions(reinterpret_cast<uint64_t>(gpu.instance.instance),
                                              reinterpret_cast<uint64_t>(physical_device.physical_device),
                                              &upscaler_device_count, &upscaler_device_names) == CAELIX_OK);
    for (uint32_t i = 0; i < upscaler_device_count; ++i)
    {
        if (!physical_device.enable_extension_if_present(upscaler_device_names[i]))
        {
            std::printf("upscaler device extension %s is not available (skipped)\n", upscaler_device_names[i]);
        }
    }

    auto device = vkb::DeviceBuilder(physical_device).build();
    if (!device)
    {
        FAIL("vkb device: " << device.error().message());
    }
    gpu.device = device.value();
    volkLoadDevice(gpu.device.device);
    gpu.queue = gpu.device.get_queue(vkb::QueueType::graphics).value();
    gpu.queue_family = gpu.device.get_queue_index(vkb::QueueType::graphics).value();

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = gpu.queue_family;
    REQUIRE(vkCreateCommandPool(gpu.device.device, &pool_info, nullptr, &gpu.pool) == VK_SUCCESS);
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    REQUIRE(vkCreateFence(gpu.device.device, &fence_info, nullptr, &gpu.fence) == VK_SUCCESS);
}

void DestroyGpu(Gpu& gpu)
{
    vkDestroyFence(gpu.device.device, gpu.fence, nullptr);
    vkDestroyCommandPool(gpu.device.device, gpu.pool, nullptr);
    vkb::destroy_device(gpu.device);
    vkb::destroy_instance(gpu.instance);
}

CaelixDevice* AdoptDevice(Gpu& gpu)
{
    CaelixDeviceAdoptDesc adopt{};
    adopt.get_instance_proc_addr = reinterpret_cast<void*>(vkGetInstanceProcAddr);
    adopt.instance = reinterpret_cast<uint64_t>(gpu.instance.instance);
    adopt.physical_device = reinterpret_cast<uint64_t>(gpu.device.physical_device.physical_device);
    adopt.device = reinterpret_cast<uint64_t>(gpu.device.device);
    adopt.graphics_queue = reinterpret_cast<uint64_t>(gpu.queue);
    adopt.graphics_queue_family = gpu.queue_family;
    CaelixDevice* device = nullptr;
    REQUIRE(caelix_device_adopt(&adopt, &device) == CAELIX_OK);
    return device;
}

void CopyToHostBarrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

void CreateReadback(Gpu& gpu, VkDeviceSize size, Readback& out)
{
    const VkDevice device = gpu.device.device;
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    REQUIRE(vkCreateBuffer(device, &buffer_info, nullptr, &out.buffer) == VK_SUCCESS);
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(gpu.device.physical_device.physical_device, &memory_properties);
    const VkMemoryPropertyFlags wanted =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
    {
        if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
            (memory_properties.memoryTypes[i].propertyFlags & wanted) == wanted)
        {
            type = i;
            break;
        }
    }
    REQUIRE(type != UINT32_MAX);
    VkMemoryAllocateInfo allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate_info.allocationSize = requirements.size;
    allocate_info.memoryTypeIndex = type;
    REQUIRE(vkAllocateMemory(device, &allocate_info, nullptr, &out.memory) == VK_SUCCESS);
    REQUIRE(vkBindBufferMemory(device, out.buffer, out.memory, 0) == VK_SUCCESS);
    void* mapped = nullptr;
    REQUIRE(vkMapMemory(device, out.memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS);
    out.words = static_cast<const uint32_t*>(mapped);
}

void DestroyReadback(Gpu& gpu, Readback& readback)
{
    vkUnmapMemory(gpu.device.device, readback.memory);
    vkDestroyBuffer(gpu.device.device, readback.buffer, nullptr);
    vkFreeMemory(gpu.device.device, readback.memory, nullptr);
}

void FlushFrame(Gpu& gpu, CaelixScene* scene, Frames& frames)
{
    const uint64_t frame = frames.next++;
    SubmitAndWait(gpu, [&](VkCommandBuffer cmd) {
        CaelixFrameDesc desc{};
        desc.command_buffer = reinterpret_cast<uint64_t>(cmd);
        desc.frame_number = frame;
        desc.safe_frame_number = frame - 1;
        REQUIRE(caelix_scene_flush(scene, &desc) == CAELIX_OK);
    });
    // The frame has completed: its staging space can be reused right away.
    REQUIRE(caelix_scene_recycle(scene, frame) == CAELIX_OK);
}

void RequireNoMessages()
{
    std::printf("validation messages: %d errors, %d warnings, %d ignored; library: %d unexpected "
                "errors, %d warnings\n",
                g_validation_errors.load(), g_validation_warnings.load(), g_validation_ignored.load(),
                g_library_errors.load(), g_library_warnings.load());
    std::fflush(stdout);
    REQUIRE(g_validation_errors.load() == 0);
    REQUIRE(g_validation_warnings.load() == 0);
    REQUIRE(g_library_errors.load() == 0);
    REQUIRE(g_library_warnings.load() == 0);
}

}  // namespace test
