// Shared Vulkan setup of the headless tests: vk-bootstrap creates the instance
// and the device the way a host would (validation layer and debug messenger
// when installed), the library adopts it through the public C ABI.
#pragma once
#include <volk.h>

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <VkBootstrap.h>
#include <catch2/catch_test_macros.hpp>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "caelix/caelix.h"

#include <atomic>
#include <cstdint>

namespace test {

// Message counters. Validation messages of other software's implicit layers
// (Overwolf) are counted as ignored.
extern std::atomic<int> g_validation_errors;
extern std::atomic<int> g_validation_warnings;
extern std::atomic<int> g_validation_ignored;
extern std::atomic<int> g_library_errors;
extern std::atomic<int> g_library_warnings;

// Library log callback: prints every message except debug, counts errors and
// warnings.
void LibraryLog(CaelixLogLevel level, const char* message, void* user_data);

// Runs `body` and requires that it logs exactly `expected` library errors.
template <typename Body>
void ExpectLibraryErrors(int expected, Body body)
{
    const int before = g_library_errors.load();
    body();
    REQUIRE(g_library_errors.load() - before == expected);
    g_library_errors -= expected;
}

struct Gpu
{
    vkb::Instance instance;
    vkb::Device device;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
};

// A Vulkan 1.3 device with every requirement of caelix_get_device_requirements.
void CreateGpu(Gpu& gpu, const char* app_name);
void DestroyGpu(Gpu& gpu);

// Adopts the test device.
CaelixDevice* AdoptDevice(Gpu& gpu);

// Records `body` into a fresh command buffer, submits it and waits for it.
template <typename Body>
void SubmitAndWait(Gpu& gpu, Body body)
{
    VkCommandBufferAllocateInfo allocate_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate_info.commandPool = gpu.pool;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    REQUIRE(vkAllocateCommandBuffers(gpu.device.device, &allocate_info, &cmd) == VK_SUCCESS);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
    body(cmd);
    REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
    VkCommandBufferSubmitInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmd_info.commandBuffer = cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmd_info;
    REQUIRE(vkQueueSubmit2(gpu.queue, 1, &submit, gpu.fence) == VK_SUCCESS);
    REQUIRE(vkWaitForFences(gpu.device.device, 1, &gpu.fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS);
    REQUIRE(vkResetFences(gpu.device.device, 1, &gpu.fence) == VK_SUCCESS);
    REQUIRE(vkResetCommandPool(gpu.device.device, gpu.pool, 0) == VK_SUCCESS);
    vkFreeCommandBuffers(gpu.device.device, gpu.pool, 1, &cmd);
}

// Transfer writes -> host reads, after readback copies.
void CopyToHostBarrier(VkCommandBuffer cmd);

// A host-visible, coherent buffer for reading GPU data back.
struct Readback
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    const uint32_t* words = nullptr;
};

void CreateReadback(Gpu& gpu, VkDeviceSize size, Readback& out);
void DestroyReadback(Gpu& gpu, Readback& readback);

// Frame bookkeeping of a test host: every submit is waited for, so the frame
// before the current one is always complete.
struct Frames
{
    uint64_t next = 1;
};

// One frame: caelix_scene_flush into a fresh command buffer, submit, wait, then
// caelix_scene_recycle of that frame.
void FlushFrame(Gpu& gpu, CaelixScene* scene, Frames& frames);

// Prints the message counters and requires that none of them counted an error
// or a warning.
void RequireNoMessages();

}  // namespace test
