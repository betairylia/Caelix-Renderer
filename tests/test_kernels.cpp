#include "test_kernels.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace test {

uint32_t FindMemoryType(Gpu& gpu, uint32_t type_bits, VkMemoryPropertyFlags wanted)
{
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(gpu.device.physical_device.physical_device, &properties);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < properties.memoryTypeCount && type == UINT32_MAX; ++i)
    {
        if ((type_bits & (1u << i)) != 0 && (properties.memoryTypes[i].propertyFlags & wanted) == wanted)
        {
            type = i;
        }
    }
    REQUIRE(type != UINT32_MAX);
    return type;
}

HostBuffer CreateHostBuffer(Gpu& gpu, VkDeviceSize size)
{
    const VkDevice device = gpu.device.device;
    HostBuffer out;
    out.size = std::max<VkDeviceSize>(16, (size + 15) & ~VkDeviceSize{15});
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = out.size;
    buffer_info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    REQUIRE(vkCreateBuffer(device, &buffer_info, nullptr, &out.buffer) == VK_SUCCESS);
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
    VkMemoryAllocateFlagsInfo flags_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate_info.pNext = &flags_info;
    allocate_info.allocationSize = requirements.size;
    allocate_info.memoryTypeIndex =
        FindMemoryType(gpu, requirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    REQUIRE(vkAllocateMemory(device, &allocate_info, nullptr, &out.memory) == VK_SUCCESS);
    REQUIRE(vkBindBufferMemory(device, out.buffer, out.memory, 0) == VK_SUCCESS);
    void* mapped = nullptr;
    REQUIRE(vkMapMemory(device, out.memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS);
    out.data = static_cast<uint8_t*>(mapped);
    std::fill(out.data, out.data + out.size, uint8_t{0});
    VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address_info.buffer = out.buffer;
    out.address = vkGetBufferDeviceAddress(device, &address_info);
    return out;
}

void DestroyHostBuffer(Gpu& gpu, HostBuffer& buffer)
{
    if (buffer.buffer != VK_NULL_HANDLE)
    {
        vkUnmapMemory(gpu.device.device, buffer.memory);
        vkDestroyBuffer(gpu.device.device, buffer.buffer, nullptr);
        vkFreeMemory(gpu.device.device, buffer.memory, nullptr);
    }
    buffer = HostBuffer{};
}

ComputeKernel::ComputeKernel(Gpu& gpu, const char* shader_dir, const char* name, uint32_t push_bytes)
    : gpu_(gpu), push_bytes_(push_bytes)
{
    const std::string path = std::string(shader_dir) + "/" + name + ".spv";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    INFO(path);
    REQUIRE(file.good());
    const std::streamsize bytes = file.tellg();
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), bytes);
    const VkDevice device = gpu.device.device;
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = words.size() * 4;
    module_info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    REQUIRE(vkCreateShaderModule(device, &module_info, nullptr, &module) == VK_SUCCESS);
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    REQUIRE(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout_) == VK_SUCCESS);
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = layout_;
    REQUIRE(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_) ==
            VK_SUCCESS);
    vkDestroyShaderModule(device, module, nullptr);
}

ComputeKernel::~ComputeKernel()
{
    vkDestroyPipeline(gpu_.device.device, pipeline_, nullptr);
    vkDestroyPipelineLayout(gpu_.device.device, layout_, nullptr);
}

void ComputeKernel::Record(VkCommandBuffer cmd, const void* push, uint32_t groups) const
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes_, push);
    vkCmdDispatch(cmd, groups, 1, 1);
}

void ComputeKernel::RecordIndirect(VkCommandBuffer cmd, const void* push, VkBuffer buffer,
                                   VkDeviceSize offset) const
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes_, push);
    vkCmdDispatchIndirect(cmd, buffer, offset);
}

void ComputeKernel::Dispatch(VkCommandBuffer cmd, const void* push, uint32_t groups) const
{
    Record(cmd, push, groups);
    ComputeBarrier(cmd);
}

void ComputeBarrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                           VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

}  // namespace test
