#include "rhi/pipeline.h"

#include "core/log.h"
#include "rhi/gpu_buffer.h"
#include "rhi/shader_loader.h"

#include <string>
#include <utility>
#include <vector>

namespace caelix {
namespace {

// Diagnostics: with the environment variable CAELIX_PIPELINE_STATS set and the host's device created
// with VK_KHR_pipeline_executable_properties (feature pipelineExecutableInfo; the Rust host does that
// under the same variable), every compute pipeline is created with CAPTURE_STATISTICS and its driver
// statistics (register count, ...) are logged at info level. The function pointer is null when the
// extension is not enabled, and then nothing changes.
bool PipelineStatsEnabled()
{
    static const bool enabled = !GetEnv("CAELIX_PIPELINE_STATS").empty();
    return enabled && vkGetPipelineExecutableStatisticsKHR != nullptr &&
           vkGetPipelineExecutablePropertiesKHR != nullptr;
}

void LogPipelineStats(VkDevice device, VkPipeline pipeline, const char* name)
{
    VkPipelineInfoKHR info{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    info.pipeline = pipeline;
    uint32_t count = 0;
    if (vkGetPipelineExecutablePropertiesKHR(device, &info, &count, nullptr) != VK_SUCCESS)
    {
        return;
    }
    std::vector<VkPipelineExecutablePropertiesKHR> executables(
        count, VkPipelineExecutablePropertiesKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    if (vkGetPipelineExecutablePropertiesKHR(device, &info, &count, executables.data()) != VK_SUCCESS)
    {
        return;
    }
    for (uint32_t e = 0; e < count; ++e)
    {
        VkPipelineExecutableInfoKHR executable{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        executable.pipeline = pipeline;
        executable.executableIndex = e;
        uint32_t stat_count = 0;
        if (vkGetPipelineExecutableStatisticsKHR(device, &executable, &stat_count, nullptr) != VK_SUCCESS)
        {
            continue;
        }
        std::vector<VkPipelineExecutableStatisticKHR> stats(
            stat_count, VkPipelineExecutableStatisticKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        if (vkGetPipelineExecutableStatisticsKHR(device, &executable, &stat_count, stats.data()) != VK_SUCCESS)
        {
            continue;
        }
        std::string line;
        for (const VkPipelineExecutableStatisticKHR& stat : stats)
        {
            line += line.empty() ? "" : ", ";
            line += stat.name;
            line += " = ";
            switch (stat.format)
            {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                line += stat.value.b32 ? "true" : "false";
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                line += std::to_string(stat.value.i64);
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                line += std::to_string(stat.value.u64);
                break;
            default:
                line += std::to_string(stat.value.f64);
                break;
            }
        }
        CAELIX_LOGI("pipeline stats %s [%s]: %s", name, executables[e].name, line.c_str());
    }
}

}  // namespace

ComputePipeline::ComputePipeline(VkDevice device, std::span<const uint32_t> spirv,
                                 uint32_t push_bytes, const char* name,
                                 VkDescriptorSetLayout set_layout)
    : device_(device), push_bytes_(push_bytes)
{
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = spirv.size_bytes();
    module_info.pCode = spirv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    VkResult result = vkCreateShaderModule(device_, &module_info, nullptr, &module);
    if (result != VK_SUCCESS)
    {
        CAELIX_LOGE("vkCreateShaderModule failed for %s: %d", name, static_cast<int>(result));
        throw VulkanError("vkCreateShaderModule failed", result);
    }

    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes_};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = push_bytes_ > 0 ? 1u : 0u;
    layout_info.pPushConstantRanges = &range;
    layout_info.setLayoutCount = set_layout != VK_NULL_HANDLE ? 1u : 0u;
    layout_info.pSetLayouts = &set_layout;
    result = vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_);
    if (result != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, module, nullptr);
        layout_ = VK_NULL_HANDLE;
        CAELIX_LOGE("vkCreatePipelineLayout failed for %s: %d", name, static_cast<int>(result));
        throw VulkanError("vkCreatePipelineLayout failed", result);
    }

    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = layout_;
    const bool stats = PipelineStatsEnabled();
    if (stats)
    {
        pipeline_info.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    }
    result =
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_);
    vkDestroyShaderModule(device_, module, nullptr);
    if (result != VK_SUCCESS)
    {
        pipeline_ = VK_NULL_HANDLE;
        vkDestroyPipelineLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
        CAELIX_LOGE("vkCreateComputePipelines failed for %s: %d", name, static_cast<int>(result));
        throw VulkanError("vkCreateComputePipelines failed", result);
    }
    if (stats)
    {
        LogPipelineStats(device_, pipeline_, name);
    }
}

ComputePipeline::~ComputePipeline()
{
    Reset();
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
      layout_(std::exchange(other.layout_, VK_NULL_HANDLE)),
      pipeline_(std::exchange(other.pipeline_, VK_NULL_HANDLE)),
      push_bytes_(std::exchange(other.push_bytes_, 0))
{
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept
{
    if (this != &other)
    {
        Reset();
        device_ = std::exchange(other.device_, VK_NULL_HANDLE);
        layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
        pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
        push_bytes_ = std::exchange(other.push_bytes_, 0);
    }
    return *this;
}

void ComputePipeline::Dispatch(VkCommandBuffer cmd, const void* push, uint32_t size,
                               uint32_t groups) const
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    if (size > 0)
    {
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, size, push);
    }
    vkCmdDispatch(cmd, groups, 1, 1);
}

void ComputePipeline::Reset()
{
    if (pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, pipeline_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
    }
    device_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    push_bytes_ = 0;
}

}  // namespace caelix
