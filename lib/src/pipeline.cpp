#include "pipeline.h"

#include "gpu_buffer.h"
#include "log.h"

#include <utility>

namespace caelix {

ComputePipeline::ComputePipeline(VkDevice device, std::span<const uint32_t> spirv,
                                 uint32_t push_bytes, const char* name)
    : device_(device), push_bytes_(push_bytes) {
  VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = spirv.size_bytes();
  module_info.pCode = spirv.data();
  VkShaderModule module = VK_NULL_HANDLE;
  VkResult result = vkCreateShaderModule(device_, &module_info, nullptr, &module);
  if (result != VK_SUCCESS) {
    CAELIX_LOGE("vkCreateShaderModule failed for %s: %d", name, static_cast<int>(result));
    throw VulkanError("vkCreateShaderModule failed", result);
  }

  const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes_};
  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.pushConstantRangeCount = push_bytes_ > 0 ? 1u : 0u;
  layout_info.pPushConstantRanges = &range;
  result = vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_);
  if (result != VK_SUCCESS) {
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
  result =
      vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_);
  vkDestroyShaderModule(device_, module, nullptr);
  if (result != VK_SUCCESS) {
    pipeline_ = VK_NULL_HANDLE;
    vkDestroyPipelineLayout(device_, layout_, nullptr);
    layout_ = VK_NULL_HANDLE;
    CAELIX_LOGE("vkCreateComputePipelines failed for %s: %d", name, static_cast<int>(result));
    throw VulkanError("vkCreateComputePipelines failed", result);
  }
}

ComputePipeline::~ComputePipeline() { Reset(); }

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
      layout_(std::exchange(other.layout_, VK_NULL_HANDLE)),
      pipeline_(std::exchange(other.pipeline_, VK_NULL_HANDLE)),
      push_bytes_(std::exchange(other.push_bytes_, 0)) {}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
  if (this != &other) {
    Reset();
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
    pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
    push_bytes_ = std::exchange(other.push_bytes_, 0);
  }
  return *this;
}

void ComputePipeline::Dispatch(VkCommandBuffer cmd, const void* push, uint32_t size,
                               uint32_t groups) const {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  if (size > 0) {
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, size, push);
  }
  vkCmdDispatch(cmd, groups, 1, 1);
}

void ComputePipeline::Reset() {
  if (pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(device_, pipeline_, nullptr);
  }
  if (layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device_, layout_, nullptr);
  }
  device_ = VK_NULL_HANDLE;
  layout_ = VK_NULL_HANDLE;
  pipeline_ = VK_NULL_HANDLE;
  push_bytes_ = 0;
}

}  // namespace caelix
