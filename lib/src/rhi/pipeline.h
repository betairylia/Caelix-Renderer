#pragma once
#include "core/vk_common.h"

#include <cstdint>
#include <span>

namespace caelix {

// A compute pipeline whose inputs are push constants and, optionally, descriptor
// sets: the kernels read and write most buffers through buffer device addresses, so
// mainly resources that have no address (acceleration structures, images) and the
// uniform buffer of the frame constants need a descriptor set. Move-only; the
// destructor destroys the pipeline and its layout (not the descriptor set layouts,
// which the caller owns). The
// caller is responsible for GPU lifetime: destroy it only when no pending GPU
// work uses it.
class ComputePipeline
{
public:
    ComputePipeline() = default;
    // Creates the shader module (entry point "main"), the layout with one push
    // constant range of `push_bytes` (compute stage; none when 0) and the descriptor
    // set layouts `set_layouts` (set_layouts[i] is set i; may be empty), and the
    // pipeline. The module is only needed for pipeline creation and is destroyed right
    // after it. Throws VulkanError.
    ComputePipeline(VkDevice device, std::span<const uint32_t> spirv, uint32_t push_bytes,
                    const char* name, std::span<const VkDescriptorSetLayout> set_layouts = {});
    ~ComputePipeline();
    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline& operator=(ComputePipeline&& other) noexcept;
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    bool valid() const
    {
        return pipeline_ != VK_NULL_HANDLE;
    }
    uint32_t push_bytes() const
    {
        return push_bytes_;
    }
    VkPipeline pipeline() const
    {
        return pipeline_;
    }
    VkPipelineLayout layout() const
    {
        return layout_;
    }

    // Binds the pipeline, pushes `size` bytes (must equal push_bytes()) and
    // dispatches `groups` x 1 x 1 workgroups.
    void Dispatch(VkCommandBuffer cmd, const void* push, uint32_t size, uint32_t groups) const;

private:
    void Reset();

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    uint32_t push_bytes_ = 0;
};

}  // namespace caelix
