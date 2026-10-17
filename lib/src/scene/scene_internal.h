#pragma once
// Helpers shared by the scene_*.cpp translation units. Not part of the Scene
// interface: include it from those files only.
#include "core/vk_common.h"

namespace caelix {

// Acceleration structures: one VkAccelerationStructureInstanceKHR per TLAS instance.
constexpr VkDeviceSize kInstanceBytes = sizeof(VkAccelerationStructureInstanceKHR);
static_assert(kInstanceBytes == 64);

// Staging alignment of records. Records are 8-byte multiples, so consecutive
// records stay contiguous in the ring and their copies merge. vkCmdCopyBuffer
// itself has no offset alignment rule.
constexpr uint64_t kRecordAlignment = 8;

// One global memory barrier.
inline void MemoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src_stage,
                          VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                          VkAccessFlags2 dst_access)
{
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

}  // namespace caelix
