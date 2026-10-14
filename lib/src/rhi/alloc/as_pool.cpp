#include "rhi/alloc/as_pool.h"

#include "core/log.h"

#include <algorithm>
#include <bit>

namespace caelix {

BlockAllocator::~BlockAllocator() {
  for (const PendingFree& item : pending_) {
    if (item.handle != VK_NULL_HANDLE) {
      vkDestroyAccelerationStructureKHR(device_, item.handle, nullptr);
    }
  }
}

void BlockAllocator::Init(VmaAllocator allocator, VkDevice device) {
  allocator_ = allocator;
  device_ = device;
}

AsBlock BlockAllocator::Allocate(VkDeviceSize bytes, AsKind kind) {
  const VkDeviceSize size = std::bit_ceil(std::max(bytes, kMinBlockBytes));
  const uint32_t size_class = static_cast<uint32_t>(std::countr_zero(size / kMinBlockBytes));
  AsBlock block;
  block.size_class = size_class;
  block.size = size;
  block.kind = kind;
  // 1. A free block of this class in any chunk.
  for (uint32_t c = 0; c < chunks_.size() && !block.valid(); ++c) {
    std::vector<std::vector<VkDeviceSize>>& lists = chunks_[c].free_by_class;
    if (size_class < lists.size() && !lists[size_class].empty()) {
      block.chunk = c;
      block.offset = lists[size_class].back();
      lists[size_class].pop_back();
    }
  }
  // 2. Bump space at the end of a chunk. Every block size is a multiple of
  //    kMinBlockBytes, so the bump pointer (a sum of block sizes) keeps every
  //    offset a multiple of 256.
  if (block.chunk == AsBlock::kNoChunk) {
    for (uint32_t c = 0; c < chunks_.size(); ++c) {
      Chunk& chunk = chunks_[c];
      if (chunk.bump + size <= chunk.buffer.size()) {
        block.chunk = c;
        block.offset = chunk.bump;
        chunk.bump += size;
        break;
      }
    }
  }
  // 3. A new chunk.
  if (block.chunk == AsBlock::kNoChunk) {
    const VkDeviceSize chunk_size = std::max(kChunkBytes, size);
    Chunk chunk;
    chunk.buffer = GpuBuffer::DeviceLocalWithUsage(
        allocator_, device_, chunk_size,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    chunk.bump = size;
    chunks_.push_back(std::move(chunk));
    chunk_bytes_ += chunk_size;
    block.chunk = static_cast<uint32_t>(chunks_.size() - 1);
    block.offset = 0;
    CAELIX_LOGD("acceleration structure pool: chunk %u created (%llu bytes)", block.chunk,
                static_cast<unsigned long long>(chunk_size));
  }
  std::vector<std::vector<VkDeviceSize>>& lists = chunks_[block.chunk].free_by_class;
  if (size_class >= lists.size()) {
    lists.resize(size_class + 1);  // Free() indexes the list directly
  }
  allocated_[static_cast<size_t>(kind)] += size;
  return block;
}

void BlockAllocator::Free(const AsBlock& block) {
  if (!block.valid()) {
    return;
  }
  chunks_[block.chunk].free_by_class[block.size_class].push_back(block.offset);
  allocated_[static_cast<size_t>(block.kind)] -= block.size;
}

void BlockAllocator::FreeAfter(const AsBlock& block, uint64_t frame,
                               VkAccelerationStructureKHR handle) {
  if (!block.valid() && handle == VK_NULL_HANDLE) {
    return;
  }
  pending_.push_back(PendingFree{frame, block, handle});
}

void BlockAllocator::Recycle(uint64_t safe_frame) {
  size_t kept = 0;
  for (size_t i = 0; i < pending_.size(); ++i) {
    const PendingFree& item = pending_[i];
    if (item.frame <= safe_frame) {
      // The handle goes first: its memory is reused once the block is free.
      if (item.handle != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR(device_, item.handle, nullptr);
      }
      Free(item.block);
    } else {
      pending_[kept++] = item;
    }
  }
  pending_.resize(kept);
}

VkAccelerationStructureKHR CreateAccelerationStructure(VkDevice device, VkBuffer buffer,
                                                       const AsBlock& block,
                                                       VkAccelerationStructureTypeKHR type) {
  VkAccelerationStructureCreateInfoKHR info{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  info.buffer = buffer;
  info.offset = block.offset;
  info.size = block.size;
  info.type = type;
  VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
  const VkResult result = vkCreateAccelerationStructureKHR(device, &info, nullptr, &handle);
  if (result != VK_SUCCESS) {
    CAELIX_LOGE("vkCreateAccelerationStructureKHR failed: %d (%llu bytes)",
                static_cast<int>(result), static_cast<unsigned long long>(block.size));
    throw VulkanError("vkCreateAccelerationStructureKHR failed", result);
  }
  return handle;
}

VkDeviceAddress AccelerationStructureAddress(VkDevice device, VkAccelerationStructureKHR handle) {
  VkAccelerationStructureDeviceAddressInfoKHR info{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  info.accelerationStructure = handle;
  return vkGetAccelerationStructureDeviceAddressKHR(device, &info);
}

}  // namespace caelix
