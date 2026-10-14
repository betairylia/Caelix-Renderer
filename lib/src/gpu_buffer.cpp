#include "gpu_buffer.h"

#include "log.h"

#include <utility>

namespace caelix {

GpuBuffer::GpuBuffer(VmaAllocator allocator, VkDevice device, VkDeviceSize size,
                     VkBufferUsageFlags usage, const VmaAllocationCreateInfo& allocation_info,
                     bool want_address) {
  VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationInfo info{};
  const VkResult result =
      vmaCreateBuffer(allocator, &buffer_info, &allocation_info, &buffer_, &allocation_, &info);
  if (result != VK_SUCCESS) {
    CAELIX_LOGE("vmaCreateBuffer failed: %d (size %llu bytes)", static_cast<int>(result),
                static_cast<unsigned long long>(size));
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    throw VulkanError("vmaCreateBuffer failed", result);
  }
  allocator_ = allocator;
  size_ = size;
  mapped_ = static_cast<uint8_t*>(info.pMappedData);
  if (want_address) {
    VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address_info.buffer = buffer_;
    address_ = vkGetBufferDeviceAddress(device, &address_info);
  }
}

GpuBuffer::~GpuBuffer() { Reset(); }

GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
    : allocator_(std::exchange(other.allocator_, VK_NULL_HANDLE)),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      size_(std::exchange(other.size_, 0)),
      address_(std::exchange(other.address_, 0)),
      mapped_(std::exchange(other.mapped_, nullptr)) {}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept {
  if (this != &other) {
    Reset();
    allocator_ = std::exchange(other.allocator_, VK_NULL_HANDLE);
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    size_ = std::exchange(other.size_, 0);
    address_ = std::exchange(other.address_, 0);
    mapped_ = std::exchange(other.mapped_, nullptr);
  }
  return *this;
}

GpuBuffer GpuBuffer::DeviceLocal(VmaAllocator allocator, VkDevice device, VkDeviceSize size) {
  VmaAllocationCreateInfo allocation_info{};
  allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  return GpuBuffer(allocator, device, size,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                   allocation_info, true);
}

GpuBuffer GpuBuffer::HostUpload(VmaAllocator allocator, VkDevice device, VkDeviceSize size) {
  VmaAllocationCreateInfo allocation_info{};
  allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
  allocation_info.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  return GpuBuffer(allocator, device, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, allocation_info,
                   false);
}

void GpuBuffer::FlushHostWrites(VkDeviceSize offset, VkDeviceSize size) const {
  if (allocation_ == VK_NULL_HANDLE || size == 0) {
    return;
  }
  const VkResult result = vmaFlushAllocation(allocator_, allocation_, offset, size);
  if (result != VK_SUCCESS) {
    throw VulkanError("vmaFlushAllocation failed", result);
  }
}

void GpuBuffer::Reset() {
  if (buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(allocator_, buffer_, allocation_);
  }
  allocator_ = VK_NULL_HANDLE;
  buffer_ = VK_NULL_HANDLE;
  allocation_ = VK_NULL_HANDLE;
  size_ = 0;
  address_ = 0;
  mapped_ = nullptr;
}

}  // namespace caelix
