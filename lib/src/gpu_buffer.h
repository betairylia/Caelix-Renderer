#pragma once
#include "vk_common.h"

#include <stdexcept>

namespace caelix {

// Thrown by internal code when a Vulkan or VMA call fails. The C ABI wrappers
// convert it to CAELIX_ERROR_OUT_OF_MEMORY or CAELIX_ERROR_VULKAN.
class VulkanError : public std::runtime_error {
 public:
  VulkanError(const char* what, VkResult result) : std::runtime_error(what), result_(result) {}
  VkResult result() const { return result_; }

 private:
  VkResult result_;
};

// A VkBuffer with its VMA allocation. Move-only; the destructor frees both.
// The caller is responsible for GPU lifetime: destroy a buffer only when no
// pending GPU work uses it.
class GpuBuffer {
 public:
  GpuBuffer() = default;
  // Creates the buffer. Throws VulkanError on failure. With `want_address`,
  // queries the VkDeviceAddress (the usage needs SHADER_DEVICE_ADDRESS).
  GpuBuffer(VmaAllocator allocator, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage,
            const VmaAllocationCreateInfo& allocation_info, bool want_address);
  ~GpuBuffer();
  GpuBuffer(GpuBuffer&& other) noexcept;
  GpuBuffer& operator=(GpuBuffer&& other) noexcept;
  GpuBuffer(const GpuBuffer&) = delete;
  GpuBuffer& operator=(const GpuBuffer&) = delete;

  // A device-local pool buffer: storage, transfer src/dst, device address.
  static GpuBuffer DeviceLocal(VmaAllocator allocator, VkDevice device, VkDeviceSize size);
  // A persistently mapped, host-visible upload buffer (transfer src).
  static GpuBuffer HostUpload(VmaAllocator allocator, VkDevice device, VkDeviceSize size);

  bool valid() const { return buffer_ != VK_NULL_HANDLE; }
  VkBuffer buffer() const { return buffer_; }
  VkDeviceSize size() const { return size_; }
  VkDeviceAddress address() const { return address_; }
  // Non-null only for buffers created with VMA_ALLOCATION_CREATE_MAPPED_BIT.
  uint8_t* mapped() const { return mapped_; }

  // Makes host writes in [offset, offset + size) visible to the device. No-op
  // on host-coherent memory.
  void FlushHostWrites(VkDeviceSize offset, VkDeviceSize size) const;

  void Reset();

 private:
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  VkDeviceSize size_ = 0;
  VkDeviceAddress address_ = 0;
  uint8_t* mapped_ = nullptr;
};

}  // namespace caelix
