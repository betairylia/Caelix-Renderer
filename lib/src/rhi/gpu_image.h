#pragma once
#include "core/vk_common.h"
#include "rhi/gpu_buffer.h"  // VulkanError

#include <cstdint>

namespace caelix {

// A 2D colour image (1 mip, 1 layer, optimal tiling) with its VMA allocation and
// one image view of the whole image. Move-only; the destructor frees all three.
// The caller is responsible for GPU lifetime (destroy an image only when no
// pending GPU work uses it) and for layout transitions: the image starts in
// VK_IMAGE_LAYOUT_UNDEFINED.
class GpuImage {
 public:
  GpuImage() = default;
  // Creates a device-local image. Throws VulkanError on failure (the C ABI
  // wrappers map VK_ERROR_OUT_OF_*_MEMORY to CAELIX_ERROR_OUT_OF_MEMORY).
  GpuImage(VmaAllocator allocator, VkDevice device, uint32_t width, uint32_t height, VkFormat format,
           VkImageUsageFlags usage, const char* name);
  ~GpuImage();
  GpuImage(GpuImage&& other) noexcept;
  GpuImage& operator=(GpuImage&& other) noexcept;
  GpuImage(const GpuImage&) = delete;
  GpuImage& operator=(const GpuImage&) = delete;

  bool valid() const { return image_ != VK_NULL_HANDLE; }
  VkImage image() const { return image_; }
  VkImageView view() const { return view_; }
  VkFormat format() const { return format_; }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  // Size of the image's memory allocation in bytes.
  VkDeviceSize allocation_bytes() const { return allocation_bytes_; }

  void Reset();

 private:
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkImage image_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  VkImageView view_ = VK_NULL_HANDLE;
  VkFormat format_ = VK_FORMAT_UNDEFINED;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  VkDeviceSize allocation_bytes_ = 0;
};

}  // namespace caelix
