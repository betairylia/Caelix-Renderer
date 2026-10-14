#include "gpu_image.h"

#include "log.h"

#include <utility>

namespace caelix {

GpuImage::GpuImage(VmaAllocator allocator, VkDevice device, uint32_t width, uint32_t height,
                   VkFormat format, VkImageUsageFlags usage, const char* name) {
  VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = format;
  image_info.extent = {width, height, 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = usage;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VmaAllocationCreateInfo allocation_info{};
  allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  VkResult result =
      vmaCreateImage(allocator, &image_info, &allocation_info, &image_, &allocation_, nullptr);
  if (result != VK_SUCCESS) {
    CAELIX_LOGE("vmaCreateImage failed for %s: %d (%ux%u, VkFormat %d)", name,
                static_cast<int>(result), width, height, static_cast<int>(format));
    image_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    throw VulkanError("vmaCreateImage failed", result);
  }
  allocator_ = allocator;
  device_ = device;
  format_ = format;
  width_ = width;
  height_ = height;

  VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = image_;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  result = vkCreateImageView(device, &view_info, nullptr, &view_);
  if (result != VK_SUCCESS) {
    view_ = VK_NULL_HANDLE;
    CAELIX_LOGE("vkCreateImageView failed for %s: %d", name, static_cast<int>(result));
    Reset();
    throw VulkanError("vkCreateImageView failed", result);
  }
}

GpuImage::~GpuImage() { Reset(); }

GpuImage::GpuImage(GpuImage&& other) noexcept
    : allocator_(std::exchange(other.allocator_, VK_NULL_HANDLE)),
      device_(std::exchange(other.device_, VK_NULL_HANDLE)),
      image_(std::exchange(other.image_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      view_(std::exchange(other.view_, VK_NULL_HANDLE)),
      format_(std::exchange(other.format_, VK_FORMAT_UNDEFINED)),
      width_(std::exchange(other.width_, 0)),
      height_(std::exchange(other.height_, 0)) {}

GpuImage& GpuImage::operator=(GpuImage&& other) noexcept {
  if (this != &other) {
    Reset();
    allocator_ = std::exchange(other.allocator_, VK_NULL_HANDLE);
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    image_ = std::exchange(other.image_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    view_ = std::exchange(other.view_, VK_NULL_HANDLE);
    format_ = std::exchange(other.format_, VK_FORMAT_UNDEFINED);
    width_ = std::exchange(other.width_, 0);
    height_ = std::exchange(other.height_, 0);
  }
  return *this;
}

void GpuImage::Reset() {
  if (view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device_, view_, nullptr);
  }
  if (image_ != VK_NULL_HANDLE) {
    vmaDestroyImage(allocator_, image_, allocation_);
  }
  allocator_ = VK_NULL_HANDLE;
  device_ = VK_NULL_HANDLE;
  image_ = VK_NULL_HANDLE;
  allocation_ = VK_NULL_HANDLE;
  view_ = VK_NULL_HANDLE;
  format_ = VK_FORMAT_UNDEFINED;
  width_ = 0;
  height_ = 0;
}

}  // namespace caelix
