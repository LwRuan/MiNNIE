#pragma once

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.h>

#include "device/device.h"

namespace Rain {
class Image {
 public:
  VkImage image_ = VK_NULL_HANDLE;
  VkImageView view_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkImageUsageFlags usages_;
  VkFormat format_;
  uint32_t width_;
  uint32_t height_;
  VkImageLayout layout_;

  VkResult InitDepthImage(Device* device, uint32_t width, uint32_t height);
  VkResult InitColorImage(Device* device, VkFormat format, uint32_t width, uint32_t height);
  VkResult CreateImage(Device* device, uint32_t width, uint32_t height,
                       VkFormat format, VkImageTiling tiling,
                       VkImageUsageFlags usages,
                       VkMemoryPropertyFlags properties);
  void TransitionLayout(Device* device, VkImageLayout new_layout);
  void Destroy(VkDevice device);
  bool HasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           format == VK_FORMAT_D24_UNORM_S8_UINT;
  }
};
};  // namespace Rain