#pragma once

#include <vulkan/vulkan.h>

#include <vector>

#include "surface/swapchain.h"

namespace Rain {
class Device {
 public:
  VkDevice device_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties physicalmem_properties_;
  VkQueue graphics_queue_ = VK_NULL_HANDLE;
  VkQueue present_queue_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> command_buffers_;
  SwapChain* swap_chain_ = nullptr;

  VkResult Init(VkPhysicalDevice physical_device,
                uint32_t graphics_queue_family_index,
                uint32_t present_queue_family_index,
                const std::vector<const char*>* layers,
                const std::vector<const char*>* extensions);
  VkResult AllocateCommandBuffers(SwapChain* swap_chain);
  VkCommandBuffer BeginSingleTimeCommands();
  void EndSingleTimeCommands(VkCommandBuffer command_buffer);
  uint32_t FindMemoryTypeIndex(uint32_t type_filter,
                               VkMemoryPropertyFlags properties);
  VkFormat FindDepthFormat();
  VkFormat FindSupportFormat(const std::vector<VkFormat>& candidates,
                             VkImageTiling tiling,
                             VkFormatFeatureFlags features);
  uint32_t GetAlignedUniformByteOffset(const uint32_t offset);
  void Destroy();
};
};  // namespace Rain