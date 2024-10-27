#pragma once

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.h>

#include "device/device.h"

namespace Rain {
class Buffer {
 public:
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkDeviceSize size_;
  VkMemoryPropertyFlags properties_;

  VkResult Allocate(Device* device, const void* data, uint64_t size,
                    VkBufferUsageFlagBits usage_flags);
  VkResult AllocateDeviceLocal(Device* device, const void* data, uint64_t size,
                               VkBufferUsageFlagBits usage_flags);
  VkResult AllocateExternal(Device* device, const void* data, uint64_t size,
                            VkBufferUsageFlagBits usage_flags);
  static VkResult CreateBuffer(Device* device, uint64_t size,
                               VkBufferUsageFlagBits usage_flags,
                               VkMemoryPropertyFlags properties,
                               VkBuffer& buffer, VkDeviceMemory& memory);
  static VkResult CopyBuffer(Device* device, VkBuffer src, VkBuffer dst,
                             VkDeviceSize size);
  void Destroy(VkDevice device);
};
};  // namespace Rain