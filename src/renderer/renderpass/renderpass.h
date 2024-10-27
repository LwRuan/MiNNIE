#pragma once

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.h>

#include "device/device.h"

namespace Rain {
class RenderPass {
public:
  VkRenderPass render_pass_;

  VkResult Init(Device* device, const VkFormat& format);
  void Destroy(VkDevice device);
};

class ImGuiRenderPass {
public:
  VkRenderPass render_pass_;

  VkResult Init(Device* device, const VkFormat& format);
  void Destroy(VkDevice device);
};
};  // namespace Rain