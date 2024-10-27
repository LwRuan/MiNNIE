#pragma once

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.h>

#include "renderscene/renderscene.h"
#include "shader/shader.h"

namespace Rain {
class Pipeline {
 public:
  Shader* shader_ = nullptr;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_;

  VkResult Init(VkDevice device, const VkExtent2D& extent,
                VkRenderPass render_pass, RenderScene* scene);
  void Destroy(VkDevice device);
};
};  // namespace Rain