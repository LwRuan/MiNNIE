#pragma once

#include <vulkan/vulkan.h>
#include <spdlog/spdlog.h>

#include <string>

namespace Rain {
class Shader {
 public:
  enum stage_t {
    SHADER_STAGE_VERTEX = 0,
    SHADER_STAGE_TESSELLATION_CONTROL,
    SHADER_STAGE_TESSELLATION_EVALUATION,
    SHADER_STAGE_GEOMETRY,
    SHADER_STAGE_FRAGMENT,
    SHADER_STAGE_COMPUTE,
    SHADER_STAGE_RAYGEN,
    SHADER_STAGE_ANY_HIT,
    SHADER_STAGE_CLOSEST_HIT,
    SHADER_STAGE_MISS,
    SHADER_STAGE_INTERSECTION,
    SHADER_STAGE_CALLABLE,
    SHADER_STAGE_TASK,
    SHADER_STAGE_MESH,

    SHADER_STAGE_NUM,
  };
  VkShaderModule modules_[SHADER_STAGE_NUM];

  Shader();
  VkResult Init(VkDevice device, const std::string& name);
  void Destroy(VkDevice device);
};
};  // namespace Rain