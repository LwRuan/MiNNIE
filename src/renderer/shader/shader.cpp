#include "shader.h"

#include <cstring>
#include <vector>

#include "helper/io.h"

namespace Rain {
Shader::Shader() {
  memset(modules_, 0, sizeof(VkShaderModule) * SHADER_STAGE_NUM);
}

VkResult Shader::Init(VkDevice device, const std::string& name) {
  const char* file_ext[SHADER_STAGE_NUM];
  file_ext[SHADER_STAGE_VERTEX] = "vert";
  file_ext[SHADER_STAGE_TESSELLATION_CONTROL] = "tess";
  file_ext[SHADER_STAGE_TESSELLATION_EVALUATION] = "tval";
  file_ext[SHADER_STAGE_GEOMETRY] = "geom";
  file_ext[SHADER_STAGE_FRAGMENT] = "frag";
  file_ext[SHADER_STAGE_COMPUTE] = "comp";
  file_ext[SHADER_STAGE_RAYGEN] = "rgen";
  file_ext[SHADER_STAGE_ANY_HIT] = "ahit";
  file_ext[SHADER_STAGE_CLOSEST_HIT] = "chit";
  file_ext[SHADER_STAGE_MISS] = "miss";
  file_ext[SHADER_STAGE_INTERSECTION] = "rint";
  file_ext[SHADER_STAGE_CALLABLE] = "call";
  file_ext[SHADER_STAGE_TASK] = "task";
  file_ext[SHADER_STAGE_MESH] = "mesh";

  for (size_t i = 0; i < SHADER_STAGE_NUM; ++i) {
    std::string file_name = "shaders/" + name + "_" + file_ext[i] + ".spv";
    std::vector<char> code = IO::ReadFile(file_name);
    if (code.size()) {
      VkShaderModuleCreateInfo create_info{};
      create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      create_info.codeSize = code.size();
      create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());
      VkResult result =
          vkCreateShaderModule(device, &create_info, nullptr, &modules_[i]);
      if (result != VK_SUCCESS) {
        spdlog::error("{} module creation failed", name + "." + file_ext[i]);
        return result;
      }
      // spdlog::debug("shader {} loaded", name + "." + file_ext[i]);
    }
  }
  return VK_SUCCESS;
}

void Shader::Destroy(VkDevice device) {
  for (size_t i = 0; i < SHADER_STAGE_NUM; ++i) {
    if (modules_[i]) {
      vkDestroyShaderModule(device, modules_[i], nullptr);
    }
  }
}
};  // namespace Rain