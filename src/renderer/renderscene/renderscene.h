#pragma once

#include <vulkan/vulkan.h>

#include <vector>

#include "buffer/buffer.h"
#include "camera/camera.h"
#include "device/device.h"
#include "mathtype.h"
#include "scene/scene.h"
#include "surface/swapchain.h"

namespace Rain {
struct GlobalUniformData {
  alignas(16) Mat4f proj; // camera
  alignas(16) Mat4f view; // camera  
  alignas(16) Vec3f ambient;
  alignas(16) Vec3f directional;
  alignas(16) Vec3f light_direction;
};

struct ModelUniformData {
  // ambient + non-transparency
  alignas(16) Vec4f Ka_d_ = Vec4f(0.2f, 0.2f, 0.2f, 1.0f);
  // diffuse
  alignas(16) Vec4f Kd_ = Vec4f(0.8f, 0.8f, 0.8f, 0.0f);
  // specular rgb + shininess
  alignas(16) Vec4f Ks_Ns_ = Vec4f(1.0f, 1.0f, 1.0f, 0.0f);
  // model transformation
  alignas(16) Mat4f model_ = Mat4f::Identity();
};

class RenderModel {
  // TODO: not optimal, use vma to alloc a big buffer, then divide to models
 public:
  bool visible_ = true;
  Object* obj_;
  ModelUniformData uniform_data_;
  std::vector<Buffer> vertex_buffers_;
  Buffer index_buffer_;
  std::vector<VkBuffer> vertex_vkbuffers_;
  std::vector<VkDeviceSize> vertex_vkbuffer_offsets_;

  uint32_t ubo_offset_;
  uint32_t ubo_size_;

  void Init(Object* obj);
  VkResult CreateBuffers(Device* device, bool is_gpu);
  void Destroy(VkDevice device);
  static std::vector<VkVertexInputBindingDescription> GetBindDescription();
  static std::vector<VkVertexInputAttributeDescription>
  GetAttributeDescriptions();
};

class RenderScene {
 public:
  Camera* camera_ = nullptr;
  Vec3f ambient_light_ = Vec3f(0.5f, 0.5f, 0.5f);
  float ambient_light_intensity_ = 1.0f;
  Vec3f directional_light_ = Vec3f(1.0f, 1.0f, 1.0f);
  Vec3f light_direction_;
  float directional_light_intensity_ = 1.0f;
  float light_x_angle_ = 45.0;
  float light_y_angle_ = 45.0;

  std::vector<Buffer> global_ubs_;  // per swap image
  std::vector<Buffer> model_ubs_;   // per swap image
  std::vector<RenderModel> models_;
  uint32_t n_swap_image_;
  uint8_t* model_ubo_data_ = nullptr;
  uint32_t model_ubo_size_;
  uint32_t n_uniform_buffer_ = 2;
  uint32_t n_uniform_texture_ = 0;

  VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> sets_;

  VkResult Init(Device* device, SwapChain* swap_chain, Scene* scene,
                bool is_gpu);
  VkResult InitUniform(Device* device, SwapChain* swap_chain);
  VkResult InitDescriptor(Device* device);
  void UpdateUniform(VkDevice device, uint32_t image_index);
  void UpdateVertex(VkDevice device);
  void BindAndDraw(VkCommandBuffer command_buffer, VkPipelineLayout layout,
                   uint32_t image_index, uint32_t model_index);
  void DestroyUniform(VkDevice device);
  void Destroy(VkDevice device);
};
};  // namespace Rain