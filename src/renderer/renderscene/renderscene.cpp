#include "renderscene.h"

#include <array>
#include <type_traits>

namespace Rain {

void RenderModel::Init(Object* obj) {
  obj_ = obj;
  obj->render_model_ = this;
  uniform_data_.Ka_d_.segment<3>(0) = obj_->material_.Ka_;
  uniform_data_.Ka_d_[3] = obj_->material_.d_;
  uniform_data_.Kd_.segment<3>(0) = obj_->material_.Kd_;
  uniform_data_.Ks_Ns_.segment<3>(0) = obj_->material_.Ks_;
  uniform_data_.Ks_Ns_[3] = obj_->material_.Ns_;
  uniform_data_.model_ = obj_->transformation_;
}

VkResult RenderModel::CreateBuffers(Device* device, bool is_gpu) {
  uint64_t size;
  VkResult result;

  // vertex buffers
  vertex_buffers_.resize(2);
  size = (uint64_t)(sizeof(Vec3)) * obj_->n_vert_;
  if (is_gpu)
    result = vertex_buffers_[0].AllocateExternal(
        device, obj_->vertices_, size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  else
    result = vertex_buffers_[0].Allocate(device, obj_->vertices_, size,
                                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  if (result != VK_SUCCESS) return result;

  size = (uint64_t)(sizeof(Vec3)) * obj_->n_vert_;
  if (is_gpu)
    result = vertex_buffers_[1].AllocateExternal(
        device, obj_->normals_, size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  else
    result = vertex_buffers_[1].Allocate(device, obj_->normals_, size,
                                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  if (result != VK_SUCCESS) return result;

  vertex_vkbuffers_.clear();
  for (auto buffer : vertex_buffers_) {
    vertex_vkbuffers_.push_back(buffer.buffer_);
  }
  vertex_vkbuffer_offsets_.clear();
  for (int i = 0; i < vertex_buffers_.size(); ++i) {
    vertex_vkbuffer_offsets_.push_back(0);
  }

  // index buffer
  size = (uint64_t)(sizeof(uint32_t)) * obj_->n_surfidx_;
  result = index_buffer_.AllocateExternal(
      device, obj_->surface_indices_, size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  if (result != VK_SUCCESS) return result;

  return VK_SUCCESS;
}

void RenderModel::Destroy(VkDevice device) {
  for (auto buffer : vertex_buffers_) {
    buffer.Destroy(device);
  }
  index_buffer_.Destroy(device);
}

std::vector<VkVertexInputBindingDescription> RenderModel::GetBindDescription() {
  std::vector<VkVertexInputBindingDescription> ret;
  VkVertexInputBindingDescription vertices_desc;
  vertices_desc.binding = 0;
  vertices_desc.stride = sizeof(Vec3);
  vertices_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  ret.push_back(vertices_desc);

  VkVertexInputBindingDescription normals_desc;
  normals_desc.binding = 1;
  normals_desc.stride = sizeof(Vec3);
  normals_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  ret.push_back(normals_desc);

  return ret;
}

std::vector<VkVertexInputAttributeDescription>
RenderModel::GetAttributeDescriptions() {
  VkFormat format;
  uint32_t normal_location;
  if constexpr (std::is_same<real, float>()) {
    format = VK_FORMAT_R32G32B32_SFLOAT;
    normal_location = 1;
  } else if constexpr (std::is_same<real, double>()) {
    format = VK_FORMAT_R64G64B64_SFLOAT;
    normal_location = 2;
  }
  std::vector<VkVertexInputAttributeDescription> ret;
  VkVertexInputAttributeDescription vertices_desc;
  vertices_desc.binding = 0;
  vertices_desc.location = 0;
  vertices_desc.format = format;
  vertices_desc.offset = 0;
  ret.push_back(vertices_desc);

  VkVertexInputAttributeDescription normals_desc;
  normals_desc.binding = 1;
  normals_desc.location = normal_location;
  normals_desc.format = format;
  normals_desc.offset = 0;
  ret.push_back(normals_desc);

  return ret;
}

VkResult RenderScene::Init(Device* device, SwapChain* swap_chain, Scene* scene,
                           bool is_gpu) {
  VkResult result;
  models_.resize(scene->objects_.size());
  uint32_t offset = 0;
  for (size_t i = 0; i < models_.size(); ++i) {
    models_[i].Init(&scene->objects_[i]);
    models_[i].ubo_offset_ = offset;
    models_[i].ubo_size_ = sizeof(ModelUniformData);
    offset += device->GetAlignedUniformByteOffset(sizeof(ModelUniformData));
    result = models_[i].CreateBuffers(device, is_gpu);
    if (result != VK_SUCCESS) {
      spdlog::error("model buffer creation failed");
      return result;
    }
  }
  model_ubo_size_ = offset;
  model_ubo_data_ = new uint8_t[model_ubo_size_];
  for (size_t i = 0; i < models_.size(); ++i) {
    memcpy(model_ubo_data_ + models_[i].ubo_offset_, &models_[i].uniform_data_,
           sizeof(ModelUniformData));
  }
  camera_ = new Camera;
  float aspect = 1.0;
  if (swap_chain->extent_.height)
    aspect = float(swap_chain->extent_.width) / swap_chain->extent_.height;
  camera_->InitData(aspect, 0.25f * PI_, 1.0f, 1000.0f, 30.0f, 0.0f, 0.3f * PI_,
                    Vec3f::Zero());
  result = InitUniform(device, swap_chain);
  if (result != VK_SUCCESS) {
    return result;
  }
  return VK_SUCCESS;
}

VkResult RenderScene::InitUniform(Device* device, SwapChain* swap_chain) {
  VkResult result;
  n_swap_image_ = swap_chain->images_.size();

  // global
  VkDeviceSize size = sizeof(GlobalUniformData);
  global_ubs_.resize(n_swap_image_);
  for (size_t i = 0; i < n_swap_image_; ++i) {
    result = global_ubs_[i].Allocate(device, nullptr, size,
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    if (result != VK_SUCCESS) {
      return result;
    }
  }
  // model
  model_ubs_.resize(n_swap_image_);
  for (size_t i = 0; i < n_swap_image_; ++i) {
    result = model_ubs_[i].AllocateDeviceLocal(
        device, model_ubo_data_, model_ubo_size_,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    if (result != VK_SUCCESS) {
      return result;
    }
  }

  result = InitDescriptor(device);
  if (result != VK_SUCCESS) {
    return result;
  }

  return VK_SUCCESS;
}

void RenderScene::UpdateUniform(VkDevice device, uint32_t image_index) {
  camera_->UpdateData();
  float theta = light_x_angle_ / 180 * PI_;
  float phi = light_y_angle_ / 180 * PI_;
  light_direction_ =
      -Vec3f(-cos(phi) * cos(theta), sin(phi), cos(phi) * sin(theta));
  void* data;
  GlobalUniformData global_data;
  global_data.proj = camera_->proj_;
  global_data.view = camera_->view_;
  global_data.ambient = ambient_light_ * ambient_light_intensity_;
  global_data.directional = directional_light_ * directional_light_intensity_;
  global_data.light_direction = light_direction_;
  vkMapMemory(device, global_ubs_[image_index].memory_, 0,
              sizeof(GlobalUniformData), 0, &data);
  memcpy(data, &global_data, sizeof(GlobalUniformData));
  vkUnmapMemory(device, global_ubs_[image_index].memory_);
}

void RenderScene::UpdateVertex(VkDevice device) {
  uint64_t size;
  for (auto model : models_) {
    void* data;
    size = (uint64_t)(sizeof(Vec3)) * model.obj_->n_vert_;
    vkMapMemory(device, model.vertex_buffers_[0].memory_, 0, size, 0, &data);
    memcpy(data, model.obj_->vertices_, size);
    vkUnmapMemory(device, model.vertex_buffers_[0].memory_);
    vkMapMemory(device, model.vertex_buffers_[1].memory_, 0, size, 0, &data);
    memcpy(data, model.obj_->normals_, size);
    vkUnmapMemory(device, model.vertex_buffers_[1].memory_);
  }
}

void RenderScene::BindAndDraw(VkCommandBuffer command_buffer,
                              VkPipelineLayout layout, uint32_t image_index,
                              uint32_t model_index) {
  vkCmdBindVertexBuffers(command_buffer, 0,
                         models_[model_index].vertex_vkbuffers_.size(),
                         models_[model_index].vertex_vkbuffers_.data(),
                         models_[model_index].vertex_vkbuffer_offsets_.data());
  vkCmdBindIndexBuffer(command_buffer,
                       models_[model_index].index_buffer_.buffer_, 0,
                       VK_INDEX_TYPE_UINT32);
  vkCmdBindDescriptorSets(
      command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1,
      &sets_[image_index * models_.size() + model_index], 0, nullptr);
  vkCmdDrawIndexed(command_buffer,
                   static_cast<uint32_t>(models_[model_index].obj_->n_surfidx_),
                   1, 0, 0, 0);
}

VkResult RenderScene::InitDescriptor(Device* device) {
  VkResult result;
  uint32_t n_uniform = n_uniform_buffer_ + n_uniform_texture_;
  std::vector<VkDescriptorSetLayoutBinding> uniform_bindings;
  uniform_bindings.resize(n_uniform);
  uint32_t idx = 0;
  for (uint32_t i = 0; i < n_uniform_buffer_; ++i) {
    uniform_bindings[idx].binding = idx;
    uniform_bindings[idx].descriptorCount = 1;
    uniform_bindings[idx].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniform_bindings[idx].pImmutableSamplers = nullptr;
    uniform_bindings[idx].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    ++idx;
  }
  for (uint32_t i = 0; i < n_uniform_texture_; ++i) {
    uniform_bindings[idx].binding = idx;
    uniform_bindings[idx].descriptorCount = 1;
    uniform_bindings[idx].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;  // image + sampler
    uniform_bindings[idx].pImmutableSamplers = nullptr;
    uniform_bindings[idx].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    ++idx;
  }

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = uniform_bindings.size();
  layout_info.pBindings = uniform_bindings.data();
  result = vkCreateDescriptorSetLayout(device->device_, &layout_info, nullptr,
                                       &layout_);
  if (result != VK_SUCCESS) {
    spdlog::error("desciptor set layout creation failed");
    return result;
  }

  std::vector<VkDescriptorPoolSize> pool_sizes;
  pool_sizes.clear();
  if (n_uniform_buffer_ > 0) {
    VkDescriptorPoolSize pool_size;
    pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size.descriptorCount =
        n_uniform_buffer_ * n_swap_image_ * models_.size();
    pool_sizes.push_back(pool_size);
  }
  if (n_uniform_texture_ > 0) {
    VkDescriptorPoolSize pool_size;
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount =
        n_uniform_texture_ * n_swap_image_ * models_.size();
    pool_sizes.push_back(pool_size);
  }

  VkDescriptorPoolCreateInfo pool_info;
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.pNext = nullptr;
  pool_info.poolSizeCount = (uint32_t)pool_sizes.size();
  pool_info.pPoolSizes = pool_sizes.data();
  pool_info.maxSets = n_swap_image_ * models_.size();
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  result = vkCreateDescriptorPool(device->device_, &pool_info, nullptr, &pool_);
  if (result != VK_SUCCESS) {
    spdlog::error("descriptor pool creation failed");
    return result;
  }

  std::vector<VkDescriptorSetLayout> layouts(n_swap_image_ * models_.size(),
                                             layout_);
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = pool_;
  alloc_info.descriptorSetCount = layouts.size();
  alloc_info.pSetLayouts = layouts.data();
  sets_.resize(layouts.size());
  result = vkAllocateDescriptorSets(device->device_, &alloc_info, sets_.data());
  if (result != VK_SUCCESS) {
    spdlog::error("descriptor sets allocation failed");
    return result;
  }

  for (size_t i = 0; i < n_swap_image_; ++i) {
    for (size_t j = 0; j < models_.size(); ++j) {
      VkDescriptorBufferInfo global_info{};
      global_info.buffer = global_ubs_[i].buffer_;
      global_info.offset = 0;
      global_info.range = sizeof(GlobalUniformData);

      VkWriteDescriptorSet global_write{};
      global_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      global_write.dstSet = sets_[i * models_.size() + j];
      global_write.dstBinding = 0;
      global_write.dstArrayElement = 0;
      global_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      global_write.descriptorCount = 1;
      global_write.pBufferInfo = &global_info;

      VkDescriptorBufferInfo model_info{};
      model_info.buffer = model_ubs_[i].buffer_;
      model_info.offset = models_[j].ubo_offset_;
      model_info.range = models_[j].ubo_size_;

      VkWriteDescriptorSet model_write{};
      model_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      model_write.dstSet = sets_[i * models_.size() + j];
      model_write.dstBinding = 1;
      model_write.dstArrayElement = 0;
      model_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      model_write.descriptorCount = 1;
      model_write.pBufferInfo = &model_info;

      std::array<VkWriteDescriptorSet, 2> writes{global_write, model_write};
      vkUpdateDescriptorSets(device->device_, writes.size(), writes.data(), 0,
                             nullptr);
    }
  }

  return VK_SUCCESS;
}

void RenderScene::DestroyUniform(VkDevice device) {
  if (layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, layout_, nullptr);
    layout_ = VK_NULL_HANDLE;
  }
  if (pool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
  }
  for (auto buffer : global_ubs_) {
    buffer.Destroy(device);
  }
  for (auto buffer : model_ubs_) {
    buffer.Destroy(device);
  }
}

void RenderScene::Destroy(VkDevice device) {
  DestroyUniform(device);
  for (auto model : models_) {
    model.Destroy(device);
  }
  delete camera_;
  delete model_ubo_data_;
}
};  // namespace Rain