#pragma once

// #include "cudaelasobjmg.h"
#include "cudaelasobjmg19.h"
#include "cudaelasobjmixedmg19.h"
#include "cudakinematicobject.h"
#include "cudaelasticobject.h"
#include "cudahelper.h"
#include "elasticmodel/stvk.h"
#include "enumtype.h"
#include "helper/yaml.h"
#include "imgui.h"
#include "mathtype.h"
#include "scene/scene.h"

namespace Rain {
class CudaSimulator {
 public:
  uint32_t n_frame_ = 0;
  Vec3 grav_ = Vec3(0, -9.8, 0);
  real damping_ = 1e-6;
  real kinematic_penalty_ = 1e7;
  real self_penalty_ = 1e7;
  uint32_t n_substep_ = 10;
  real substep_size_ = 1e-5;
  std::vector<CudaElasticObject*> elastic_objs_;
  std::vector<CudaKinematicObject*> kinematic_objs_;
  cudaStream_t stream_;

  void Init(Scene* scene, const YAML::Node& config, VkDevice device,
            uint8_t* vkDeviceUUID, size_t UUID_SIZE);
  void InitCuda(uint8_t* vkDeviceUUID, size_t UUID_SIZE);
  void Reset();
  void Update(float dt);
  void ShowUI();
  void Destroy();
};
};  // namespace Rain
