#pragma once

#include "elasticmodel/stvk.h"
#include "elasticmodel/pd.h"
#include "elasticmodel/neohookean.h"
#include "elasticobject.h"
#include "kinematicobject.h"
#include "enumtype.h"
#include "helper/yaml.h"
#include "imgui.h"
#include "mathtype.h"
#include "scene/scene.h"

namespace Rain {

class Simulator {
 public:
  uint32_t n_frame_ = 0;
  Vec3 grav_;
  real damping_;
  uint32_t n_substep_;
  real substep_size_;
  std::vector<ElasticObject*> elastic_objs_;
  std::vector<KinematicObject*> kinematic_objs_;
  void Init(Scene* scene, const YAML::Node& config);
  void Reset();
  void Update(float dt);
  void ShowUI();
  void Destroy();
};
};  // namespace Rain
