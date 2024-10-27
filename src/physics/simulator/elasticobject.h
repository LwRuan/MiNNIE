#pragma once

#include "elasticmodel/neohookean.h"
#include "elasticmodel/pd.h"
#include "elasticmodel/stvk.h"
#include "imgui.h"
#include "kinematicobject.h"
#include "scene/scene.h"

namespace Rain {
class ElasticObject {
 public:
  Object* obj_;
  uint32_t n_tet_;
  uint32_t n_vert_;
  Vec3* verts_;
  uint32_t* tets_;
  real* masses_ = nullptr;
  real* volumes_ = nullptr;
  Mat3* Dm_inv_ = nullptr;
  Vec3* initial_vertices_ = nullptr;
  Vec3* velocities_ = nullptr;
  Vec3* forces_ = nullptr;
  real density_ = 1e3;
  real young_ = 5e5;
  real poisson_ = 0.45;
  ElasticModel* elastic_model_ = nullptr;
  bool output_frame_data_ = false;
  bool self_collision_ = false;

  void Init(Object* obj, real density, ElasticModelType type, real young,
            real poisson);
  virtual void Reset();
  virtual void Update(float dt, const Vec3& grav, real damping,
                      uint32_t n_substep, real substep_size, uint32_t n_frame,
                      const std::vector<KinematicObject*>& kobjs);
  virtual void ShowUI();
  virtual void Destroy();
};
};  // namespace Rain