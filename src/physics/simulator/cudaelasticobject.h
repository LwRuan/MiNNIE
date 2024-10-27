#pragma once

#include <string>

#include "cudahelper.h"
#include "cudakinematicobject.h"
#include "elasticmodel/neohookean.h"
#include "elasticmodel/pd.h"
#include "elasticmodel/stvk.h"
#include "elasticmodel/corotation.h"
#include "elasticmodel/neohookeanlog.h"
#include "enumtype.h"
#include "helper/yaml.h"
#include "mathtype.h"
#include "scene/scene.h"

namespace Rain {
struct Joint {
  std::string name;
  int parent;
  Vec3 offset;
  Vec3 axis;
  real rot = 0.;
  real rot_init;
  Vec2 rot_limit;
  Vec3 bone_start;
  Vec3 bone_end;
  real bone_radius;
};

class CudaElasticObject {
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
  // Vec3* velocities_ = nullptr;
  // Vec3* forces_ = nullptr;
  real density_ = 1e3;
  // real young_ = 5e5;
  // real poisson_ = 0.45;
  ElasticModel* elastic_model_ = nullptr;

  cudaExternalMemory_t vert_mem_;
  Vec3* dvertices_ = nullptr;
  cudaExternalMemory_t norm_mem_;
  Vec3* dnormals_ = nullptr;
  cudaExternalMemory_t face_mem_;
  uint32_t* dfaces_ = nullptr;
  uint32_t* dindices_ = nullptr;
  Vec3* dvelocities_ = nullptr;
  Vec3* dforces_ = nullptr;
  real* dmasses_ = nullptr;
  bool* dfixed_ = nullptr;
  // fixed vertices
  Vec3* dfixed_verts_ = nullptr;
  real* dvolumes_ = nullptr;
  Mat3* dDm_inv_ = nullptr;

  bool output_frame_data_ = false;
  bool self_collision_ = false;

  int32_t n_joint_ = 0;
  Joint* skeleton_ = nullptr;
  JointTransform* joint_trans_ = nullptr;
  int32_t* bone_ids_ = nullptr;
  Vec3* local_pos_ = nullptr;
  JointTransform* djoint_trans_ = nullptr;
  int32_t* dbone_ids_ = nullptr;
  Vec3* dlocal_pos_ = nullptr;

  // cuda config
  int threads_per_block_ = 64;
  int vert_threads_per_block_ = 64;
  int tet_threads_per_block_ = 64;
  int face_threads_per_block_ = 64;
  int vert_blocks_;
  int tet_blocks_;
  int face_blocks_;

  virtual void Init(VkDevice device, Object* obj, real density,
                    ElasticModelType type, real young, real poisson,
                    Joint* skeleton = nullptr, int32_t n_joint = 0);
  virtual void Reset();
  virtual void ShowUI();
  virtual void Update(cudaStream_t stream, float dt, const Vec3& grav,
                      real damping, real kinematic_penalty, real self_penalty,
                      const std::vector<CudaKinematicObject*>& kobjs,
                      uint32_t n_substep, real substep_size, uint32_t n_frame);
  virtual void Destroy();

  void UpdateSkeleton();
  Vec3 GetGlobalPos(const int i, const Vec3& pos) const;
};
};  // namespace Rain