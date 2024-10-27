#pragma once

#include "cudahelper.h"
#include "enumtype.h"
#include "mathtype.h"
#include "scene/scene.h"

namespace Rain {
class CudaKinematicObject {
 public:
  Object* obj_;

  Mat3 rotation_ = Mat3::Identity();
  Vec3 translation_ = Vec3::Zero();
  Vec3 pivot_;
  Vec3 velocity_;
  Vec3 angular_velocity_;
  Vec3* rel_pos_;
  int start_frame_;
  int end_frame_;

  cudaExternalMemory_t vert_mem_;
  Vec3* dvertices_ = nullptr;
  cudaExternalMemory_t norm_mem_;
  Vec3* dnormals_ = nullptr;
  cudaExternalMemory_t face_mem_;
  uint32_t* dfaces_ = nullptr;
  Vec3* drel_pos_;

  int vert_threads_per_block_ = 64;
  int vert_blocks_;

  void Init(VkDevice device, Object* obj, const Vec3& pivot, const Vec3& vel,
            const Vec3& angular_vel, int st_frame, int ed_frame);
  void Reset();
  void Update(cudaStream_t stream, float dt, uint32_t n_substep,
              real substep_size, uint32_t n_frame);
  void ShowUI();
  void Destroy();
};
};  // namespace Rain