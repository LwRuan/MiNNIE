#include "cudaelasticobject.h"
#include "cudasim/explicit.h"

namespace Rain {
void CudaElasticObject::Update(cudaStream_t stream, float Dt, const Vec3& grav,
                               real damping, real kinematic_penalty, real self_penalty,
                               const std::vector<CudaKinematicObject*>& kobjs,
                               uint32_t n_substep, real substep_size,
                               uint32_t n_frame) {
  for (int step = 0; step < n_substep; ++step) {
    real dt = std::min((real)Dt / n_substep, substep_size);
    // compute force
    CUDA::ClearArray<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
        dforces_, (uint32_t)obj_->n_vert_);
    CUDA::ComputeForceExplicitNeoHookean<<<tet_blocks_, tet_threads_per_block_,
                                           0, stream>>>(
        dindices_, dvertices_, dDm_inv_, dvolumes_, dforces_,
        (uint32_t)obj_->n_ele_, (uint32_t)obj_->n_vert_, elastic_model_->mu_,
        elastic_model_->lambda_);
    CUDA::UpdateSimpletic<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
        dvertices_, dvelocities_, dfixed_, dforces_, dmasses_,
        (uint32_t)obj_->n_vert_, grav, damping, dt);
  }
  // Update normal
  CUDA::ClearArray<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
      dnormals_, (uint32_t)obj_->n_vert_);
  CUDA::UpdateFaceNormals<<<face_blocks_, face_threads_per_block_, 0, stream>>>(
      dnormals_, dvertices_, dfaces_, (uint32_t)obj_->n_face_);
  CUDA::UpdateVertNormals<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
      dnormals_, (uint32_t)obj_->n_vert_);
}
};  // namespace Rain