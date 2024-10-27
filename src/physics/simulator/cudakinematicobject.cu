#include "cudakinematicobject.h"

namespace Rain {
namespace CUDA {
static __global__ void UpdateVertices(const Vec3 trans, const Mat3 rot,
                                      const Vec3* rel_pos, Vec3* X,
                                      const uint32_t n_vert) {
  int v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  X[v] = trans + rot * rel_pos[v];
}

static __global__ void TorusHack(real da, Vec3* X,
                                 const uint32_t n_vert) {
  int v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  Vec3 dir = Vec3(X[v][0], 0, X[v][2]);
  dir = dir.normalized();
  X[v] += da * dir;
}
}  // namespace CUDA

void CudaKinematicObject::Update(cudaStream_t stream, float Dt,
                                 uint32_t n_substep, real substep_size,
                                 uint32_t n_frame) {
  if (n_frame < start_frame_) return;
  if ((end_frame_ != -1) && n_frame >= end_frame_) return;
  // real dt = std::min((real)Dt, n_substep * substep_size);
  real dt = n_substep * substep_size;
  real theta = angular_velocity_.norm() * dt;
  Vec3 n = angular_velocity_.normalized();
  Mat3 cross;
  cross << 0, -n.z(), n.y(), n.z(), 0, -n.x(), -n.y(), n.x(), 0;
  Mat3 r = std::cos(theta) * Mat3::Identity() + std::sin(theta) * cross +
           (1 - std::cos(theta)) * n * n.transpose();
  rotation_ = r * rotation_;
  translation_ = translation_ + velocity_ * dt;
  CUDA::UpdateVertices<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
      translation_, rotation_, drel_pos_, dvertices_, obj_->n_vert_);
  
  // {// HACK: for the bunny-torus example
  //   if (n_frame <= 160) {
  //     real da = -0.01;
  //     Vec3 center;
  //     obj_->implicit_geo_params_[6] += da;
  //     center << obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
  //         obj_->implicit_geo_params_[2];
  //     CUDA::TorusHack<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
  //         da, drel_pos_, obj_->n_vert_);
  //   }
  // }
}
};  // namespace Rain