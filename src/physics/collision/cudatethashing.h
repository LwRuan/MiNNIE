#pragma once

#include "mathtype.h"
#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <thrust/device_vector.h>

namespace Rain {
class CudaTetHashing {
 public:
  uint32_t n_vert_;
  uint32_t n_tet_;
  uint32_t n_hash_;
  Vec3 grid_min_;
  Vec3 grid_max_;
  real grid_size_;

  uint32_t n_colli_ = 0;

  // cuda
  Vec3* dverts_;
  uint32_t* dtets_;
  uint32_t* dhashing_ = nullptr;
  uint32_t* dvert_ids_ = nullptr;
  Vec3* dAABB_min_ = nullptr;
  Vec3* dAABB_max_ = nullptr;
  real* dAABB_size_ = nullptr;

  Vec3* dgrid_min_ = nullptr;
  Vec3* dgrid_max_ = nullptr;
  real* dgrid_size_ = nullptr;

  uint32_t max_n_colli_;
  uint32_t* dcolli_pairs_ = nullptr;
  uint32_t* dn_colli_ = nullptr;
  bool* dtet_sign_ = nullptr;
  bool* dis_vert_surf_ = nullptr;

  int threads_per_block_ = 64;
  int vert_threads_per_block_ = 64;
  int tet_threads_per_block_ = 64;
  int vert_blocks_;
  int tet_blocks_;

  void Init(Vec3* dverts, bool* tet_sign, bool* is_vert_surf, uint32_t* dtets, uint32_t n_vert, uint32_t n_tet,
            uint32_t n_hash, uint32_t max_n_colli);
  void Hashing(cudaStream_t stream);
  void GetVertTetCollisionList(cudaStream_t stream);
  void Destroy();
};
};  // namespace Rain