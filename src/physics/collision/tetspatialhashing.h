#pragma once

#include "mathtype.h"

namespace Rain {
class TetSpatialHashing {
 public:
  // using CellObj = std::pair<uint32_t, uint32_t>;
  using CellVert = std::pair<uint32_t, uint32_t>;
  using VertTet = std::pair<uint32_t, uint32_t>;
  uint32_t n_vert_;
  uint32_t n_ele_;
  uint32_t n_face_;
  Vec3* verts_ = nullptr;
  uint32_t* indices_ = nullptr;
  uint32_t* surf_ids_ = nullptr;
  bool* is_surf_ = nullptr;

  Vec3 grid_min_;
  Vec3 grid_max_;
  real grid_size_;
  
  Vec3* AABBs_;
  uint32_t n_hash_ = 0;
  int32_t* hashing_ = nullptr;
  int32_t* next_vert_ = nullptr;
  uint32_t n_colli_ = 0;
  VertTet* colli_pairs_ = nullptr;

  void Init(Vec3* verts, uint32_t* indices, uint32_t* surf_ids, bool* is_surf,
            uint32_t n_vert, uint32_t n_ele, uint32_t n_face, uint32_t n_hash);
  void Hashing();
  void GetSelfCollisionList();
  void Destroy();

  uint32_t HashMap(uint32_t x, uint32_t y, uint32_t z, uint32_t n);
};
};  // namespace Rain