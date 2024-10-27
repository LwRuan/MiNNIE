#pragma once

#include "mathtype.h"

#include <algorithm>

namespace Rain {
struct ColliPair {
  // if vertex-face, p0 for vertex, p1p2p3 for triangle
  // if edge-edge, p0p1 for edge1, p2p3 for edge2
  uint32_t p0;
  uint32_t p1;
  uint32_t p2;
  uint32_t p3;
  // if vertex-face, w1 w2 for triangle weights
  // if edge-edge, w1 for edge1, w2 for edge2
  real w1;
  real w2;
  int type; // 0 : vertex face, 1 : edge edge
};
class SpatialHashing {
 public:
  using CellObj = std::pair<uint32_t, uint32_t>;
  using Pair = std::pair<uint32_t, uint32_t>;
  uint32_t n_vert_;
  Vec3* verts_old_;
  Vec3* verts_new_;
  uint32_t n_ele_;
  uint32_t* indices_;
  Vec3 grid_min_;
  Vec3 grid_max_;
  real grid_size_;

  uint32_t n_hash_;
  CellObj* hashing_ = nullptr;
  uint32_t n_cell_;
  uint32_t* cell_offs_ = nullptr;
  uint32_t n_colli_;
  ColliPair* colli_pairs_ = nullptr;
  

  void Init(Vec3* verts_old, Vec3* verts_new, uint32_t* indices, uint32_t n_vert, uint32_t n_ele);
  void Hashing();
  void GetSelfCollisionList();
  void Destroy();
};
};  // namespace Rain