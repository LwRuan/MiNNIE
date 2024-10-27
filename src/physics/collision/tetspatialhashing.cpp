#include "tetspatialhashing.h"

#include <algorithm>
#include <set>

#include "geometry/tetmesh.h"

namespace Rain {
void TetSpatialHashing::Init(Vec3* verts, uint32_t* indices, uint32_t* surf_ids,
                             bool* is_surf, uint32_t n_vert, uint32_t n_ele,
                             uint32_t n_face, uint32_t n_hash) {
  verts_ = verts;
  indices_ = indices;
  surf_ids_ = surf_ids;
  n_vert_ = n_vert;
  n_ele_ = n_ele;
  n_face_ = n_face;
  is_surf_ = is_surf;
  n_hash_ = n_hash;

  AABBs_ = new Vec3[2 * n_ele_];
  hashing_ = new int32_t[n_hash_];
  colli_pairs_ = new VertTet[n_vert_* 10000];
  next_vert_ = new int32_t[n_vert_];
}

void TetSpatialHashing::Hashing() {
  // generate grid
  grid_min_ = 1e16 * Vec3::Ones();
  grid_max_ = -1e16 * Vec3::Ones();

  real max_ele_size = 0;
  for (uint32_t e = 0; e < n_ele_; ++e) {
    Vec3 ele_bbox_min = 1e16 * Vec3::Ones();
    Vec3 ele_bbox_max = -1e16 * Vec3::Ones();
    for (uint32_t t = 0; t < 4; ++t) {
      uint32_t i = indices_[e * 4 + t];
      ele_bbox_min = ele_bbox_min.cwiseMin(verts_[i]);
      ele_bbox_max = ele_bbox_max.cwiseMax(verts_[i]);
    }
    AABBs_[2 * e] = ele_bbox_min;
    AABBs_[2 * e + 1] = ele_bbox_max;
    grid_min_ = grid_min_.cwiseMin(ele_bbox_min);
    grid_max_ = grid_max_.cwiseMax(ele_bbox_max);
    Vec3 delta = ele_bbox_max - ele_bbox_min;
    max_ele_size = std::max(delta.maxCoeff(), max_ele_size);
  }
  // expand the grid a bit
  Vec3 bbox_size = grid_max_ - grid_min_;
  grid_min_ -= 0.01 * bbox_size;
  grid_max_ += 0.01 * bbox_size;
  bbox_size = grid_max_ - grid_min_;

  grid_size_ = max_ele_size / 2;
  // grid_size_ = std::max(grid_size_, bbox_size.maxCoeff() / 128);

  memset(next_vert_, 0xff, sizeof(int32_t) * n_vert_);
  memset(hashing_, 0xff, sizeof(int32_t) * n_hash_);
  for (uint32_t i = 0; i < n_vert_; ++i) {
    Vec3i cell = ((verts_[i] - grid_min_) / grid_size_).cast<int>();
    uint32_t key = HashMap(cell.x(), cell.y(), cell.z(), n_hash_);
    if (hashing_[key] == -1) {  // no vert in cell
      hashing_[key] = i;
    } else {  // insert vert
      next_vert_[i] = hashing_[key];
      hashing_[key] = i;
    }
  }
}

void TetSpatialHashing::GetSelfCollisionList() {
  n_colli_ = 0;
  memset(colli_pairs_, 0, sizeof(VertTet) * n_vert_ * 10000);
  for (uint32_t e = 0; e < n_ele_; ++e) {
    uint32_t* tet = &indices_[4 * e];
    const Vec3& bbox_min = AABBs_[2 * e];
    const Vec3& bbox_max = AABBs_[2 * e + 1];
    Vec3i low_cell = ((bbox_min - grid_min_) / grid_size_).cast<int>();
    Vec3i up_cell = ((bbox_max - grid_min_) / grid_size_).cast<int>();
    for (uint32_t idx = low_cell(0); idx <= up_cell(0); ++idx) {
      for (uint32_t idy = low_cell(1); idy <= up_cell(1); ++idy) {
        for (uint32_t idz = low_cell(2); idz <= up_cell(2); ++idz) {
          uint32_t key = HashMap(idx, idy, idz, n_hash_);
          int32_t v = hashing_[key];
          while (v != -1) {
            if ((v != tet[0]) && (v != tet[1]) && (v != tet[2]) &&
                (v != tet[3])) {
              if (Tet::VertexInside(verts_[v], verts_[tet[0]], verts_[tet[1]],
                                    verts_[tet[2]], verts_[tet[3]])) {
                colli_pairs_[n_colli_++] = {v, e};
              }
            }
            v = next_vert_[v];
          }
        }
      }
    }
  }
}

void TetSpatialHashing::Destroy() {
  if (AABBs_) delete[] AABBs_;
  if (hashing_) delete[] hashing_;
  if (next_vert_) delete[] next_vert_;
  if (colli_pairs_) delete[] colli_pairs_;
}

uint32_t TetSpatialHashing::HashMap(uint32_t x, uint32_t y, uint32_t z,
                                    uint32_t n) {
  return ((x * 73856093u) ^ (y * 19349663u) ^ (z * 83492791u)) % n;
}

// void TetSpatialHashing::Init(Vec3* verts, uint32_t* indices, uint32_t*
// surf_ids,
//                              bool* is_surf, uint32_t n_vert, uint32_t n_ele,
//                              uint32_t n_face) {
//   verts_ = verts;
//   indices_ = indices;
//   surf_ids_ = surf_ids;
//   n_vert_ = n_vert;
//   n_ele_ = n_ele;
//   n_face_ = n_face;
//   is_surf_ = is_surf;

//   grid_min_ = 1e16 * Vec3::Ones();
//   grid_max_ = -1e16 * Vec3::Ones();

//   // std::cout << n_ele_ << std::endl;
//   // decide grid size, bigger than an element
//   // also insure grid number in one dim < 256
//   real max_ele_size = 0;
//   for (uint32_t e = 0; e < n_ele_; ++e) {
//     Vec3 ele_bbox_min = 1e16 * Vec3::Ones();
//     Vec3 ele_bbox_max = -1e16 * Vec3::Ones();
//     for (uint32_t t = 0; t < 4; ++t) {
//       uint32_t i = indices_[e * 4 + t];
//       ele_bbox_min = ele_bbox_min.cwiseMin(verts_[i]);
//       ele_bbox_max = ele_bbox_max.cwiseMax(verts_[i]);
//     }
//     grid_min_ = grid_min_.cwiseMin(ele_bbox_min);
//     grid_max_ = grid_max_.cwiseMax(ele_bbox_max);
//     Vec3 delta = ele_bbox_max - ele_bbox_min;
//     max_ele_size = std::max(delta.maxCoeff(), max_ele_size);
//   }
//   // expand the grid a bit
//   Vec3 bbox_size = grid_max_ - grid_min_;
//   grid_min_ -= 0.01 * bbox_size;
//   grid_max_ += 0.01 * bbox_size;
//   bbox_size = grid_max_ - grid_min_;

//   grid_size_ = max_ele_size / 2;
//   grid_size_ = std::max(grid_size_, bbox_size.maxCoeff() / 128);

//   hashing_ = new CellObj[27 * n_ele_];
//   cell_offs_ = new uint32_t[27 * n_ele_ + 1];
// }

// void TetSpatialHashing::Hashing() {
//   n_hash_ = 0;
//   for (uint32_t e = 0; e < n_ele_; ++e) {
//     Vec3 ele_bbox_min = 1e16 * Vec3::Ones();
//     Vec3 ele_bbox_max = -1e16 * Vec3::Ones();
//     for (uint32_t t = 0; t < 4; ++t) {
//       uint32_t i = indices_[e * 4 + t];
//       ele_bbox_min = ele_bbox_min.cwiseMin(verts_[i]);
//       ele_bbox_max = ele_bbox_max.cwiseMax(verts_[i]);
//     }
//     Vec3i low_cell = ((ele_bbox_min - grid_min_) / grid_size_).cast<int>();
//     Vec3i up_cell = ((ele_bbox_max - grid_min_) / grid_size_).cast<int>();
//     for (uint32_t idx = low_cell(0); idx <= up_cell(0); ++idx) {
//       for (uint32_t idy = low_cell(1); idy <= up_cell(1); ++idy) {
//         for (uint32_t idz = low_cell(2); idz <= up_cell(2); ++idz) {
//           uint32_t hash = idx | (idy << 8) | (idz << 16);
//           hashing_[n_hash_++] = std::make_pair(hash, e);
//         }
//       }
//     }
//   }
//   std::sort(hashing_, hashing_ + n_hash_);

//   n_cell_ = 0;
//   uint32_t cur_hash = 0xffffffff;
//   for (uint32_t i = 0; i < n_hash_; ++i) {
//     if (hashing_[i].first != cur_hash) {
//       cur_hash = hashing_[i].first;
//       cell_offs_[n_cell_] = i;
//       ++n_cell_;
//     }
//   }
//   cell_offs_[n_cell_] = n_hash_;
// }

// void TetSpatialHashing::GetSelfCollisionList() {
//   n_colli_ = 0;
//   if (colli_pairs_) delete[] colli_pairs_;
//   uint32_t n_pair = 0;
//   uint32_t n_overlap = 0;
//   for (uint32_t c = 0; c < n_cell_; ++c) {
//     uint32_t n_in_cell = cell_offs_[c + 1] - cell_offs_[c];
//     n_pair += n_in_cell * (n_in_cell - 1) / 2;
//     n_overlap = std::max(n_overlap, n_in_cell);
//   }
//   colli_pairs_ = new VertTet[8 * n_pair];

//   for (uint32_t c = 0; c < n_cell_; ++c) {
//     for (uint32_t i1 = cell_offs_[c]; i1 < cell_offs_[c + 1]; ++i1) {
//       for (uint32_t i2 = i1 + 1; i2 < cell_offs_[c + 1]; ++i2) {
//         uint32_t e1 = hashing_[i1].second;
//         uint32_t e2 = hashing_[i2].second;
//         bool is_adj = false;
//         uint32_t* tet1 = &indices_[4 * e1];
//         uint32_t* tet2 = &indices_[4 * e2];
//         for (uint32_t j1 = 0; j1 < 4; ++j1)
//           for (uint32_t j2 = 0; j2 < 4; ++j2)
//             if (tet1[j1] == tet2[j2]) is_adj = true;
//         if (is_adj) continue;

//         for (uint32_t j = 0; j < 4; ++j) {
//           if (is_surf_[tet1[j]])
//             if (Tet::VertexInside(verts_[tet1[j]], verts_[tet2[0]],
//                                   verts_[tet2[1]], verts_[tet2[2]],
//                                   verts_[tet2[3]])) {
//               colli_pairs_[n_colli_++] = VertTet{tet1[j], e2};
//               std::cout << tet1[j] << " " << tet2[0] << " " << tet2[1] << " "
//               << tet2[2] << " " << tet2[3] << std::endl;
//             }
//           if (is_surf_[tet2[j]])
//             if (Tet::VertexInside(verts_[tet2[j]], verts_[tet1[0]],
//                                   verts_[tet1[1]], verts_[tet1[2]],
//                                   verts_[tet1[3]])) {
//               colli_pairs_[n_colli_++] = VertTet{tet2[j], e1};
//               std::cout << tet2[j] << " " << tet1[0] << " " << tet1[1] << " "
//               << tet1[2] << " " << tet1[3] << std::endl;
//             }
//         }
//       }
//     }
//   }
// }

// void TetSpatialHashing::Destroy() {
//   if (hashing_) delete[] hashing_;
//   if (cell_offs_) delete[] cell_offs_;
//   if (colli_pairs_) delete[] colli_pairs_;
// }
};  // namespace Rain