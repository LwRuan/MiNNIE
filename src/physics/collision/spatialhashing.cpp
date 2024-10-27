#include "spatialhashing.h"

#include "geometry/CTCD.h"
#include "geometry/triangle.h"
namespace Rain {
void SpatialHashing::Init(Vec3* verts_old, Vec3* verts_new, uint32_t* indices,
                          uint32_t n_vert, uint32_t n_ele) {
  n_vert_ = n_vert;
  n_ele_ = n_ele;
  verts_old_ = verts_old;
  verts_new_ = verts_new;
  indices_ = indices;

  grid_min_ = 1e16 * Vec3::Ones();
  grid_max_ = -1e16 * Vec3::Ones();

  // std::cout << n_ele_ << std::endl;
  // decide grid size, bigger than an element
  // also insure grid number in one dim < 256
  real max_ele_size = 0;
  for (uint32_t e = 0; e < n_ele_; ++e) {
    Vec3 ele_bbox_min = 1e16 * Vec3::Ones();
    Vec3 ele_bbox_max = -1e16 * Vec3::Ones();
    for (uint32_t t = 0; t < 3; ++t) {
      uint32_t i = indices_[e * 3 + t];
      ele_bbox_min = ele_bbox_min.cwiseMin(verts_old_[i]);
      ele_bbox_min = ele_bbox_min.cwiseMin(verts_new_[i]);
      ele_bbox_max = ele_bbox_max.cwiseMax(verts_old_[i]);
      ele_bbox_max = ele_bbox_max.cwiseMax(verts_new_[i]);
    }
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

  grid_size_ = 1.1 * max_ele_size;
  grid_size_ = std::max(grid_size_, bbox_size.maxCoeff() / 128);
  
  std::cout << grid_size_ << ", " << bbox_size.transpose()  << std::endl;

  hashing_ = new CellObj[8 * n_ele_];
  cell_offs_ = new uint32_t[8 * n_ele_ + 1];
}

void SpatialHashing::Hashing() {
  n_hash_ = 0;
  for (uint32_t e = 0; e < n_ele_; ++e) {
    Vec3 ele_bbox_min = 1e16 * Vec3::Ones();
    Vec3 ele_bbox_max = -1e16 * Vec3::Ones();
    for (uint32_t t = 0; t < 3; ++t) {
      uint32_t i = indices_[e * 3 + t];
      ele_bbox_min = ele_bbox_min.cwiseMin(verts_old_[i]);
      ele_bbox_min = ele_bbox_min.cwiseMin(verts_new_[i]);
      ele_bbox_max = ele_bbox_max.cwiseMax(verts_old_[i]);
      ele_bbox_max = ele_bbox_max.cwiseMax(verts_new_[i]);
    }
    Vec3i low_cell = ((ele_bbox_min - grid_min_) / grid_size_).cast<int>();
    Vec3i up_cell = ((ele_bbox_max - grid_min_) / grid_size_).cast<int>();
    for (uint32_t idx = low_cell(0); idx <= up_cell(0); ++idx) {
      for (uint32_t idy = low_cell(1); idy <= up_cell(1); ++idy) {
        for (uint32_t idz = low_cell(2); idz <= up_cell(2); ++idz) {
          uint32_t hash = idx | (idy << 8) | (idz << 16);
          hashing_[n_hash_++] = std::make_pair(hash, e);
        }
      }
    }
  }
  std::cout << n_hash_ << std::endl;
  std::sort(hashing_, hashing_ + n_hash_);

  n_cell_ = 0;
  uint32_t cur_hash = 0xffffffff;
  for (uint32_t i = 0; i < n_hash_; ++i) {
    if (hashing_[i].first != cur_hash) {
      cur_hash = hashing_[i].first;
      cell_offs_[n_cell_] = i;
      ++n_cell_;
    }
  }
  cell_offs_[n_cell_] = n_hash_;
  std::cout << n_cell_ << std::endl;
}

void SpatialHashing::GetSelfCollisionList() {
  n_colli_ = 0;
  if (colli_pairs_) delete[] colli_pairs_;
  uint32_t n_pair = 0;
  uint32_t n_overlap = 0;
  for (uint32_t c = 0; c < n_cell_; ++c) {
    uint32_t n_in_cell = cell_offs_[c + 1] - cell_offs_[c];
    n_pair += n_in_cell * (n_in_cell - 1) / 2;
    n_overlap = std::max(n_overlap, n_in_cell);
  }
  std::cout << n_overlap << std::endl;
  colli_pairs_ = new ColliPair[n_pair * 15];
  const double eta = 1e-6;
  for (uint32_t c = 0; c < n_cell_; ++c) {
    for (uint32_t i1 = cell_offs_[c]; i1 < cell_offs_[c + 1]; ++i1) {
      for (uint32_t i2 = i1 + 1; i2 < cell_offs_[c + 1]; ++i2) {
        uint32_t e1 = hashing_[i1].second;
        uint32_t e2 = hashing_[i2].second;
        uint32_t* tri1 = &indices_[3 * e1];
        uint32_t* tri2 = &indices_[3 * e2];
        bool is_adj = false;
        for (uint32_t j1 = 0; j1 < 3; ++j1)
          for (uint32_t j2 = 0; j2 < 3; ++j2)
            if (tri1[j1] == tri2[j2]) is_adj = true;
        if (is_adj) continue;
        // 6 vertex-face and 9 edge-edge
        // vertex face
        for (uint32_t j1 = 0; j1 < 3; ++j1) {
          double t;
          bool is_colli = Evouga::CTCD::vertexFaceCTCD(
              verts_old_[tri1[j1]].cast<double>(),
              verts_old_[tri2[0]].cast<double>(),
              verts_old_[tri2[1]].cast<double>(),
              verts_old_[tri2[2]].cast<double>(),
              verts_new_[tri1[j1]].cast<double>(),
              verts_new_[tri2[0]].cast<double>(),
              verts_new_[tri2[1]].cast<double>(),
              verts_new_[tri2[2]].cast<double>(), eta, t);
          if (is_colli) {
            // std::cout << "v" << std::flush;
            // compute weight
            Vec3 p = verts_old_[tri1[j1]] * t + verts_new_[tri1[j1]] * (1 - t);
            Vec3 v1 = verts_old_[tri2[0]] * t + verts_new_[tri2[0]] * (1 - t);
            Vec3 v2 = verts_old_[tri2[1]] * t + verts_new_[tri2[1]] * (1 - t);
            Vec3 v3 = verts_old_[tri2[2]] * t + verts_new_[tri2[2]] * (1 - t);
            real w1, w2;
            Triangle::BaryCentricWeight(p, v1, v2, v3, w1, w2);
            colli_pairs_[n_colli_++] =
                ColliPair{tri1[j1], tri2[0], tri2[1], tri2[2], w1, w2, 0};
          }

          is_colli = Evouga::CTCD::vertexFaceCTCD(
              verts_old_[tri2[j1]].cast<double>(),
              verts_old_[tri1[0]].cast<double>(),
              verts_old_[tri1[1]].cast<double>(),
              verts_old_[tri1[2]].cast<double>(),
              verts_new_[tri2[j1]].cast<double>(),
              verts_new_[tri1[0]].cast<double>(),
              verts_new_[tri1[1]].cast<double>(),
              verts_new_[tri1[2]].cast<double>(), eta, t);
          if (is_colli) {
            // std::cout << "v" << std::flush;
            // compute weight
            Vec3 p = verts_old_[tri2[j1]] * t + verts_new_[tri2[j1]] * (1 - t);
            Vec3 v1 = verts_old_[tri1[0]] * t + verts_new_[tri1[0]] * (1 - t);
            Vec3 v2 = verts_old_[tri1[1]] * t + verts_new_[tri1[1]] * (1 - t);
            Vec3 v3 = verts_old_[tri1[2]] * t + verts_new_[tri1[2]] * (1 - t);
            real w1, w2;
            Triangle::BaryCentricWeight(p, v1, v2, v3, w1, w2);
            colli_pairs_[n_colli_++] =
                ColliPair{tri2[j1], tri1[0], tri1[1], tri1[2], w1, w2, 0};
          }
        }
        // edge edge
        for (uint32_t j1 = 0; j1 < 3; ++j1) {
          for (uint32_t j2 = 0; j2 < 3; ++j2) {
            // edge 1
            uint32_t idx1 = 3 * e1 + j1;
            uint32_t idx2 = 3 * e1 + (j1 + 1) % 3;
            // edge 2
            uint32_t idx3 = 3 * e2 + j2;
            uint32_t idx4 = 3 * e2 + (j2 + 1) % 3;
            double t;
            bool is_colli = Evouga::CTCD::edgeEdgeCTCD(
                verts_old_[idx1].cast<double>(),
                verts_old_[idx2].cast<double>(),
                verts_old_[idx3].cast<double>(),
                verts_old_[idx4].cast<double>(),
                verts_new_[idx1].cast<double>(),
                verts_new_[idx2].cast<double>(),
                verts_new_[idx3].cast<double>(),
                verts_new_[idx4].cast<double>(), eta, t);
            if (is_colli) {
              // std::cout << "e" << std::flush;
              Vec3 v1 = t * verts_old_[idx1] + (1 - t) * verts_new_[idx1];
              Vec3 v2 = t * verts_old_[idx2] + (1 - t) * verts_new_[idx2];
              Vec3 v3 = t * verts_old_[idx3] + (1 - t) * verts_new_[idx3];
              Vec3 v4 = t * verts_old_[idx4] + (1 - t) * verts_new_[idx4];
              real w1, w2;
              Triangle::EdgeEdgeCrossWeight(v1, v2, v3, v4, w1, w2);
              colli_pairs_[n_colli_++] =
                  ColliPair{idx1, idx2, idx3, idx4, w1, w2, 1};
            }
          }
        }
      }
    }
  }
}

void SpatialHashing::Destroy() {
  if (hashing_) delete[] hashing_;
  if (cell_offs_) delete[] cell_offs_;
  if (colli_pairs_) delete[] colli_pairs_;
}
};  // namespace Rain