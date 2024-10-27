#include "elasobjmg19.h"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SVD>
#include <Eigen/SparseCholesky>
#include <fstream>
#include <queue>

namespace Rain {
void ElasobjMG19::Init(ElasobjMG19InitInfo* info) {
  spdlog::info("begin elastic object initialization");
  ElasticObject::Init(info->obj, info->density, info->type, info->young,
                      info->poisson);
  dt_ = info->dt;
  n_iter_ = info->n_iter;
  control_mag_ = info->control_mag;
  relaxation_ = info->relaxation;
  n_layer_ = info->n_layer;
  n_handle_ = new uint32_t[n_layer_ + 1];
  deletion_queue_.push_back([=]() { delete[] n_handle_; });
  n_handle_[n_layer_] = n_vert_;
  memcpy(n_handle_, info->n_handles, sizeof(uint32_t) * n_layer_);
  A_as_LDU_ = new bool[n_layer_ + 1];
  deletion_queue_.push_back([=]() { delete[] A_as_LDU_; });
  memcpy(A_as_LDU_, info->A_as_LDU, sizeof(bool) * (n_layer_ + 1));
  A_as_dense_ = new bool[n_layer_ + 1];
  deletion_queue_.push_back([=]() { delete[] A_as_dense_; });
  memcpy(A_as_dense_, info->A_as_dense, sizeof(bool) * (n_layer_ + 1));
  if (!A_as_LDU_[n_layer_]) {
    spdlog::error("system matrix must be stored in LDU format");
    exit(1);
  }
  operations_.clear();
  int cur_layer = n_layer_;
  for (int i = 0; i < info->n_op; ++i) {
    operations_.push_back(info->operations[i]);
    if (info->operations[i].type == MGOpType::GS) {
      if (!A_as_dense_[cur_layer] && !A_as_LDU_[cur_layer]) {
        spdlog::error("Gauss Seidel operation only for dense or LDU matrix");
        exit(1);
      }
    }
    if (info->operations[i].type == MGOpType::DS) --cur_layer;
    if (info->operations[i].type == MGOpType::US) ++cur_layer;
    if (cur_layer < 0 || cur_layer > n_layer_) {
      spdlog::error("invalid operation sequence");
      exit(1);
    }
  }

  dim_ = new uint32_t[n_layer_ + 1];
  deletion_queue_.push_back([=]() { delete[] dim_; });
  dim_[n_layer_] = 3 * n_vert_;
  for (int l = 0; l < n_layer_; ++l) dim_[l] = 12 * n_handle_[l];
  std::string dims_str;
  for (int l = 0; l <= n_layer_; ++l) {
    dims_str += std::to_string(dim_[l]) + " ";
  }
  spdlog::info("multigrid matrix dims: {}", dims_str);
  BuildV2T();
  spdlog::info("select handles and reorder");
  BuildGraph();
  SelectHandles();
  ReOrder();
  spdlog::info("compute matrix");
  ComputeUMatrices(false);
  ComputeAMatrices(false);
  BuildUpdateAuxiliary();
  BuildCollisionAuxiliary();
  spdlog::info("fix diagonal");
  FixDiag();
  spdlog::info("end elastic object initialization");
}

void ElasobjMG19::Reset() { ElasticObject::Reset(); }

void ElasobjMG19::Destroy() {
  ElasticObject::Destroy();
  for (auto it = deletion_queue_.rbegin(); it != deletion_queue_.rend(); it++)
    (*it)();
  deletion_queue_.clear();
}

void ElasobjMG19::BuildV2T() {
  using VT = std::pair<uint32_t, uint32_t>;
  VT* vts = new VT[n_tet_ * 4];
  v2t_ids_ = new uint32_t[n_tet_ * 4];
  deletion_queue_.push_back([=]() { delete[] v2t_ids_; });
  v2t_off_ = new uint32_t[n_vert_ + 1];
  deletion_queue_.push_back([=]() { delete[] v2t_off_; });
  for (auto t = 0; t < n_tet_ * 4; ++t) {
    vts[t] = std::make_pair(tets_[t], t);
  }
  std::sort(vts, vts + n_tet_ * 4);
  std::optional<uint32_t> v{};
  for (auto i = 0; i < n_tet_ * 4; ++i) {
    if (vts[i].first != v) {
      v = vts[i].first;
      v2t_off_[v.value()] = i;
    }
    v2t_ids_[i] = vts[i].second;
  }
  v2t_off_[n_vert_] = n_tet_ * 4;
  delete[] vts;
}

void ElasobjMG19::BuildGraph() {
  using Edge = std::pair<uint32_t, uint32_t>;
  Edge* tedges = new Edge[n_tet_ * 12];
  uint32_t tmp1 = 0, tmp2 = 0;
  for (auto t = 0; t < n_tet_; ++t)
    for (auto i = 0; i < 4; ++i)
      for (auto j = 0; j < 4; ++j) {
        if (i != j) {
          tedges[tmp1++] = std::make_pair(tets_[4 * t + i], tets_[4 * t + j]);
        }
      }
  std::sort(tedges, tedges + n_tet_ * 12);

  v2e_off_ = new uint32_t[n_vert_ + 1];
  deletion_queue_.push_back([=]() { delete[] v2e_off_; });
  edge_to_ = new uint32_t[n_tet_ * 12];
  deletion_queue_.push_back([=]() { delete[] edge_to_; });
  edge_len_ = new real[n_tet_ * 12];
  deletion_queue_.push_back([=]() { delete[] edge_len_; });
  tmp1 = 0, tmp2 = 0;
  std::optional<uint32_t> pre_from{};
  std::optional<uint32_t> pre_to{};
  for (auto i = 0; i < n_tet_ * 12; ++i) {
    const Edge& tedge = tedges[i];
    if (tedge.first == pre_from && tedge.second == pre_to) continue;
    if (tedge.first != pre_from) v2e_off_[tmp1++] = tmp2;
    edge_to_[tmp2] = tedge.second;
    edge_len_[tmp2] = (verts_[tedge.first] - verts_[tedge.second]).norm();
    ++tmp2;
    pre_from = tedge.first, pre_to = tedge.second;
  }
  v2e_off_[n_vert_] = tmp2;
  delete[] tedges;
  // {  // Test Code
  //   std::cout << "graph built, vertex to edge list:" << std::endl;
  //   for (auto i = 0; i <= n_vert_; ++i) {
  //     std::cout << v2e_off_[i] << " ";
  //   }
  //   std::cout << std::endl;
  // }
}

void ElasobjMG19::SelectHandles() {
  if (!n_layer_) return;
  handle_ids_.clear();
  handle_ = new uint32_t*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteArray2D(handle_, n_layer_ + 1); });
  // vertex to index in handle_ids_
  uint32_t* v2h = new uint32_t[n_vert_ + 1];
  LenFrom* shortest = new LenFrom[n_vert_];
  LenFrom* pre_shortest = new LenFrom[n_vert_];
  for (auto i = 0; i < n_vert_; ++i)
    shortest[i] = std::make_pair(std::numeric_limits<real>::infinity(), 0);
  uint32_t init_handle = 0;
  ComputeShortestPath(init_handle, shortest);
  v2h[init_handle] = handle_ids_.size();
  handle_ids_.push_back(init_handle);
  handle_[0] = nullptr;
  for (int l = 0; l < n_layer_; ++l) {
    while (handle_ids_.size() < n_handle_[l]) {
      uint32_t next_handle = std::distance(
          shortest, std::max_element(shortest, shortest + n_vert_));
      v2h[next_handle] = handle_ids_.size();
      handle_ids_.push_back(next_handle);
      ComputeShortestPath(next_handle, shortest);
    }
    if (l) {  // except the bottom layer
      handle_[l] = new uint32_t[n_handle_[l]];
      for (auto i = 0; i < n_handle_[l]; ++i) {
        handle_[l][i] = v2h[pre_shortest[handle_ids_[i]].second];
      }
    }
    memcpy(pre_shortest, shortest, sizeof(LenFrom) * n_vert_);
  }
  handle_[n_layer_] = new uint32_t[n_vert_];
  for (auto i = 0; i < n_vert_; ++i)
    handle_[n_layer_][i] = v2h[shortest[i].second];
  // {  // Test Code
  //   std::cout << "handles selected" << std::endl;
  //   for (int l = n_layer_; l > 0; --l) {
  //     std::cout << "layer " << l << ": ";
  //     for (auto i = 0; i < n_handle_[l]; ++i) {
  //       std::cout << handle_[l][i] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }
  delete[] v2h;
  delete[] shortest;
  delete[] pre_shortest;
}

void ElasobjMG19::ReOrder() {
  if (!reordered_) return;
  using Edge = std::pair<uint32_t, uint32_t>;
  h2m_ = new uint32_t*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteArray2D(h2m_, n_layer_ + 1); });
  m2h_ = new uint32_t*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteArray2D(m2h_, n_layer_ + 1); });
  c2h_off_ = new uint32_t*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteArray2D(c2h_off_, n_layer_ + 1); });
  n_color_ = new uint32_t[n_layer_ + 1];
  deletion_queue_.push_back([=]() { delete[] n_color_; });

  uint32_t* color = new uint32_t[n_vert_];
  Edge* tmp = new Edge[v2e_off_[n_vert_] * 2];
  uint32_t* tmp_v2e_off = new uint32_t[n_vert_ + 1];
  uint32_t* tmp_edge_to = new uint32_t[v2e_off_[n_vert_]];
  for (int l = n_layer_; l >= 0; --l) {
    if (l == n_layer_) {
      memcpy(tmp_v2e_off, v2e_off_, sizeof(uint32_t) * (n_vert_ + 1));
      memcpy(tmp_edge_to, edge_to_, sizeof(uint32_t) * v2e_off_[n_vert_]);
    } else {  // build adjacent graph for coarse level
      uint32_t* tt_v2e_off = new uint32_t[n_handle_[l] + 1];
      uint32_t* tt_edge_to = new uint32_t[tmp_v2e_off[n_handle_[l + 1]] * 2];
      uint32_t tt_n_edge = 0;
      for (auto i = 0; i < n_handle_[l + 1]; ++i)
        for (auto j = tmp_v2e_off[i]; j < tmp_v2e_off[i + 1]; ++j)
          if (handle_[l + 1][i] != handle_[l + 1][tmp_edge_to[j]]) {
            tmp[tt_n_edge++] = std::make_pair(handle_[l + 1][i],
                                              handle_[l + 1][tmp_edge_to[j]]);
            tmp[tt_n_edge++] = std::make_pair(handle_[l + 1][tmp_edge_to[j]],
                                              handle_[l + 1][i]);
          }
      std::sort(tmp, tmp + tt_n_edge);
      std::optional<uint32_t> pre_from{};
      std::optional<uint32_t> pre_to{};
      uint32_t t1 = 0, t2 = 0;
      for (auto i = 0; i < tt_n_edge; ++i) {
        if (tmp[i].first == pre_from && tmp[i].second == pre_to) continue;
        if (tmp[i].first != pre_from) tt_v2e_off[t1++] = t2;
        tt_edge_to[t2++] = tmp[i].second;
        pre_from = tmp[i].first;
        pre_to = tmp[i].second;
      }
      tt_v2e_off[t1] = t2;
      if (t1 == 0) tt_v2e_off[1] = 0;  // 1 handle, no edge
      delete[] tmp_v2e_off;
      delete[] tmp_edge_to;
      tmp_v2e_off = tt_v2e_off;
      tmp_edge_to = tt_edge_to;
    }
    h2m_[l] = new uint32_t[n_handle_[l]];
    m2h_[l] = new uint32_t[n_handle_[l]];
    c2h_off_[l] = new uint32_t[n_handle_[l] + 1];
    Coloring(tmp_v2e_off, tmp_edge_to, color, n_handle_[l], 5);
    for (auto i = 0; i < n_handle_[l]; ++i)
      tmp[i] = std::make_pair(color[i], i);
    std::sort(tmp, tmp + n_handle_[l]);
    std::optional<uint32_t> pre{};
    n_color_[l] = 0;
    for (auto i = 0; i < n_handle_[l]; ++i) {
      if (tmp[i].first != pre) {
        c2h_off_[l][n_color_[l]++] = i;
        pre = tmp[i].first;
      }
      h2m_[l][tmp[i].second] = i;
      m2h_[l][i] = tmp[i].second;
    }
    c2h_off_[l][n_color_[l]] = n_handle_[l];
  }
  delete[] tmp;
  delete[] color;
  delete[] tmp_v2e_off;
  delete[] tmp_edge_to;
}

void ElasobjMG19::ComputeUMatrices(bool tofile) {
  if (!n_layer_) return;
  Uf_.resize(dim_[n_layer_], dim_[n_layer_ - 1]);
  Uf_.setZero();
  U_ = new ESpMat[n_layer_ - 1];
  deletion_queue_.push_back([=]() { delete[] U_; });
  int l = n_layer_ - 1;
  std::vector<ETriplet> coeff;
  for (auto r = 0; r < n_vert_; ++r) {
    uint32_t v = r;
    uint32_t c = handle_[n_layer_][r];
    if (reordered_) {
      v = m2h_[n_layer_][r];
      c = h2m_[l][handle_[n_layer_][v]];
    }
    for (auto d = 0; d < 3; ++d) {
      for (auto t = 0; t < 4; ++t) {
        coeff.push_back(ETriplet(3 * r + d, 12 * c + 4 * d + t,
                                 (t == 3) ? 1 : verts_[v][t]));
      }
    }
  }
  Uf_.setFromTriplets(coeff.begin(), coeff.end());
  Uf_.makeCompressed();
  coeff.clear();
  --l;
  for (; l >= 0; --l) {
    U_[l].resize(dim_[l + 1], dim_[l]);
    U_[l].setZero();
    for (auto r = 0; r < n_handle_[l + 1]; ++r) {
      uint32_t c = handle_[l + 1][r];
      if (reordered_) c = h2m_[l][handle_[l + 1][m2h_[l + 1][r]]];
      for (auto t = 0; t < 12; ++t) {
        coeff.push_back(ETriplet(12 * r + t, 12 * c + t, 1.0));
      }
    }
    U_[l].setFromTriplets(coeff.begin(), coeff.end());
    coeff.clear();
    U_[l].makeCompressed();
  }

  if (tofile) {
    std::ofstream file("Uf.txt");
    file << Uf_.rows() << " " << Uf_.cols() << " " << Uf_.nonZeros()
         << std::endl;
    for (uint32_t k = 0; k < Uf_.outerSize(); ++k) {
      for (ESpMat::InnerIterator it(Uf_, k); it; ++it) {
        file << it.row() << " " << it.col() << " " << it.value() << std::endl;
      }
    }
    file.close();
  }
}

void ElasobjMG19::ComputeAMatrices(bool tofile) {
  real inv_dt = 1 / dt_;
  std::map<RowCol, Mat3> block_coo;
  // energy gradient part
  for (auto t = 0; t < n_tet_; ++t) {
    uint32_t* tet = &tets_[t * 4];
    Eigen::Matrix<real, 4, 3> idm;
    idm.block<3, 3>(0, 0) = Dm_inv_[t];
    idm(3, 0) = -idm(0, 0) - idm(1, 0) - idm(2, 0);
    idm(3, 1) = -idm(0, 1) - idm(1, 1) - idm(2, 1);
    idm(3, 2) = -idm(0, 2) - idm(1, 2) - idm(2, 2);
    Mat3 Ds;
    Ds.col(0) = verts_[tet[0]] - verts_[tet[3]];
    Ds.col(1) = verts_[tet[1]] - verts_[tet[3]];
    Ds.col(2) = verts_[tet[2]] - verts_[tet[3]];
    Mat3 F = Ds * Dm_inv_[t];
    Mat9 dPdF;
    elastic_model_->GetdPdF(F, dPdF);
    for (auto i = 0; i < 4; ++i) {
      for (auto j = 0; j < 4; ++j) {
        uint32_t r = tet[i];
        uint32_t c = tet[j];
        if (reordered_) {
          r = h2m_[n_layer_][tet[i]];
          c = h2m_[n_layer_][tet[j]];
        }
        RowCol key = std::make_pair(r, c);
        auto iter = block_coo.find(key);
        if (iter == block_coo.end())
          iter = block_coo.insert(std::make_pair(key, Mat3::Zero())).first;
        real dot = 0;
        Mat3 tH = Mat3::Zero();
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            tH += idm(i, a) * dPdF.block<3, 3>(3 * a, 3 * b) * idm(j, b);
          }
        }
        iter->second += tH * volumes_[t];
        // for (auto k = 0; k < 3; ++k) dot += idm(i, k) * idm(j, k);
        // dot *= volumes_[t] * elasticity_;
        // iter->second += dot * Mat3::Identity();
      }
    }
  }
  // inertia part
  for (auto i = 0; i < n_vert_; ++i) {
    uint32_t v = i;
    if (reordered_) v = m2h_[n_layer_][i];
    real diag = masses_[v] * inv_dt * inv_dt;
    if (obj_->is_fixed_[v]) diag += control_mag_;
    block_coo[std::make_pair(i, i)] += diag * Mat3::Identity();
  }

  Af_coo_.clear();
  for (const auto& iter : block_coo) {
    uint32_t r = iter.first.first;
    uint32_t c = iter.first.second;
    for (int x = 0; x < 3; ++x) {
      for (int y = 0; y < 3; ++y) {
        Af_coo_.push_back(ETriplet(3 * r + x, 3 * c + y, iter.second(x, y)));
      }
    }
  }

  Af_.resize(dim_[n_layer_], dim_[n_layer_]);
  Af_.setFromTriplets(Af_coo_.begin(), Af_coo_.end());
  Af_.makeCompressed();
  if (!n_layer_) return;

  A_ = new ESpMat[n_layer_];
  deletion_queue_.push_back([=]() { delete[] A_; });
  for (int l = n_layer_ - 1; l >= 0; --l) {
    A_[l].resize(dim_[l], dim_[l]);
    if (l == n_layer_ - 1) {
      A_[l] = Uf_.transpose() * Af_ * Uf_;
    } else {
      A_[l] = U_[l].transpose() * A_[l + 1] * U_[l];
    }
    A_[l].makeCompressed();
  }

  solver_.compute(Af_);

  if (tofile) {
    // output Af
    std::ofstream file("Af_cpu.txt");
    file << Af_.rows() << " " << Af_.cols() << " " << Af_.nonZeros()
         << std::endl;
    for (uint32_t k = 0; k < Af_.outerSize(); ++k) {
      for (ESpMat::InnerIterator it(Af_, k); it; ++it) {
        file << it.row() << " " << it.col() << " " << it.value() << std::endl;
      }
    }
    file.close();
  }
}

void ElasobjMG19::FixDiag() {
  if (!n_layer_) return;
  // n_layer_ - 1
  std::vector<Mat4> evd4x4;
  evd4x4.resize(n_handle_[n_layer_ - 1]);
  std::fill(evd4x4.begin(), evd4x4.end(), Mat4::Zero());
  std::vector<uint32_t> control_num;
  control_num.resize(n_handle_[n_layer_ - 1]);
  std::fill(control_num.begin(), control_num.end(), 0);
  for (uint32_t i = 0; i < n_vert_; ++i) {
    uint32_t idx = handle_[n_layer_][i];
    if (reordered_) idx = h2m_[n_layer_ - 1][idx];
    Vec4 v1 = Vec4::Unit(3);
    v1.segment<3>(0) = verts_[i];
    evd4x4[idx] += v1 * v1.transpose();
    ++control_num[idx];
  }
  std::vector<Mat4> diag_fix;
  diag_fix.resize(n_handle_[n_layer_ - 1]);
  std::fill(diag_fix.begin(), diag_fix.end(), Mat4::Zero());
  for (auto i = 0; i < n_handle_[n_layer_ - 1]; ++i) {
    Eigen::JacobiSVD<Mat4> svd(evd4x4[i],
                               Eigen::ComputeFullU | Eigen::ComputeFullV);
    for (auto j = 0; j < 4; ++j) {
      if (j >= control_num[i]) {
        spdlog::info("rank + 1");
        const Vec4& sv = svd.matrixV().col(j);
        real norm = sv.squaredNorm();
        diag_fix[i] += sv * sv.transpose() / norm * 1000;
      }
    }
  }
  std::vector<ETriplet> coeff;
  for (auto i = 0; i < n_handle_[n_layer_ - 1]; ++i) {
    for (auto c = 0; c < 3; ++c) {
      for (auto di = 0; di < 4; ++di) {
        for (auto dj = 0; dj < 4; ++dj) {
          coeff.push_back(ETriplet(12 * i + c * 4 + di, 12 * i + c * 4 + dj,
                                   diag_fix[i](di, dj)));
        }
      }
    }
  }
  UtAU_diag_fix_.resize(dim_[n_layer_ - 1], dim_[n_layer_ - 1]);
  UtAU_diag_fix_.setFromTriplets(coeff.begin(), coeff.end());
  UtAU_diag_fix_.makeCompressed();
}

void ElasobjMG19::BuildUpdateAuxiliary() {
  diag_add_ = new ESpMat[n_layer_ + 1];
  deletion_queue_.push_back([=]() { delete[] diag_add_; });
  for (int l = 0; l < n_layer_ + 1; ++l) {
    diag_add_[l].resize(dim_[l], dim_[l]);
  }
  Af_add_.resize(dim_[n_layer_], dim_[n_layer_]);
  Af_composed_.resize(dim_[n_layer_], dim_[n_layer_]);
  old_verts_ = new Vec3[n_vert_];
  deletion_queue_.push_back([=]() { delete[] old_verts_; });
  inertia_verts_ = new Vec3[n_vert_];
  deletion_queue_.push_back([=]() { delete[] inertia_verts_; });
  fixed_verts_ = new Vec3[n_vert_];
  memcpy(fixed_verts_, verts_, sizeof(Vec3) * n_vert_);
  deletion_queue_.push_back([=]() { delete[] fixed_verts_; });
  tet_grads_ = new real[n_tet_ * 12];
  deletion_queue_.push_back([=]() { delete[] tet_grads_; });
  rhs_ = new real*[n_layer_ + 1];
  for (int l = 0; l < n_layer_ + 1; ++l) {
    rhs_[l] = new real[dim_[l]];
  }
  deletion_queue_.push_back([=]() { DeleteArray2D(rhs_, n_layer_ + 1); });
  lhs_ = new real*[n_layer_ + 1];
  for (int l = 0; l < n_layer_ + 1; ++l) {
    lhs_[l] = new real[dim_[l]];
  }
  deletion_queue_.push_back([=]() { DeleteArray2D(lhs_, n_layer_ + 1); });
  tmp_ = new real*[n_layer_ + 1];
  for (int l = 0; l < n_layer_ + 1; ++l) {
    tmp_[l] = new real[dim_[l]];
  }
  deletion_queue_.push_back([=]() { DeleteArray2D(tmp_, n_layer_ + 1); });
  P_ = new real*[n_layer_ + 1];
  for (int l = 0; l < n_layer_ + 1; ++l) {
    P_[l] = new real[dim_[l]];
  }
  deletion_queue_.push_back([=]() { DeleteArray2D(P_, n_layer_ + 1); });
  R_ = new real*[n_layer_ + 1];
  for (int l = 0; l < n_layer_ + 1; ++l) {
    R_[l] = new real[dim_[l]];
  }
  deletion_queue_.push_back([=]() { DeleteArray2D(R_, n_layer_ + 1); });
}

void ElasobjMG19::Update(float dt, const Vec3& grav, real damping,
                         uint32_t n_substep, real substep_size,
                         uint32_t n_frame,
                         const std::vector<KinematicObject*>& kobjs) {
  // no reorder
  Af_add_.setZero();
  Af_add_coo_.clear();
  for (int l = 0; l < n_layer_ + 1; ++l) {
    diag_add_[l].setZero();
  }
  for (uint32_t v = 0; v < n_vert_; ++v) {
    uint32_t i = v;
    if (reordered_) i = h2m_[n_layer_][v];
    if (obj_->is_fixed_[v]) {
      Af_add_coo_.push_back(ETriplet(3 * i + 0, 3 * i + 0, control_mag_));
      Af_add_coo_.push_back(ETriplet(3 * i + 1, 3 * i + 1, control_mag_));
      Af_add_coo_.push_back(ETriplet(3 * i + 2, 3 * i + 2, control_mag_));
    } else if (v == obj_->selected_idx_) {
      Af_add_coo_.push_back(ETriplet(3 * i + 0, 3 * i + 0, control_mag_));
      Af_add_coo_.push_back(ETriplet(3 * i + 1, 3 * i + 1, control_mag_));
      Af_add_coo_.push_back(ETriplet(3 * i + 2, 3 * i + 2, control_mag_));
    }
  }

  for (int step = 0; step < n_substep; ++step) {
    real dt = substep_size;
    memcpy(old_verts_, verts_, sizeof(Vec3) * n_vert_);
    for (uint32_t i = 0; i < n_vert_; ++i) {
      velocities_[i] += grav * dt;
      velocities_[i] -= damping * velocities_[i];
      verts_[i] += velocities_[i] * dt;
    }
    memcpy(inertia_verts_, verts_, sizeof(Vec3) * n_vert_);

    for (int iter = 0; iter < n_iter_; ++iter) {
      if (elastic_model_->type_ == ElasticModelType::NeoHookean) {
        // TODO: Hessian
      }

      // rhs
      memset(rhs_[n_layer_], 0, sizeof(real) * dim_[n_layer_]);
      for (int t = 0; t < n_tet_; ++t) {
        const uint32_t& v1 = tets_[4 * t];
        const uint32_t& v2 = tets_[4 * t + 1];
        const uint32_t& v3 = tets_[4 * t + 2];
        const uint32_t& v4 = tets_[4 * t + 3];
        Mat3 Ds;
        Ds.col(0) = verts_[v1] - verts_[v4];
        Ds.col(1) = verts_[v2] - verts_[v4];
        Ds.col(2) = verts_[v3] - verts_[v4];
        Mat3 F = Ds * Dm_inv_[t];
        Mat3 P;
        elastic_model_->GetPiola(F, P);
        Eigen::Matrix<real, 4, 3> G;
        G.block<3, 3>(0, 0) = Dm_inv_[t];
        G.block<1, 3>(3, 0) = -Vec3::Ones().transpose() * Dm_inv_[t];
        Eigen::Matrix<real, 3, 4> T = -volumes_[t] * P * G.transpose();
        for (int i = 0; i < 12; ++i) tet_grads_[12 * t + i] = T(i % 3, i / 3);
      }
      real dt_inv_sq = 1 / dt / dt;
      for (int v = 0; v < n_vert_; ++v) {
        uint32_t i = v;
        if (reordered_) i = h2m_[n_layer_][v];
        for (int32_t idx = v2t_off_[v]; idx < v2t_off_[v + 1]; ++idx) {
          rhs_[n_layer_][3 * i + 0] += tet_grads_[v2t_ids_[idx] * 3 + 0];
          rhs_[n_layer_][3 * i + 1] += tet_grads_[v2t_ids_[idx] * 3 + 1];
          rhs_[n_layer_][3 * i + 2] += tet_grads_[v2t_ids_[idx] * 3 + 2];
        }
        if (obj_->is_fixed_[v]) {
          rhs_[n_layer_][3 * i + 0] +=
              control_mag_ * (fixed_verts_[v](0) - verts_[v](0));
          rhs_[n_layer_][3 * i + 1] +=
              control_mag_ * (fixed_verts_[v](1) - verts_[v](1));
          rhs_[n_layer_][3 * i + 2] +=
              control_mag_ * (fixed_verts_[v](2) - verts_[v](2));
        } else if (v == obj_->selected_idx_) {
          rhs_[n_layer_][3 * i + 0] +=
              control_mag_ * (obj_->control_pos_(0) - verts_[v](0));
          rhs_[n_layer_][3 * i + 1] +=
              control_mag_ * (obj_->control_pos_(1) - verts_[v](1));
          rhs_[n_layer_][3 * i + 2] +=
              control_mag_ * (obj_->control_pos_(2) - verts_[v](2));
        }
        rhs_[n_layer_][3 * i + 0] +=
            masses_[v] * dt_inv_sq * (inertia_verts_[v](0) - verts_[v](0));
        rhs_[n_layer_][3 * i + 1] +=
            masses_[v] * dt_inv_sq * (inertia_verts_[v](1) - verts_[v](1));
        rhs_[n_layer_][3 * i + 2] +=
            masses_[v] * dt_inv_sq * (inertia_verts_[v](2) - verts_[v](2));
      }

      Eigen::Map<VecX> rhs(rhs_[n_layer_], 3 * n_vert_);
      Eigen::Map<VecX> lhs(lhs_[n_layer_], 3 * n_vert_);

      if (output_data_) {
        std::ofstream file("rhsf.txt");
        file << rhs << std::endl;
        file.close();
      }

      // collision
      real k_penalty = 2e7;
      real k_d = 1e2;
      if (self_collision_) {
        spatial_hashing_.Hashing();
        spatial_hashing_.GetSelfCollisionList();
        colli_vert_pairs_.resize(spatial_hashing_.n_colli_ * 2);
        colli_v2e_.resize(n_vert_);
        colli_edge_next_.resize(spatial_hashing_.n_colli_ * 2);
        std::fill(colli_edge_next_.begin(), colli_edge_next_.end(), -1);
        std::fill(colli_v2e_.begin(), colli_v2e_.end(), -1);
      }
      n_colli_pair_ = spatial_hashing_.n_colli_;

      if (self_collision_) {
        for (uint32_t c = 0; c < spatial_hashing_.n_colli_; ++c) {
          uint32_t v = spatial_hashing_.colli_pairs_[c].first;
          uint32_t t = spatial_hashing_.colli_pairs_[c].second;
          real min_dist = 1e16;
          uint32_t p = 0;
          for (uint32_t i = 0; i < closest_surf_vert_[5 * t]; ++i) {
            uint32_t vi = closest_surf_vert_[5 * t + i + 1];
            real dist = (verts_[v] - verts_[vi]).norm();
            if (dist < min_dist) {
              min_dist = dist;
              p = vi;
            }
          }
          Vec3 n = obj_->normals_[p];
          Mat3 nnt = n * n.transpose();
          colli_vert_pairs_[2 * c] = {v, p};
          colli_vert_pairs_[2 * c + 1] = {p, v};
          for (uint32_t x = 0; x < 3; ++x) {
            for (uint32_t y = 0; y < 3; ++y) {
              Af_add_coo_.push_back(
                  ETriplet(3 * p + x, 3 * p + y, k_penalty * nnt(x, y)));
              Af_add_coo_.push_back(
                  ETriplet(3 * v + x, 3 * v + y, k_penalty * nnt(x, y)));
              Af_add_coo_.push_back(
                  ETriplet(3 * p + x, 3 * v + y, -k_penalty * nnt(x, y)));
              Af_add_coo_.push_back(
                  ETriplet(3 * v + x, 3 * p + y, -k_penalty * nnt(x, y)));
            }
          }
          // for (uint32_t x = 0; x < 3; ++x) {
          //   Af_add_coo_.push_back(ETriplet(3 * p + x, 3 * p + x, k_penalty));
          //   Af_add_coo_.push_back(ETriplet(3 * v + x, 3 * v + x, k_penalty));
          //   Af_add_coo_.push_back(ETriplet(3 * p + x, 3 * v + x,
          //   -k_penalty)); Af_add_coo_.push_back(ETriplet(3 * v + x, 3 * p +
          //   x, -k_penalty));
          // }
          Vec3 fv = -k_penalty * n.dot(verts_[v] - verts_[p]) * n;
          Vec3 fp = k_penalty * n.dot(verts_[v] - verts_[p]) * n;
          fv += k_d * n.dot(velocities_[p] - velocities_[v]) * n;
          fp -= k_d * n.dot(velocities_[p] - velocities_[v]) * n;
          // Vec3 fv = -k_penalty * (verts_[v] - verts_[p]);
          // Vec3 fp = k_penalty * (verts_[v] - verts_[p]);
          for (uint32_t x = 0; x < 3; ++x) {
            rhs_[n_layer_][3 * v + x] += fv(x);
            rhs_[n_layer_][3 * p + x] += fp(x);
          }
        }
      }

      for (uint32_t i = 0; i < colli_vert_pairs_.size(); ++i) {
        uint32_t s = colli_vert_pairs_[i].first;
        uint32_t t = colli_vert_pairs_[i].second;
        colli_edge_next_[i] = colli_v2e_[s];
        colli_v2e_[s] = i;
      }

      // SelectHandlesWithCollision();
      // ComputeUMatrices(false);
      for (uint32_t v = 0; v < n_vert_; ++v) {
        for (int i = 0; i < kobjs.size(); ++i) {
          if (kobjs[i]->GetSignedDistance(verts_[v]) < 0) {
            Vec3 normal = kobjs[i]->GetClosestNormal(verts_[v]);
            Vec3 target = kobjs[i]->GetClosestSurfacePosition(verts_[v]);
            Vec3 f = -k_penalty * normal.dot(verts_[v] - target) * normal;
            //!TODO: damping
            // Vec3 dv = kobjs[i]->GetVelocity(target, n_frame) - velocities_[v];
            // dv = dv - dv.dot(normal) * normal;
            // f = f + dv.normalized() * std::min(1e-1 * dv.norm(), 0.5 * f.norm());
            Mat3 nnt = normal * normal.transpose();
            for (uint32_t x = 0; x < 3; ++x) {
              for (uint32_t y = 0; y < 3; ++y) {
                Af_add_coo_.push_back(
                    ETriplet(3 * v + x, 3 * v + y, k_penalty * nnt(x, y)));
              }
            }
            for (uint32_t x = 0; x < 3; ++x) {
              rhs_[n_layer_][3 * v + x] += f(x);
            }
          }
        }
      }
      // A = A + diag_add_[n_layer_];
      Af_add_.setFromTriplets(Af_add_coo_.begin(), Af_add_coo_.end());
      Af_composed_ = Af_ + Af_add_;
      Af_composed_.makeCompressed();
      Af_dense_ = MatX(Af_composed_);
      if (output_data_) {
        spdlog::info("output system matrix");
        std::ofstream file("A.txt");
        file << Af_composed_.rows() << " " << Af_composed_.cols() << " "
             << Af_composed_.nonZeros() << std::endl;
        for (uint32_t k = 0; k < Af_composed_.outerSize(); ++k) {
          for (ESpMat::InnerIterator it(Af_composed_, k); it; ++it) {
            file << it.row() << " " << it.col() << " " << it.value()
                 << std::endl;
          }
        }
        file.close();
        file.open("Af.txt");
        file << Af_.rows() << " " << Af_.cols() << " " << Af_.nonZeros()
             << std::endl;
        for (uint32_t k = 0; k < Af_.outerSize(); ++k) {
          for (ESpMat::InnerIterator it(Af_, k); it; ++it) {
            file << it.row() << " " << it.col() << " " << it.value()
                 << std::endl;
          }
        }
        file.close();
        spdlog::info("output rhs vector");
        file.open("rhs.txt");
        file << rhs << std::endl;
        file.close();
        spdlog::info("output U matrix");
        file.open("U.txt");
        file << Uf_.rows() << " " << Uf_.cols() << " " << Uf_.nonZeros()
             << std::endl;
        for (uint32_t k = 0; k < Uf_.outerSize(); ++k) {
          for (ESpMat::InnerIterator it(Uf_, k); it; ++it) {
            file << it.row() << " " << it.col() << " " << it.value()
                 << std::endl;
          }
        }
        file.close();
      }
      // if (frame_cnt == 0) {
      //   std::ofstream file("b_cpu.txt");
      //   for (uint32_t i = 0; i < dim_[n_layer_]; ++i) {
      //     file << rhs_[n_layer_][i] << std::endl;
      //   }
      //   file.close();
      // }

      if (n_layer_ > 0)
        A_[n_layer_ - 1] = Uf_.transpose() * Af_composed_ * Uf_;
      for (int l = n_layer_ - 2; l >= 0; --l)
        A_[l] = U_[l].transpose() * A_[l + 1] * U_[l];


      memset(lhs_[n_layer_], 0, sizeof(real) * dim_[n_layer_]);
      memcpy(R_[n_layer_], rhs_[n_layer_], sizeof(real) * dim_[n_layer_]);
      int32_t cur_layer = n_layer_;
      real tol = 1e-6;
      // Call Eigen solve
      // // Eigen::ConjugateGradient<ESpMat> solver;
      // Eigen::SimplicialLDLT<ESpMat> solver;
      // solver.compute(Af_composed_);
      // VecX X_star = solver.solve(rhs);
      PerformGSIteration(cur_layer, 3, tol);
      // real e1 = (X_star - lhs).norm();
      DownSample(cur_layer);
      DirectSolve(cur_layer);
      UpSample(cur_layer);
      // real e2 = (X_star - lhs).norm();
      PerformGSIteration(cur_layer, 3, tol);
      // std::cout << e1 << " " << e2 << " " << e2 / e1 << std::endl;
      for (uint32_t v = 0; v < n_vert_; ++v) {
        uint32_t i = v;
        if (reordered_) i = h2m_[n_layer_][v];
        verts_[v] += lhs.segment<3>(3 * i);
      }
    }
    for (uint32_t v = 0; v < n_vert_; ++v) {
      velocities_[v] = (verts_[v] - old_verts_[v]) / dt;
    }
  }
  obj_->UpdateNormal();
  std::string filename = "output/" + std::to_string(n_frame) + ".obj";
  obj_->WriteObj(filename);
  std::cout << verts_[0].transpose() << std::endl;
  output_data_ = false;
}

void ElasobjMG19::PerformGSIteration(int32_t& l, const int32_t max_iter,
                                     const real tol) {
  Eigen::Map<VecX> tmp(tmp_[l], dim_[l]);
  Eigen::Map<VecX> R(R_[l], dim_[l]);
  Eigen::Map<VecX> X(lhs_[l], dim_[l]);
  if (l == n_layer_) {
    tmp = R;
    for (int r = Af_dense_.rows() - 1; r >= 0; --r) {
      if (r < Af_dense_.rows() - 1) {
        tmp.segment(r, 1) = tmp.segment(r, 1) - Af_dense_.block(r, r + 1, 1, Af_dense_.rows() - 1 - r) * tmp.segment(r + 1, Af_dense_.rows() - 1 - r);
      }
      tmp(r) /= Af_dense_(r, r);
    }
    X += tmp;
    R -= Af_composed_ * tmp;
    tmp = R;
    for (int r = 0; r < Af_dense_.rows(); ++r) {
      if (r > 0) {
        tmp.segment(r, 1) = tmp.segment(r, 1) - Af_dense_.block(r, 0, 1, r) * tmp.segment(0, r);
      }
      tmp(r) /= Af_dense_(r, r);
    }
    X += tmp;
    R -= Af_composed_ * tmp;
  } else {
    spdlog::error("Gauss Seidel for low level not implemented");
    exit(1);
  }
}

void ElasobjMG19::DownSample(int32_t& l) {
  Eigen::Map<VecX> R(R_[l], dim_[l]);
  Eigen::Map<VecX> R1(R_[l - 1], dim_[l - 1]);
  Eigen::Map<VecX> B(rhs_[l - 1], dim_[l - 1]);
  if (l == n_layer_) {
    B = Uf_.transpose() * R;
  } else {
    B = U_[l - 1].transpose() * R;
  }
  R1 = B;
  memset(lhs_[l - 1], 0, sizeof(real) * dim_[l - 1]);
  --l;
}

void ElasobjMG19::UpSample(int32_t& l) {
  Eigen::Map<VecX> X(lhs_[l], dim_[l]);
  Eigen::Map<VecX> tmp(tmp_[l + 1], dim_[l + 1]);
  Eigen::Map<VecX> X1(lhs_[l + 1], dim_[l + 1]);
  Eigen::Map<VecX> R(R_[l + 1], dim_[l + 1]);
  if (l == n_layer_ - 1) {
    tmp = Uf_ * X;
    R -= Af_composed_ * tmp;
  } else {
    tmp = U_[l] * X;
    R -= A_[l + 1] * tmp; 
  }
  X1 += tmp;
  ++l;
}

void ElasobjMG19::DirectSolve(int32_t& l) {
  Eigen::Map<VecX> X(lhs_[l], dim_[l]);
  Eigen::Map<VecX> B(rhs_[l], dim_[l]);
  Eigen::Map<VecX> R(R_[l], dim_[l]);
  if (l == n_layer_) {
    Eigen::SimplicialLDLT<ESpMat> solver;
    solver.compute(Af_composed_);
    X = solver.solve(B);
    R = B - Af_composed_ * X;
  } else {
    Eigen::SimplicialLDLT<ESpMat> solver;
    solver.compute(A_[l]);
    X = solver.solve(B);
    R = B - A_[l] * X;
  }
}

void ElasobjMG19::BuildCollisionAuxiliary() {
  closest_surf_vert_ = new uint32_t[5 * n_tet_];
  deletion_queue_.push_back([=]() { delete[] closest_surf_vert_; });
  is_vert_surf_ = new bool[n_vert_];
  memset(is_vert_surf_, 0, sizeof(bool) * n_vert_);
  deletion_queue_.push_back([=]() { delete[] is_vert_surf_; });

  int32_t n_remain = n_tet_;
  std::vector<bool> vis(n_tet_, false);
  std::set<uint32_t> surf_ids;
  for (uint32_t i = 0; i < obj_->n_surfidx_; ++i)
    surf_ids.insert(obj_->surface_indices_[i]);
  for (uint32_t v = 0; v < n_vert_; ++v) {
    if (surf_ids.find(v) != surf_ids.end()) is_vert_surf_[v] = true;
  }

  for (uint32_t t = 0; t < n_tet_; ++t) {
    uint32_t off = 0;
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t v = tets_[4 * t + i];
      if (is_vert_surf_[v]) {
        closest_surf_vert_[5 * t + off + 1] = v;
        ++off;
      }
    }
    closest_surf_vert_[5 * t] = off;
    if (off != 0) {
      vis[t] = true;
      --n_remain;
    }
  }

  while (n_remain > 0) {
    for (uint32_t t = 0; t < n_tet_; ++t) {
      if (vis[t]) continue;
      uint32_t nvis = 0;
      for (uint32_t i = 0; i < 4; ++i) {
        uint32_t v = tets_[4 * t + i];
        for (uint32_t j = v2t_off_[v]; j < v2t_off_[v + 1]; ++j) {
          uint32_t t2 = v2t_ids_[j] / 4;
          if (vis[t2]) ++nvis;
        }
      }
      if (nvis == 0) continue;
      std::set<uint32_t> candidates;
      for (uint32_t i = 0; i < 4; ++i) {
        uint32_t v = tets_[4 * t + i];
        for (uint32_t j = v2t_off_[v]; j < v2t_off_[v + 1]; ++j) {
          uint32_t t2 = v2t_ids_[j] / 4;
          if (vis[t2]) {
            for (uint32_t k = 0; k < closest_surf_vert_[5 * t2]; ++k)
              candidates.insert(closest_surf_vert_[5 * t2 + k + 1]);
          }
        }
      }
      Vec3 center = Vec3::Zero();
      for (uint32_t i = 0; i < 4; ++i) center += verts_[tets_[4 * t + i]] / 4;

      real mindist = 1e16;
      uint32_t minidx = 0;
      for (uint32_t c : candidates) {
        real dist = (verts_[c] - center).norm();
        if (dist < mindist) {
          mindist = dist;
          minidx = c;
        }
      }
      closest_surf_vert_[5 * t] = 1;
      closest_surf_vert_[5 * t + 1] = minidx;
      vis[t] = true;
      --n_remain;
    }
  }

  spatial_hashing_.Init(verts_, tets_, obj_->surface_indices_, is_vert_surf_,
                        n_vert_, n_tet_, obj_->n_face_, 300000);
  deletion_queue_.push_back([=]() { spatial_hashing_.Destroy(); });
}

void ElasobjMG19::SelectHandlesWithCollision() {
  if (!n_layer_) return;
  handle_ids_.clear();
  if (handle_) {
    DeleteArray2D(handle_, n_layer_ + 1);
  } else if (handle_ == nullptr) {
    deletion_queue_.push_back([=]() { DeleteArray2D(handle_, n_layer_ + 1); });
  }
  handle_ = new uint32_t*[n_layer_ + 1];
  // vertex to index in handle_ids_
  uint32_t* v2h = new uint32_t[n_vert_ + 1];
  LenFrom* shortest = new LenFrom[n_vert_];
  LenFrom* pre_shortest = new LenFrom[n_vert_];
  for (auto i = 0; i < n_vert_; ++i)
    shortest[i] = std::make_pair(std::numeric_limits<real>::infinity(), 0);
  uint32_t init_handle = 0;
  ComputeShortestPathWithCollision(init_handle, shortest);
  v2h[init_handle] = handle_ids_.size();
  handle_ids_.push_back(init_handle);
  handle_[0] = nullptr;
  for (int l = 0; l < n_layer_; ++l) {
    while (handle_ids_.size() < n_handle_[l]) {
      uint32_t next_handle = std::distance(
          shortest, std::max_element(shortest, shortest + n_vert_));
      v2h[next_handle] = handle_ids_.size();
      handle_ids_.push_back(next_handle);
      ComputeShortestPathWithCollision(next_handle, shortest);
    }
    if (l) {  // except the bottom layer
      handle_[l] = new uint32_t[n_handle_[l]];
      for (auto i = 0; i < n_handle_[l]; ++i) {
        handle_[l][i] = v2h[pre_shortest[handle_ids_[i]].second];
      }
    }
    memcpy(pre_shortest, shortest, sizeof(LenFrom) * n_vert_);
  }
  handle_[n_layer_] = new uint32_t[n_vert_];
  for (auto i = 0; i < n_vert_; ++i)
    handle_[n_layer_][i] = v2h[shortest[i].second];
  // {  // Test Code
  //   std::cout << "handles selected" << std::endl;
  //   for (int l = n_layer_; l > 0; --l) {
  //     std::cout << "layer " << l << ": ";
  //     for (auto i = 0; i < n_handle_[l]; ++i) {
  //       std::cout << handle_[l][i] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }
  delete[] v2h;
  delete[] shortest;
  delete[] pre_shortest;
}

void ElasobjMG19::ComputeShortestPath(uint32_t source, LenFrom* shortest) {
  struct LenFromTo {
    LenFrom lf;
    uint32_t to;
    bool operator<(const LenFromTo& r) const {
      return r.lf.first < lf.first;  // for priority_queue
    }
  };
  std::priority_queue<LenFromTo> pq;
  pq.push(LenFromTo{LenFrom{0, source}, source});
  while (!pq.empty()) {
    LenFromTo top = pq.top();
    pq.pop();
    real dist = top.lf.first;
    uint32_t from = top.lf.second;
    uint32_t to = top.to;
    if (dist > shortest[to].first) continue;
    shortest[to] = top.lf;
    for (auto i = v2e_off_[to]; i < v2e_off_[to + 1]; ++i) {
      uint32_t next = edge_to_[i];
      real nd = dist + edge_len_[i];
      if (nd < shortest[next].first)
        pq.push(LenFromTo{LenFrom{nd, from}, next});
    }
  }
}

void ElasobjMG19::ComputeShortestPathWithCollision(uint32_t source,
                                                   LenFrom* shortest) {
  struct LenFromTo {
    LenFrom lf;
    uint32_t to;
    bool operator<(const LenFromTo& r) const {
      return r.lf.first < lf.first;  // for priority_queue
    }
  };
  std::priority_queue<LenFromTo> pq;
  pq.push(LenFromTo{LenFrom{0, source}, source});
  while (!pq.empty()) {
    LenFromTo top = pq.top();
    pq.pop();
    real dist = top.lf.first;
    uint32_t from = top.lf.second;
    uint32_t to = top.to;
    if (dist > shortest[to].first) continue;
    shortest[to] = top.lf;
    for (auto i = v2e_off_[to]; i < v2e_off_[to + 1]; ++i) {
      uint32_t next = edge_to_[i];
      real nd = dist + edge_len_[i];
      if (nd < shortest[next].first)
        pq.push(LenFromTo{LenFrom{nd, from}, next});
    }
    int tmp = colli_v2e_[to];
    while (tmp != -1) {
      uint32_t next = colli_vert_pairs_[tmp].second;
      real nd = dist + (verts_[next] - verts_[to]).norm();
      if (nd < shortest[next].first)
        pq.push(LenFromTo{LenFrom{nd, from}, next});
      tmp = colli_edge_next_[tmp];
    }
  }
}

uint32_t ElasobjMG19::Coloring(uint32_t* v2e_off, uint32_t* edge_to,
                               uint32_t* color, uint32_t n_vert,
                               uint32_t n_color) {
  std::srand(0);
  bool* fixed = new bool[n_vert];
  memset(fixed, 0, sizeof(bool) * n_vert);
  uint32_t unknown = n_vert;
  uint32_t pre_unknown = n_vert;
  uint32_t tnc = n_color;  // tmp n_color
  bool need_expand;
  uint32_t stuck = 0;
  while (unknown) {
    need_expand = false;
    bool* used = new bool[tnc];
    for (auto i = 0; i < n_vert; ++i) {
      if (fixed[i]) continue;
      memset(used, 0, sizeof(bool) * tnc);
      for (auto j = v2e_off[i]; j < v2e_off[i + 1]; ++j) {
        uint32_t t = edge_to[j];
        if (fixed[t]) used[color[t]] = true;
      }
      uint32_t available = 0;
      for (auto j = 0; j < tnc; ++j)
        if (!used[j]) ++available;
      if (available == 0) {
        need_expand = true;
        color[i] = tnc;
      } else {
        uint32_t idx = rand() % available;
        for (auto j = 0; j < tnc; ++j) {
          if (used[j]) continue;
          if (!idx) {
            color[i] = j;
            break;
          }
          --idx;
        }
      }
    }
    for (auto i = 0; i < n_vert; ++i) {
      if (fixed[i]) continue;
      bool valid = true;
      for (auto j = v2e_off[i]; j < v2e_off[i + 1]; ++j)
        valid &= (color[i] != color[edge_to[j]]);
      if (valid) {
        fixed[i] = true;
        --unknown;
      }
    }
    delete[] used;
    if (need_expand)
      ++tnc;
    else {
      if (pre_unknown == unknown)
        ++stuck;
      else
        stuck = 0;
      if (stuck == 4) {
        ++tnc;
        stuck = 0;
      }
    }
    pre_unknown = unknown;
  }
  delete[] fixed;
  return tnc;
}

template <typename _type>
void ElasobjMG19::DeleteArray2D(_type** ptr, uint32_t size) {
  if (ptr == nullptr) return;
  for (auto i = 0; i < size; ++i)
    if (ptr[i] != nullptr) delete[] ptr[i];
  delete[] ptr;
}

void ElasobjMG19::ShowUI() {
  ElasticObject::ShowUI();
  ImGui::Text("#collision pairs: %d", n_colli_pair_);
  if (ImGui::Button("output data")) {
    output_data_ = true;
  }
}
};  // namespace Rain