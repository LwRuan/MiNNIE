#include "cudaelasobjmixedmg19.h"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <fstream>
#include <iostream>
#include <queue>

namespace Rain {
void CudaElasobjMixedMG19::Init(CudaElasobjMixedMG19InitInfo* info) {
  spdlog::info("begin elastic object initialization");
  CudaElasticObject::Init(info->device, info->obj, info->density, info->type,
                          info->young, info->poisson, info->skeleton,
                          info->n_joint);
  rest_vol_ = 0.;
  for (int i = 0; i < n_tet_; ++i) rest_vol_ += volumes_[i];
  mu_ = elastic_model_->mu_;
  lambda_inv_ = elastic_model_->lambda_inv_;
  InitCudaContext();
  dt_ = info->dt;
  quasi_static_ = info->quasi_static;
  minres_iter_ = info->minres_iter;
  p_smooth_ = info->p_smooth;
  p_scale_ = info->p_scale;
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
  for (int l = 1; l <= n_layer_; ++l) {
    if (A_as_dense_[l]) {
      spdlog::error("only the coarsest layer can be stored in dense format");
      exit(1);
    }
  }
  operations_.clear();
  int cur_layer = n_layer_;
  for (int i = 0; i < info->n_op; ++i) {
    operations_.push_back(info->operations[i]);
    if (info->operations[i].type == MGOpType::DS) --cur_layer;
    if (info->operations[i].type == MGOpType::US) ++cur_layer;
    if (cur_layer < 0 || cur_layer > n_layer_) {
      spdlog::error("invalid operation sequence");
      exit(1);
    }
  }

  CheckCuda(cudaMalloc(&dfixed_verts_, sizeof(Vec3) * n_vert_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dfixed_verts_)); });
  CheckCuda(cudaMemcpy(dfixed_verts_, verts_, sizeof(Vec3) * n_vert_,
                       cudaMemcpyHostToDevice));

  CheckCuda(cudaMalloc(&drest_verts_, sizeof(Vec3) * n_vert_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(drest_verts_)); });
  CheckCuda(cudaMemcpy(drest_verts_, verts_, sizeof(Vec3) * n_vert_,
                       cudaMemcpyHostToDevice));

  pressure_ = new real[n_vert_];
  deletion_queue_.push_back([=]() { delete[] pressure_; });
  memset(pressure_, 0, sizeof(real) * n_vert_);

  dim_ = new uint32_t[n_layer_ + 1];
  deletion_queue_.push_back([=]() { delete[] dim_; });
  dim_[n_layer_] = 4 * n_vert_;
  for (int l = 0; l < n_layer_; ++l) dim_[l] = 16 * n_handle_[l];
  std::string dims_str;
  for (int l = 0; l <= n_layer_; ++l) {
    dims_str += std::to_string(dim_[l]) + " ";
  }
  spdlog::info("multigrid matrix dims: {}", dims_str);

  marker_ = new int32_t[n_vert_];
  memset(marker_, 0, sizeof(int32_t) * n_vert_);
  deletion_queue_.push_back([=]() { delete[] marker_; });

  CheckCuda(cudaMalloc(&dtforce_, sizeof(Vec3)));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dtforce_)); });

  BuildV2T();
  BuildGraph();
  ComputeNormalSign();
  SelectHandles(false);
  Colorize();
  is_vert_surf_ = new bool[n_vert_];
  memset(is_vert_surf_, 0, sizeof(bool) * n_vert_);
  deletion_queue_.push_back([=]() { delete[] is_vert_surf_; });
  std::set<uint32_t> surf_ids;
  for (uint32_t i = 0; i < obj_->n_surfidx_; ++i)
    surf_ids.insert(obj_->surface_indices_[i]);
  for (const uint32_t v : surf_ids) is_vert_surf_[v] = true;
  ComputeUMatrices(false);
  ComputeAMatrices(false);
  BuildUpdateAuxiliary();
  FixDiag();
  ToDevice();
  if (self_collision_) {
    BuildCollisionAuxiliary();
    hashing_ = new CudaTetHashing();
    hashing_->Init(dvertices_, dtet_sign_, dis_vert_surf_, dindices_, n_vert_,
                   n_tet_, n_vert_, MAXN_COLLI);
    deletion_queue_.push_back([=]() {
      hashing_->Destroy();
      delete[] hashing_;
    });
  }
  CheckCuda(cudaDeviceSynchronize());
  spdlog::info("end elastic object initialization");
  timer_.Reset();
}

void CudaElasobjMixedMG19::Reset() {
  CudaElasticObject::Reset();

  // { // HACK: for the cube-random example
  //   for (int i = 0; i < n_vert_; ++i) {
  //     if (obj_->is_fixed_[i]) continue;
  //     obj_->vertices_[i] = 2. * Vec3::Random();
  //   }
  // }

  CheckCuda(cudaMemcpy(dfixed_, obj_->is_fixed_, sizeof(bool) * n_vert_,
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(dfixed_verts_, obj_->vertices_,
                       sizeof(Vec3) * obj_->n_vert_, cudaMemcpyHostToDevice));
  obj_->UpdateNormal();
  CheckCuda(cudaMemcpy(dvertices_, obj_->vertices_,
                       sizeof(Vec3) * obj_->n_vert_, cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(dnormals_, obj_->normals_, sizeof(Vec3) * obj_->n_vert_,
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(dpressure_, pressure_, sizeof(real) * obj_->n_vert_,
                       cudaMemcpyHostToDevice));
}

void CudaElasobjMixedMG19::Destroy() {
  CudaElasticObject::Destroy();
  for (auto it = deletion_queue_.rbegin(); it != deletion_queue_.rend(); it++)
    (*it)();
  deletion_queue_.clear();
}

void CudaElasobjMixedMG19::InitCudaContext() {
  cublasCreate(&cublas_handle_);
  cusparseCreate(&cusparse_handle_);
  cusolverDnCreate(&cusolverDn_handle_);
  cusolverDnCreateParams(&cusolverDn_params_);
  cusolverSpCreate(&cusolverSp_handle_);

  cusparseCreateMatDescr(&descr_);
  cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatDiagType(descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);

  cusparseCreateMatDescr(&descrU_);
  cusparseSetMatType(descrU_, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  cusparseSetMatIndexBase(descrU_, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatDiagType(descrU_, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatFillMode(descrU_, CUSPARSE_FILL_MODE_UPPER);

  cusparseCreateMatDescr(&descrL_);
  cusparseSetMatType(descrL_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrL_, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatDiagType(descrL_, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatFillMode(descrL_, CUSPARSE_FILL_MODE_LOWER);
}

void CudaElasobjMixedMG19::BuildV2T() {
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

void CudaElasobjMixedMG19::BuildGraph() {
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

  v2e_off_ = new uint32_t*[n_layer_ + 1];
  v2e_off_[n_layer_] = new uint32_t[n_vert_ + 1];
  deletion_queue_.push_back([=]() { DeleteArray2D(v2e_off_, n_layer_ + 1); });
  edge_to_ = new uint32_t*[n_layer_ + 1];
  edge_to_[n_layer_] = new uint32_t[n_tet_ * 12];
  deletion_queue_.push_back([=]() { DeleteArray2D(edge_to_, n_layer_ + 1); });
  edge_len_ = new real[n_tet_ * 12];
  deletion_queue_.push_back([=]() { delete[] edge_len_; });
  tmp1 = 0, tmp2 = 0;
  std::optional<uint32_t> pre_from{};
  std::optional<uint32_t> pre_to{};
  for (auto i = 0; i < n_tet_ * 12; ++i) {
    const Edge& tedge = tedges[i];
    if (tedge.first == pre_from && tedge.second == pre_to) continue;
    if (tedge.first != pre_from) v2e_off_[n_layer_][tmp1++] = tmp2;
    edge_to_[n_layer_][tmp2] = tedge.second;
    edge_len_[tmp2] = (verts_[tedge.first] - verts_[tedge.second]).norm();
    ++tmp2;
    pre_from = tedge.first, pre_to = tedge.second;
  }
  v2e_off_[n_layer_][n_vert_] = tmp2;
  delete[] tedges;
}

void CudaElasobjMixedMG19::TetColorize() {
  tet_color_ = new uint32_t[n_tet_];
  deletion_queue_.push_back([=]() { delete[] tet_color_; });
  std::vector<uint32_t> edge_off;
  std::vector<uint32_t> edge_to;
  edge_off.push_back(0);
  for (int t = 0; t < n_tet_; ++t) {
    for (int i = 0; i < 4; ++i) {
      int v = tets_[4 * t + i];
      for (int idx = v2t_off_[v]; idx < v2t_off_[v + 1]; ++idx) {
        int t2 = v2t_ids_[idx] / 4;
        if (t2 == t) continue;
        edge_to.push_back(t2);
      }
    }
    edge_off.push_back(edge_to.size());
  }
  n_tet_color_ = 0;
  std::ifstream file("coloring3.txt");
  for (int i = 0; i < n_tet_; ++i) {
    file >> tet_color_[i];
    n_tet_color_ = std::max(tet_color_[i], n_tet_color_);
  }
  file.close();
  n_tet_color_ += 1;
  std::cout << n_tet_color_ << std::endl;

  CheckCuda(cudaMalloc(&dtet_color_, sizeof(uint32_t) * n_tet_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dtet_color_)); });
  CheckCuda(cudaMemcpy(dtet_color_, tet_color_, sizeof(uint32_t) * n_tet_,
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMalloc(&dupdated_, sizeof(int32_t) * n_vert_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dupdated_)); });
}

void CudaElasobjMixedMG19::Colorize() {
  using Edge = std::pair<uint32_t, uint32_t>;
  Edge* tmp = new Edge[v2e_off_[n_layer_][n_vert_] * 2];
  n_color_ = new uint32_t[n_layer_ + 1];
  deletion_queue_.push_back([=]() { delete[] n_color_; });
  color_ = new uint32_t*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteArray2D(color_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    n_color_[l] = 0;
    color_[l] = new uint32_t[n_handle_[l]];
  }
  uint32_t* tmp_v2e_off = new uint32_t[n_vert_ + 1];
  uint32_t* tmp_edge_to = new uint32_t[v2e_off_[n_layer_][n_vert_]];

  for (int l = n_layer_; l >= 0; --l) {
    if (l == n_layer_) {
      memcpy(tmp_v2e_off, v2e_off_[n_layer_], sizeof(uint32_t) * (n_vert_ + 1));
      memcpy(tmp_edge_to, edge_to_[n_layer_],
             sizeof(uint32_t) * v2e_off_[n_layer_][n_vert_]);
    } else {
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
        if (tmp[i].first != pre_from) {
          for (; t1 < tmp[i].first; ++t1) {
            tt_v2e_off[t1] = t2;
          }
          tt_v2e_off[t1++] = t2;
        }
        tt_edge_to[t2++] = tmp[i].second;
        pre_from = tmp[i].first;
        pre_to = tmp[i].second;
      }
      for (; t1 < n_handle_[l] + 1; ++t1) tt_v2e_off[t1] = t2;
      delete[] tmp_v2e_off;
      delete[] tmp_edge_to;
      tmp_v2e_off = tt_v2e_off;
      tmp_edge_to = tt_edge_to;
    }
    if (l < n_layer_) {
      v2e_off_[l] = new uint32_t[n_handle_[l] + 1];
      edge_to_[l] = new uint32_t[tmp_v2e_off[n_handle_[l]]];
      memcpy(v2e_off_[l], tmp_v2e_off, sizeof(uint32_t) * (n_handle_[l] + 1));
      memcpy(edge_to_[l], tmp_edge_to,
             sizeof(uint32_t) * tmp_v2e_off[n_handle_[l]]);
    }
    n_color_[l] =
        GraphColoring(tmp_v2e_off, tmp_edge_to, color_[l], n_handle_[l], 5);
  }
  delete[] tmp;
  delete[] tmp_v2e_off;
  delete[] tmp_edge_to;
}

void CudaElasobjMixedMG19::SelectHandles(bool tofile) {
  if (!n_layer_) return;
  handle_ids_.clear();
  handle_ = new uint32_t*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteArray2D(handle_, n_layer_ + 1); });
  // vertex to index in handle_ids_
  uint32_t* v2h = new uint32_t[n_vert_ + 1];
  LenFrom* shortest = new LenFrom[n_vert_];
  LenFrom* pre_shortest = new LenFrom[n_vert_];
  for (auto i = 0; i < n_vert_; ++i) shortest[i] = std::make_pair(1e16, 0);
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
  if (tofile) {
    std::ofstream file("handles.txt");
    for (uint32_t i = 0; i < n_handle_[n_layer_]; ++i) {
      file << handle_[n_layer_][i] << std::endl;
    }
    file.close();

    std::ofstream file1("vertices.txt");
    for (uint32_t i = 0; i < n_vert_; ++i) {
      file1 << verts_[i].transpose() << std::endl;
    }
    file1.close();
  }
  delete[] v2h;
  delete[] shortest;
  delete[] pre_shortest;
}

void CudaElasobjMixedMG19::ComputeUMatrices(bool tofile) {
  if (!n_layer_) return;
  Uf_ = new SpMat;
  deletion_queue_.push_back([=]() {
    Uf_->Destroy();
    delete Uf_;
  });
  U_ = new SpMat[n_layer_ - 1];
  deletion_queue_.push_back([=]() {
    for (int i = 0; i + 1 < n_layer_; ++i) U_[i].Destroy();
    delete[] U_;
  });
  // bottom layers
  int l = n_layer_ - 1;
  Uf_->Init(4 * n_vert_, 16 * n_handle_[l]);
  Uf_->nnz_ = 4 * dim_[n_layer_];
  Uf_->csr_row_ = new uint32_t[dim_[n_layer_] + 1];
  Uf_->deletion_queue_.push_back([=]() { delete[] Uf_->csr_row_; });
  Uf_->csr_col_ = new uint32_t[4 * dim_[n_layer_]];
  Uf_->deletion_queue_.push_back([=]() { delete[] Uf_->csr_col_; });
  Uf_->csr_val_ = new real[4 * dim_[n_layer_]];
  Uf_->deletion_queue_.push_back([=]() { delete[] Uf_->csr_val_; });
  uint32_t i1 = 0;
  for (auto r = 0; r < n_vert_; ++r) {
    uint32_t v = r;
    uint32_t c = handle_[n_layer_][r];
    for (auto d = 0; d < 4; ++d) {
      Uf_->csr_row_[4 * r + d] = i1;
      for (auto t = 0; t < 4; ++t) {
        Uf_->csr_col_[i1] = 16 * c + 4 * d + t;
        Uf_->csr_val_[i1] =
            (t == 3) ? 1 : verts_[v][t] - verts_[handle_ids_[c]][t];
        ++i1;
      }
    }
  }
  Uf_->csr_row_[dim_[n_layer_]] = i1;
  --l;
  for (; l >= 0; --l) {
    U_[l].Init(16 * n_handle_[l + 1], 16 * n_handle_[l]);
    U_[l].csr_row_ = new uint32_t[dim_[l + 1] + 1];
    U_[l].deletion_queue_.push_back([=]() { delete[] U_[l].csr_row_; });
    U_[l].csr_col_ = new uint32_t[dim_[l + 1]];
    U_[l].deletion_queue_.push_back([=]() { delete[] U_[l].csr_col_; });
    U_[l].csr_val_ = new real[dim_[l + 1]];
    U_[l].deletion_queue_.push_back([=]() { delete[] U_[l].csr_val_; });
    U_[l].nnz_ = dim_[l + 1];
    uint32_t i1 = 0;
    for (auto r = 0; r < n_handle_[l + 1]; ++r) {
      uint32_t c = handle_[l + 1][r];
      for (auto t = 0; t < 16; ++t) {
        U_[l].csr_row_[16 * r + t] = i1;
        U_[l].csr_col_[i1] = 16 * c + t;
        U_[l].csr_val_[i1] = 1.0;
        ++i1;
      }
    }
    U_[l].csr_row_[dim_[l + 1]] = i1;
  }

  if (tofile) {
    spdlog::info("begin writing Us");
    for (int l = 0; l + 1 < n_layer_; ++l) {
      std::ofstream f("U" + std::to_string(l) + ".txt");
      f << U_[l].rows_ << " " << U_[l].cols_ << " " << U_[l].nnz_ << std::endl;
      U_[l].OutputCsr(f);
      f.close();
    }
    std::ofstream f("Uf.txt");
    f << Uf_->rows_ << " " << Uf_->cols_ << " " << Uf_->nnz_ << std::endl;
    Uf_->OutputCsr(f);
    f.close();
    spdlog::info("end writing Us");
  }
}

void CudaElasobjMixedMG19::ComputeAMatrices(bool tofile) {
  real inv_dt = 1 / dt_;
  std::map<RowCol, Mat4> block_coo;

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
    elastic_model_->GetMixeddPdF(F, dPdF);

    real pk = p_smooth_ * volumes_[t] / 80. / mu_;
    real pm = volumes_[t] * lambda_inv_ / 20.;
    if ((0.5 - elastic_model_->poisson_) < 1e-6) {
      real lam_inv = 1e-5 / elastic_model_->young_;    
      pm = volumes_[t] * lam_inv / 20.;
    }
    for (auto i = 0; i < 4; ++i) {
      Vec3 dxi1 = verts_[tet[(i + 1) % 4]] - verts_[tet[(i + 3) % 4]];
      Vec3 dxi2 = verts_[tet[(i + 2) % 4]] - verts_[tet[(i + 3) % 4]];
      Vec3 ni = dxi1.cross(dxi2) / 6. * normal_sign_[4 * t + i];
      for (auto j = 0; j < 4; ++j) {
        Vec3 dxj1 = verts_[tet[(j + 1) % 4]] - verts_[tet[(j + 3) % 4]];
        Vec3 dxj2 = verts_[tet[(j + 2) % 4]] - verts_[tet[(j + 3) % 4]];
        Vec3 nj = dxj1.cross(dxj2) / 6. * normal_sign_[4 * t + j];
        uint32_t r = tet[i];
        uint32_t c = tet[j];
        RowCol key = std::make_pair(r, c);
        auto iter = block_coo.find(key);
        if (iter == block_coo.end())
          iter = block_coo.insert(std::make_pair(key, Mat4::Zero())).first;
        real dot = 0;
        Mat4 tH = Mat4::Zero();
        // distortion Hessian, A
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            tH.block<3, 3>(0, 0) +=
                idm(i, a) * dPdF.block<3, 3>(3 * a, 3 * b) * idm(j, b);
          }
        }
        tH.block<3, 3>(0, 0) *= volumes_[t];
        // volume gradient, G and Gt
        tH.block<1, 3>(3, 0) = p_scale_ * nj.transpose() / 4.;
        tH.block<3, 1>(0, 3) = p_scale_ * ni / 4.;
        if (elastic_model_->type_ == ElasticModelType::StVK) {
          Eigen::Matrix<real, 3, 4> tmpg = F * idm.transpose();
          tH.block<1, 3>(3, 0) =
              p_scale_ * tmpg.col(j).transpose() / 4. * volumes_[t];
          tH.block<3, 1>(0, 3) = p_scale_ * tmpg.col(i) / 4. * volumes_[t];
        } else if (elastic_model_->type_ == ElasticModelType::Corotation) {
          const Eigen::JacobiSVD<Mat3, Eigen::NoQRPreconditioner> svd(
              F, Eigen::ComputeFullU | Eigen::ComputeFullV);
          Mat3 U = svd.matrixU();
          Mat3 V = svd.matrixV();
          Vec3 S = svd.singularValues();
          {
            Mat3 L = Mat3::Identity();
            L(2, 2) = (U * V.transpose()).determinant();
            real detU = U.determinant();
            real detV = V.determinant();
            if (detU < 0. && detV > 0.) U = U * L;
            if (detU > 0. && detV < 0.) V = V * L;
            S(2) = S(2) * L(2, 2);
          }
          Eigen::Matrix<real, 3, 4> tmpg = U * V.transpose() * idm.transpose();
          tH.block<1, 3>(3, 0) = p_scale_ * tmpg.col(j).transpose() / 4. * volumes_[t];
          tH.block<3, 1>(0, 3) = p_scale_ * tmpg.col(i) / 4. * volumes_[t];
        }
        // volume Hessian, -C
        if (i == j)
          tH(3, 3) -= (3 * pk + 2 * pm) * p_scale_ * p_scale_;
        else
          tH(3, 3) -= (pm - pk) * p_scale_ * p_scale_;
        iter->second += tH;
      }
    }
  }

  // inertia part
  for (auto i = 0; i < n_vert_; ++i) {
    real diag = masses_[i] * inv_dt * inv_dt;
    if (obj_->is_fixed_[i]) diag += control_mag_;
    block_coo[std::make_pair(i, i)].block<3, 3>(0, 0) +=
        diag * Mat3::Identity();
  }

  // Af
  Af_ = new BSMat<4, 4>;
  deletion_queue_.push_back([=]() {
    Af_->Destroy();
    delete Af_;
  });
  Af_->Init(n_vert_, n_vert_, A_as_LDU_[n_layer_], A_as_dense_[n_layer_]);
  Af_->FromCooMapSym(block_coo);
  if (!n_layer_) return;

  // the rest As
  std::map<RowCol, Mat16> block_coo2;
  for (const auto& iter : block_coo) {
    uint32_t r = iter.first.first;
    uint32_t c = iter.first.second;
    const Mat4& a44 = iter.second;

    uint32_t i = r;
    uint32_t j = c;
    uint32_t rd = handle_[n_layer_][i];
    uint32_t cd = handle_[n_layer_][j];
    RowCol key2 = std::make_pair(rd, cd);
    auto iter2 = block_coo2.find(key2);
    if (iter2 == block_coo2.end())
      iter2 = block_coo2.insert(std::make_pair(key2, Mat16::Zero())).first;
    Vec4 v1 = Vec4::Unit(3);
    v1.segment<3>(0) = verts_[i] - verts_[handle_ids_[rd]];
    Vec4 v2 = Vec4::Unit(3);
    v2.segment<3>(0) = verts_[j] - verts_[handle_ids_[cd]];
    Mat4 v12 = v1 * v2.transpose();
    for (int vi = 0; vi < 4; ++vi) {
      for (int vj = 0; vj < 4; ++vj) {
        iter2->second.block<4, 4>(4 * vi, 4 * vj) += a44(vi, vj) * v12;
      }
    }
  }
  A_ = new BSMat<16, 16>[n_layer_];
  deletion_queue_.push_back([=]() {
    for (int i = 0; i < n_layer_; ++i) A_[i].Destroy();
    delete[] A_;
  });
  for (int l = n_layer_ - 1; l >= 0; --l) {
    A_[l].Init(n_handle_[l], n_handle_[l], A_as_LDU_[l], A_as_dense_[l]);
    A_[l].FromCooMapSym(block_coo2);
    if (l) {  // not the last layer
      std::map<RowCol, Mat16> tmp_coo;
      for (const auto& iter : block_coo2) {
        uint32_t r = iter.first.first;
        uint32_t c = iter.first.second;
        uint32_t rd = handle_[l][r];
        uint32_t cd = handle_[l][c];
        const Mat16& v = iter.second;
        RowCol key = std::make_pair(rd, cd);
        auto tmp_iter = tmp_coo.find(key);
        if (tmp_iter == tmp_coo.end())
          tmp_iter = tmp_coo.insert(std::make_pair(key, Mat16::Zero())).first;
        tmp_iter->second += v;
      }
      block_coo2.swap(tmp_coo);
    }

    // compute C and C_inv
    if (l == 0 && A_as_dense_[l]) {
      int n = n_handle_[0];
      C_ = new real[16 * n * n];
      memset(C_, 0, sizeof(real) * 16 * n * n);
      deletion_queue_.push_back([=]() { delete[] C_; });
      C_inv_ = new real[16 * n * n];
      memset(C_inv_, 0, sizeof(real) * 16 * n * n);
      deletion_queue_.push_back([=]() { delete[] C_inv_; });
      for (const auto& coo : block_coo2) {
        uint32_t r = coo.first.first;
        uint32_t c = coo.first.second;
        const Mat16& v = coo.second;
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            C_[(4 * r + i) + 4 * n * (4 * c + j)] -= v(12 + i, 12 + j);
          }
        }
      }
      Eigen::Map<MatX> C_map(C_, 4 * n, 4 * n);
      Eigen::Map<MatX> C_inv_map(C_inv_, 4 * n, 4 * n);
      C_inv_map = C_map.inverse();
    }
  }

  // save to files
  if (tofile) {
    spdlog::info("begin writing As");
    std::ofstream f("Af.txt");
    f << Af_->rows_ << " " << Af_->cols_ << " " << Af_->nnz_ << std::endl;
    Af_->OutputBlockCoo(f);
    f.close();
    for (int l = 0; l < n_layer_; ++l) {
      std::ofstream fl("A" + std::to_string(l) + ".txt");
      fl << A_[l].rows_ << " " << A_[l].cols_ << " " << A_[l].nnz_ << std::endl;
      A_[l].OutputBlockCoo(fl);
      fl.close();
    }
    spdlog::info("end writing As");
  }
}

void CudaElasobjMixedMG19::BuildUpdateAuxiliary() {
  assert(Af_ && Af_->bcoo_row_);
  std::map<RowCol, uint32_t> block_off;
  for (auto i = 0; i < Af_->bnnz_; ++i) {
    block_off[std::make_pair(Af_->bcoo_row_[i], Af_->bcoo_col_[i])] = i;
  }
  t2off_ = new uint32_t[n_tet_ * 16];
  deletion_queue_.push_back([=]() { delete[] t2off_; });
  d2off_ = new uint32_t[n_vert_];
  deletion_queue_.push_back([=]() { delete[] d2off_; });
  for (auto i = 0; i < n_vert_; ++i) {
    d2off_[i] = block_off[std::make_pair(i, i)];
  }
  uint32_t e = 0;
  for (auto t = 0; t < n_tet_; ++t)
    for (auto u = 0; u < 4; ++u)
      for (auto v = 0; v < 4; ++v) {
        uint32_t r = tets_[4 * t + u];
        uint32_t c = tets_[4 * t + v];
        t2off_[e++] = block_off[std::make_pair(r, c)];
      }
  if (n_layer_ > 0) {
    diag_XXt_ = new real[n_vert_ * 16];
    deletion_queue_.push_back([=]() { delete[] diag_XXt_; });
    for (auto r = 0; r < n_vert_; ++r) {
      Vec4 tmp = Vec4::Unit(3);
      tmp.segment<3>(0) = verts_[r] - verts_[handle_ids_[handle_[n_layer_][r]]];
      Mat4 XXt = tmp * tmp.transpose();
      for (int a = 0; a < 16; ++a) diag_XXt_[16 * r + a] = XXt(a / 4, a % 4);
    }

    XXt_ = new real[Af_->bnnz_ * 16];
    deletion_queue_.push_back([=]() { delete[] XXt_; });
    red_off_ = new uint32_t*[n_layer_];
    deletion_queue_.push_back([=]() { DeleteArray2D(red_off_, n_layer_); });
    red_half_off_ = new uint32_t*[n_layer_];
    deletion_queue_.push_back(
        [=]() { DeleteArray2D(red_half_off_, n_layer_); });
    red_n_half_ = new uint32_t[n_layer_];
    deletion_queue_.push_back([=]() { delete[] red_n_half_; });

    A_low_off_ = new uint32_t*[n_layer_];
    deletion_queue_.push_back([=]() { DeleteArray2D(A_low_off_, n_layer_); });
    A_diag_off_ = new uint32_t*[n_layer_];
    deletion_queue_.push_back([=]() { DeleteArray2D(A_diag_off_, n_layer_); });
    A_mirror_off_ = new uint32_t*[n_layer_];
    deletion_queue_.push_back(
        [=]() { DeleteArray2D(A_mirror_off_, n_layer_); });

    std::map<RowCol, uint32_t> block_offset_UtAU;
    BSMat<16, 16>* UtAU = &A_[n_layer_ - 1];
    for (uint32_t i = 0; i < UtAU->bnnz_; ++i) {
      block_offset_UtAU[std::make_pair(UtAU->bcoo_row_[i],
                                       UtAU->bcoo_col_[i])] = i;
      if (UtAU->as_dense_)
        block_offset_UtAU[std::make_pair(UtAU->bcoo_row_[i],
                                         UtAU->bcoo_col_[i])] =
            UtAU->bcoo_row_[i] * UtAU->bcols_ + UtAU->bcoo_col_[i];
    }
    red_off_[n_layer_ - 1] = new uint32_t[Af_->bnnz_];
    red_n_half_[n_layer_ - 1] = 0;
    red_half_off_[n_layer_ - 1] = new uint32_t[Af_->bnnz_];
    for (uint32_t b = 0; b < Af_->bnnz_; ++b) {
      uint32_t r = Af_->bcoo_row_[b];
      uint32_t c = Af_->bcoo_col_[b];
      uint32_t i = r;
      uint32_t j = c;
      uint32_t rd = handle_[n_layer_][i];
      uint32_t cd = handle_[n_layer_][j];
      // XXt
      Vec4 xi = Vec4::Unit(3);
      xi.segment<3>(0) = verts_[i] - verts_[handle_ids_[rd]];
      Vec4 xj = Vec4::Unit(3);
      xj.segment<3>(0) = verts_[j] - verts_[handle_ids_[cd]];
      Mat4 xxt = xi * xj.transpose();
      for (int a = 0; a < 16; ++a) XXt_[16 * b + a] = xxt(a / 4, a % 4);
      // reduction offsets
      red_off_[n_layer_ - 1][b] = block_offset_UtAU[std::make_pair(rd, cd)];
      if (rd >= cd) {
        uint32_t n = red_n_half_[n_layer_ - 1];
        red_half_off_[n_layer_ - 1][n] = b;
        ++red_n_half_[n_layer_ - 1];
      }
    }
    A_low_off_[n_layer_ - 1] = new uint32_t[UtAU->low_bnnz_];
    A_diag_off_[n_layer_ - 1] = new uint32_t[UtAU->diag_bnnz_];
    A_mirror_off_[n_layer_ - 1] = new uint32_t[UtAU->low_bnnz_];
    uint32_t i_l = 0, i_d = 0;
    for (uint32_t b = 0; b < UtAU->bnnz_; ++b) {
      uint32_t r = UtAU->bcoo_row_[b];
      uint32_t c = UtAU->bcoo_col_[b];
      if (r > c) {
        A_low_off_[n_layer_ - 1][i_l] = block_offset_UtAU[std::make_pair(r, c)];
        A_mirror_off_[n_layer_ - 1][i_l] =
            block_offset_UtAU[std::make_pair(c, r)];
        ++i_l;
      } else if (r == c) {
        A_diag_off_[n_layer_ - 1][i_d] =
            block_offset_UtAU[std::make_pair(r, c)];
        ++i_d;
      }
    }

    for (int l = n_layer_ - 2; l >= 0; --l) {
      red_off_[l] = new uint32_t[A_[l + 1].bnnz_];
      red_n_half_[l] = 0;
      red_half_off_[l] = new uint32_t[A_[l + 1].bnnz_];
      block_offset_UtAU.clear();
      for (uint32_t b = 0; b < A_[l].bnnz_; ++b) {
        block_offset_UtAU[std::make_pair(A_[l].bcoo_row_[b],
                                         A_[l].bcoo_col_[b])] = b;
        if (A_[l].as_dense_)
          block_offset_UtAU[std::make_pair(A_[l].bcoo_row_[b],
                                           A_[l].bcoo_col_[b])] =
              A_[l].bcoo_row_[b] * A_[l].bcols_ + A_[l].bcoo_col_[b];
      }
      for (uint32_t b = 0; b < A_[l + 1].bnnz_; ++b) {
        uint32_t r = A_[l + 1].bcoo_row_[b];
        uint32_t c = A_[l + 1].bcoo_col_[b];
        uint32_t i = r, j = c;
        uint32_t rd = handle_[l + 1][i];
        uint32_t cd = handle_[l + 1][j];
        red_off_[l][b] = block_offset_UtAU[std::make_pair(rd, cd)];
        if (rd >= cd) red_half_off_[l][red_n_half_[l]++] = b;
      }
      A_low_off_[l] = new uint32_t[A_[l].low_bnnz_];
      A_diag_off_[l] = new uint32_t[A_[l].diag_bnnz_];
      A_mirror_off_[l] = new uint32_t[A_[l].low_bnnz_];
      i_l = i_d = 0;
      for (uint32_t b = 0; b < A_[l].bnnz_; ++b) {
        uint32_t r = A_[l].bcoo_row_[b];
        uint32_t c = A_[l].bcoo_col_[b];
        if (r > c) {
          A_low_off_[l][i_l] = block_offset_UtAU[std::make_pair(r, c)];
          A_mirror_off_[l][i_l] = block_offset_UtAU[std::make_pair(c, r)];
          ++i_l;
        } else if (r == c) {
          A_diag_off_[l][i_d] = block_offset_UtAU[std::make_pair(r, c)];
          ++i_d;
        }
      }
    }
  }
}

void CudaElasobjMixedMG19::FixDiag() {
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
    Vec4 v1 = Vec4::Unit(3);
    v1.segment<3>(0) = verts_[i] - verts_[handle_ids_[idx]];
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
        diag_fix[i] += sv * sv.transpose() / norm * 1e3;
      }
    }
  }
  UtAU_diag_fix_ = new real[n_handle_[n_layer_ - 1] * 256];
  deletion_queue_.push_back([=]() { delete[] UtAU_diag_fix_; });
  memset(UtAU_diag_fix_, 0, sizeof(real) * n_handle_[n_layer_ - 1] * 256);
  for (auto i = 0; i < n_handle_[n_layer_ - 1]; ++i) {
    for (auto c = 0; c < 4; ++c) {
      for (auto di = 0; di < 4; ++di) {
        for (auto dj = 0; dj < 4; ++dj) {
          if (c < 3)
            UtAU_diag_fix_[256 * i + (c * 4 + di) * 16 + (c * 4 + dj)] =
                diag_fix[i](di, dj);
          else
            UtAU_diag_fix_[256 * i + (c * 4 + di) * 16 + (c * 4 + dj)] =
                -diag_fix[i](di, dj);
        }
      }
    }
  }
}

void CudaElasobjMixedMG19::BuildCollisionAuxiliary() {
  bool* tet_sign = new bool[n_tet_];
  for (uint32_t t = 0; t < n_tet_; ++t) {
    const uint32_t& v1 = tets_[4 * t];
    const uint32_t& v2 = tets_[4 * t + 1];
    const uint32_t& v3 = tets_[4 * t + 2];
    const uint32_t& v4 = tets_[4 * t + 3];
    Mat3 Dm;
    Dm.col(0) = verts_[v1] - verts_[v4];
    Dm.col(1) = verts_[v2] - verts_[v4];
    Dm.col(2) = verts_[v3] - verts_[v4];
    tet_sign[t] = Dm.determinant() > 0;
  }

  CheckCuda(cudaMalloc(&dtet_sign_, sizeof(bool) * n_tet_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dtet_sign_)); });
  CheckCuda(cudaMemcpy(dtet_sign_, tet_sign, sizeof(bool) * n_tet_,
                       cudaMemcpyHostToDevice));

  closest_surf_vert_ = new int32_t[4 * n_tet_];
  memset(closest_surf_vert_, 0xff, sizeof(int32_t) * 4 * n_tet_);
  deletion_queue_.push_back([=]() { delete[] closest_surf_vert_; });

  int32_t n_remain = n_tet_;
  std::vector<bool> vis(n_tet_, false);
  for (uint32_t t = 0; t < n_tet_; ++t) {
    uint32_t off = 0;
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t v = tets_[4 * t + i];
      if (is_vert_surf_[v]) {
        closest_surf_vert_[4 * t + off] = v;
        ++off;
      }
    }
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
            for (uint32_t k = 0; k < 4; ++k) {
              if (closest_surf_vert_[4 * t2 + k] < 0) break;
              candidates.insert(closest_surf_vert_[4 * t2 + k]);
            }
          }
        }
      }
      Vec3 center{0, 0, 0};
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
      closest_surf_vert_[4 * t] = minidx;
      vis[t] = true;
      --n_remain;
    }
  }

  CheckCuda(cudaMalloc(&dis_vert_surf_, sizeof(bool) * n_vert_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dis_vert_surf_)); });
  CheckCuda(cudaMemcpy(dis_vert_surf_, is_vert_surf_, sizeof(bool) * n_vert_,
                       cudaMemcpyHostToDevice));

  CheckCuda(cudaMalloc(&dclosest_surf_vert_, sizeof(int32_t) * 4 * n_tet_));
  deletion_queue_.push_back(
      [=]() { CheckCuda(cudaFree(dclosest_surf_vert_)); });
  CheckCuda(cudaMemcpy(dclosest_surf_vert_, closest_surf_vert_,
                       sizeof(int32_t) * 4 * n_tet_, cudaMemcpyHostToDevice));
  delete[] tet_sign;

  degree_ = new int32_t*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteArray2D(degree_, n_layer_ + 1); });
  min_degree_ = new int32_t[n_layer_ + 1];
  deletion_queue_.push_back([=]() { delete[] min_degree_; });
  max_degree_ = new int32_t[n_layer_ + 1];
  deletion_queue_.push_back([=]() { delete[] max_degree_; });
  for (int l = 0; l < n_layer_ + 1; ++l) {
    min_degree_[l] = 100000;
    max_degree_[l] = 0;
    degree_[l] = new int32_t[n_handle_[l]];
    for (int i = 0; i < n_handle_[l]; ++i) {
      degree_[l][i] = v2e_off_[l][i + 1] - v2e_off_[l][i];
      if (degree_[l][i] < min_degree_[l]) min_degree_[l] = degree_[l][i];
      if (degree_[l][i] > max_degree_[l]) max_degree_[l] = degree_[l][i];
    }
    spdlog::info("layer {} min degree {} max degree {}", l, min_degree_[l],
                 max_degree_[l]);
  }
  ddegree_ = new int32_t*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(ddegree_, n_layer_ + 1); });
  for (int l = 0; l < n_layer_ + 1; ++l) {
    CheckCuda(cudaMalloc(&ddegree_[l], sizeof(int32_t) * n_handle_[l]));
    CheckCuda(cudaMemcpy(ddegree_[l], degree_[l],
                         sizeof(int32_t) * n_handle_[l],
                         cudaMemcpyHostToDevice));
  }

  dhas_color_ = new int32_t*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(dhas_color_, n_layer_ + 1); });
  dpalette_ = new bool*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(dpalette_, n_layer_ + 1); });
  dpalette_size_ = new int32_t*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(dpalette_size_, n_layer_ + 1); });
  drand_states_ = new curandState*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteArray2D(drand_states_, n_layer_ + 1); });
  for (int l = 0; l < n_layer_ + 1; ++l) {
    CheckCuda(cudaMalloc(&dhas_color_[l], sizeof(int32_t) * n_handle_[l]));
    CheckCuda(cudaMalloc(&dpalette_[l],
                         sizeof(bool) * n_handle_[l] * max_degree_[l]));
    CheckCuda(cudaMalloc(&dpalette_size_[l], sizeof(int32_t) * n_handle_[l]));
    CheckCuda(
        cudaMalloc(&drand_states_[l], sizeof(curandState) * n_handle_[l]));
  }

  dcolli_pairs_ = new int32_t*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(dcolli_pairs_, n_layer_ + 1); });
  dcolli_hessian_ = new real*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(dcolli_hessian_, n_layer_ + 1); });
  dcolli_v2e_ = new int32_t*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(dcolli_v2e_, n_layer_ + 1); });
  dcolli_next_edge_ = new int32_t*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(dcolli_next_edge_, n_layer_ + 1); });
  dcolli_edge_to_ = new int32_t*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(dcolli_edge_to_, n_layer_ + 1); });
  for (int l = 0; l < n_layer_ + 1; ++l) {
    CheckCuda(cudaMalloc(&dcolli_pairs_[l], sizeof(int32_t) * MAXN_COLLI * 2));
    if (l == n_layer_) {
      CheckCuda(
          cudaMalloc(&dcolli_hessian_[l], sizeof(real) * MAXN_COLLI * 16));
      CheckCuda(
          cudaMemset(dcolli_hessian_[l], 0, sizeof(real) * MAXN_COLLI * 16));
    } else {
      CheckCuda(
          cudaMalloc(&dcolli_hessian_[l], sizeof(real) * MAXN_COLLI * 256));
      CheckCuda(
          cudaMemset(dcolli_hessian_[l], 0, sizeof(real) * MAXN_COLLI * 256));
    }
    CheckCuda(cudaMalloc(&dcolli_v2e_[l], sizeof(int32_t) * n_handle_[l]));
    CheckCuda(
        cudaMalloc(&dcolli_next_edge_[l], sizeof(int32_t) * MAXN_COLLI * 2));
    CheckCuda(
        cudaMalloc(&dcolli_edge_to_[l], sizeof(int32_t) * MAXN_COLLI * 2));
  }
}

void CudaElasobjMixedMG19::ToDevice() {
  CheckCuda(cudaMalloc(&dpressure_, sizeof(real) * n_vert_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dpressure_)); });
  CheckCuda(cudaMemset(dpressure_, 0, sizeof(real) * n_vert_));
  CheckCuda(cudaMalloc(&dtet_vol_, sizeof(real) * n_tet_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dtet_vol_)); });
  if (Uf_) Uf_->CsrToDevice(cusparse_handle_);
  for (int l = 0; l + 1 < n_layer_; ++l) U_[l].CsrToDevice(cusparse_handle_);
  if (Af_) Af_->BCooSymToDevice(cusparse_handle_);
  for (int l = 0; l < n_layer_; ++l) {
    A_[l].BCooSymToDevice(cusparse_handle_);
    if (A_[l].as_dense_)
      A_[l].LDLTToDevice(cusolverDn_handle_, cusolverDn_params_);
  }

  ddiag_add_ = new real*[n_layer_ + 1];
  for (int l = 0; l < n_layer_; ++l) {
    CheckCuda(cudaMalloc(&ddiag_add_[l], sizeof(real) * n_handle_[l] * 256));
  }
  CheckCuda(cudaMalloc(&ddiag_add_[n_layer_], sizeof(real) * n_vert_ * 16));
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(ddiag_add_, n_layer_ + 1); });
  CheckCuda(cudaMalloc(&dd2off_, sizeof(int32_t) * n_vert_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dd2off_)); });
  CheckCuda(cudaMemcpy(dd2off_, d2off_, sizeof(int32_t) * n_vert_,
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMalloc(&dt2off_, sizeof(int32_t) * 16 * n_tet_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dt2off_)); });
  CheckCuda(cudaMemcpy(dt2off_, t2off_, sizeof(int32_t) * 16 * n_tet_,
                       cudaMemcpyHostToDevice));

  if (n_layer_ > 0 && A_as_dense_[0]) {
    int n = n_handle_[0] * 4;
    CheckCuda(cudaMalloc(&dC_inv_, sizeof(real) * n * n));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dC_inv_)); });
    CheckCuda(cudaMemcpy(dC_inv_, C_inv_, sizeof(real) * n * n,
                         cudaMemcpyHostToDevice));
    int m = n_handle_[0] * 12;
    CheckCuda(cudaMalloc(&dGt_, sizeof(real) * m * n));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGt_)); });
    CheckCuda(cusolverDnXpotrf_bufferSize(
        cusolverDn_handle_, cusolverDn_params_, CUBLAS_FILL_MODE_LOWER,
        (int64_t)m, CudaRealType, A_[0].dden_val_, (int64_t)dim_[0],
        CudaRealType, &chol_dev_buffer_size_, &chol_host_buffer_size_));
    CheckCuda(cudaMalloc(&dchol_info_, sizeof(int32_t)));
    CheckCuda(cudaMemset(dchol_info_, 0, sizeof(int32_t)));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dchol_info_)); });
    chol_host_buffer_ = ::operator new(chol_host_buffer_size_);
    deletion_queue_.push_back([=]() { delete[] chol_host_buffer_; });
    CheckCuda(cudaMalloc(&chol_dev_buffer_, chol_dev_buffer_size_));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(chol_dev_buffer_)); });
    real* chol_fix = new real[m];
    for (int i = 0; i < m; ++i) chol_fix[i] = real(1e-4);
    CheckCuda(cudaMalloc(&dchol_fix_, sizeof(real) * m));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dchol_fix_)); });
    CheckCuda(cudaMemcpy(dchol_fix_, chol_fix, sizeof(real) * m,
                         cudaMemcpyHostToDevice));
    delete[] chol_fix;
  }

  if (n_layer_ > 0) {
    CheckCuda(cudaMalloc(&dUtAU_diag_fix_,
                         sizeof(real) * n_handle_[n_layer_ - 1] * 256));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dUtAU_diag_fix_)); });
    CheckCuda(cudaMemcpy(dUtAU_diag_fix_, UtAU_diag_fix_,
                         sizeof(real) * n_handle_[n_layer_ - 1] * 256,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMalloc(&ddiag_XXt_, sizeof(real) * n_vert_ * 16));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(ddiag_XXt_)); });
    CheckCuda(cudaMemcpy(ddiag_XXt_, diag_XXt_, sizeof(real) * n_vert_ * 16,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMalloc(&dXXt_, sizeof(real) * Af_->bnnz_ * 16));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dXXt_)); });
    CheckCuda(cudaMemcpy(dXXt_, XXt_, sizeof(real) * Af_->bnnz_ * 16,
                         cudaMemcpyHostToDevice));
    dred_off_ = new int32_t*[n_layer_];
    deletion_queue_.push_back(
        [=]() { DeleteCudaArray2D(dred_off_, n_layer_); });
    dred_half_off_ = new int32_t*[n_layer_];
    deletion_queue_.push_back(
        [=]() { DeleteCudaArray2D(dred_half_off_, n_layer_); });
    dA_low_off_ = new int32_t*[n_layer_];
    deletion_queue_.push_back(
        [=]() { DeleteCudaArray2D(dA_low_off_, n_layer_); });
    dA_diag_off_ = new int32_t*[n_layer_];
    deletion_queue_.push_back(
        [=]() { DeleteCudaArray2D(dA_diag_off_, n_layer_); });
    dA_mirror_off_ = new int32_t*[n_layer_];
    deletion_queue_.push_back(
        [=]() { DeleteCudaArray2D(dA_mirror_off_, n_layer_); });
    for (int l = 0; l < n_layer_; ++l) {
      size_t red_size = sizeof(int32_t) * Af_->bnnz_;
      if (l != n_layer_ - 1) red_size = sizeof(int32_t) * A_[l + 1].bnnz_;
      CheckCuda(cudaMalloc(&dred_off_[l], red_size));
      CheckCuda(cudaMemcpy(dred_off_[l], red_off_[l], red_size,
                           cudaMemcpyHostToDevice));
      CheckCuda(
          cudaMalloc(&dred_half_off_[l], sizeof(int32_t) * red_n_half_[l]));
      CheckCuda(cudaMemcpy(dred_half_off_[l], red_half_off_[l],
                           sizeof(int32_t) * red_n_half_[l],
                           cudaMemcpyHostToDevice));
      CheckCuda(cudaMalloc(&dA_low_off_[l], sizeof(int32_t) * A_[l].low_bnnz_));
      CheckCuda(cudaMemcpy(dA_low_off_[l], A_low_off_[l],
                           sizeof(int32_t) * A_[l].low_bnnz_,
                           cudaMemcpyHostToDevice));
      CheckCuda(
          cudaMalloc(&dA_diag_off_[l], sizeof(int32_t) * A_[l].diag_bnnz_));
      CheckCuda(cudaMemcpy(dA_diag_off_[l], A_diag_off_[l],
                           sizeof(int32_t) * A_[l].diag_bnnz_,
                           cudaMemcpyHostToDevice));
      CheckCuda(
          cudaMalloc(&dA_mirror_off_[l], sizeof(int32_t) * A_[l].low_bnnz_));
      CheckCuda(cudaMemcpy(dA_mirror_off_[l], A_mirror_off_[l],
                           sizeof(int32_t) * A_[l].low_bnnz_,
                           cudaMemcpyHostToDevice));
    }
    dhandle_ = new int32_t*[n_layer_ + 1];
    deletion_queue_.push_back(
        [=]() { DeleteCudaArray2D(dhandle_, n_layer_ + 1); });
    dhandle_[0] = nullptr;
    for (int l = 1; l <= n_layer_; ++l) {
      CheckCuda(cudaMalloc(&dhandle_[l], sizeof(int32_t) * n_handle_[l]));
      CheckCuda(cudaMemcpy(dhandle_[l], handle_[l],
                           sizeof(int32_t) * n_handle_[l],
                           cudaMemcpyHostToDevice));
    }
    CheckCuda(cudaMalloc(&dhandle_ids_, sizeof(int32_t) * handle_ids_.size()));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dhandle_ids_)); });
    CheckCuda(cudaMemcpy(dhandle_ids_, handle_ids_.data(),
                         sizeof(int32_t) * handle_ids_.size(),
                         cudaMemcpyHostToDevice));
  }
  dcolor_ = new int32_t*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(dcolor_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    CheckCuda(cudaMalloc(&dcolor_[l], sizeof(int32_t) * n_handle_[l]));
    CheckCuda(cudaMemcpy(dcolor_[l], color_[l], sizeof(int32_t) * n_handle_[l],
                         cudaMemcpyHostToDevice));
  }
  dv2e_off_ = new int32_t*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(dv2e_off_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    CheckCuda(cudaMalloc(&dv2e_off_[l], sizeof(int32_t) * (n_handle_[l] + 1)));
    CheckCuda(cudaMemcpy(dv2e_off_[l], v2e_off_[l],
                         sizeof(int32_t) * (n_handle_[l] + 1),
                         cudaMemcpyHostToDevice));
  }
  dedge_to_ = new int32_t*[n_layer_ + 1];
  deletion_queue_.push_back(
      [=]() { DeleteCudaArray2D(dedge_to_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    int n = v2e_off_[l][n_handle_[l]];
    CheckCuda(cudaMalloc(&dedge_to_[l], sizeof(int32_t) * n));
    CheckCuda(cudaMemcpy(dedge_to_[l], edge_to_[l], sizeof(int32_t) * n,
                         cudaMemcpyHostToDevice));
  }
  CheckCuda(cudaMalloc(&dv2t_ids_, sizeof(int32_t) * 4 * n_tet_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dv2t_ids_)); });
  CheckCuda(cudaMemcpy(dv2t_ids_, v2t_ids_, sizeof(int32_t) * 4 * n_tet_,
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMalloc(&dv2t_off_, sizeof(int32_t) * (n_vert_ + 1)));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dv2t_off_)); });
  CheckCuda(cudaMemcpy(dv2t_off_, v2t_off_, sizeof(int32_t) * (n_vert_ + 1),
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMalloc(&dnormal_sign_, sizeof(int32_t) * 4 * n_tet_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dnormal_sign_)); });
  CheckCuda(cudaMemcpy(dnormal_sign_, normal_sign_,
                       sizeof(int32_t) * 4 * n_tet_, cudaMemcpyHostToDevice));
  CheckCuda(
      cudaMalloc(&dAf_XXt_, sizeof(real) * red_n_half_[n_layer_ - 1] * 256));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dAf_XXt_)); });

  CheckCuda(cudaMalloc(&dold_verts_, sizeof(Vec3) * n_vert_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dold_verts_)); });
  CheckCuda(cudaMalloc(&dinertia_verts_, sizeof(Vec3) * n_vert_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dinertia_verts_)); });
  CheckCuda(cudaMalloc(&dtet_grad_, sizeof(real) * n_tet_ * 12));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dtet_grad_)); });
  CheckCuda(cudaMalloc(&dtet_p_grad_, sizeof(real) * n_tet_ * 4));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dtet_p_grad_)); });
  drhs_ = new real*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteCudaArray2D(drhs_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    CheckCuda(cudaMalloc(&drhs_[l], sizeof(real) * dim_[l]));
  }
  dlhs_ = new real*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteCudaArray2D(dlhs_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    CheckCuda(cudaMalloc(&dlhs_[l], sizeof(real) * dim_[l]));
  }
  dtmp_ = new real*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteCudaArray2D(dtmp_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    CheckCuda(cudaMalloc(&dtmp_[l], sizeof(real) * dim_[l]));
  }
  dtmp2_ = new real*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteCudaArray2D(dtmp2_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    CheckCuda(cudaMalloc(&dtmp2_[l], sizeof(real) * dim_[l]));
  }
  dtAP_ = new real*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteCudaArray2D(dtAP_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    CheckCuda(cudaMalloc(&dtAP_[l], sizeof(real) * dim_[l]));
  }
  dP_ = new real*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteCudaArray2D(dP_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    CheckCuda(cudaMalloc(&dP_[l], sizeof(real) * dim_[l]));
  }
  dR_ = new real*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteCudaArray2D(dR_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    CheckCuda(cudaMalloc(&dR_[l], sizeof(real) * dim_[l]));
  }
  dAP_ = new real*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteCudaArray2D(dAP_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    CheckCuda(cudaMalloc(&dAP_[l], sizeof(real) * dim_[l]));
  }
  dZ_ = new real*[n_layer_ + 1];
  deletion_queue_.push_back([=]() { DeleteCudaArray2D(dZ_, n_layer_ + 1); });
  for (int l = 0; l <= n_layer_; ++l) {
    CheckCuda(cudaMalloc(&dZ_[l], sizeof(real) * dim_[l]));
  }

  rhs_descr_ = new cusparseDnVecDescr_t[n_layer_ + 1];
  deletion_queue_.push_back([=]() {
    for (int l = 0; l <= n_layer_; ++l)
      CheckCuda(cusparseDestroyDnVec(rhs_descr_[l]));
    delete[] rhs_descr_;
  });
  for (int l = 0; l <= n_layer_; ++l)
    CheckCuda(
        cusparseCreateDnVec(&rhs_descr_[l], dim_[l], drhs_[l], CudaRealType));
  lhs_descr_ = new cusparseDnVecDescr_t[n_layer_ + 1];
  deletion_queue_.push_back([=]() {
    for (int l = 0; l <= n_layer_; ++l)
      CheckCuda(cusparseDestroyDnVec(lhs_descr_[l]));
    delete[] lhs_descr_;
  });
  for (int l = 0; l <= n_layer_; ++l)
    CheckCuda(
        cusparseCreateDnVec(&lhs_descr_[l], dim_[l], dlhs_[l], CudaRealType));
  tmp_descr_ = new cusparseDnVecDescr_t[n_layer_ + 1];
  deletion_queue_.push_back([=]() {
    for (int l = 0; l <= n_layer_; ++l)
      CheckCuda(cusparseDestroyDnVec(tmp_descr_[l]));
    delete[] tmp_descr_;
  });
  for (int l = 0; l <= n_layer_; ++l)
    CheckCuda(
        cusparseCreateDnVec(&tmp_descr_[l], dim_[l], dtmp_[l], CudaRealType));
  R_descr_ = new cusparseDnVecDescr_t[n_layer_ + 1];
  deletion_queue_.push_back([=]() {
    for (int l = 0; l <= n_layer_; ++l)
      CheckCuda(cusparseDestroyDnVec(R_descr_[l]));
    delete[] R_descr_;
  });
  for (int l = 0; l <= n_layer_; ++l)
    CheckCuda(cusparseCreateDnVec(&R_descr_[l], dim_[l], dR_[l], CudaRealType));
}

void CudaElasobjMixedMG19::ComputeNormalSign() {
  normal_sign_ = new int32_t[n_tet_ * 4];
  deletion_queue_.push_back([=]() { delete[] normal_sign_; });
  for (auto t = 0; t < n_tet_; ++t) {
    uint32_t* tet = &tets_[t * 4];
    for (auto a = 0; a < 4; ++a) {
      Vec3 dx1 = verts_[tet[(a + 1) % 4]] - verts_[tet[(a + 3) % 4]];
      Vec3 dx2 = verts_[tet[(a + 2) % 4]] - verts_[tet[(a + 3) % 4]];
      Vec3 n = dx1.cross(dx2);
      real dot = n.dot(verts_[tet[a]] - verts_[tet[(a + 3) % 4]]);
      normal_sign_[4 * t + a] = (dot > 0.) ? 1 : -1;
    }
  }
}

uint32_t CudaElasobjMixedMG19::GraphColoring(uint32_t* v2e_off,
                                             uint32_t* edge_to, uint32_t* color,
                                             uint32_t n_vert,
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

void CudaElasobjMixedMG19::ComputeShortestPath(uint32_t source,
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
    for (auto i = v2e_off_[n_layer_][to]; i < v2e_off_[n_layer_][to + 1]; ++i) {
      uint32_t next = edge_to_[n_layer_][i];
      real nd = dist + edge_len_[i];
      if (nd < shortest[next].first)
        pq.push(LenFromTo{LenFrom{nd, from}, next});
    }
  }
}

template <typename _type>
void CudaElasobjMixedMG19::DeleteArray2D(_type** ptr, uint32_t size) {
  if (ptr) return;
  for (auto i = 0; i < size; ++i)
    if (ptr[i]) delete[] ptr[i];
  delete[] ptr;
}

template <typename _type>
void CudaElasobjMixedMG19::DeleteCudaArray2D(_type** ptr, uint32_t size) {
  if (ptr) return;
  for (auto i = 0; i < size; ++i) CheckCuda(cudaFree(ptr[i]));
  delete[] ptr;
}
};  // namespace Rain