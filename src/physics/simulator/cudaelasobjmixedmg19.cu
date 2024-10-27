#include <filesystem>
#include <fstream>
#include <iomanip>

#include "imgui.h"
#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <thrust/device_vector.h>

#include "cudaelasobjmixedmg19.h"
#include "cudasim/explicit.h"
#include "cudasim/kinematiccollision.h"
#include "cudasim/mixedmultigrid.h"
#include "cudasim/multigrid.h"
#include "cudasim/selfcollision.h"

namespace Rain {
void CudaElasobjMixedMG19::ShowUI() {
  CudaElasticObject::ShowUI();
  CUDA::ComputeTetVolume<<<tet_blocks_, tet_threads_per_block_>>>(
      dvertices_, dindices_, dtet_vol_, n_tet_);
  thrust::device_ptr<real> vol_ptr(dtet_vol_);
  real vol =
      thrust::reduce(vol_ptr, vol_ptr + n_tet_, 0., thrust::plus<real>());
  vol_diff_ = (vol - rest_vol_) / rest_vol_ * 100.;
  ImGui::Text("Volume Change: %.4f%%", vol_diff_);
  thrust::device_ptr<real> pressure_ptr(dpressure_);
  real pmin =
      thrust::reduce(pressure_ptr, pressure_ptr + n_vert_,
                     std::numeric_limits<real>::max(), thrust::minimum<real>());
  real pmax =
      thrust::reduce(pressure_ptr, pressure_ptr + n_vert_,
                     std::numeric_limits<real>::min(), thrust::maximum<real>());
  ImGui::Text("Pressure: %.2e ~ %.2e", pmin, pmax);
  if (self_collision_) {
    ImGui::Text("#collision pairs: %d", hashing_->n_colli_);
    for (int l = n_layer_; l > 0; --l) {
      ImGui::Text("#color at layer %d: %d", l, n_color_[l]);
    }
  }
  float relaxation = (float)relaxation_;
  ImGui::SliderFloat("relaxation", &relaxation, 0., 1.);
  relaxation_ = (real)relaxation;
  float p_smooth = (float)p_smooth_;
  ImGui::SliderFloat("p_smooth", &p_smooth, 0., 1.);
  p_smooth_ = (real)p_smooth;
}

__global__ static void OscillationHack(const bool* fixed, Vec3* fixed_X,
                                       const Vec3* X, const int n_frame,
                                       const real dt, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  real freq = 10.;
  real amp = 1.;
  int target = 4 * PI_ / freq / dt;
  if (n_frame > target) return;
  if (fixed[v] && fixed_X[v][1] > 1.) {
    fixed_X[v][0] += amp * freq * dt * sin(freq * n_frame * dt);
  }
}

void CudaElasobjMixedMG19::Update(
    cudaStream_t stream, float Dt, const Vec3& grav, real damping,
    real kinematic_penalty, real self_penalty,
    const std::vector<CudaKinematicObject*>& kobjs, uint32_t n_substep,
    real substep_size, uint32_t n_frame) {
  if (minres_iter_ > 0) {
    UpdateMINRES(stream, Dt, grav, damping, kinematic_penalty, self_penalty,
                 kobjs, n_substep, substep_size, n_frame);
    return;
  }

  if (n_joint_ > 0) UpdateSkeleton();

  // Hack for the bunny example
  // OscillationHack<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
  //     dfixed_, dfixed_verts_, dvertices_, n_frame, dt_, n_vert_);

  for (int step = 0; step < n_substep; ++step) {
    real dt = dt_;
    CheckCuda(cudaMemcpyAsync(dold_verts_, dvertices_, sizeof(Vec3) * n_vert_,
                              cudaMemcpyDeviceToDevice, stream));
    if (!quasi_static_)
      CUDA::UpdateBasic<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
          dvertices_, dvelocities_, n_vert_, grav, damping, dt);
    CheckCuda(cudaMemcpyAsync(dinertia_verts_, dvertices_,
                              sizeof(Vec3) * n_vert_, cudaMemcpyDeviceToDevice,
                              stream));
    for (auto iter = 0; iter < n_iter_; ++iter) {
      // Hessian
      if (elastic_model_->type_ == ElasticModelType::NeoHookean) {
        CheckCuda(cudaMemsetAsync(Af_->dbcoo_val_, 0, sizeof(real) * Af_->nnz_,
                                  stream));
        CUDA::HessianMixedNeoHookeanClamped<<<
            tet_blocks_, tet_threads_per_block_, 0, stream>>>(
            dvertices_, dpressure_, dindices_, dDm_inv_, dvolumes_, dt2off_,
            Af_->dbcoo_val_, dnormal_sign_, mu_, lambda_inv_, p_smooth_,
            p_scale_, n_tet_);
        if (!quasi_static_)
          CUDA::InertiaHessianMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                      stream>>>(
              dmasses_, dd2off_, Af_->dbcoo_val_, 1 / dt_, n_vert_);
        Af_->UpdateDiag(stream);
      } else if (elastic_model_->type_ == ElasticModelType::Corotation) {
        CheckCuda(cudaMemsetAsync(Af_->dbcoo_val_, 0, sizeof(real) * Af_->nnz_,
                                  stream));
        CUDA::HessianMixedCorotationClamped<<<
            tet_blocks_, tet_threads_per_block_, 0, stream>>>(
            dvertices_, dpressure_, dindices_, dDm_inv_, dvolumes_, dt2off_,
            Af_->dbcoo_val_, dnormal_sign_, mu_, lambda_inv_, p_smooth_,
            p_scale_, n_tet_);
        if (!quasi_static_)
          CUDA::InertiaHessianMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                      stream>>>(
              dmasses_, dd2off_, Af_->dbcoo_val_, 1 / dt_, n_vert_);
      } else {
        spdlog::error("not implemented for mixed multigrid");
        exit(1);
      }

      for (int l = n_layer_ - 1; l >= 0; --l) {
        if (A_as_dense_[l]) {
          CheckCuda(cudaMemsetAsync(A_[l].dden_val_, 0,
                                    sizeof(real) * A_[l].rows_ * A_[l].cols_,
                                    stream));
        } else {
          CheckCuda(cudaMemsetAsync(A_[l].dbcoo_val_, 0,
                                    sizeof(real) * A_[l].nnz_, stream));
        }
        if (l == n_layer_ - 1) {
          uint32_t n = red_n_half_[l];
          int32_t n_red_thread = 4;
          int32_t n_red_block = (16 * n + n_red_thread - 1) / n_red_thread;
          CUDA::AfKroneckerXXtHalfMixed<<<n_red_block, n_red_thread * 16>>>(
              Af_->dbcoo_val_, dXXt_, dred_half_off_[l], dAf_XXt_, n);
          if (A_as_dense_[l]) {
            if (l == 0 && direct_shur_) {
              CUDA::
                  AfDenseReductionShurMixed<<<n_red_block, n_red_thread * 16>>>(
                      dAf_XXt_, dred_off_[l], dred_half_off_[l],
                      A_[l].dden_val_, n, n_handle_[l]);
              CUDA::AsDenseMirrorShurMixed<<<A_[l].low_bnnz_, 256>>>(
                  A_[l].dden_val_, dA_low_off_[l], n_handle_[l],
                  A_[l].low_bnnz_);
            } else {
              CUDA::AfDenseReductionMixed<<<n_red_block, n_red_thread * 16>>>(
                  dAf_XXt_, dred_off_[l], dred_half_off_[l], A_[l].dden_val_, n,
                  n_handle_[l]);
              CUDA::AsDenseMirrorMixed<<<A_[l].low_bnnz_, 256>>>(
                  A_[l].dden_val_, dA_low_off_[l], n_handle_[l],
                  A_[l].low_bnnz_);
            }
          } else {
            CUDA::AfSparseReductionMixed<<<n_red_block, n_red_thread * 16>>>(
                dAf_XXt_, dred_off_[l], dred_half_off_[l], A_[l].dbcoo_val_, n);
            CUDA::AsSparseMirrorMixed<<<A_[l].low_bnnz_, 256>>>(
                A_[l].dbcoo_val_, dA_low_off_[l], dA_mirror_off_[l],
                A_[l].low_bnnz_);
          }
        } else {
          uint32_t n = A_[l + 1].bnnz_;
          if (l || !A_[l].as_dense_) {  // As to sparse reduction
            CUDA::AsSparseReductionMixed<<<n, 256>>>(
                A_[l + 1].dbcoo_val_, dred_off_[l], A_[l].dbcoo_val_, n);
          } else {  // As to dense reduction
            if (l == 0 && A_[l].as_dense_ && direct_shur_) {
              CUDA::AsDenseReductionShurMixed<<<n, 256>>>(
                  A_[l + 1].dbcoo_val_, dred_off_[l], A_[l].dden_val_,
                  n_handle_[l], n);
            } else {
              CUDA::AsDenseReductionMixed<<<n, 256>>>(
                  A_[l + 1].dbcoo_val_, dred_off_[l], A_[l].dden_val_,
                  n_handle_[l], n);
            }
          }
        }
        A_[l].UpdateDiag(stream);
      }

      CheckCuda(cudaMemset(dtet_grad_, 0, sizeof(real) * n_tet_ * 12));
      CheckCuda(cudaMemset(dtet_p_grad_, 0, sizeof(real) * n_tet_ * 4));
      real psmooth = p_smooth_;
      if (quasi_static_) {
        if (abs(vol_diff_) < 1.) psmooth = 0.;
      }
      if (elastic_model_->type_ == ElasticModelType::NeoHookean) {
        CUDA::TetGradientMixedNeoHookean<<<tet_blocks_, tet_threads_per_block_,
                                           0, stream>>>(
            dvertices_, dindices_, dpressure_, dnormal_sign_, dDm_inv_,
            dvolumes_, dtet_grad_, dtet_p_grad_, mu_, lambda_inv_, psmooth,
            p_scale_, n_tet_);
      } else if (elastic_model_->type_ == ElasticModelType::Corotation) {
        CUDA::TetGradientMixedCorotation<<<tet_blocks_, tet_threads_per_block_,
                                           0, stream>>>(
            dvertices_, dindices_, dpressure_, dnormal_sign_, dDm_inv_,
            dvolumes_, dtet_grad_, dtet_p_grad_, mu_, lambda_inv_, psmooth,
            p_scale_, n_tet_);
      } else {
        spdlog::error("not implemented for mixed multigrid");
        exit(1);
      }
      CheckCuda(cudaMemset(drhs_[n_layer_], 0, sizeof(real) * dim_[n_layer_]));
      if (obj_->selected_idx_.has_value()) {
        CUDA::MixedEnergyGradientWithCtrl<<<
            vert_blocks_, vert_threads_per_block_, 0, stream>>>(
            dtet_grad_, dtet_p_grad_, dv2t_ids_, dv2t_off_, dfixed_,
            dfixed_verts_, dvertices_, drhs_[n_layer_], control_mag_,
            obj_->selected_idx_.value(), obj_->control_pos_, n_vert_);
      } else {
        CUDA::MixedEnergyGradient<<<vert_blocks_, vert_threads_per_block_, 0,
                                    stream>>>(
            dtet_grad_, dtet_p_grad_, dv2t_ids_, dv2t_off_, dfixed_,
            dfixed_verts_, dvertices_, dmasses_, drhs_[n_layer_], control_mag_,
            n_vert_);
      }
      if (!quasi_static_)
        CUDA::InertiaGradientMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                     stream>>>(dinertia_verts_, dvertices_,
                                               dmasses_, drhs_[n_layer_],
                                               1 / dt_, n_vert_);
      if (n_joint_ > 0) {
        CUDA::SkeletonGradientMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                      stream>>>(
            drhs_[n_layer_], dvertices_, dbone_ids_, djoint_trans_, dlocal_pos_,
            control_mag_, n_vert_);
      }

      for (int l = 0; l < n_layer_; ++l)
        CheckCuda(
            cudaMemset(ddiag_add_[l], 0, sizeof(real) * n_handle_[l] * 256));
      CheckCuda(
          cudaMemset(ddiag_add_[n_layer_], 0, sizeof(real) * n_vert_ * 16));
      if (obj_->selected_idx_.has_value()) {
        CUDA::UpdateAfDiagWithCtrlMixed<<<vert_blocks_, vert_threads_per_block_,
                                          0, stream>>>(
            ddiag_add_[n_layer_], dfixed_, control_mag_,
            obj_->selected_idx_.value(), n_vert_);
      } else {
        CUDA::UpdateAfDiagMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                  stream>>>(ddiag_add_[n_layer_], dfixed_,
                                            control_mag_, n_vert_);
      }
      // skeleton
      if (n_joint_ > 0) {
        CUDA::UpdateAfDiagSkeletonMixed<<<vert_blocks_, vert_threads_per_block_,
                                          0, stream>>>(
            ddiag_add_[n_layer_], dbone_ids_, control_mag_, n_vert_);
      }
      if (self_collision_) {
        SelfCollision(self_penalty, stream);
        for (int l = n_layer_; l > 0; --l) {
          Vivace(l, stream);
        }
      }

      KinematicCollision(kinematic_penalty, kobjs, stream);

      for (int l = n_layer_ - 1; l >= 0; --l) {
        if (l == n_layer_ - 1) {  // UTAU
          real one = real(1.);
          CheckCuda(cublasAxpyEx(cublas_handle_, n_handle_[l] * 256, &one,
                                 CudaRealType, dUtAU_diag_fix_, CudaRealType, 1,
                                 ddiag_add_[l], CudaRealType, 1, CudaRealType));
          int32_t reduction_blocks =
              (16 * n_vert_ + threads_per_block_ - 1) / threads_per_block_;
          CUDA::UpdateUtAUDiagMixed<<<reduction_blocks, threads_per_block_, 0,
                                      stream>>>(
              ddiag_add_[l], ddiag_add_[n_layer_], ddiag_XXt_, dhandle_[l + 1],
              n_vert_);
        } else {
          int32_t UltAUl_blocks =
              (n_handle_[l + 1] + UltAUl_threads_per_block_ - 1) /
              UltAUl_threads_per_block_;
          CUDA::UpdateUltAUlDiagMixed<<<
              UltAUl_blocks, dim3(UltAUl_threads_per_block_, 256), 0, stream>>>(
              ddiag_add_[l], ddiag_add_[l + 1], dhandle_[l + 1],
              n_handle_[l + 1]);
        }

        if (l == 0 && A_as_dense_[l]) {
          if (l == n_layer_) {
            spdlog::error("please add at least one layer");
            exit(1);
          }
          if (!direct_shur_) {
            timer_.Tick();
            // dense matrix at the coarsest level, add diag part and factorize
            CUDA::
                UpdateDenseDiagMixed<<<n_handle_[l], dim3(1, 256), 0, stream>>>(
                    A_[l].dden_val_, ddiag_add_[l], n_handle_[l], A_[l].rows_);
            if (self_collision_) {
              CUDA::SelfCollisionCoarseDenseHessianMixed<<<
                  hashing_->n_colli_, dim3(1, 144), 0, stream>>>(
                  A_[l].dden_val_, dcolli_pairs_[l], dcolli_hessian_[l],
                  dim_[l], hashing_->n_colli_);
            }
            CheckCuda(cudaMemcpyAsync(A_[l].dldlt_val_, A_[l].dden_val_,
                                      sizeof(real) * A_[l].rows_ * A_[l].cols_,
                                      cudaMemcpyDeviceToDevice, stream));
#ifdef REAL_AS_DOUBLE
            CheckCuda(cusolverDnDsytrf(
                cusolverDn_handle_, CUBLAS_FILL_MODE_LOWER, dim_[l],
                A_[l].dldlt_val_, dim_[l], A_[l].dldlt_ipiv_,
                (double*)A_[l].ldlt_dev_buffer_, A_[l].ldlt_dev_buffer_size_,
                A_[l].dldlt_info_));
#else
            CheckCuda(cusolverDnSsytrf(
                cusolverDn_handle_, CUBLAS_FILL_MODE_LOWER, dim_[l],
                A_[l].dldlt_val_, dim_[l], A_[l].dldlt_ipiv_,
                (float*)A_[l].ldlt_dev_buffer_, A_[l].ldlt_dev_buffer_size_,
                A_[l].dldlt_info_));
#endif
            int n_block =
                (dim_[l] + threads_per_block_ - 1) / threads_per_block_;
            CUDA::Int32toInt64<<<n_block, threads_per_block_, 0, stream>>>(
                A_[l].dldlt_ipiv_, A_[l].dldlt_ipiv_64_, dim_[l]);
            timer_.Tick();
          } else {  // shur complement
            CUDA::UpdateDenseDiagShurMixed<<<n_handle_[l], dim3(1, 256), 0,
                                             stream>>>(
                A_[l].dden_val_, ddiag_add_[l], n_handle_[l], A_[l].rows_);
            if (self_collision_) {
              CUDA::SelfCollisionCoarseDenseHessianShurMixed<<<
                  hashing_->n_colli_, dim3(1, 144), 0, stream>>>(
                  A_[l].dden_val_, dcolli_pairs_[l], dcolli_hessian_[l],
                  dim_[l], hashing_->n_colli_);
            }
            CheckCuda(cudaMemcpyAsync(A_[l].dldlt_val_, A_[l].dden_val_,
                                      sizeof(real) * A_[l].rows_ * A_[l].cols_,
                                      cudaMemcpyDeviceToDevice, stream));
            int n = n_handle_[l];
            real one = real(1.);
            real zero = real(0.);
            const real* GT = &A_[l].dldlt_val_[dim_[l] * 12 * n];
            const real* G = &A_[l].dldlt_val_[12 * n];
#ifdef REAL_AS_DOUBLE
            // T <- G^T @ C^-1
            CheckCuda(cublasDsymm(cublas_handle_, CUBLAS_SIDE_RIGHT,
                                  CUBLAS_FILL_MODE_LOWER, 12 * n, 4 * n, &one,
                                  dC_inv_, 4 * n, GT, dim_[l], &zero, dGt_,
                                  12 * n));
            // A <- A + G^T @ C^-1 @ G = A + T @ G
            CheckCuda(cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                  12 * n, 12 * n, 4 * n, &one, dGt_, 12 * n, G,
                                  dim_[l], &one, A_[l].dldlt_val_, dim_[l]));
#else
            CheckCuda(cublasSsymm(cublas_handle_, CUBLAS_SIDE_RIGHT,
                                  CUBLAS_FILL_MODE_LOWER, 12 * n, 4 * n, &one,
                                  dC_inv_, 4 * n, GT, dim_[l], &zero, dGt_,
                                  12 * n));
            CheckCuda(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                  12 * n, 12 * n, 4 * n, &one, dGt_, 12 * n, G,
                                  dim_[l], &one, A_[l].dldlt_val_, dim_[l]));
#endif
            // A factorization
            CheckCuda(cublasAxpyEx(cublas_handle_, 12 * n, &one, CudaRealType,
                                   dchol_fix_, CudaRealType, 1,
                                   A_[l].dldlt_val_, CudaRealType, dim_[l] + 1,
                                   CudaRealType));
            CheckCuda(cusolverDnXpotrf(
                cusolverDn_handle_, cusolverDn_params_, CUBLAS_FILL_MODE_LOWER,
                12 * n, CudaRealType, A_[l].dldlt_val_, dim_[l], CudaRealType,
                chol_dev_buffer_, chol_dev_buffer_size_, chol_host_buffer_,
                chol_host_buffer_size_, dchol_info_));
          }
        }
      }

      CheckCuda(cudaMemset(dlhs_[n_layer_], 0, sizeof(real) * dim_[n_layer_]));
      CheckCuda(cudaMemcpyAsync(dR_[n_layer_], drhs_[n_layer_],
                                sizeof(real) * dim_[n_layer_],
                                cudaMemcpyDeviceToDevice, stream));

      int32_t cur_layer = n_layer_;
      real tol = 1e-6;

      for (MGOpConfig op : operations_) {
        switch (op.type) {
          case MGOpType::GS:
            PerformGSIteration(cur_layer, op.max_iter, tol, stream);
            break;
          case MGOpType::DS:
            DownSample(cur_layer, stream);
            break;
          case MGOpType::Direct:
            if (direct_shur_)
              DirectSolveShur(cur_layer, stream);
            else
              DirectSolve(cur_layer, stream);
            break;
          case MGOpType::US:
            UpSample(cur_layer, stream);
            break;
          default:
            break;
        }
      }

      CUDA::UpdatePosPressureMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                     stream>>>(
          dvertices_, dpressure_, (real)1., dlhs_[n_layer_], p_scale_, n_vert_);
    }  // end of iteration
    if (!quasi_static_)
      CUDA::UpdateVelFromPos<<<vert_blocks_, vert_threads_per_block_, 0,
                               stream>>>(dvelocities_, dvertices_, dold_verts_,
                                         1 / dt_, n_vert_);
  }
  // Update normal
  CUDA::ClearArray<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
      dnormals_, (uint32_t)obj_->n_vert_);
  CUDA::UpdateFaceNormals<<<face_blocks_, face_threads_per_block_, 0, stream>>>(
      dnormals_, dvertices_, dfaces_, (uint32_t)obj_->n_face_);
  CUDA::UpdateVertNormals<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
      dnormals_, (uint32_t)obj_->n_vert_);
  CheckCuda(cudaMemcpy(obj_->vertices_, dvertices_, sizeof(Vec3) * n_vert_,
                       cudaMemcpyDeviceToHost));
  if (output_frame_data_) {
    std::ofstream fout("vertices_" + std::to_string(n_frame) + ".txt");
    for (int i = 0; i < n_vert_; ++i) {
      fout << obj_->vertices_[i][0] << " " << obj_->vertices_[i][1] << " "
           << obj_->vertices_[i][2] << std::endl;
    }
    fout.close();
    CheckCuda(cudaMemcpy(pressure_, dpressure_, sizeof(real) * n_vert_,
                         cudaMemcpyDeviceToHost));
    obj_->WriteObj(obj_->name_ + std::to_string(n_frame) + ".obj");
    // output pressure
    std::ofstream fout2("pressure_" + std::to_string(n_frame) + ".txt");
    for (int i = 0; i < n_vert_; ++i) {
      fout2 << pressure_[i] << std::endl;
    }
    fout2.close();
  }
  output_frame_data_ = false;
}

void CudaElasobjMixedMG19::UpdateMINRES(
    cudaStream_t stream, float Dt, const Vec3& grav, real damping,
    real kinematic_penalty, real self_penalty,
    const std::vector<CudaKinematicObject*>& kobjs, uint32_t n_substep,
    real substep_size, uint32_t n_frame) {
  for (int step = 0; step < n_substep; ++step) {
    real dt = dt_;
    CheckCuda(cudaMemcpyAsync(dold_verts_, dvertices_, sizeof(Vec3) * n_vert_,
                              cudaMemcpyDeviceToDevice, stream));
    if (!quasi_static_)
      CUDA::UpdateBasic<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
          dvertices_, dvelocities_, n_vert_, grav, damping, dt);
    CheckCuda(cudaMemcpyAsync(dinertia_verts_, dvertices_,
                              sizeof(Vec3) * n_vert_, cudaMemcpyDeviceToDevice,
                              stream));
    for (auto iter = 0; iter < n_iter_; ++iter) {
      // Hessian
      if (elastic_model_->type_ == ElasticModelType::NeoHookean) {
        CheckCuda(cudaMemsetAsync(Af_->dbcoo_val_, 0, sizeof(real) * Af_->nnz_,
                                  stream));
        CUDA::HessianMixedNeoHookeanClamped<<<
            tet_blocks_, tet_threads_per_block_, 0, stream>>>(
            dvertices_, dpressure_, dindices_, dDm_inv_, dvolumes_, dt2off_,
            Af_->dbcoo_val_, dnormal_sign_, mu_, lambda_inv_, p_smooth_,
            p_scale_, n_tet_);
        if (!quasi_static_)
          CUDA::InertiaHessianMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                      stream>>>(
              dmasses_, dd2off_, Af_->dbcoo_val_, 1 / dt_, n_vert_);
        Af_->UpdateDiag(stream);
      } else if (elastic_model_->type_ == ElasticModelType::StVK) {
        CheckCuda(cudaMemsetAsync(Af_->dbcoo_val_, 0, sizeof(real) * Af_->nnz_,
                                  stream));
        CUDA::HessianMixedStVKClamped<<<tet_blocks_, tet_threads_per_block_, 0,
                                        stream>>>(
            dvertices_, dpressure_, dindices_, dDm_inv_, dvolumes_, dt2off_,
            Af_->dbcoo_val_, dnormal_sign_, mu_, lambda_inv_, p_smooth_,
            p_scale_, n_tet_);
        if (!quasi_static_)
          CUDA::InertiaHessianMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                      stream>>>(
              dmasses_, dd2off_, Af_->dbcoo_val_, 1 / dt_, n_vert_);
      } else if (elastic_model_->type_ == ElasticModelType::Corotation) {
        CheckCuda(cudaMemsetAsync(Af_->dbcoo_val_, 0, sizeof(real) * Af_->nnz_,
                                  stream));
        CUDA::HessianMixedCorotationClamped<<<
            tet_blocks_, tet_threads_per_block_, 0, stream>>>(
            dvertices_, dpressure_, dindices_, dDm_inv_, dvolumes_, dt2off_,
            Af_->dbcoo_val_, dnormal_sign_, mu_, lambda_inv_, p_smooth_,
            p_scale_, n_tet_);
        if (!quasi_static_)
          CUDA::InertiaHessianMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                      stream>>>(
              dmasses_, dd2off_, Af_->dbcoo_val_, 1 / dt_, n_vert_);
      } else if (elastic_model_->type_ == ElasticModelType::NeoHookeanLog) {
        CheckCuda(cudaMemsetAsync(Af_->dbcoo_val_, 0, sizeof(real) * Af_->nnz_,
                                  stream));
        CUDA::HessianMixedNeoHookeanLogClamped<<<
            tet_blocks_, tet_threads_per_block_, 0, stream>>>(
            dvertices_, dpressure_, dindices_, dDm_inv_, dvolumes_, dt2off_,
            Af_->dbcoo_val_, dnormal_sign_, mu_, lambda_inv_, p_smooth_,
            p_scale_, n_tet_);
        if (!quasi_static_)
          CUDA::InertiaHessianMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                      stream>>>(
              dmasses_, dd2off_, Af_->dbcoo_val_, 1 / dt_, n_vert_);
        Af_->UpdateDiag(stream);
      } else {
        spdlog::error("not implemented for mixed multigrid");
        exit(1);
      }

      CheckCuda(cudaMemset(dtet_grad_, 0, sizeof(real) * n_tet_ * 12));
      CheckCuda(cudaMemset(dtet_p_grad_, 0, sizeof(real) * n_tet_ * 4));
      if (elastic_model_->type_ == ElasticModelType::NeoHookean) {
        CUDA::TetGradientMixedNeoHookean<<<tet_blocks_, tet_threads_per_block_,
                                           0, stream>>>(
            dvertices_, dindices_, dpressure_, dnormal_sign_, dDm_inv_,
            dvolumes_, dtet_grad_, dtet_p_grad_, mu_, lambda_inv_, p_smooth_,
            p_scale_, n_tet_);
      } else if (elastic_model_->type_ == ElasticModelType::StVK) {
        CUDA::TetGradientMixedStVK<<<tet_blocks_, tet_threads_per_block_, 0,
                                     stream>>>(
            dvertices_, dindices_, dpressure_, dnormal_sign_, dDm_inv_,
            dvolumes_, dtet_grad_, dtet_p_grad_, mu_, lambda_inv_, p_smooth_,
            p_scale_, n_tet_);
      } else if (elastic_model_->type_ == ElasticModelType::Corotation) {
        CUDA::TetGradientMixedCorotation<<<tet_blocks_, tet_threads_per_block_,
                                           0, stream>>>(
            dvertices_, dindices_, dpressure_, dnormal_sign_, dDm_inv_,
            dvolumes_, dtet_grad_, dtet_p_grad_, mu_, lambda_inv_, p_smooth_,
            p_scale_, n_tet_);
      } else if (elastic_model_->type_ == ElasticModelType::NeoHookeanLog) {
        CUDA::TetGradientMixedNeoHookeanLog<<<
            tet_blocks_, tet_threads_per_block_, 0, stream>>>(
            dvertices_, dindices_, dpressure_, dnormal_sign_, dDm_inv_,
            dvolumes_, dtet_grad_, dtet_p_grad_, mu_, lambda_inv_, p_smooth_,
            p_scale_, n_tet_);
      } else {
        spdlog::error("not implemented for mixed multigrid");
        exit(1);
      }
      CheckCuda(cudaMemset(drhs_[n_layer_], 0, sizeof(real) * dim_[n_layer_]));
      if (obj_->selected_idx_.has_value()) {
        CUDA::MixedEnergyGradientWithCtrl<<<
            vert_blocks_, vert_threads_per_block_, 0, stream>>>(
            dtet_grad_, dtet_p_grad_, dv2t_ids_, dv2t_off_, dfixed_,
            dfixed_verts_, dvertices_, drhs_[n_layer_], control_mag_,
            obj_->selected_idx_.value(), obj_->control_pos_, n_vert_);
      } else {
        CUDA::MixedEnergyGradient<<<vert_blocks_, vert_threads_per_block_, 0,
                                    stream>>>(
            dtet_grad_, dtet_p_grad_, dv2t_ids_, dv2t_off_, dfixed_,
            dfixed_verts_, dvertices_, dmasses_, drhs_[n_layer_], control_mag_,
            n_vert_);
      }
      if (!quasi_static_)
        CUDA::InertiaGradientMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                     stream>>>(dinertia_verts_, dvertices_,
                                               dmasses_, drhs_[n_layer_],
                                               1 / dt_, n_vert_);

      CheckCuda(
          cudaMemset(ddiag_add_[n_layer_], 0, sizeof(real) * n_vert_ * 16));
      if (obj_->selected_idx_.has_value()) {
        CUDA::UpdateAfDiagWithCtrlMixed<<<vert_blocks_, vert_threads_per_block_,
                                          0, stream>>>(
            ddiag_add_[n_layer_], dfixed_, control_mag_,
            obj_->selected_idx_.value(), n_vert_);
      } else {
        CUDA::UpdateAfDiagMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                  stream>>>(ddiag_add_[n_layer_], dfixed_,
                                            control_mag_, n_vert_);
      }
      KinematicCollision(kinematic_penalty, kobjs, stream);

      CheckCuda(cudaMemset(dlhs_[n_layer_], 0, sizeof(real) * dim_[n_layer_]));

      real* b = drhs_[n_layer_];
      real* x = dlhs_[n_layer_];
      real* r1 = dR_[n_layer_];
      real* r2 = dtmp_[n_layer_];
      real* y = dtAP_[n_layer_];
      real* w = dP_[n_layer_];
      real* w1 = dtmp2_[n_layer_];
      real* v = dAP_[n_layer_];
      real* w2 = dZ_[n_layer_];
      real alfa, beta, beta1, oldb, dbar, oldeps, epsln, gbar, phi, phibar,
          delta, gamma, cs, sn;
      CheckCuda(cudaMemcpyAsync(y, b, sizeof(real) * dim_[n_layer_],
                                cudaMemcpyDeviceToDevice, stream));
      CheckCuda(cudaMemsetAsync(w, 0, sizeof(real) * dim_[n_layer_], stream));
      CheckCuda(cudaMemsetAsync(w2, 0, sizeof(real) * dim_[n_layer_], stream));
      CheckCuda(cudaMemcpyAsync(r1, b, sizeof(real) * dim_[n_layer_],
                                cudaMemcpyDeviceToDevice, stream));
      CheckCuda(cudaMemcpyAsync(r2, b, sizeof(real) * dim_[n_layer_],
                                cudaMemcpyDeviceToDevice, stream));

#ifdef REAL_AS_DOUBLE
      CheckCuda(cublasDdot(cublas_handle_, dim_[n_layer_], b, 1, y, 1, &beta1));
#else
      CheckCuda(cublasSdot(cublas_handle_, dim_[n_layer_], b, 1, y, 1, &beta1));
#endif
      beta1 = sqrt(beta1);
      oldb = 0.;
      beta = beta1;
      dbar = 0.;
      epsln = 0.;
      phibar = beta1;
      cs = -1.;
      sn = 0.;
      real one = (real)1.;
      int miter = 0;
      for (; miter < minres_iter_; ++miter) {
        real s = 1. / beta;
        CheckCuda(cudaMemsetAsync(v, 0, sizeof(real) * dim_[n_layer_], stream));
#ifdef REAL_AS_DOUBLE
        CheckCuda(cublasDaxpy(cublas_handle_, dim_[n_layer_], &s, y, 1, v, 1));
#else
        CheckCuda(cublasSaxpy(cublas_handle_, dim_[n_layer_], &s, y, 1, v, 1));
#endif
        CheckCuda(cudaMemsetAsync(y, 0, sizeof(real) * dim_[n_layer_], stream));
        ComputeAP(n_layer_, y, &one, v, stream);
        if (self_collision_)
          ComputeSelfcollisionOffAP(n_layer_, y, &one, v, stream);
        if (miter > 0) {
          real fac = -beta / oldb;
#ifdef REAL_AS_DOUBLE
          CheckCuda(
              cublasDaxpy(cublas_handle_, dim_[n_layer_], &fac, r1, 1, y, 1));
#else
          CheckCuda(
              cublasSaxpy(cublas_handle_, dim_[n_layer_], &fac, r1, 1, y, 1));
#endif
        }
#ifdef REAL_AS_DOUBLE
        CheckCuda(
            cublasDdot(cublas_handle_, dim_[n_layer_], v, 1, y, 1, &alfa));
#else
        CheckCuda(
            cublasSdot(cublas_handle_, dim_[n_layer_], v, 1, y, 1, &alfa));
#endif
        real fac = -alfa / beta;
#ifdef REAL_AS_DOUBLE
        CheckCuda(
            cublasDaxpy(cublas_handle_, dim_[n_layer_], &fac, r2, 1, y, 1));
#else
        CheckCuda(
            cublasSaxpy(cublas_handle_, dim_[n_layer_], &fac, r2, 1, y, 1));
#endif
        CheckCuda(cudaMemcpyAsync(r1, r2, sizeof(real) * dim_[n_layer_],
                                  cudaMemcpyDeviceToDevice, stream));
        CheckCuda(cudaMemcpyAsync(r2, y, sizeof(real) * dim_[n_layer_],
                                  cudaMemcpyDeviceToDevice, stream));
        oldb = beta;
#ifdef REAL_AS_DOUBLE
        CheckCuda(
            cublasDdot(cublas_handle_, dim_[n_layer_], r2, 1, y, 1, &beta));
#else
        CheckCuda(
            cublasSdot(cublas_handle_, dim_[n_layer_], r2, 1, y, 1, &beta));
#endif
        beta = sqrt(beta);

        oldeps = epsln;
        delta = cs * dbar + sn * alfa;
        gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;

        gamma = sqrt(gbar * gbar + beta * beta);
        gamma = max(gamma, 1e-15);
        cs = gbar / gamma;
        sn = beta / gamma;
        phi = cs * phibar;
        phibar = sn * phibar;

        CheckCuda(cudaMemcpyAsync(w1, w2, sizeof(real) * dim_[n_layer_],
                                  cudaMemcpyDeviceToDevice, stream));
        CheckCuda(cudaMemcpyAsync(w2, w, sizeof(real) * dim_[n_layer_],
                                  cudaMemcpyDeviceToDevice, stream));
        CheckCuda(cudaMemsetAsync(w, 0, sizeof(real) * dim_[n_layer_], stream));
        real fac1 = 1. / gamma;
#ifdef REAL_AS_DOUBLE
        CheckCuda(
            cublasDaxpy(cublas_handle_, dim_[n_layer_], &fac1, v, 1, w, 1));
#else
        CheckCuda(
            cublasSaxpy(cublas_handle_, dim_[n_layer_], &fac1, v, 1, w, 1));
#endif
        real fac2 = -oldeps / gamma;
#ifdef REAL_AS_DOUBLE
        CheckCuda(
            cublasDaxpy(cublas_handle_, dim_[n_layer_], &fac2, w1, 1, w, 1));
#else
        CheckCuda(
            cublasSaxpy(cublas_handle_, dim_[n_layer_], &fac2, w1, 1, w, 1));
#endif
        real fac3 = -delta / gamma;
#ifdef REAL_AS_DOUBLE
        CheckCuda(
            cublasDaxpy(cublas_handle_, dim_[n_layer_], &fac3, w2, 1, w, 1));
#else
        CheckCuda(
            cublasSaxpy(cublas_handle_, dim_[n_layer_], &fac3, w2, 1, w, 1));
#endif
#ifdef REAL_AS_DOUBLE
        CheckCuda(
            cublasDaxpy(cublas_handle_, dim_[n_layer_], &phi, w, 1, x, 1));
#else
        CheckCuda(
            cublasSaxpy(cublas_handle_, dim_[n_layer_], &phi, w, 1, x, 1));
#endif
        if (phibar / beta1 < 1e-6 || phibar < 1.e1) {
          break;
        }
      }
      CUDA::UpdatePosPressureMixed<<<vert_blocks_, vert_threads_per_block_, 0,
                                     stream>>>(
          dvertices_, dpressure_, (real)1., dlhs_[n_layer_], p_scale_, n_vert_);
    }  // end of iteration
    if (!quasi_static_)
      CUDA::UpdateVelFromPos<<<vert_blocks_, vert_threads_per_block_, 0,
                               stream>>>(dvelocities_, dvertices_, dold_verts_,
                                         1 / dt_, n_vert_);
  }
  // Update normal
  CUDA::ClearArray<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
      dnormals_, (uint32_t)obj_->n_vert_);
  CUDA::UpdateFaceNormals<<<face_blocks_, face_threads_per_block_, 0, stream>>>(
      dnormals_, dvertices_, dfaces_, (uint32_t)obj_->n_face_);
  CUDA::UpdateVertNormals<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
      dnormals_, (uint32_t)obj_->n_vert_);
  CheckCuda(cudaMemcpy(obj_->vertices_, dvertices_, sizeof(Vec3) * n_vert_,
                       cudaMemcpyDeviceToHost));

  if (output_frame_data_) {
    // output vertices
    std::ofstream fout("vert_" + std::to_string(n_frame) + ".txt");
    for (int i = 0; i < n_vert_; ++i) {
      fout << obj_->vertices_[i][0] << " " << obj_->vertices_[i][1] << " "
           << obj_->vertices_[i][2] << std::endl;
    }
    fout.close();
    CheckCuda(cudaMemcpy(pressure_, dpressure_, sizeof(real) * n_vert_,
                         cudaMemcpyDeviceToHost));
    obj_->WriteObj("mesh_" + std::to_string(n_frame) + ".obj");
    // output pressure
    std::ofstream fout2("pressure_" + std::to_string(n_frame) + ".txt");
    for (int i = 0; i < n_vert_; ++i) {
      fout2 << pressure_[i] << std::endl;
    }
    fout2.close();
  }
  output_frame_data_ = false;
}

void CudaElasobjMixedMG19::KinematicCollision(
    real k_penalty, const std::vector<CudaKinematicObject*>& kobjs,
    cudaStream_t stream) {
  for (const auto& obj : kobjs) {
    if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::None) {
      // TODO: explicit mesh
      spdlog::error(
          "Explicit collision mesh not implemented for mixed multigrid");
      exit(1);
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Sphere) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      real r = params[3];
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      CUDA::KinematicCollisionSphereMixed<<<
          vert_blocks_, vert_threads_per_block_, 0, stream>>>(
          dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], k_penalty, c, r,
          obj->obj_->implicit_geo_sign_, n_vert_);
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Torus) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      Vec3 d{params[3], params[4], params[5]};
      real a = params[6];
      real r = params[7];
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      d = (obj->rotation_ * d).normalized();
      CUDA::KinematicCollisionTorusMixed<<<
          vert_blocks_, vert_threads_per_block_, 0, stream>>>(
          dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], k_penalty, c, d, a,
          r, n_vert_);
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Cylinder) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      Vec3 d{params[3], params[4], params[5]};
      real r = params[6];
      real h = params[7];
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      d = (obj->rotation_ * d).normalized();
      CUDA::KinematicCollisionCylinderMixed<<<
          vert_blocks_, vert_threads_per_block_, 0, stream>>>(
          dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], k_penalty, c, d, r,
          h, n_vert_);
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Plane) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      Vec3 n{params[3], params[4], params[5]};
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      n = (obj->rotation_ * n).normalized();
      CUDA::KinematicCollisionPlaneMixed<<<
          vert_blocks_, vert_threads_per_block_, 0, stream>>>(
          dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], k_penalty, c, n,
          n_vert_);
    }
  }
}

void CudaElasobjMixedMG19::SelfCollision(real k_penalty, cudaStream_t stream) {
  hashing_->Hashing(stream);
  hashing_->GetVertTetCollisionList(stream);
  int32_t n_block =
      (hashing_->n_colli_ + threads_per_block_ - 1) / threads_per_block_;
  CUDA::MakeSelfCollisionMixed<<<n_block, threads_per_block_, 0, stream>>>(
      drhs_[n_layer_], dcolli_pairs_[n_layer_], dcolli_hessian_[n_layer_],
      ddiag_add_[n_layer_], dvertices_, dnormals_, dindices_,
      hashing_->dcolli_pairs_, dclosest_surf_vert_, k_penalty, n_vert_,
      hashing_->n_colli_);
  CheckCuda(cudaMemsetAsync(dcolli_v2e_[n_layer_], 0xff,
                            sizeof(int32_t) * n_vert_, stream));
  CUDA::BuildSelfCollisionGraph<<<n_block, threads_per_block_, 0, stream>>>(
      dcolli_pairs_[n_layer_], dcolli_v2e_[n_layer_],
      dcolli_next_edge_[n_layer_], dcolli_edge_to_[n_layer_],
      hashing_->n_colli_);
  // reduction
  // diag_add reduction for GS is done outside of this function
  // the off diagonal reduction is done here
  // note that fine level's off diag might reduce to coarse level's diag
  // and the graph reduction for Vivace
  int32_t l = n_layer_ - 1;
  if (l >= 0) {  // XXt reduction
    CUDA::SelfCollisionFineOffReductionMixed<<<n_block, threads_per_block_, 0,
                                               stream>>>(
        dcolli_pairs_[l], dcolli_hessian_[l], dcolli_pairs_[l + 1],
        dcolli_hessian_[l + 1], drest_verts_, dhandle_[l + 1], dhandle_ids_,
        hashing_->n_colli_);
    CUDA::SelfCollisionFineDiagReductionMixed<<<n_block, threads_per_block_, 0,
                                                stream>>>(
        ddiag_add_[l], dcolli_pairs_[l + 1], dcolli_hessian_[l + 1],
        drest_verts_, dhandle_[l + 1], dhandle_ids_, hashing_->n_colli_);
    CheckCuda(cudaMemsetAsync(dcolli_v2e_[l], 0xff,
                              sizeof(int32_t) * n_handle_[l], stream));
    CUDA::BuildSelfCollisionGraph<<<n_block, threads_per_block_, 0, stream>>>(
        dcolli_pairs_[l], dcolli_v2e_[l], dcolli_next_edge_[l],
        dcolli_edge_to_[l], hashing_->n_colli_);
  }
  l -= 1;
  for (; l >= 0; --l) {
    CUDA::SelfCollisionCoarseOffReductionMixed<<<hashing_->n_colli_,
                                                 dim3(1, 256), 0, stream>>>(
        dcolli_pairs_[l], dcolli_hessian_[l], dcolli_pairs_[l + 1],
        dcolli_hessian_[l + 1], dhandle_[l + 1], hashing_->n_colli_);
    CUDA::SelfCollisionCoarseDiagReductionMixed<<<hashing_->n_colli_,
                                                  dim3(1, 256), 0, stream>>>(
        ddiag_add_[l], dcolli_pairs_[l + 1], dcolli_hessian_[l + 1],
        dhandle_[l + 1], hashing_->n_colli_);
    CUDA::BuildSelfCollisionGraph<<<n_block, threads_per_block_, 0, stream>>>(
        dcolli_pairs_[l], dcolli_v2e_[l], dcolli_next_edge_[l],
        dcolli_edge_to_[l], hashing_->n_colli_);
  }
}

void CudaElasobjMixedMG19::Vivace(int32_t l, cudaStream_t stream) {
  int32_t n = n_handle_[l];
  int32_t n_block = (n + threads_per_block_ - 1) / threads_per_block_;
  int32_t* color = dcolor_[l];
  int32_t* v2e_off = dv2e_off_[l];
  int32_t* edge_to = dedge_to_[l];
  int32_t shrink = min_degree_[l];
  int32_t maxp = max_degree_[l];
  int32_t minp = (l == n_layer_ ? 10 : 5);
  int32_t* degree = ddegree_[l];
  bool* palette = dpalette_[l];
  int32_t* psize = dpalette_size_[l];
  int32_t* has_color = dhas_color_[l];
  int32_t* c_v2e = dcolli_v2e_[l];
  int32_t* c_next_edge = dcolli_next_edge_[l];
  int32_t* c_edge_to = dcolli_edge_to_[l];
  curandState* rand_state = drand_states_[l];
  int32_t n_has = 0, pre_n_has = 0, stuck = 0;
  CUDA::InitRandState<<<n_block, threads_per_block_, 0, stream>>>(rand_state,
                                                                  n);

  CheckCuda(cudaMemsetAsync(has_color, 0, sizeof(int32_t) * n, stream));
  CheckCuda(cudaMemsetAsync(palette, 0, sizeof(bool) * maxp * n, stream));
  CUDA::VivacePass1<<<n_block, threads_per_block_, 0, stream>>>(
      psize, degree, shrink, maxp, minp, n);
  while (n_has < n) {
    CUDA::VivacePass2<<<n_block, threads_per_block_, 0, stream>>>(
        color, rand_state, has_color, palette, psize, maxp, n);
    // CUDA::VivacePass3<<<n_block, threads_per_block_, 0, stream>>>(
    //     has_color, palette, color, v2e_off, edge_to, maxp, n);
    CUDA::VivacePass3WithSelfCollision<<<n_block, threads_per_block_, 0,
                                         stream>>>(
        has_color, palette, color, v2e_off, edge_to, c_v2e, c_next_edge,
        c_edge_to, maxp, n);
    thrust::device_ptr<int32_t> has_color_ptr(has_color);
    pre_n_has = n_has;
    n_has = thrust::reduce(has_color_ptr, has_color_ptr + n, 0,
                           thrust::plus<int32_t>());
    CUDA::VivacePass4<<<n_block, threads_per_block_, 0, stream>>>(
        palette, psize, has_color, maxp, n);
    if (pre_n_has == n_has)
      ++stuck;
    else
      stuck = 0;
    if (stuck == 3) {
      stuck = 0;
      CUDA::VivacePassStuck<<<n_block, threads_per_block_, 0, stream>>>(
          palette, psize, has_color, maxp, n);
    }
  }

  thrust::device_ptr<int32_t> color_ptr(color);
  n_color_[l] =
      thrust::reduce(color_ptr, color_ptr + n, 0, thrust::maximum<int32_t>());
  n_color_[l] += 1;
}

// Gauss Seidel
void CudaElasobjMixedMG19::PerformGSIteration(int32_t& l,
                                              const int32_t max_iter,
                                              const real tol,
                                              cudaStream_t stream) {
  real r1;
  real minus_one = real(-1);
  real one = real(1);
  real* P = dP_[l];
  real* R = dR_[l];
  real* lhs = dlhs_[l];
  real minus_relax = real(-relaxation_);

  for (int k = 0; k < max_iter; ++k) {
    CheckCuda(cudaMemset(P, 0, sizeof(real) * dim_[l]));
    for (int c = 0; c < n_color_[l]; ++c) {
      if (l == n_layer_) {
        CUDA::
            GSAfIncMixed<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
                P, Af_->ddiag_val_, ddiag_add_[l], Af_->dbcsr_val_,
                Af_->dbcsr_row_, Af_->dbcsr_col_, R, dcolor_[l], c, relaxation_,
                n_vert_);
        if (self_collision_) {
          int n_block = (hashing_->n_colli_ + threads_per_block_ - 1) /
                        threads_per_block_;
          CUDA::SelfCollisionFineOffGSAPMixed<<<n_block, threads_per_block_, 0,
                                                stream>>>(
              R, dcolli_pairs_[l], dcolli_hessian_[l], dcolor_[l], c, P, -1,
              hashing_->n_colli_);
        }
      } else {
        CUDA::GSAsIncMixed<<<n_handle_[l], 16, 0, stream>>>(
            P, A_[l].ddiag_val_, ddiag_add_[l], A_[l].dbcsr_val_,
            A_[l].dbcsr_row_, A_[l].dbcsr_col_, R, dcolor_[l], c, relaxation_);
        if (self_collision_) {
          CUDA::SelfCollisionCoarseOffGSAPMixed<<<hashing_->n_colli_,
                                                  dim3(1, 16), 0, stream>>>(
              R, dcolli_pairs_[l], dcolli_hessian_[l], dcolor_[l], c, P, -1,
              hashing_->n_colli_);
        }
      }
    }
#ifdef REAL_AS_DOUBLE
    CheckCuda(cublasDaxpy(cublas_handle_, dim_[l], &one, P, 1, lhs, 1));
#else
    CheckCuda(cublasSaxpy(cublas_handle_, dim_[l], &one, P, 1, lhs, 1));
#endif
    ComputeAP(l, R, &minus_one, P, stream);
  }
}

void CudaElasobjMixedMG19::UpSample(int32_t& l, cudaStream_t stream) {
  real one = real(1);
  real zero = real(0);
  real minus_one = real(-1);
  if (l == n_layer_ - 1) {
    size_t buffer_size = 0;
    CheckCuda(cusparseSpMV_bufferSize(
        cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        Uf_->spmat_descr_, lhs_descr_[l], &zero, tmp_descr_[l + 1],
        CudaRealType, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
    void* buffer = nullptr;
    CheckCuda(cudaMalloc(&buffer, buffer_size));
    CheckCuda(cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &one, Uf_->spmat_descr_, lhs_descr_[l], &zero,
                           tmp_descr_[l + 1], CudaRealType,
                           CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    CheckCuda(cudaFree(buffer));
  } else {
    size_t buffer_size = 0;
    CheckCuda(cusparseSpMV_bufferSize(
        cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        U_[l].spmat_descr_, lhs_descr_[l], &zero, tmp_descr_[l + 1],
        CudaRealType, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
    void* buffer = nullptr;
    CheckCuda(cudaMalloc(&buffer, buffer_size));
    CheckCuda(cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &one, U_[l].spmat_descr_, lhs_descr_[l], &zero,
                           tmp_descr_[l + 1], CudaRealType,
                           CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    CheckCuda(cudaFree(buffer));
  }
  CheckCuda(cublasAxpyEx(cublas_handle_, dim_[l + 1], &one, CudaRealType,
                         dtmp_[l + 1], CudaRealType, 1, dlhs_[l + 1],
                         CudaRealType, 1, CudaRealType));
  ComputeAP(l + 1, dR_[l + 1], &minus_one, dtmp_[l + 1], stream);
  if (self_collision_)
    ComputeSelfcollisionOffAP(l + 1, dR_[l + 1], &minus_one, dtmp_[l + 1],
                              stream);
  ++l;
}

void CudaElasobjMixedMG19::DownSample(int32_t& l, cudaStream_t stream) {
  real one = real(1);
  real zero = real(0);
  if (l == n_layer_) {
    size_t buffer_size = 0;
    CheckCuda(cusparseSpMV_bufferSize(
        cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        Uf_->trans_descr_, R_descr_[l], &zero, rhs_descr_[l - 1], CudaRealType,
        CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
    void* buffer = nullptr;
    CheckCuda(cudaMalloc(&buffer, buffer_size));
    CheckCuda(cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &one, Uf_->trans_descr_, R_descr_[l], &zero,
                           rhs_descr_[l - 1], CudaRealType,
                           CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    CheckCuda(cudaFree(buffer));
  } else {
    size_t buffer_size = 0;
    CheckCuda(cusparseSpMV_bufferSize(
        cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        U_[l - 1].trans_descr_, R_descr_[l], &zero, rhs_descr_[l - 1],
        CudaRealType, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
    void* buffer = nullptr;
    CheckCuda(cudaMalloc(&buffer, buffer_size));
    CheckCuda(cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &one, U_[l - 1].trans_descr_, R_descr_[l], &zero,
                           rhs_descr_[l - 1], CudaRealType,
                           CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    CheckCuda(cudaFree(buffer));
  }
  CheckCuda(cudaMemcpyAsync(dR_[l - 1], drhs_[l - 1],
                            sizeof(real) * dim_[l - 1],
                            cudaMemcpyDeviceToDevice, stream));
  CheckCuda(cudaMemset(dlhs_[l - 1], 0, sizeof(real) * dim_[l - 1]));
  --l;
}

void CudaElasobjMixedMG19::DirectSolve(int32_t& l, cudaStream_t stream) {
  real minus_one = real(-1);
  CheckCuda(cudaMemcpyAsync(dlhs_[l], drhs_[l], sizeof(real) * dim_[l],
                            cudaMemcpyDeviceToDevice, stream));
  size_t host_buffer_size = 0;
  size_t dev_buffer_size = 0;
  CheckCuda(cusolverDnXsytrs_bufferSize(
      cusolverDn_handle_, CUBLAS_FILL_MODE_LOWER, dim_[l], 1, CudaRealType,
      A_[l].dldlt_val_, dim_[l], A_[l].dldlt_ipiv_64_, CudaRealType, dlhs_[l],
      dim_[l], &dev_buffer_size, &host_buffer_size));
  void* host_buffer = ::operator new(host_buffer_size);
  void* dev_buffer = nullptr;
  CheckCuda(cudaMalloc(&dev_buffer, dev_buffer_size));
  CheckCuda(cusolverDnXsytrs(cusolverDn_handle_, CUBLAS_FILL_MODE_LOWER,
                             dim_[l], 1, CudaRealType, A_[l].dldlt_val_,
                             dim_[l], A_[l].dldlt_ipiv_64_, CudaRealType,
                             dlhs_[l], dim_[l], dev_buffer, dev_buffer_size,
                             host_buffer, host_buffer_size, A_[l].dldlt_info_));
  CheckCuda(cudaMemsetAsync(dR_[l], 0, sizeof(real) * dim_[l], stream));
  delete[] host_buffer;
  CheckCuda(cudaFree(dev_buffer));
}

void CudaElasobjMixedMG19::DirectSolveShur(int32_t& l, cudaStream_t stream) {
  int n_block = (dim_[l] + threads_per_block_ - 1) / threads_per_block_;
  CUDA::CoarseAOS2SOAMixed<<<n_block, threads_per_block_, 0, stream>>>(
      dP_[l], drhs_[l], dim_[l]);
  int n = n_handle_[l];
  real* f = dP_[l];
  real* g = dP_[l] + 12 * n;
  real one = real(1.);
  real zero = real(0.);
  real minus_one = real(-1.);
#ifdef REAL_AS_DOUBLE
  CheckCuda(cublasDgemv(cublas_handle_, CUBLAS_OP_N, 12 * n, 4 * n, &one, dGt_,
                        12 * n, g, 1, &one, f, 1));
#else
  CheckCuda(cublasSgemv(cublas_handle_, CUBLAS_OP_N, 12 * n, 4 * n, &one, dGt_,
                        12 * n, g, 1, &one, f, 1));
#endif
  CheckCuda(cusolverDnXpotrs(cusolverDn_handle_, cusolverDn_params_,
                             CUBLAS_FILL_MODE_LOWER, 12 * n, 1, CudaRealType,
                             A_[l].dldlt_val_, dim_[l], CudaRealType, f,
                             dim_[l], dchol_info_));
  CheckCuda(cudaMemcpyAsync(dZ_[l], dP_[l], sizeof(real) * dim_[l],
                            cudaMemcpyDeviceToDevice, stream));
#ifdef REAL_AS_DOUBLE
  CheckCuda(cublasDgemv(cublas_handle_, CUBLAS_OP_T, 12 * n, 4 * n, &one, dGt_,
                        12 * n, f, 1, &zero, g, 1));
  CheckCuda(cublasDsymv(cublas_handle_, CUBLAS_FILL_MODE_LOWER, 4 * n,
                        &minus_one, dC_inv_, 4 * n, dZ_[l] + 12 * n, 1, &one, g,
                        1));
#else
  CheckCuda(cublasSgemv(cublas_handle_, CUBLAS_OP_T, 12 * n, 4 * n, &one, dGt_,
                        12 * n, f, 1, &zero, g, 1));
  CheckCuda(cublasSsymv(cublas_handle_, CUBLAS_FILL_MODE_LOWER, 4 * n,
                        &minus_one, dC_inv_, 4 * n, dZ_[l] + 12 * n, 1, &one, g,
                        1));
#endif
  CUDA::CoarseSOA2AOSMixed<<<n_block, threads_per_block_, 0, stream>>>(
      dlhs_[l], dP_[l], dim_[l]);
  CheckCuda(cudaMemsetAsync(dR_[l], 0, sizeof(real) * dim_[l], stream));
}

real CudaElasobjMixedMG19::ComputeDistortionEnergy(cudaStream_t stream) {
  real E;
  real* dE;
  CheckCuda(cudaMalloc(&dE, sizeof(real)));
  CheckCuda(cudaMemset(dE, 0, sizeof(real)));
  CUDA::ComputeDistortionEnergyMixed<<<tet_blocks_, tet_threads_per_block_, 0,
                                       stream>>>(
      dvertices_, dindices_, dvolumes_, dDm_inv_, dE, mu_, n_tet_);
  CheckCuda(cudaMemcpy(&E, dE, sizeof(real), cudaMemcpyDeviceToHost));
  CheckCuda(cudaFree(dE));
  return E;
}

void CudaElasobjMixedMG19::ComputeAP(int32_t l, real* AP, const real* alpha,
                                     const real* P, cudaStream_t stream) {
  real one = real(1);
  if (l == n_layer_) {
    if (Af_->as_LDU_) {
#ifdef REAL_AS_DOUBLE
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->low_bnnz_, alpha, descr_,
                               Af_->dlow_bcsr_val_, Af_->dlow_bcsr_row_,
                               Af_->dlow_bcsr_col_, 4, P, &one, AP));
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->diag_bnnz_, alpha, descr_,
                               Af_->ddiag_bcsr_val_, Af_->ddiag_bcsr_row_,
                               Af_->ddiag_bcsr_col_, 4, P, &one, AP));
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->up_bnnz_, alpha, descr_,
                               Af_->dup_bcsr_val_, Af_->dup_bcsr_row_,
                               Af_->dup_bcsr_col_, 4, P, &one, AP));
#else
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->low_bnnz_, alpha, descr_,
                               Af_->dlow_bcsr_val_, Af_->dlow_bcsr_row_,
                               Af_->dlow_bcsr_col_, 4, P, &one, AP));
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->diag_bnnz_, alpha, descr_,
                               Af_->ddiag_bcsr_val_, Af_->ddiag_bcsr_row_,
                               Af_->ddiag_bcsr_col_, 4, P, &one, AP));
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->up_bnnz_, alpha, descr_,
                               Af_->dup_bcsr_val_, Af_->dup_bcsr_row_,
                               Af_->dup_bcsr_col_, 4, P, &one, AP));
#endif
      int n_block = (n_vert_ * 4 + threads_per_block_ - 1) / threads_per_block_;
      CUDA::AfDiagMulVecMixed<<<n_block, threads_per_block_, 0, stream>>>(
          AP, ddiag_add_[l], P, *alpha, n_vert_);
    } else if (Af_->as_dense_) {
#ifdef REAL_AS_DOUBLE
      CheckCuda(cublasDgemv(cublas_handle_, CUBLAS_OP_N, Af_->rows_, Af_->cols_,
                            alpha, Af_->dden_val_, Af_->rows_, P, 1, &one, AP,
                            1));
#else
      CheckCuda(cublasSgemv(cublas_handle_, CUBLAS_OP_N, Af_->rows_, Af_->cols_,
                            alpha, Af_->dden_val_, Af_->rows_, P, 1, &one, AP,
                            1));
#endif
    } else {
#ifdef REAL_AS_DOUBLE
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->bnnz_, alpha, descr_,
                               Af_->dbcsr_val_, Af_->dbcsr_row_,
                               Af_->dbcsr_col_, 4, P, &one, AP));
#else
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->bnnz_, alpha, descr_,
                               Af_->dbcsr_val_, Af_->dbcsr_row_,
                               Af_->dbcsr_col_, 4, P, &one, AP));
#endif
      int n_block = (n_vert_ * 4 + threads_per_block_ - 1) / threads_per_block_;
      CUDA::AfDiagMulVecMixed<<<n_block, threads_per_block_, 0, stream>>>(
          AP, ddiag_add_[l], P, *alpha, n_vert_);
    }
  } else {  // coarser level
    if (A_[l].as_dense_) {
#ifdef REAL_AS_DOUBLE
      CheckCuda(cublasDgemv(cublas_handle_, CUBLAS_OP_N, A_[l].rows_,
                            A_[l].cols_, alpha, A_[l].dden_val_, A_[l].rows_, P,
                            1, &one, AP, 1));
#else
      CheckCuda(cublasSgemv(cublas_handle_, CUBLAS_OP_N, A_[l].rows_,
                            A_[l].cols_, alpha, A_[l].dden_val_, A_[l].rows_, P,
                            1, &one, AP, 1));
#endif
    } else if (A_[l].as_LDU_) {
#ifdef REAL_AS_DOUBLE
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].low_bnnz_, alpha, descr_,
                               A_[l].dlow_bcsr_val_, A_[l].dlow_bcsr_row_,
                               A_[l].dlow_bcsr_col_, 16, P, &one, AP));
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].diag_bnnz_, alpha, descr_,
                               A_[l].ddiag_bcsr_val_, A_[l].ddiag_bcsr_row_,
                               A_[l].ddiag_bcsr_col_, 16, P, &one, AP));
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].up_bnnz_, alpha, descr_,
                               A_[l].dup_bcsr_val_, A_[l].dup_bcsr_row_,
                               A_[l].dup_bcsr_col_, 16, P, &one, AP));
#else
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].low_bnnz_, alpha, descr_,
                               A_[l].dlow_bcsr_val_, A_[l].dlow_bcsr_row_,
                               A_[l].dlow_bcsr_col_, 16, P, &one, AP));
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].diag_bnnz_, alpha, descr_,
                               A_[l].ddiag_bcsr_val_, A_[l].ddiag_bcsr_row_,
                               A_[l].ddiag_bcsr_col_, 16, P, &one, AP));
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].up_bnnz_, alpha, descr_,
                               A_[l].dup_bcsr_val_, A_[l].dup_bcsr_row_,
                               A_[l].dup_bcsr_col_, 16, P, &one, AP));
#endif
      int n_block =
          (n_handle_[l] * 16 + threads_per_block_ - 1) / threads_per_block_;
      CUDA::AsDiagMulVecMixed<<<n_block, threads_per_block_, 0, stream>>>(
          AP, ddiag_add_[l], P, *alpha, n_handle_[l]);
    } else {
#ifdef REAL_AS_DOUBLE
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].bnnz_, alpha, descr_,
                               A_[l].dbcsr_val_, A_[l].dbcsr_row_,
                               A_[l].dbcsr_col_, 16, P, &one, AP));
#else
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].bnnz_, alpha, descr_,
                               A_[l].dbcsr_val_, A_[l].dbcsr_row_,
                               A_[l].dbcsr_col_, 16, P, &one, AP));
#endif
      int n_block =
          (n_handle_[l] * 16 + threads_per_block_ - 1) / threads_per_block_;
      CUDA::AsDiagMulVecMixed<<<n_block, threads_per_block_, 0, stream>>>(
          AP, ddiag_add_[l], P, *alpha, n_handle_[l]);
    }
  }
}

void CudaElasobjMixedMG19::ComputeSelfcollisionOffAP(int32_t l, real* AP,
                                                     const real* alpha,
                                                     const real* P,
                                                     cudaStream_t stream) {
  if (l == n_layer_ && !Af_->as_dense_) {
    int n_block =
        (hashing_->n_colli_ + threads_per_block_ - 1) / threads_per_block_;
    CUDA::
        SelfCollisionFineOffAPMixed<<<n_block, threads_per_block_, 0, stream>>>(
            AP, dcolli_pairs_[l], dcolli_hessian_[l], P, *alpha,
            hashing_->n_colli_);
  } else if (l < n_layer_ && !A_[l].as_dense_) {
    CUDA::SelfCollisionCoarseOffAPMixed<<<hashing_->n_colli_, dim3(1, 16), 0,
                                          stream>>>(AP, dcolli_pairs_[l],
                                                    dcolli_hessian_[l], P,
                                                    *alpha, hashing_->n_colli_);
  }
}

};  // namespace Rain