// #include <windows.h>

#include <fstream>

#include "cudaelasobjmg19.h"
#include "cudasim/explicit.h"
#include "cudasim/kinematiccollision.h"
#include "cudasim/multigrid.h"
#include "cudasim/selfcollision.h"
#include "imgui.h"

namespace Rain {
void CudaElasobjMG19::ShowUI() {
  CudaElasticObject::ShowUI();
  CUDA::ComputeTetVolume<<<tet_blocks_, tet_threads_per_block_>>>(
      dvertices_, dindices_, dtet_vol_, n_tet_);
  thrust::device_ptr<real> vol_ptr(dtet_vol_);
  real vol =
      thrust::reduce(vol_ptr, vol_ptr + n_tet_, 0., thrust::plus<real>());
  real vol_diff = (vol - rest_vol_) / rest_vol_ * 100.;
  ImGui::Text("Volume Change: %.2f%%", vol_diff);
  if (self_collision_) {
    ImGui::Text("#collision pairs: %d", hashing_->n_colli_);
    for (int l = n_layer_; l > 0; --l) {
      ImGui::Text("#color at layer %d: %d", l, n_color_[l]);
    }
  }
  ImGui::Text("line search iters: %d", n_line_iter_);
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

void CudaElasobjMG19::Update(cudaStream_t stream, float Dt, const Vec3& grav,
                             real damping, real kinematic_penalty,
                             real self_penalty,
                             const std::vector<CudaKinematicObject*>& kobjs,
                             uint32_t n_substep, real substep_size,
                             uint32_t n_frame) {
  
  // Hack for the bunny example
  // OscillationHack<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
  //     dfixed_, dfixed_verts_, dvertices_, n_frame, dt_, n_vert_);

  if (n_joint_ > 0) UpdateSkeleton();
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
      real before_energy = 0;
      if (line_search_)
        before_energy = ComputeEnergy(dinertia_verts_, dvertices_,
                                      kinematic_penalty, kobjs, stream);
      if (elastic_model_->type_ == ElasticModelType::NeoHookean) {
        CheckCuda(cudaMemsetAsync(Af_->dbcoo_val_, 0, sizeof(real) * Af_->nnz_,
                                  stream));
        CUDA::HessianNeoHookeanClamped<<<tet_blocks_, tet_threads_per_block_, 0,
                                         stream>>>(
            dvertices_, dindices_, dDm_inv_, dvolumes_, dt2off_,
            Af_->dbcoo_val_, elastic_model_->mu_, elastic_model_->lambda_,
            n_tet_);
        if (!quasi_static_)
          CUDA::InteriaHessian<<<vert_blocks_, vert_threads_per_block_, 0,
                                 stream>>>(dmasses_, dd2off_, Af_->dbcoo_val_,
                                           1 / dt_, n_vert_);
        Af_->UpdateDiag(stream);
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
          int32_t n_red_block = (9 * n + n_red_thread - 1) / n_red_thread;
          CUDA::AfKroneckerXXtHalf<<<n_red_block, n_red_thread * 16>>>(
              Af_->dbcoo_val_, dXXt_, dred_half_off_[l], dAf_XXt_, n);
          if (A_as_dense_[l]) {  // Af to dense reduction
            CUDA::AfDenseReduction<<<n_red_block, n_red_thread * 16>>>(
                dAf_XXt_, dred_off_[l], dred_half_off_[l], A_[l].dden_val_, n,
                n_handle_[l]);
            CUDA::AsDenseMirror<<<A_[l].low_bnnz_, 144>>>(
                A_[l].dden_val_, dA_low_off_[l], n_handle_[l], A_[l].low_bnnz_);
          } else {  // Af to sparse reduction
            CUDA::AfSparseReduction<<<n_red_block, n_red_thread * 16>>>(
                dAf_XXt_, dred_off_[l], dred_half_off_[l], A_[l].dbcoo_val_, n);
            CUDA::AsSparseMirror<<<A_[l].low_bnnz_, 144>>>(
                A_[l].dbcoo_val_, dA_low_off_[l], dA_mirror_off_[l],
                A_[l].low_bnnz_);
          }
        } else {
          uint32_t n = A_[l + 1].bnnz_;
          if (l || !A_[l].as_dense_) {  // As to sparse reduction
            CUDA::AsSparseReduction<<<n, 144>>>(
                A_[l + 1].dbcoo_val_, dred_off_[l], A_[l].dbcoo_val_, n);
          } else {  // As to dense reduction
            CUDA::AsDenseReduction<<<n, 144>>>(A_[l + 1].dbcoo_val_,
                                               dred_off_[l], A_[l].dden_val_,
                                               n_handle_[l], n);
          }
        }
        A_[l].UpdateDiag(stream);
      }

      CheckCuda(cudaMemset(dtet_grad_, 0, sizeof(real) * n_tet_ * 12));
      if (elastic_model_->type_ == ElasticModelType::PD) {
        CUDA::TetGradientPD<<<tet_blocks_, tet_threads_per_block_, 0, stream>>>(
            dvertices_, dindices_, dDm_inv_, dvolumes_, dtet_grad_,
            elastic_model_->mu_, n_tet_);
      } else if (elastic_model_->type_ == ElasticModelType::NeoHookean) {
        CUDA::TetGradientNeoHookean<<<tet_blocks_, tet_threads_per_block_, 0,
                                      stream>>>(
            dvertices_, dindices_, dDm_inv_, dvolumes_, dtet_grad_,
            elastic_model_->mu_, elastic_model_->lambda_, n_tet_);
      }

      CheckCuda(cudaMemset(drhs_[n_layer_], 0, sizeof(real) * dim_[n_layer_]));
      if (obj_->selected_idx_.has_value()) {
        // std::cout << obj_->selected_idx_.value() << std::endl;
        if constexpr (reordered_) {
          CUDA::EnergyGradientWithCtrlReordered<<<
              vert_blocks_, vert_threads_per_block_, 0, stream>>>(
              dtet_grad_, dv2t_ids_, dv2t_off_, dfixed_, dfixed_verts_,
              dvertices_, dm2h_[n_layer_], drhs_[n_layer_], control_mag_,
              obj_->selected_idx_.value(), obj_->control_pos_, n_vert_);
        } else {
          CUDA::EnergyGradientWithCtrl<<<vert_blocks_, vert_threads_per_block_,
                                         0, stream>>>(
              dtet_grad_, dv2t_ids_, dv2t_off_, dfixed_, dfixed_verts_,
              dvertices_, drhs_[n_layer_], control_mag_,
              obj_->selected_idx_.value(), obj_->control_pos_, n_vert_);
        }
      } else {
        if constexpr (reordered_) {
          CUDA::EnergyGradientReordered<<<vert_blocks_, vert_threads_per_block_,
                                          0, stream>>>(
              dtet_grad_, dv2t_ids_, dv2t_off_, dfixed_, dfixed_verts_,
              dvertices_, dm2h_[n_layer_], drhs_[n_layer_], control_mag_,
              n_vert_);
        } else {
          CUDA::EnergyGradient<<<vert_blocks_, vert_threads_per_block_, 0,
                                 stream>>>(
              dtet_grad_, dv2t_ids_, dv2t_off_, dfixed_, dfixed_verts_,
              dvertices_, dmasses_, drhs_[n_layer_], control_mag_, n_vert_);
        }
      }

      if (!quasi_static_) {
        if constexpr (reordered_) {
          CUDA::InertiaGradientReordered<<<
              vert_blocks_, vert_threads_per_block_, 0, stream>>>(
              dinertia_verts_, dvertices_, dm2h_[n_layer_], dmasses_,
              drhs_[n_layer_], 1 / dt_, n_vert_);
        } else {
          CUDA::InertiaGradient<<<vert_blocks_, vert_threads_per_block_, 0,
                                  stream>>>(dinertia_verts_, dvertices_,
                                            dmasses_, drhs_[n_layer_], 1 / dt_,
                                            n_vert_);
        }
      }

      if (n_joint_ > 0) {
        CUDA::SkeletonGradient<<<vert_blocks_, vert_threads_per_block_, 0,
                                 stream>>>(drhs_[n_layer_], dvertices_,
                                           dbone_ids_, djoint_trans_,
                                           dlocal_pos_, control_mag_, n_vert_);
      }

      for (int l = 0; l < n_layer_; ++l)
        CheckCuda(
            cudaMemset(ddiag_add_[l], 0, sizeof(real) * n_handle_[l] * 144));
      CheckCuda(
          cudaMemset(ddiag_add_[n_layer_], 0, sizeof(real) * n_vert_ * 9));

      if (n_joint_ > 0) {
        CUDA::UpdateAfDiagSkeleton<<<vert_blocks_, vert_threads_per_block_, 0,
                                     stream>>>(ddiag_add_[n_layer_], dbone_ids_,
                                               control_mag_, n_vert_);
      }

      if (obj_->selected_idx_.has_value()) {
        if constexpr (reordered_) {
          CUDA::UpdateAfDiagWithCtrlReordered<<<
              vert_blocks_, vert_threads_per_block_, 0, stream>>>(
              ddiag_add_[n_layer_], dh2m_[n_layer_], dfixed_, control_mag_,
              obj_->selected_idx_.value(), n_vert_);
        } else {
          CUDA::UpdateAfDiagWithCtrl<<<vert_blocks_, vert_threads_per_block_, 0,
                                       stream>>>(
              ddiag_add_[n_layer_], dfixed_, control_mag_,
              obj_->selected_idx_.value(), n_vert_);
        }
      } else {
        if constexpr (reordered_) {
          CUDA::UpdateAfDiagReordered<<<vert_blocks_, vert_threads_per_block_,
                                        0, stream>>>(ddiag_add_[n_layer_],
                                                     dh2m_[n_layer_], dfixed_,
                                                     control_mag_, n_vert_);
        } else {
          CUDA::UpdateAfDiag<<<vert_blocks_, vert_threads_per_block_, 0,
                               stream>>>(ddiag_add_[n_layer_], dfixed_,
                                         control_mag_, n_vert_);
        }
      }

      if (self_collision_) {
        SelfCollision(self_penalty, stream);
        for (int l = n_layer_; l > 0; --l) {
          Vivace(l, stream);
        }
      }

      KinematicCollision(kinematic_penalty, kobjs, stream);

      for (int l = n_layer_ - 1; l >= 0; --l) {
        if (l == n_layer_ - 1) {
          real one = real(1.);
          CheckCuda(cublasAxpyEx(cublas_handle_, n_handle_[l] * 144, &one,
                                 CudaRealType, dUtAU_diag_fix_, CudaRealType, 1,
                                 ddiag_add_[l], CudaRealType, 1, CudaRealType));
          int32_t reduction_blocks =
              (9 * n_vert_ + threads_per_block_ - 1) / threads_per_block_;
          CUDA::UpdateUtAUDiag<<<reduction_blocks, threads_per_block_, 0,
                                 stream>>>(ddiag_add_[l], ddiag_add_[n_layer_],
                                           ddiag_XXt_, dmfine2coarse_[l],
                                           n_vert_);
        } else {  // UltAUl
          int32_t UltAUl_blocks =
              (n_handle_[l + 1] + UltAUl_threads_per_block_ - 1) /
              UltAUl_threads_per_block_;
          CUDA::UpdateUltAUlDiag<<<
              UltAUl_blocks, dim3(UltAUl_threads_per_block_, 144), 0, stream>>>(
              ddiag_add_[l], ddiag_add_[l + 1], dmfine2coarse_[l],
              n_handle_[l + 1]);
        }

        if (l == 0 && A_as_dense_[l]) {
          CheckCuda(cudaMemcpyAsync(A_[l].dchol_val_, A_[l].dden_val_,
                                    sizeof(real) * A_[l].rows_ * A_[l].rows_,
                                    cudaMemcpyDeviceToDevice, stream));
          int32_t UltAUl_blocks =
              (n_handle_[l] + UltAUl_threads_per_block_ - 1) /
              UltAUl_threads_per_block_;
          CUDA::UpdateDenseDiag<<<UltAUl_blocks,
                                  dim3(UltAUl_threads_per_block_, 144)>>>(
              A_[l].dchol_val_, ddiag_add_[l], n_handle_[l], A_[l].rows_);
          if (self_collision_) {
            CUDA::SelfCollisionCoarseDenseHessian<<<
                hashing_->n_colli_, dim3(1, 144), 0, stream>>>(
                A_[l].dden_val_, dcolli_pairs_[l], dcolli_hessian_[l],
                dim_[l], hashing_->n_colli_);
          }
          real one = 1.0;
          CheckCuda(cublasAxpyEx(cublas_handle_, dim_[l], &one, CudaRealType,
                                 A_[l].dchol_fixed_, CudaRealType, 1,
                                 A_[l].dchol_val_, CudaRealType, dim_[l] + 1,
                                 CudaRealType));
          CheckCuda(cusolverDnXpotrf(
              cusolverDn_handle_, cusolverDn_params_, CUBLAS_FILL_MODE_LOWER,
              dim_[l], CudaRealType, A_[l].dchol_val_, dim_[l], CudaRealType,
              A_[l].chol_dev_buffer_, A_[l].chol_dev_buffer_size_,
              A_[l].chol_host_buffer_, A_[l].chol_host_buffer_size_,
              A_[l].dchol_info_));
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
          case MGOpType::Jacobi:
            PerformJacobiIteration(cur_layer, op.max_iter, tol, stream);
            break;
          case MGOpType::DS:
            DownSample(cur_layer, stream);
            break;
          case MGOpType::Direct:
            DirectSolve(cur_layer, stream);
            break;
          case MGOpType::US:
            UpSample(cur_layer, stream);
            break;
          default:
            break;
        }
      }

      if (line_search_) {
        real alpha = 1;
        const real c = 0.5;
        const real tau = 0.5;
        const int max_line_search = 3;
        real m = 0;
#ifdef REAL_AS_DOUBLE
        CheckCuda(cublasDdot(cublas_handle_, n_vert_ * 3, dlhs_[n_layer_], 1,
                             drhs_[n_layer_], 1, &m));
#else
        CheckCuda(cublasSdot(cublas_handle_, n_vert_ * 3, dlhs_[n_layer_], 1,
                             drhs_[n_layer_], 1, &m));
#endif
        m = c * m;
        real old_alpha = 0;
        for (n_line_iter_ = 0; n_line_iter_ < max_line_search; ++n_line_iter_) {
          if constexpr (reordered_) {
            CUDA::UpdatedXReordered<<<vert_blocks_, vert_threads_per_block_, 0,
                                      stream>>>(dvertices_, alpha - old_alpha,
                                                dlhs_[n_layer_],
                                                dm2h_[n_layer_], n_vert_);
          } else {
            CUDA::
                UpdatedX<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
                    dvertices_, alpha - old_alpha, dlhs_[n_layer_], n_vert_);
          }
          real alpha_energy = ComputeEnergy(dinertia_verts_, dvertices_,
                                            kinematic_penalty, kobjs, stream);
          if (before_energy - alpha_energy >= alpha * m) break;
          old_alpha = alpha;
          alpha *= tau;
        }
      } else {  // no line search
        if constexpr (reordered_) {
          CUDA::UpdatedXReordered<<<vert_blocks_, vert_threads_per_block_, 0,
                                    stream>>>(dvertices_, 1.0, dlhs_[n_layer_],
                                              dm2h_[n_layer_], n_vert_);
        } else {
          CUDA::UpdatedX<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
              dvertices_, 1., dlhs_[n_layer_], n_vert_);
        }
      }
    }
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
    obj_->WriteObj(obj_->name_ + std::to_string(n_frame) + ".obj");
  }
  output_frame_data_ = false;
}

void CudaElasobjMG19::UpdateXXt(const Vec3* X, cudaStream_t stream) {
  if (reordered_) {
    // diag_XXt
    CUDA::UpdateDiagXXtReordered<<<vert_blocks_, vert_threads_per_block_, 0,
                                   stream>>>(X, dm2h_[n_layer_], ddiag_XXt_,
                                             n_vert_);
    // XXt
    int blocks = (Af_->bnnz_ + threads_per_block_ - 1) / threads_per_block_;
    CUDA::UpdateXXtReordered<<<blocks, threads_per_block_, 0, stream>>>(
        X, dm2h_[n_layer_], Af_->dbcoo_row_, Af_->dbcoo_col_, dXXt_,
        Af_->bnnz_);
  } else {
    // diag_XXt
    CUDA::UpdateDiagXXt<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
        X, ddiag_XXt_, n_vert_);
    // XXt
    int blocks = (Af_->bnnz_ + threads_per_block_ - 1) / threads_per_block_;
    CUDA::UpdateXXt<<<blocks, threads_per_block_, 0, stream>>>(
        X, Af_->dbcoo_row_, Af_->dbcoo_col_, dXXt_, Af_->bnnz_);
  }
}

void CudaElasobjMG19::UpdateU(const Vec3* X, cudaStream_t stream) {
  if (reordered_) {
    CUDA::
        UpdateUReordered<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
            X, dm2h_[n_layer_], Uf_->dcsr_val_, n_vert_);
  } else {
    CUDA::UpdateU<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
        X, Uf_->dcsr_val_, n_vert_);
  }
}

void CudaElasobjMG19::KinematicCollision(
    real k_penalty, const std::vector<CudaKinematicObject*>& kobjs,
    cudaStream_t stream) {
  for (const auto& obj : kobjs) {
    if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::None) {
      // TODO: explicit mesh
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Sphere) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      real r = params[3];
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      if constexpr (reordered_) {
        CUDA::KinematicCollisionSphereReordered<<<
            vert_blocks_, vert_threads_per_block_, 0, stream>>>(
            dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], dh2m_[n_layer_],
            k_penalty, c, r, n_vert_);
      } else {
        CUDA::KinematicCollisionSphere<<<vert_blocks_, vert_threads_per_block_,
                                         0, stream>>>(
            dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], k_penalty, c, r,
            n_vert_);
      }
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Torus) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      Vec3 d{params[3], params[4], params[5]};
      real a = params[6];
      real r = params[7];
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      d = (obj->rotation_ * d).normalized();
      if constexpr (reordered_) {
        CUDA::KinematicCollisionTorusReordered<<<
            vert_blocks_, vert_threads_per_block_, 0, stream>>>(
            dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], dh2m_[n_layer_],
            k_penalty, c, d, a, r, n_vert_);
      } else {
        CUDA::KinematicCollisionTorus<<<vert_blocks_, vert_threads_per_block_,
                                        0, stream>>>(
            dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], k_penalty, c, d,
            a, r, n_vert_);
      }
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Cylinder) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      Vec3 d{params[3], params[4], params[5]};
      real r = params[6];
      real h = params[7];
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      d = (obj->rotation_ * d).normalized();
      if constexpr (reordered_) {
        CUDA::KinematicCollisionCylinderReordered<<<
            vert_blocks_, vert_threads_per_block_, 0, stream>>>(
            dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], dh2m_[n_layer_],
            k_penalty, c, d, r, h, n_vert_);
      } else {
        CUDA::KinematicCollisionCylinder<<<
            vert_blocks_, vert_threads_per_block_, 0, stream>>>(
            dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], k_penalty, c, d,
            r, h, n_vert_);
      }
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Plane) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      Vec3 n{params[3], params[4], params[5]};
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      n = (obj->rotation_ * n).normalized();
      if constexpr (reordered_) {
        CUDA::KinematicCollisionPlaneReordered<<<
            vert_blocks_, vert_threads_per_block_, 0, stream>>>(
            dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], dh2m_[n_layer_],
            k_penalty, c, n, n_vert_);
      } else {
        CUDA::KinematicCollisionPlane<<<vert_blocks_, vert_threads_per_block_,
                                        0, stream>>>(
            dvertices_, drhs_[n_layer_], ddiag_add_[n_layer_], k_penalty, c, n,
            n_vert_);
      }
    }
  }
}

void CudaElasobjMG19::SelfCollision(real k_penalty, cudaStream_t stream) {
  hashing_->Hashing(stream);
  hashing_->GetVertTetCollisionList(stream);
  int32_t n_block =
      (hashing_->n_colli_ + threads_per_block_ - 1) / threads_per_block_;
  CUDA::MakeSelfCollision<<<n_block, threads_per_block_, 0, stream>>>(
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
    CUDA::SelfCollisionFineOffReduction<<<n_block, threads_per_block_, 0,
                                               stream>>>(
        dcolli_pairs_[l], dcolli_hessian_[l], dcolli_pairs_[l + 1],
        dcolli_hessian_[l + 1], drest_verts_, dhandle_[l + 1], dhandle_ids_,
        hashing_->n_colli_);
    CUDA::SelfCollisionFineDiagReduction<<<n_block, threads_per_block_, 0,
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
    CUDA::SelfCollisionCoarseOffReduction<<<hashing_->n_colli_,
                                                 dim3(1, 144), 0, stream>>>(
        dcolli_pairs_[l], dcolli_hessian_[l], dcolli_pairs_[l + 1],
        dcolli_hessian_[l + 1], dhandle_[l + 1], hashing_->n_colli_);
    CUDA::SelfCollisionCoarseDiagReduction<<<hashing_->n_colli_,
                                                  dim3(1, 144), 0, stream>>>(
        ddiag_add_[l], dcolli_pairs_[l + 1], dcolli_hessian_[l + 1],
        dhandle_[l + 1], hashing_->n_colli_);
    CUDA::BuildSelfCollisionGraph<<<n_block, threads_per_block_, 0, stream>>>(
        dcolli_pairs_[l], dcolli_v2e_[l], dcolli_next_edge_[l],
        dcolli_edge_to_[l], hashing_->n_colli_);
  }
}

void CudaElasobjMG19::Vivace(int32_t l, cudaStream_t stream) {
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

real CudaElasobjMG19::ComputeEnergy(
    const Vec3* inertia_X, const Vec3* X, const real k_penalty,
    const std::vector<CudaKinematicObject*>& kobjs, cudaStream_t stream) {
  real E;
  real* dE;
  CheckCuda(cudaMalloc(&dE, sizeof(real)));
  CheckCuda(cudaMemset(dE, 0, sizeof(real)));
  if (elastic_model_->type_ == ElasticModelType::PD) {
    CUDA::EnergyPD<<<tet_blocks_, threads_per_block_, 0, stream>>>(
        X, dindices_, dDm_inv_, dvolumes_, elastic_model_->mu_, dE, n_tet_);
  } else if (elastic_model_->type_ == ElasticModelType::NeoHookean) {
    CUDA::EnergyNeoHookean<<<tet_blocks_, threads_per_block_, 0, stream>>>(
        X, dindices_, dDm_inv_, dvolumes_, elastic_model_->mu_,
        elastic_model_->lambda_, dE, n_tet_);
  }
  if (obj_->selected_idx_.has_value()) {
    CUDA::EnergyFixedWithCtrl<<<vert_blocks_, threads_per_block_, 0, stream>>>(
        dfixed_, dfixed_verts_, X, control_mag_, obj_->selected_idx_.value(),
        obj_->control_pos_, dE, n_vert_);
  } else {
    CUDA::EnergyFixed<<<vert_blocks_, threads_per_block_, 0, stream>>>(
        dfixed_, dfixed_verts_, X, control_mag_, dE, n_vert_);
  }
  if (!quasi_static_)
    CUDA::EnergyInertia<<<vert_blocks_, threads_per_block_, 0, stream>>>(
        inertia_X, X, dmasses_, 1 / dt_, dE, n_vert_);
  KinematicCollisionEnergy(k_penalty, kobjs, dE, stream);
  CheckCuda(cudaMemcpy(&E, dE, sizeof(real), cudaMemcpyDeviceToHost));
  CheckCuda(cudaFree(dE));
  return E;
}

void CudaElasobjMG19::KinematicCollisionEnergy(
    real k_penalty, const std::vector<CudaKinematicObject*>& kobjs, real* E,
    cudaStream_t stream) {
  for (const auto& obj : kobjs) {
    if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::None) {
      // TODO: explicit mesh
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Sphere) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      real r = params[3];
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      CUDA::KinematicCollisionSphereEnergy<<<
          vert_blocks_, vert_threads_per_block_, 0, stream>>>(
          dvertices_, E, k_penalty, c, r, n_vert_);
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Torus) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      Vec3 d{params[3], params[4], params[5]};
      real a = params[6];
      real r = params[7];
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      d = (obj->rotation_ * d).normalized();

      CUDA::KinematicCollisionTorusEnergy<<<
          vert_blocks_, vert_threads_per_block_, 0, stream>>>(
          dvertices_, E, k_penalty, c, d, a, r, n_vert_);
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Cylinder) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      Vec3 d{params[3], params[4], params[5]};
      real r = params[6];
      real h = params[7];
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      d = (obj->rotation_ * d).normalized();
      CUDA::KinematicCollisionCylinderEnergy<<<
          vert_blocks_, vert_threads_per_block_, 0, stream>>>(
          dvertices_, E, k_penalty, c, d, r, h, n_vert_);
    } else if (obj->obj_->implicit_geo_type_ == ImplicitGeoType::Plane) {
      const auto& params = obj->obj_->implicit_geo_params_;
      Vec3 c{params[0], params[1], params[2]};
      Vec3 n{params[3], params[4], params[5]};
      c = obj->rotation_ * (c - obj->pivot_) + obj->translation_;
      n = (obj->rotation_ * n).normalized();
      CUDA::KinematicCollisionPlaneEnergy<<<
          vert_blocks_, vert_threads_per_block_, 0, stream>>>(
          dvertices_, E, k_penalty, c, n, n_vert_);
    }
  }
}

void CudaElasobjMG19::ComputeGradientReordered(const Vec3* inertia_X,
                                               const Vec3* X, real* out,
                                               cudaStream_t stream) {
  // side effect: dtet_grad_
  // minus gradient
  if (elastic_model_->type_ == ElasticModelType::PD) {
    CUDA::TetGradientPD<<<tet_blocks_, tet_threads_per_block_, 0, stream>>>(
        X, dindices_, dDm_inv_, dvolumes_, dtet_grad_, elastic_model_->mu_,
        n_tet_);
  } else if (elastic_model_->type_ == ElasticModelType::NeoHookean) {
    CUDA::TetGradientNeoHookean<<<tet_blocks_, tet_threads_per_block_, 0,
                                  stream>>>(X, dindices_, dDm_inv_, dvolumes_,
                                            dtet_grad_, elastic_model_->mu_,
                                            elastic_model_->lambda_, n_tet_);
  }

  if (obj_->selected_idx_.has_value()) {
    CUDA::EnergyGradientWithCtrlReordered<<<
        vert_blocks_, vert_threads_per_block_, 0, stream>>>(
        dtet_grad_, dv2t_ids_, dv2t_off_, dfixed_, dfixed_verts_, X,
        dm2h_[n_layer_], out, control_mag_, obj_->selected_idx_.value(),
        obj_->control_pos_, n_vert_);
  } else {
    CUDA::EnergyGradientReordered<<<vert_blocks_, vert_threads_per_block_, 0,
                                    stream>>>(
        dtet_grad_, dv2t_ids_, dv2t_off_, dfixed_, dfixed_verts_, X,
        dm2h_[n_layer_], out, control_mag_, n_vert_);
  }

  CUDA::InertiaGradientReordered<<<vert_blocks_, vert_threads_per_block_, 0,
                                   stream>>>(inertia_X, X, dm2h_[n_layer_],
                                             dmasses_, out, 1 / dt_, n_vert_);
}

void CudaElasobjMG19::TestHessian(cudaStream_t stream) {
  // test Hessian in Af_, use gradient in rhs_
  std::cout << "---------Test Hessian---------" << std::endl;
  Vec3* dtmp_X;
  CheckCuda(cudaMalloc(&dtmp_X, sizeof(Vec3) * n_vert_));
  CheckCuda(cudaMemcpy(dtmp_X, dvertices_, sizeof(Vec3) * n_vert_,
                       cudaMemcpyDeviceToDevice));
  real* dtmp_grad;
  CheckCuda(cudaMalloc(&dtmp_grad, sizeof(real) * 3 * n_vert_));
  CheckCuda(cudaMemset(dtmp_grad, 0, sizeof(real) * 3 * n_vert_));
  Vec3* delta;
  delta = new Vec3[n_vert_];
  for (int i = 0; i < n_vert_; ++i) {
    delta[i] = 1e-8 * Vec3::Random();
  }
  real* ddelta;
  CheckCuda(cudaMalloc(&ddelta, sizeof(real) * 3 * n_vert_));
  CheckCuda(cudaMemcpy(ddelta, delta, sizeof(real) * 3 * n_vert_,
                       cudaMemcpyHostToDevice));
  real* dHx;
  CheckCuda(cudaMalloc(&dHx, sizeof(real) * 3 * n_vert_));
  CheckCuda(cudaMemset(dHx, 0, sizeof(real) * 3 * n_vert_));
  real one = 1.0;
  real minus_one = -1;
  ComputeAP(n_layer_, dHx, &minus_one, ddelta, stream);
  real gradient_norm = 0;
  CheckCuda(cublasNrm2Ex(cublas_handle_, 3 * n_vert_, drhs_[n_layer_],
                         CudaRealType, 1, &gradient_norm, CudaRealType,
                         CudaRealType));
  std::cout << "gradient normal: " << gradient_norm << std::endl;
  real analytical_diff = 0;
  CheckCuda(cublasNrm2Ex(cublas_handle_, 3 * n_vert_, dHx, CudaRealType, 1,
                         &analytical_diff, CudaRealType, CudaRealType));
  std::cout << "analytical gradient difference: " << analytical_diff
            << std::endl;
  CUDA::UpdatedXReordered<<<vert_blocks_, threads_per_block_, 0, stream>>>(
      dtmp_X, 1.0, ddelta, dm2h_[n_layer_], n_vert_);
  ComputeGradientReordered(dinertia_verts_, dtmp_X, dtmp_grad, stream);
  CheckCuda(cublasAxpyEx(cublas_handle_, 3 * n_vert_, &minus_one, CudaRealType,
                         drhs_[n_layer_], CudaRealType, 1, dtmp_grad,
                         CudaRealType, 1, CudaRealType));
  real numerical_diff = 0;
  CheckCuda(cublasNrm2Ex(cublas_handle_, 3 * n_vert_, dtmp_grad, CudaRealType,
                         1, &numerical_diff, CudaRealType, CudaRealType));
  std::cout << "numerical gradient difference: " << numerical_diff << std::endl;
  real analytical_numerical = 0;
  CheckCuda(cublasAxpyEx(cublas_handle_, 3 * n_vert_, &minus_one, CudaRealType,
                         dHx, CudaRealType, 1, dtmp_grad, CudaRealType, 1,
                         CudaRealType));
  CheckCuda(cublasNrm2Ex(cublas_handle_, 3 * n_vert_, dtmp_grad, CudaRealType,
                         1, &analytical_numerical, CudaRealType, CudaRealType));
  std::cout << "numerical-analytical residual: " << analytical_numerical
            << std::endl;
  CheckCuda(cudaFree(dtmp_X));
  CheckCuda(cudaFree(dtmp_grad));
  CheckCuda(cudaFree(ddelta));
  CheckCuda(cudaFree(dHx));
  delete[] delta;
  std::cout << "-------End Test Hessian-------" << std::endl;
}

void CudaElasobjMG19::PerformGSIteration(int32_t& l, const int32_t max_iter,
                                         const real tol, cudaStream_t stream) {
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
        CUDA::GSAfInc<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
            P, Af_->ddiag_val_, ddiag_add_[l], Af_->dbcsr_val_, Af_->dbcsr_row_,
            Af_->dbcsr_col_, R, dcolor_[l], c, n_vert_);
        if (self_collision_) {
          int n_block = (hashing_->n_colli_ + threads_per_block_ - 1) /
                        threads_per_block_;
          CUDA::SelfCollisionFineOffGSAP<<<n_block, threads_per_block_, 0,
                                                stream>>>(
              R, dcolli_pairs_[l], dcolli_hessian_[l], dcolor_[l], c, P, -1,
              hashing_->n_colli_);
        }
      } else {
        CUDA::GSAsInc<<<n_handle_[l], 12, 0, stream>>>(
            P, A_[l].ddiag_val_, ddiag_add_[l], A_[l].dbcsr_val_,
            A_[l].dbcsr_row_, A_[l].dbcsr_col_, R, dcolor_[l], c);
        if (self_collision_) {
          CUDA::SelfCollisionCoarseOffGSAP<<<hashing_->n_colli_,
                                                  dim3(1, 12), 0, stream>>>(
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

void CudaElasobjMG19::PerformJacobiIteration(const int32_t l,
                                             const int32_t max_iter,
                                             const real tol,
                                             cudaStream_t stream) {
  real* R = dR_[l];
  real* P = dP_[l];
  real* lhs = dlhs_[l];
  real minus_relax = real(-relaxation_);
  for (int k = 0; k < max_iter; ++k) {
    CheckCuda(cudaMemset(P, 0, sizeof(real) * dim_[l]));
    if (l == n_layer_) {
      CUDA::JacobiAf<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
          P, Af_->ddiag_val_, ddiag_add_[l], R, n_vert_);
    } else {
      CUDA::JacobiAs<<<n_handle_[l], 12, 0, stream>>>(P, A_[l].ddiag_val_,
                                                      ddiag_add_[l], R);
    }
#ifdef REAL_AS_DOUBLE
    CheckCuda(cublasDaxpy(cublas_handle_, dim_[l], &relaxation_, P, 1, lhs, 1));
#else
    CheckCuda(cublasSaxpy(cublas_handle_, dim_[l], &relaxation_, P, 1, lhs, 1));
#endif
    ComputeAP(l, R, &minus_relax, P, stream);
  }
}

void CudaElasobjMG19::DownSample(int32_t& l, cudaStream_t stream) {
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

void CudaElasobjMG19::UpSample(int32_t& l, cudaStream_t stream) {
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

void CudaElasobjMG19::DirectSolve(int32_t& l, cudaStream_t stream) {
  real minus_one = real(-1);
  if (l == n_layer_) {
    // TODO: PD
  } else if (l == 0 && n_layer_ != 0) {
    // dense solver at coarsest
    // int32_t nan_info = CheckNan(drhs_[l], dim_[l]);
    // if (nan_info > 0) std::cout << nan_info << " nan(s) in rhs" << std::endl;
    CheckCuda(cudaMemcpyAsync(dlhs_[l], drhs_[l], sizeof(real) * dim_[l],
                              cudaMemcpyDeviceToDevice, stream));
    // nan_info = CheckNan(A_[l].dchol_val_, dim_[l] * dim_[l]);
    // if (nan_info > 0) std::cout << nan_info << " nan(s) in A" << std::endl;
    CheckCuda(cusolverDnXpotrs(cusolverDn_handle_, cusolverDn_params_,
                               CUBLAS_FILL_MODE_LOWER, dim_[l], 1, CudaRealType,
                               A_[l].dchol_val_, dim_[l], CudaRealType,
                               dlhs_[l], dim_[l], A_[l].dchol_info_));
    // int32_t nan_info = CheckNan(dlhs_[l], dim_[l]);
    // if (nan_info > 0) std::cout << nan_info << " nan(s) in lhs" << std::endl;
    ComputeAP(l, dR_[l], &minus_one, dlhs_[l], stream);
  }
}

void CudaElasobjMG19::ComputeAP(int32_t l, real* AP, const real* alpha,
                                const real* P, cudaStream_t stream) {
  real one = real(1);
  CheckCuda(cudaMemset(dtAP_[l], 0, sizeof(real) * dim_[l]));
  if (l == n_layer_) {  // Af
    if (Af_->as_LDU_) {
#ifdef REAL_AS_DOUBLE
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->low_bnnz_, alpha, descr_,
                               Af_->dlow_bcsr_val_, Af_->dlow_bcsr_row_,
                               Af_->dlow_bcsr_col_, 3, P, &one, AP));
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->diag_bnnz_, alpha, descr_,
                               Af_->ddiag_bcsr_val_, Af_->ddiag_bcsr_row_,
                               Af_->ddiag_bcsr_col_, 3, P, &one, AP));
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->up_bnnz_, alpha, descr_,
                               Af_->dup_bcsr_val_, Af_->dup_bcsr_row_,
                               Af_->dup_bcsr_col_, 3, P, &one, AP));
#else
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->low_bnnz_, alpha, descr_,
                               Af_->dlow_bcsr_val_, Af_->dlow_bcsr_row_,
                               Af_->dlow_bcsr_col_, 3, P, &one, AP));
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->diag_bnnz_, alpha, descr_,
                               Af_->ddiag_bcsr_val_, Af_->ddiag_bcsr_row_,
                               Af_->ddiag_bcsr_col_, 3, P, &one, AP));
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->up_bnnz_, alpha, descr_,
                               Af_->dup_bcsr_val_, Af_->dup_bcsr_row_,
                               Af_->dup_bcsr_col_, 3, P, &one, AP));
#endif
    } else {
#ifdef REAL_AS_DOUBLE
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->bnnz_, alpha, descr_,
                               Af_->dbcsr_val_, Af_->dbcsr_row_,
                               Af_->dbcsr_col_, 3, P, &one, AP));
#else
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, Af_->brows_,
                               Af_->bcols_, Af_->bnnz_, alpha, descr_,
                               Af_->dbcsr_val_, Af_->dbcsr_row_,
                               Af_->dbcsr_col_, 3, P, &one, AP));
#endif
    }
    // diag addition
    CUDA::AfDiagAddMulVec<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
        dtAP_[l], ddiag_add_[l], P, n_vert_);
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
                               A_[l].dlow_bcsr_col_, 12, P, &one, AP));
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].diag_bnnz_, alpha, descr_,
                               A_[l].ddiag_bcsr_val_, A_[l].ddiag_bcsr_row_,
                               A_[l].ddiag_bcsr_col_, 12, P, &one, AP));
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].up_bnnz_, alpha, descr_,
                               A_[l].dup_bcsr_val_, A_[l].dup_bcsr_row_,
                               A_[l].dup_bcsr_col_, 12, P, &one, AP));
#else
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].low_bnnz_, alpha, descr_,
                               A_[l].dlow_bcsr_val_, A_[l].dlow_bcsr_row_,
                               A_[l].dlow_bcsr_col_, 12, P, &one, AP));
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].diag_bnnz_, alpha, descr_,
                               A_[l].ddiag_bcsr_val_, A_[l].ddiag_bcsr_row_,
                               A_[l].ddiag_bcsr_col_, 12, P, &one, AP));
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].up_bnnz_, alpha, descr_,
                               A_[l].dup_bcsr_val_, A_[l].dup_bcsr_row_,
                               A_[l].dup_bcsr_col_, 12, P, &one, AP));
#endif
    } else {
#ifdef REAL_AS_DOUBLE
      CheckCuda(cusparseDbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].bnnz_, alpha, descr_,
                               A_[l].dbcsr_val_, A_[l].dbcsr_row_,
                               A_[l].dbcsr_col_, 12, P, &one, AP));
#else
      CheckCuda(cusparseSbsrmv(cusparse_handle_, CUSPARSE_DIRECTION_ROW,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, A_[l].brows_,
                               A_[l].bcols_, A_[l].bnnz_, alpha, descr_,
                               A_[l].dbcsr_val_, A_[l].dbcsr_row_,
                               A_[l].dbcsr_col_, 12, P, &one, AP));
#endif
    }
    // diag addition
    int n_block = (n_handle_[l] + threads_per_block_ - 1) / threads_per_block_;
    CUDA::AsDiagAddMulVec<<<n_block, threads_per_block_, 0, stream>>>(
        dtAP_[l], ddiag_add_[l], P, n_handle_[l]);
  }
  CheckCuda(cublasAxpyEx(cublas_handle_, dim_[l], alpha, CudaRealType, dtAP_[l],
                         CudaRealType, 1, AP, CudaRealType, 1, CudaRealType));
}

void CudaElasobjMG19::ComputeSelfcollisionOffAP(int32_t l, real* AP,
                                                     const real* alpha,
                                                     const real* P,
                                                     cudaStream_t stream) {
  if (l == n_layer_ && !Af_->as_dense_) {
    int n_block =
        (hashing_->n_colli_ + threads_per_block_ - 1) / threads_per_block_;
    CUDA::
        SelfCollisionFineOffAP<<<n_block, threads_per_block_, 0, stream>>>(
            AP, dcolli_pairs_[l], dcolli_hessian_[l], P, *alpha,
            hashing_->n_colli_);
  } else if (l < n_layer_ && !A_[l].as_dense_) {
    CUDA::SelfCollisionCoarseOffAP<<<hashing_->n_colli_, dim3(1, 12), 0,
                                          stream>>>(AP, dcolli_pairs_[l],
                                                    dcolli_hessian_[l], P,
                                                    *alpha, hashing_->n_colli_);
  }
}
};  // namespace Rain