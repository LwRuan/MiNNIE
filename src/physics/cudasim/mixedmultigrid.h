#pragma once

#include "cudahelper.h"
#include "cudamath.h"
#include "mathtype.h"

namespace Rain {
namespace CUDA {
__global__ void HessianMixedNeoHookean(
    const Vec3* X, const real* P, const uint32_t* tet, const Mat3* Dm_inv,
    const real* vol, const int32_t* t2off, real* bcoo_val,
    const int32_t* normal_sign, const real mu, const real lambda_inv,
    const real p_smooth, const real p_scale, const uint32_t n_tet);

__global__ void HessianMixedNeoHookeanClamped(
    const Vec3* X, const real* P, const uint32_t* tet, const Mat3* Dm_inv,
    const real* vol, const int32_t* t2off, real* bcoo_val,
    const int32_t* normal_sign, const real mu, const real lambda_inv,
    const real p_smooth, const real p_scale, const uint32_t n_tet);

__global__ void HessianMixedStVKClamped(
    const Vec3* X, const real* P, const uint32_t* tet, const Mat3* Dm_inv,
    const real* vol, const int32_t* t2off, real* bcoo_val,
    const int32_t* normal_sign, const real mu, const real lambda_inv,
    const real p_smooth, const real p_scale, const uint32_t n_tet);

__global__ void HessianMixedCorotationClamped(
    const Vec3* X, const real* P, const uint32_t* tet, const Mat3* Dm_inv,
    const real* vol, const int32_t* t2off, real* bcoo_val,
    const int32_t* normal_sign, const real mu, const real lambda_inv,
    const real p_smooth, const real p_scale, const uint32_t n_tet);

__global__ void HessianMixedNeoHookeanLogClamped(
    const Vec3* X, const real* P, const uint32_t* tet, const Mat3* Dm_inv,
    const real* vol, const int32_t* t2off, real* bcoo_val,
    const int32_t* normal_sign, const real mu, const real lambda_inv,
    const real p_smooth, const real p_scale, const uint32_t n_tet);

__global__ void InertiaHessianMixed(const real* M, const int32_t* d2off,
                                    real* bcoo_val, const real dt_inv,
                                    const uint32_t n_vert);

__global__ void AfKroneckerXXtHalfMixed(const real* Af, const real* XXt,
                                        const int32_t* half_off, real* Af_XXt,
                                        const uint32_t n_half);

__global__ void AfDenseReductionMixed(const real* Af_XXt,
                                      const int32_t* red_off,
                                      const int32_t* half_off, real* UtAU_dense,
                                      const int32_t n_half,
                                      const int32_t n_handle);

__global__ void AfDenseReductionShurMixed(
    const real* Af_XXt, const int32_t* red_off, const int32_t* half_off,
    real* UtAU_dense, const int32_t n_half, const int32_t n_handle);

__global__ void AsDenseMirrorMixed(real* A_dense, const int32_t* low_off,
                                   const int32_t n_handle,
                                   const int32_t low_nnz);

__global__ void AsDenseMirrorShurMixed(real* A_dense, const int32_t* low_off,
                                       const int32_t n_handle,
                                       const int32_t low_nnz);

__global__ void AfSparseReductionMixed(const real* Af_XXt,
                                       const int32_t* red_off,
                                       const int32_t* half_off,
                                       real* UtAU_sparse, const int32_t n_half);

__global__ void AsSparseMirrorMixed(real* A_sparse, const int32_t* low_off,
                                    const int32_t* mirror_off,
                                    const int32_t low_nnz);

__global__ void AsSparseReductionMixed(const real* A_upper,
                                       const int32_t* red_off, real* A,
                                       const int32_t upper_nnz);

__global__ void AsDenseReductionMixed(const real* A_upper,
                                      const int32_t* red_off, real* A,
                                      const int32_t n_handle,
                                      const int32_t upper_nnz);

__global__ void AsDenseReductionShurMixed(const real* A_upper,
                                          const int32_t* red_off, real* A,
                                          const int32_t n_handle,
                                          const int32_t upper_nnz);

__global__ void TetGradientMixedNeoHookean(
    const Vec3* X, const uint32_t* tet, const real* pressure,
    const int32_t* normal_sign, const Mat3* Dm_inv, const real* vol, real* grad,
    real* p_grad, const real mu, const real lam_inv, const real p_smooth,
    const real p_scale, const uint32_t n_tet);

__global__ void TetGradientMixedStVK(const Vec3* X, const uint32_t* tet,
                                     const real* pressure,
                                     const int32_t* normal_sign,
                                     const Mat3* Dm_inv, const real* vol,
                                     real* grad, real* p_grad, const real mu,
                                     const real lam_inv, const real p_smooth,
                                     const real p_scale, const uint32_t n_tet);

__global__ void TetGradientMixedCorotation(
    const Vec3* X, const uint32_t* tet, const real* pressure,
    const int32_t* normal_sign, const Mat3* Dm_inv, const real* vol, real* grad,
    real* p_grad, const real mu, const real lam_inv, const real p_smooth,
    const real p_scale, const uint32_t n_tet);

__global__ void TetGradientMixedNeoHookeanLog(
    const Vec3* X, const uint32_t* tet, const real* pressure,
    const int32_t* normal_sign, const Mat3* Dm_inv, const real* vol, real* grad,
    real* p_grad, const real mu, const real lam_inv, const real p_smooth,
    const real p_scale, const uint32_t n_tet);

__global__ void MixedEnergyGradient(
    const real* tet_grad, const real* tet_p_grad, const int32_t* v2t_ids,
    const int32_t* v2t_off, const bool* fixed, const Vec3* fixed_X,
    const Vec3* X, const real* mass, real* grad, const real control_mag,
    const int32_t n_vert);

__global__ void MixedEnergyGradientWithCtrl(
    const real* tet_grad, const real* tet_p_grad, const int32_t* v2t_ids,
    const int32_t* v2t_off, const bool* fixed, const Vec3* fixed_X,
    const Vec3* X, real* grad, const real control_mag, int ctrl_vert,
    Vec3 ctrl_pos, const int32_t n_vert);

__global__ void SkeletonGradientMixed(real* grad, const Vec3* X,
                                      const int32_t* bone_ids,
                                      const JointTransform* trans,
                                      const Vec3* local_pos, const real ctr_mag,
                                      const int32_t n_vert);

__global__ void InertiaGradientMixed(const Vec3* inertia_X, const Vec3* X,
                                     const real* M, real* grad,
                                     const real dt_inv, const int32_t n_vert);

__global__ void UpdateAfDiagWithCtrlMixed(real* Af_diag_add, bool* fixed,
                                          const real control_mag,
                                          int32_t ctrl_vert, int32_t n_vert);

__global__ void UpdateAfDiagMixed(real* Af_diag_add, bool* fixed,
                                  const real control_mag, int32_t n_vert);

__global__ void UpdateAfDiagSkeletonMixed(real* Af_diag_add,
                                          const int32_t* bone_id,
                                          const real ctr_mag,
                                          const int32_t n_vert);

__global__ void UpdateUtAUDiagMixed(real* UtAU_diag_add,
                                    const real* Af_diag_add,
                                    const real* diag_XXt,
                                    const int32_t* update_off, int32_t n_vert);

__global__ void UpdateUltAUlDiagMixed(real* diag_add,
                                      const real* diag_add_finer,
                                      const int32_t* update_off,
                                      int32_t n_handle);

__global__ void UpdateDenseDiagMixed(real* den_val, const real* diag_add,
                                     int32_t n_handle, int32_t rows);

__global__ void UpdateDenseDiagShurMixed(real* den_val, const real* diag_add,
                                         int32_t n_handle, int32_t rows);

__global__ void CoarseAOS2SOAMixed(real* P, const real* B, const int32_t dim);

__global__ void CoarseSOA2AOSMixed(real* P, const real* B, const int32_t dim);

__global__ void GSAfMixed(real* X, const real* Af_diag, const real* Af_diag_add,
                          const real* b, const int32_t* colors, const int c,
                          const int32_t n_vert);

__global__ void GSAfIncMixed(real* X, const real* Af_diag,
                             const real* Af_diag_add, const real* bcsr_val,
                             const int32_t* bcsr_row, const int32_t* bcsr_col,
                             const real* b, const int32_t* colors, const int c,
                             const real relax, const int32_t n_vert);

__global__ void CellVankaAfMixed(real* X, const real* Af_diag,
                                 const real* Af_diag_add, const real* bcsr_val,
                                 const int32_t* bcsr_row,
                                 const int32_t* bcsr_col, const real* b,
                                 const uint32_t* ind, const uint32_t* colors,
                                 const int c, const real relax,
                                 int32_t* updated);

__global__ void CellVankaJacobiMixed(real* X, const real* Af_diag,
                                     const real* Af_diag_add,
                                     const real* bcsr_val,
                                     const int32_t* bcsr_row,
                                     const int32_t* bcsr_col, const real* b,
                                     const uint32_t* ind, const real relax);

__global__ void GSAfDecMixed(real* X, const real* Af_diag,
                             const real* Af_diag_add, const real* bcsr_val,
                             const int32_t* bcsr_row, const int32_t* bcsr_col,
                             const real* b, const int32_t* colors, const int c,
                             const real relax, const int32_t n_vert);

__global__ void GSAsMixed(real* X, const real* diag, const real* diag_add,
                          const real* b, const int32_t* colors, const int c);

__global__ void GSAsIncMixed(real* X, const real* diag, const real* diag_add,
                             const real* bcsr_val, const int32_t* bcsr_row,
                             const int32_t* bcsr_col, const real* b,
                             const int32_t* colors, const int c,
                             const real relax);

__global__ void GSAsDecMixed(real* X, const real* diag, const real* diag_add,
                             const real* bcsr_val, const int32_t* bcsr_row,
                             const int32_t* bcsr_col, const real* b,
                             const int32_t* colors, const int c,
                             const real relax);

__global__ void hVankaAfMixed(real* X, const real* Af_diag,
                              const real* Af_diag_add, const real* bcsr_val,
                              const int32_t* bcsr_row, const int32_t* bcsr_col,
                              const real* b, const int32_t* colors, const int c,
                              const real relax, const int32_t n_vert);

__global__ void RestrictedVankaAfMixed(
    real* X, const real* Af_diag, const real* Af_diag_add, const real* bcsr_val,
    const int32_t* bcsr_row, const int32_t* bcsr_col, const real* b,
    const int32_t* colors, const int c, const int32_t n_vert);

__global__ void InexactUzawaXAfMixed(real* X, const real* Af_diag,
                                     const real* Af_diag_add, const real* b,
                                     const int32_t n_vert);

__global__ void InexactUzawaGSXAfMixed(
    real* X, const real* Af_diag, const real* Af_diag_add, const real* bcsr_val,
    const int32_t* bcsr_row, const int32_t* bcsr_col, const real* b,
    const int32_t* colors, const int c, const real relax, const int32_t n_vert);

__global__ void InexactUzawaXAsMixed(real* X, const real* diag,
                                     const real* diag_add, const real* b);

__global__ void InexactUzawaPAfMixed(real* P, const real* Af_diag,
                                     const real* Af_diag_add,
                                     const real* bcsr_val,
                                     const int32_t* bcsr_row,
                                     const int32_t* bcsr_col, const real* b,
                                     const int32_t n_vert);

__global__ void KaczmarzIteration(real* P, const real* Af_diag_add,
                                  const real* bcsr_val, const int32_t* bcsr_row,
                                  const int32_t* bcsr_col, const real* b,
                                  const int32_t n_vert);

__global__ void AfDiagMulVecMixed(real* Y, const real* Af_diag_add,
                                  const real* X, const real alpha,
                                  const int32_t n_vert);

__global__ void AsDiagMulVecMixed(real* Y, const real* As_diag_add,
                                  const real* X, const real alpha,
                                  const int32_t n_handle);

__global__ void UpdatePosPressureMixed(Vec3* X, real* p, const real alpha,
                                       const real* dX, const real p_scale,
                                       const int32_t n_vert);

__global__ void ComputeDistortionEnergyMixed(const Vec3* X, const uint32_t* tet,
                                             const real* vol,
                                             const Mat3* Dm_inv, real* E,
                                             const real mu,
                                             const uint32_t n_tet);

__global__ void MakeSelfCollisionMixed(
    real* grad, int32_t* vv_pairs, real* hessians, real* diag_add,
    const Vec3* X, const Vec3* norms, const uint32_t* tets,
    const uint32_t* vt_pairs, const int32_t* closest_surf_vert,
    const real k_penalty, const int32_t n_vert, const int32_t n_colli);

__global__ void SelfCollisionFineOffReductionMixed(
    int32_t* coarse_pairs, real* coarse_hessian, const int32_t* fine_pairs,
    const real* fine_hessian, const Vec3* pos_rest, const int32_t* handle,
    const int32_t* handle_ids, const int32_t n_colli);

__global__ void SelfCollisionFineDiagReductionMixed(
    real* diag_add, const int32_t* fine_pairs, const real* fine_hessian,
    const Vec3* pos_rest, const int32_t* handle, const int32_t* handle_ids,
    const int32_t n_colli);

__global__ void SelfCollisionCoarseOffReductionMixed(
    int32_t* coarse_pairs, real* coarse_hessian, const int32_t* fine_pairs,
    const real* fine_hessian, const int32_t* handle, const int32_t n_colli);

__global__ void SelfCollisionCoarseDiagReductionMixed(real* diag_add,
                                                      const int32_t* fine_pairs,
                                                      const real* fine_hessian,
                                                      const int32_t* handle,
                                                      const int32_t n_colli);

__global__ void SelfCollisionFineOffAPMixed(real* AP, const int32_t* pairs,
                                            const real* hessian, const real* P,
                                            const real alpha,
                                            const int32_t n_colli);

__global__ void SelfCollisionFineOffGSAPMixed(
    real* AP, const int32_t* pairs, const real* hessian, const int32_t* color,
    const int32_t c, const real* P, const real alpha, const int32_t n_colli);

__global__ void SelfCollisionCoarseOffAPMixed(real* AP, const int32_t* pairs,
                                              const real* hessian,
                                              const real* P, const real alpha,
                                              const int32_t n_colli);

__global__ void SelfCollisionCoarseOffGSAPMixed(
    real* AP, const int32_t* pairs, const real* hessian, const int32_t* color,
    const int32_t c, const real* P, const real alpha, const int32_t n_colli);

__global__ void SelfCollisionFineDenseHessianMixed(real* den,
                                                   const int32_t* pairs,
                                                   const real* hessian,
                                                   const int32_t dim,
                                                   const int32_t n_colli);

__global__ void SelfCollisionCoarseDenseHessianMixed(real* den,
                                                     const int32_t* pairs,
                                                     const real* hessian,
                                                     const int32_t dim,
                                                     const int32_t n_colli);

__global__ void SelfCollisionCoarseDenseHessianShurMixed(real* den,
                                                         const int32_t* pairs,
                                                         const real* hessian,
                                                         const int32_t dim,
                                                         const int32_t n_colli);

// helper functions
// fuck you Nvidia
__global__ void Int32toInt64(const int32_t* in, int64_t* out, int32_t n);
};  // namespace CUDA
};  // namespace Rain