#pragma once

#include "cudahelper.h"
#include "cudamath.h"
#include "mathtype.h"

namespace Rain {
namespace CUDA {
// update position with velocity, apply gravity
__global__ void UpdateBasic(Vec3* X, Vec3* V, int32_t n_vert, Vec3 grav,
                            real damping, real dt);
// update Af's diag for fixed vetices and control
__global__ void UpdateAfDiagReordered(real* Af_diag_add, int32_t* h2m,
                                      bool* fixed, const real control_mag,
                                      int32_t n_vert);

__global__ void UpdateAfDiag(real* Af_diag_add, bool* fixed,
                             const real control_mag, int32_t n_vert);

__global__ void UpdateAfDiagWithCtrlReordered(real* Af_diag_add, int32_t* h2m,
                                              bool* fixed,
                                              const real control_mag,
                                              int32_t ctrl_vert,
                                              int32_t n_vert);

__global__ void UpdateAfDiagWithCtrl(real* Af_diag_add, bool* fixed,
                                     const real control_mag, int32_t ctrl_vert,
                                     int32_t n_vert);

// update UtAU's diag from Af, also fix rank
__global__ void UpdateUtAUDiag(real* UtAU_diag_add, const real* Af_diag_add,
                               const real* diag_XXt, const int32_t* update_off,
                               int32_t n_vert);
// propagate diag addition to coarse levels
__global__ void UpdateUltAUlDiag(real* diag_add, const real* diag_add_finer,
                                 const int32_t* update_off, int32_t n_handle);

__global__ void DerivativeDiagReduction(real* down_diag, const real* up_diag,
                                        const real* diag_XXt,
                                        const int32_t* update_off,
                                        const int32_t up_dof,
                                        const int32_t n_handle);

__global__ void SameDiagReduction(real* down_diag, const real* up_diag,
                                  const int32_t* update_off, const int32_t dof,
                                  const int32_t n_handle);

// update diag for dense matrix (coarse level)
__global__ void UpdateDenseDiag(real* den_val, const real* diag_add,
                                int32_t n_handle, int32_t rows);

__global__ void UpdateDenseDiag(real* den_val, const real* diag_add,
                                const int32_t dof, const int32_t n_handle,
                                const int32_t rows);

__global__ void HessianNeoHookean(const Vec3* X, const uint32_t* tet,
                                  const Mat3* Dm_inv, const real* vol,
                                  const int32_t* t2off, real* bcoo_val,
                                  const real mu, const real lambda,
                                  const uint32_t n_tet);

__global__ void HessianNeoHookeanClamped(const Vec3* X, const uint32_t* tet,
                                         const Mat3* Dm_inv, const real* vol,
                                         const int32_t* t2off, real* bcoo_val,
                                         const real mu, const real lambda,
                                         const uint32_t n_tet);

__global__ void EnergyNeoHookean(const Vec3* X, const uint32_t* tet,
                                 const Mat3* Dm_inv, const real* vol,
                                 const real mu, const real lambda, real* out,
                                 const uint32_t n_tet);

__global__ void EnergyPD(const Vec3* X, const uint32_t* tet, const Mat3* Dm_inv,
                         const real* vol, const real mu, real* out,
                         const uint32_t n_tet);

__global__ void EnergyFixed(const bool* fixed, const Vec3* fixed_X,
                            const Vec3* X, const real control_mag, real* out,
                            const int32_t n_vert);

__global__ void EnergyFixedWithCtrl(const bool* fixed, const Vec3* fixed_X,
                                    const Vec3* X, const real control_mag,
                                    int ctrl_vert, Vec3 ctrl_pos, real* out,
                                    const int32_t n_vert);

__global__ void EnergyInertia(const Vec3* inertia_X, const Vec3* X,
                              const real* M, const real dt_inv, real* out,
                              const uint32_t n_vert);

__global__ void EnergyGrav(const Vec3* X, const real* M, const Vec3 grav,
                           real* out, const uint32_t n_vert);

__global__ void UpdateDiagXXt(const Vec3* X, real* diag_XXt,
                              const uint32_t n_vert);

__global__ void UpdateDiagXXtReordered(const Vec3* X, const int32_t* m2h,
                                       real* diag_XXt, const uint32_t n_vert);

__global__ void UpdateXXt(const Vec3* X, const int32_t* bcoo_row,
                          const int32_t* bcoo_col, real* XXt,
                          const uint32_t bnnz);

__global__ void UpdateXXtReordered(const Vec3* X, const int32_t* m2h,
                                   const int32_t* bcoo_row,
                                   const int32_t* bcoo_col, real* XXt,
                                   const uint32_t bnnz);

__global__ void UpdateU(const Vec3* X, real* csr_val, const uint32_t n_vert);

__global__ void UpdateUReordered(const Vec3* X, const int32_t* m2h,
                                 real* csr_val, const uint32_t n_vert);

__global__ void HessianPD(const Mat3* Dm_inv, const real* vol,
                          const int32_t* t2off, real* bcoo_val, const real mu,
                          const uint32_t n_tet);

__global__ void InteriaHessian(const real* M, const int32_t* d2off,
                               real* bcoo_val, const real dt_inv,
                               const uint32_t n_vert);

__global__ void AfKroneckerXXtHalf(const real* Af, const real* XXt,
                                   const int32_t* half_off, real* Af_XXt,
                                   const uint32_t n_half);

__global__ void AfDenseReduction(const real* Af_XXt, const int32_t* red_off,
                                 const int32_t* half_off, real* UtAU_dense,
                                 const int32_t n_half, const int32_t n_handle);

__global__ void AfSparseReduction(const real* Af_XXt, const int32_t* red_off,
                                  const int32_t* half_off, real* UtAU_sparse,
                                  const int32_t n_half);

__global__ void AsDenseMirror(real* A_dense, const int32_t* low_off,
                              const int32_t n_handle, const int32_t low_nnz);

__global__ void AsSparseMirror(real* A_sparse, const int32_t* low_off,
                               const int32_t* mirror_off,
                               const int32_t low_nnz);

__global__ void AsSparseReduction(const real* A_upper, const int32_t* red_off,
                                  real* A, const int32_t upper_nnz);

__global__ void AsDenseReduction(const real* A_upper, const int32_t* red_off,
                                 real* A, const int32_t n_handle,
                                 const int32_t upper_nnz);

// Projective Dynamics's local step, compute -gradients(forces) for each tet
__global__ void TetGradientPD(const Vec3* X, const uint32_t* tet,
                              const Mat3* Dm_inv, const real* vol, real* grad,
                              const real mu, const uint32_t n_tet);
// Stable Neo Hookean's local step, compute -gradients(forces) for each tet
__global__ void TetGradientNeoHookean(const Vec3* X, const uint32_t* tet,
                                      const Mat3* Dm_inv, const real* vol,
                                      real* grad, const real mu,
                                      const real lambda, const uint32_t n_tet);

// collect -gradients(forces) to vertices, add forces for control/collision
__global__ void EnergyGradientReordered(
    const real* tet_grad, const int32_t* v2t_ids, const int32_t* v2t_off,
    const bool* fixed, const Vec3* fixed_X, const Vec3* X, const int32_t* m2h,
    real* grad, const real control_mag, const int32_t n_vert);

__global__ void EnergyGradient(const real* tet_grad, const int32_t* v2t_ids,
                               const int32_t* v2t_off, const bool* fixed,
                               const Vec3* fixed_X, const Vec3* X,
                               const real* mass, real* grad,
                               const real control_mag, const int32_t n_vert);

__global__ void EnergyGradientWithCtrlReordered(
    const real* tet_grad, const int32_t* v2t_ids, const int32_t* v2t_off,
    const bool* fixed, const Vec3* fixed_X, const Vec3* X, const int32_t* m2h,
    real* grad, const real control_mag, int ctrl_vert, Vec3 ctrl_pos,
    const int32_t n_vert);

__global__ void EnergyGradientWithCtrl(
    const real* tet_grad, const int32_t* v2t_ids, const int32_t* v2t_off,
    const bool* fixed, const Vec3* fixed_X, const Vec3* X, real* grad,
    const real control_mag, int ctrl_vert, Vec3 ctrl_pos, const int32_t n_vert);

// inertia energy -grandient
__global__ void InertiaGradientReordered(const Vec3* inertia_X, const Vec3* X,
                                         const int32_t* m2h, const real* M,
                                         real* grad, const real dt_inv,
                                         const int32_t n_vert);

__global__ void InertiaGradient(const Vec3* inertia_X, const Vec3* X,
                                const real* M, real* grad, const real dt_inv,
                                const int32_t n_vert);

// colored GS diag inverse for Af
__global__ void ColoredGSAf(real* X, const real* Af_diag,
                            const real* Af_diag_add, const real* b,
                            const int32_t base, const int32_t n_in_color);
// y += diag * x
__global__ void AfDiagAddMulVec(real* Y, const real* Af_diag_add, const real* X,
                                const int32_t n_vert);
// y += diag * x
__global__ void AsDiagAddMulVec(real* Y, const real* As_diag_add, const real* X,
                                const int32_t n_handle);
__global__ void ADiagAddMulVec(real* Y, const real* diag_add, const real* X,
                               const int32_t block_size,
                               const int32_t n_handle);
// colored GS diag inverse for As
__global__ void ColoredGSAs(real* X, const real* diag, const real* diag_add,
                            const real* b, const int32_t base);
// extern shared memory: sizeof(real) * block_size * 4
__global__ void ColoredGSAs(real* X, const real* diag, const real* diag_add,
                            const real* b, const int32_t block_size,
                            const int32_t base);

__global__ void GSAf(real* X, const real* Af_diag, const real* Af_diag_add,
                     const real* b, const int32_t* colors, const int c,
                     const int32_t n_vert);

__global__ void GSAs(real* X, const real* diag, const real* diag_add,
                     const real* B, const int32_t* colors, const int c);

__global__ void GSAfInc(real* X, const real* Af_diag, const real* Af_diag_add,
                        const real* bcsr_val, const int32_t* bcsr_row,
                        const int32_t* bcsr_col, const real* b,
                        const int32_t* colors, const int c,
                        const int32_t n_vert);

__global__ void GSAsInc(real* X, const real* diag, const real* diag_add,
                        const real* bcsr_val, const int32_t* bcsr_row,
                        const int32_t* bcsr_col, const real* B,
                        const int32_t* colors, const int c);

// Jacobi iteration
__global__ void JacobiAf(real* X, const real* Af_diag, const real* Af_diag_add,
                         const real* b, const int32_t n_vert);
__global__ void JacobiAs(real* X, const real* diag, const real* diag_add,
                         const real* b);
// extern shared memory: sizeof(real) * block_size * 4
__global__ void JacobiAs(real* X, const real* diag, const real* diag_add,
                         const real* b, const int32_t block_size);
// update deltaX after multigrid solving
__global__ void UpdatedXReordered(Vec3* X, const real alpha, const real* dX,
                                  const int32_t* m2h, const int32_t n_vert);

__global__ void UpdatedX(Vec3* X, const real alpha, const real* dX,
                         const int32_t n_vert);

// update velocity from position difference
__global__ void UpdateVelFromPos(Vec3* V, const Vec3* X, const Vec3* old_X,
                                 const real dt_inv, const int32_t n_vert);

__global__ void SkeletonGradient(real* grad, const Vec3* X,
                                 const int32_t* bone_ids,
                                 const JointTransform* trans,
                                 const Vec3* local_pos, const real ctr_mag,
                                 const int32_t n_vert);

__global__ void UpdateAfDiagSkeleton(real* Af_diag_add, const int32_t* bone_id,
                                     const real ctr_mag, const int32_t n_vert);


__global__ void MakeSelfCollision(
    real* grad, int32_t* vv_pairs, real* hessians, real* diag_add,
    const Vec3* X, const Vec3* norms, const uint32_t* tets,
    const uint32_t* vt_pairs, const int32_t* closest_surf_vert,
    const real k_penalty, const int32_t n_vert, const int32_t n_colli);

__global__ void SelfCollisionFineOffReduction(
    int32_t* coarse_pairs, real* coarse_hessian, const int32_t* fine_pairs,
    const real* fine_hessian, const Vec3* pos_rest, const int32_t* handle,
    const int32_t* handle_ids, const int32_t n_colli);

__global__ void SelfCollisionFineDiagReduction(
    real* diag_add, const int32_t* fine_pairs, const real* fine_hessian,
    const Vec3* pos_rest, const int32_t* handle, const int32_t* handle_ids,
    const int32_t n_colli);

__global__ void SelfCollisionCoarseOffReduction(
    int32_t* coarse_pairs, real* coarse_hessian, const int32_t* fine_pairs,
    const real* fine_hessian, const int32_t* handle, const int32_t n_colli);

__global__ void SelfCollisionCoarseDiagReduction(real* diag_add,
                                                const int32_t* fine_pairs,
                                                const real* fine_hessian,
                                                const int32_t* handle,
                                                const int32_t n_colli);

__global__ void SelfCollisionCoarseDenseHessian(real* den,
                                                     const int32_t* pairs,
                                                     const real* hessian,
                                                     const int32_t dim,
                                                     const int32_t n_colli);

__global__ void SelfCollisionFineOffGSAP(
    real* AP, const int32_t* pairs, const real* hessian, const int32_t* color,
    const int32_t c, const real* P, const real alpha, const int32_t n_colli);

__global__ void SelfCollisionCoarseOffGSAP(
    real* AP, const int32_t* pairs, const real* hessian, const int32_t* color,
    const int32_t c, const real* P, const real alpha, const int32_t n_colli);

__global__ void SelfCollisionFineOffAP(real* AP, const int32_t* pairs,
                                        const real* hessian, const real* P,
                                        const real alpha,
                                        const int32_t n_colli);

__global__ void SelfCollisionCoarseOffAP(real* AP, const int32_t* pairs,
                                          const real* hessian,
                                          const real* P, const real alpha,
                                          const int32_t n_colli);
};  // namespace CUDA
};  // namespace Rain