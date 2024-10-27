#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include <deque>
#include <functional>
#include <vector>

#include "collision/cudatethashing.h"
#include "configtype.h"
#include "cudaelasticobject.h"
#include "cudakinematicobject.h"
#include "steptimer.h"

namespace Rain {
struct CudaElasobjMixedMG19InitInfo {
  VkDevice device;
  Object* obj;
  real density;
  ElasticModelType type;
  real young;
  real poisson;
  real control_mag;
  real relaxation;
  uint32_t n_layer;
  // for bottom to top
  uint32_t* n_handles;
  bool* A_as_LDU;
  bool* A_as_dense;
  real dt;
  bool quasi_static;
  int minres_iter;
  real p_smooth;
  real p_scale;
  uint32_t n_iter;
  uint32_t n_op;
  MGOpConfig* operations;
  Joint* skeleton;
  int32_t n_joint;
};

class CudaElasobjMixedMG19 : public CudaElasticObject {
 public:
  virtual void Init(CudaElasobjMixedMG19InitInfo* info);
  virtual void Reset();
  virtual void ShowUI();
  virtual void Update(cudaStream_t stream, float dt, const Vec3& grav,
                      real damping, real kinematic_penalty, real self_penalty,
                      const std::vector<CudaKinematicObject*>& kobjs,
                      uint32_t n_substep, real substep_size, uint32_t n_frame);
  void UpdateMINRES(cudaStream_t stream, float dt, const Vec3& grav,
                    real damping, real kinematic_penalty, real self_penalty,
                    const std::vector<CudaKinematicObject*>& kobjs,
                    uint32_t n_substep, real substep_size, uint32_t n_frame);
  virtual void Destroy();
  std::deque<std::function<void()>> deletion_queue_;

  using RowCol = std::pair<uint32_t, uint32_t>;
  using LenFrom = std::pair<real, uint32_t>;
  static const uint32_t MAXN_COLLI = 100000;

  StepTimer timer_;

  // vertex to tet indices multimapping
  uint32_t* v2t_ids_ = nullptr;  // indices
  uint32_t* v2t_off_ = nullptr;  // offset
  // adjacent graph from tet
  uint32_t** v2e_off_ = nullptr;
  uint32_t** edge_to_ = nullptr;
  real* edge_len_ = nullptr;
  int32_t* normal_sign_ = nullptr;
  // tet to matrix block offsets
  uint32_t* t2off_ = nullptr;
  // diag to matrix block offsets
  uint32_t* d2off_ = nullptr;
  uint32_t* n_color_ = nullptr;
  uint32_t** color_ = nullptr;
  real rest_vol_;
  // for fixed vertex constriants
  real control_mag_;
  // for system matrix
  real dt_;
  // relaxation for solver
  real relaxation_;
  // # iterations
  uint32_t n_iter_;
  real mu_;
  real lambda_inv_;
  real p_scale_;
  real p_smooth_;
  bool quasi_static_;
  int minres_iter_;
  bool direct_shur_ = true;

  // # multigrid layers except the original layer
  uint32_t n_layer_;
  // # handles from bottom to top
  uint32_t* n_handle_ = nullptr;
  // matrix dims from bottom to top
  uint32_t* dim_ = nullptr;
  bool* A_as_LDU_ = nullptr;
  bool* A_as_dense_ = nullptr;
  std::vector<MGOpConfig> operations_;
  // handles mapping for each level
  uint32_t** handle_ = nullptr;
  // handle indices
  std::vector<uint32_t> handle_ids_;
  // precomputed 4x4: XXt
  real* diag_XXt_ = nullptr;
  real* XXt_ = nullptr;
  // reduction block offsets
  uint32_t** red_off_ = nullptr;
  uint32_t** red_half_off_ = nullptr;
  uint32_t* red_n_half_ = nullptr;
  uint32_t** A_low_off_ = nullptr;
  uint32_t** A_diag_off_ = nullptr;
  uint32_t** A_mirror_off_ = nullptr;
  // diag fix
  real* UtAU_diag_fix_ = nullptr;

  // interpolation matrix at finest
  SpMat* Uf_ = nullptr;
  // interpolation matrices below
  SpMat* U_ = nullptr;
  // system matrix A at finest
  BSMat<4, 4>* Af_ = nullptr;
  // system matrix As below
  BSMat<16, 16>* A_ = nullptr;

  // cuda context
  cublasHandle_t cublas_handle_;
  cusparseHandle_t cusparse_handle_;
  cusolverDnHandle_t cusolverDn_handle_;
  cusolverDnParams_t cusolverDn_params_;
  cusolverSpHandle_t cusolverSp_handle_;

  cusparseMatDescr_t descr_;
  cusparseMatDescr_t descrU_;
  cusparseMatDescr_t descrL_;

  real* pressure_ = nullptr;
  real* dpressure_ = nullptr;

  // vertex to tet indices multimapping
  int32_t* dv2t_ids_ = nullptr;  // indices
  int32_t* dv2t_off_ = nullptr;  // offset
  int32_t** dv2e_off_ = nullptr;
  int32_t** dedge_to_ = nullptr;
  bool* dtet_sign_ = nullptr;
  bool* dis_vert_surf_ = nullptr;
  int32_t* dnormal_sign_ = nullptr;

  Vec3* drest_verts_ = nullptr;
  // data for update
  Vec3* dold_verts_ = nullptr;
  Vec3* dinertia_verts_ = nullptr;
  real* dtet_grad_ = nullptr;
  real* dtet_p_grad_ = nullptr;
  real* dtet_vol_ = nullptr;
  // Af Kronecker XXt
  real* dAf_XXt_ = nullptr;
  real** drhs_ = nullptr;
  real** dlhs_ = nullptr;
  real** dtmp_ = nullptr;
  real** dtmp2_ = nullptr;
  real** dtAP_ = nullptr;
  real** dP_ = nullptr;
  real** dR_ = nullptr;
  real** dAP_ = nullptr;
  real** dZ_ = nullptr;

  real* C_ = nullptr;
  real* C_inv_ = nullptr;
  // real* C_inv_sym_ = nullptr;

  cusparseDnVecDescr_t* rhs_descr_;
  cusparseDnVecDescr_t* lhs_descr_;
  cusparseDnVecDescr_t* tmp_descr_;
  cusparseDnVecDescr_t* R_descr_;

  int32_t reduction_threads_per_block_ = 8;
  int32_t UltAUl_threads_per_block_ = 1;

  // self collision
  CudaTetHashing* hashing_ = nullptr;
  int32_t* closest_surf_vert_ = nullptr;
  int32_t* dclosest_surf_vert_ = nullptr;
  bool* is_vert_surf_ = nullptr;
  int32_t** degree_ = nullptr;
  int32_t* min_degree_ = nullptr;
  int32_t* max_degree_ = nullptr;
  int32_t** ddegree_ = nullptr;
  int32_t** dhas_color_ = nullptr;
  bool** dpalette_ = nullptr;
  int32_t** dpalette_size_ = nullptr;
  curandState** drand_states_ = nullptr;
  int32_t** dcolli_pairs_ = nullptr;
  real** dcolli_hessian_ = nullptr;
  int32_t** dcolli_v2e_ = nullptr;
  int32_t** dcolli_next_edge_ = nullptr;
  int32_t** dcolli_edge_to_ = nullptr;

  // diag addition
  real** ddiag_add_ = nullptr;
  // tet to matrix block offsets
  int32_t* dt2off_ = nullptr;
  // diag to matrix block offsets
  int32_t* dd2off_ = nullptr;
  // diag fix
  real* dUtAU_diag_fix_ = nullptr;
  // precomputed 4x4: XXt
  real* ddiag_XXt_ = nullptr;
  real* dXXt_ = nullptr;
  // reduction block offsets
  int32_t** dred_off_ = nullptr;
  int32_t** dred_half_off_ = nullptr;
  int32_t** dA_low_off_ = nullptr;
  int32_t** dA_diag_off_ = nullptr;
  int32_t** dA_mirror_off_ = nullptr;
  int32_t** dhandle_ = nullptr;
  int32_t* dhandle_ids_ = nullptr;
  int32_t** dcolor_ = nullptr;
  // real* dC_inv_sym_ = nullptr;
  real* dC_inv_ = nullptr;
  real* dGt_ = nullptr;

  size_t chol_host_buffer_size_ = 0;
  size_t chol_dev_buffer_size_ = 0;
  int32_t* dchol_info_;
  void* chol_dev_buffer_;
  void* chol_host_buffer_;
  real* dchol_fix_ = nullptr;

  uint32_t* tet_color_ = nullptr;
  uint32_t* dtet_color_ = nullptr;
  uint32_t n_tet_color_ = 0;
  int32_t* dupdated_ = nullptr;

  int32_t* marker_ = nullptr;
  Vec3 tforce_ = Vec3(0, 0, 0);
  Vec3* dtforce_ = nullptr;

  real vol_diff_ = 0.;

  // init cuda context
  void InitCudaContext();
  // build vertex to tet indices multimapping
  void BuildV2T();
  // build adjacent graph from tet
  void BuildGraph();
  // select handles for each level
  void SelectHandles(bool tofile);
  // compute interpolation matrx
  void ComputeUMatrices(bool tofile);
  // compute LHS matrix
  void ComputeAMatrices(bool tofile);
  // build update matrix auxiliary data
  void BuildUpdateAuxiliary();
  void BuildCollisionAuxiliary();
  void FixDiag();
  void Colorize();
  void TetColorize();
  // data to device
  void ToDevice();
  void ComputeAlpha();

  void KinematicCollision(real k_penalty,
                          const std::vector<CudaKinematicObject*>& kobjs,
                          cudaStream_t stream);
  void SelfCollision(real self_penalty, cudaStream_t stream);
  void Vivace(int32_t layer, cudaStream_t stream);
  void KaczmarzIteration(int32_t& layer, const int32_t max_iter, const real tol,
                         cudaStream_t stream);
  void PerformGSIteration(int32_t& layer, const int32_t max_iter,
                          const real tol, cudaStream_t stream);
  void CellVankaIteration(int32_t& layer, const int32_t max_iter,
                          const real tol, cudaStream_t stream);
  void InexactUzawaIteration(int32_t& layer, const int32_t max_iter,
                             const real tol, cudaStream_t stream);
  void RestrictedVankaIteration(int32_t& layer, const int32_t max_iter,
                                const real tol, cudaStream_t stream);
  void DownSample(int32_t& layer, cudaStream_t stream);
  void UpSample(int32_t& layer, cudaStream_t stream);
  void DirectSolve(int32_t& layer, cudaStream_t stream);
  void DirectSolveShur(int32_t& layer, cudaStream_t stream);
  void ComputeAP(int32_t l, real* AP, const real* alpha, const real* P,
                 cudaStream_t stream);
  void ComputeSelfcollisionOffAP(int32_t l, real* AP, const real* alpha,
                                 const real* P, cudaStream_t stream);
  real ComputeDistortionEnergy(cudaStream_t stream);
  // helper functions
  uint32_t GraphColoring(uint32_t* v2e_off, uint32_t* edge_to, uint32_t* color,
                         uint32_t n_vert, uint32_t n_color);
  void ComputeShortestPath(uint32_t source, LenFrom* shortest);
  void ComputeNormalSign();
  template <typename _type>
  void DeleteArray2D(_type** ptr, uint32_t size);
  template <typename _type>
  void DeleteCudaArray2D(_type** ptr, uint32_t size);
};
};  // namespace Rain