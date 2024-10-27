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
struct CudaElasobjMG19InitInfo {
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
  bool quasi_static;
  bool* A_as_LDU;
  bool* A_as_dense;
  real dt;
  uint32_t n_iter;
  uint32_t n_op;
  MGOpConfig* operations;
  Joint* skeleton;
  int32_t n_joint;
};

class CudaElasobjMG19 : public CudaElasticObject {
 public:
  virtual void Init(CudaElasobjMG19InitInfo* info);
  virtual void Reset();
  virtual void ShowUI();
  virtual void Update(cudaStream_t stream, float dt, const Vec3& grav,
                      real damping, real kinematic_penalty, real self_penalty,
                      const std::vector<CudaKinematicObject*>& kobjs,
                      uint32_t n_substep, real substep_size, uint32_t n_frame);
  void Update_bk(cudaStream_t stream, float dt, const Vec3& grav, real damping,
                 real kinematic_penalty, real self_penalty,
                 const std::vector<CudaKinematicObject*>& kobjs,
                 uint32_t n_substep, real substep_size, uint32_t n_frame);
  virtual void Destroy();

  StepTimer timer_;

  using RowCol = std::pair<uint32_t, uint32_t>;
  using LenFrom = std::pair<real, uint32_t>;
  static const uint32_t MAXN_COLLI = 1000000;
  // for Gauss-Seidel
  static const bool reordered_ = false;
  // for line search
  static const bool line_search_ = false;
  bool quasi_static_;
  // for fixed vertex constriants
  real control_mag_;
  // for system matrix
  real dt_;
  // relaxation for solver
  real relaxation_;
  // # iterations
  uint32_t n_iter_;
  uint32_t n_line_iter_ = 0;
  // vertex to tet indices multimapping
  uint32_t* v2t_ids_ = nullptr;  // indices
  uint32_t* v2t_off_ = nullptr;  // offset
  // adjacent graph from tet
  uint32_t** v2e_off_ = nullptr;
  uint32_t** edge_to_ = nullptr;
  real* edge_len_ = nullptr;
  // # multigrid layers except the original layer
  uint32_t n_layer_;
  // # handles from bottom to top
  uint32_t* n_handle_ = nullptr;
  // matrix dims from bottom to top
  uint32_t* dim_ = nullptr;
  // handles mapping for each level
  uint32_t** handle_ = nullptr;
  // handle indices
  std::vector<uint32_t> handle_ids_;
  // handle to reordered matrix indice
  uint32_t** h2m_ = nullptr;
  // reordered matrix indice to handle
  uint32_t** m2h_ = nullptr;
  uint32_t** color_ = nullptr;
  // Gauss-Seidel color to handles multimapping
  uint32_t** c2h_off_ = nullptr;
  // # Gauss-Seidel colors for each level
  uint32_t* n_color_ = nullptr;
  // tet to matrix block offsets
  uint32_t* t2off_ = nullptr;
  // diag to matrix block offsets
  uint32_t* d2off_ = nullptr;
  // # store system matrix in LDU format
  bool* A_as_LDU_ = nullptr;
  bool* A_as_dense_ = nullptr;

  // interpolation matrix at finest
  // BSMat<3, 12>* Uf_ = nullptr;
  SpMat* Uf_ = nullptr;
  // interpolation matrices below
  // BSMat<12, 12>* U_ = nullptr;
  SpMat* U_ = nullptr;
  // system matrix A at finest
  BSMat<3, 3>* Af_ = nullptr;
  // system matrix As below
  BSMat<12, 12>* A_ = nullptr;

  // diag fix
  real* UtAU_diag_fix_ = nullptr;
  // precomputed 4x4: XXt
  real* diag_XXt_ = nullptr;
  real* XXt_ = nullptr;
  // matrix indices to coarser matrix indices
  uint32_t** mfine2coarse_ = nullptr;
  // reduction block offsets
  uint32_t** red_off_ = nullptr;
  uint32_t** red_half_off_ = nullptr;
  uint32_t* red_n_half_ = nullptr;
  uint32_t** A_low_off_ = nullptr;
  uint32_t** A_diag_off_ = nullptr;
  uint32_t** A_mirror_off_ = nullptr;

  // cuda context
  cublasHandle_t cublas_handle_;
  cusparseHandle_t cusparse_handle_;
  cusolverDnHandle_t cusolverDn_handle_;
  cusolverDnParams_t cusolverDn_params_;
  cusolverSpHandle_t cusolverSp_handle_;

  cusparseMatDescr_t descr_;
  cusparseMatDescr_t descrU_;
  cusparseMatDescr_t descrL_;

  // diag fix
  real* dUtAU_diag_fix_ = nullptr;
  // diag addition
  real** ddiag_add_ = nullptr;
  // handle to reordered matrix indice
  int32_t** dh2m_ = nullptr;
  int32_t** dm2h_ = nullptr;
  // tet to matrix block offsets
  int32_t* dt2off_ = nullptr;
  // diag to matrix block offsets
  int32_t* dd2off_ = nullptr;
  // precomputed 4x4: XXt
  real* ddiag_XXt_ = nullptr;
  real* dXXt_ = nullptr;
  // reduction block offsets
  int32_t** dred_off_ = nullptr;
  int32_t** dred_half_off_ = nullptr;
  int32_t** dA_low_off_ = nullptr;
  int32_t** dA_diag_off_ = nullptr;
  int32_t** dA_mirror_off_ = nullptr;
  // matrix indices to coarser matrix indices
  int32_t** dmfine2coarse_ = nullptr;
  // vertex to tet indices multimapping
  int32_t** dhandle_ = nullptr;
  int32_t* dhandle_ids_ = nullptr;
  int32_t* dv2t_ids_ = nullptr;  // indices
  int32_t* dv2t_off_ = nullptr;  // offset
  int32_t** dv2e_off_ = nullptr;
  int32_t** dedge_to_ = nullptr;
  bool* dtet_sign_ = nullptr;

  Vec3* drest_verts_ = nullptr;
  // data for update
  Vec3* dold_verts_ = nullptr;
  Vec3* dinertia_verts_ = nullptr;
  real* dtet_grad_ = nullptr;
  // Af Kronecker XXt
  real* dAf_XXt_ = nullptr;
  real** drhs_ = nullptr;
  real** dlhs_ = nullptr;
  real** dtmp_ = nullptr;
  real** dtAP_ = nullptr;
  real** dP_ = nullptr;
  real** dR_ = nullptr;
  real** dAP_ = nullptr;
  real** dZ_ = nullptr;

  cusparseDnVecDescr_t* rhs_descr_;
  cusparseDnVecDescr_t* lhs_descr_;
  cusparseDnVecDescr_t* tmp_descr_;
  cusparseDnVecDescr_t* R_descr_;

  int32_t reduction_threads_per_block_ = 8;
  int32_t UltAUl_threads_per_block_ = 1;

  std::vector<MGOpConfig> operations_;

  std::deque<std::function<void()>> deletion_queue_;

  // self collision
  bool* dis_vert_surf_ = nullptr;
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
  int32_t** dcolor_ = nullptr;

  real rest_vol_;
  real* dtet_vol_ = nullptr;

  int32_t* marker_ = nullptr;
  Vec3 tforce_ = Vec3(0, 0, 0);
  Vec3* dtforce_ = nullptr;

  // init cuda context
  void InitCudaContext();
  // build vertex to tet indices multimapping
  void BuildV2T();
  // build adjacent graph from tet
  void BuildGraph();
  // select handles for each level
  void SelectHandles(bool tofile);
  // reorder for Gauss-Seidel
  // void ReOrder();
  void Colorize();
  // compute interpolation matrx
  void ComputeUMatrices(bool tofile);
  // compute LHS matrix
  void ComputeAMatrices(bool tofile);
  // build update matrix auxiliary data
  void BuildUpdateAuxiliary();
  // data to device
  void ToDevice();
  // fix diag
  void FixDiag();

  void KinematicCollision(real k_penalty,
                          const std::vector<CudaKinematicObject*>& kobjs,
                          cudaStream_t stream);
  void KinematicCollisionEnergy(real k_penalty,
                                const std::vector<CudaKinematicObject*>& kobjs,
                                real* E, cudaStream_t stream);
  void SelfCollision(real k_penalty, cudaStream_t stream);
  void Vivace(int32_t layer, cudaStream_t stream);
  // Gauss Seidel
  void PerformGSIteration(int32_t& layer, const int32_t max_iter,
                          const real tol, cudaStream_t stream);

  void PerformGSIteration_bk(int32_t& layer, const int32_t max_iter,
                             const real tol, cudaStream_t stream);
  // Jacobi
  void PerformJacobiIteration(const int32_t layer, const int32_t max_iter,
                              const real tol, cudaStream_t stream);
  // down sample
  void DownSample(int32_t& layer, cudaStream_t stream);
  // direct solve
  void DirectSolve(int32_t& layer, cudaStream_t stream);
  // up sample
  void UpSample(int32_t& layer, cudaStream_t stream);

  // helper functions
  uint32_t GraphColoring(uint32_t* v2e_off, uint32_t* edge_to, uint32_t* color,
                         uint32_t n_vert, uint32_t n_color);
  void ComputeShortestPath(uint32_t source, LenFrom* shortest);
  // coloring for Gauss-Seidel
  uint32_t Coloring(uint32_t* v2e_off, uint32_t* edge_to, uint32_t* color,
                    uint32_t n_vert, uint32_t n_color);
  // compute AP
  void ComputeAP(int32_t l, real* AP, const real* alpha, const real* P,
                 cudaStream_t stream);
  void ComputeSelfcollisionOffAP(int32_t l, real* AP, const real* alpha,
                                 const real* P, cudaStream_t stream);
  real ComputeEnergy(const Vec3* inertia_X, const Vec3* X, const real k_penalty,
                     const std::vector<CudaKinematicObject*>& kobjs,
                     cudaStream_t stream);
  void ComputeGradientReordered(const Vec3* inertia_X, const Vec3* X, real* out,
                                cudaStream_t stream);
  // void TestGradient(cudaStream_t stream);
  void TestHessian(cudaStream_t stream);
  void BuildCollisionAuxiliary();
  void UpdateXXt(const Vec3* X, cudaStream_t stream);
  void UpdateU(const Vec3* X, cudaStream_t stream);
  template <typename _type>
  void DeleteArray2D(_type** ptr, uint32_t size);
  template <typename _type>
  void DeleteCudaArray2D(_type** ptr, uint32_t size);
};
};  // namespace Rain