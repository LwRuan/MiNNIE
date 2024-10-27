#pragma once

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <deque>
#include <functional>
#include <map>
#include <vector>

#include "collision/spatialhashing.h"
#include "collision/tetspatialhashing.h"
#include "configtype.h"
#include "elasticobject.h"
#include "imgui.h"

namespace Rain {
struct ElasobjMG19InitInfo {
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
  uint32_t n_iter;
  uint32_t n_op;
  MGOpConfig* operations;
};

class ElasobjMG19 : public ElasticObject {
 public:
  void Init(ElasobjMG19InitInfo* info);
  virtual void Reset();
  virtual void Update(float dt, const Vec3& grav, real damping,
                      uint32_t n_substep, real substep_size, uint32_t n_frame,
                      const std::vector<KinematicObject*>& kobjs);
  virtual void Destroy();

  using RowCol = std::pair<uint32_t, uint32_t>;
  using LenFrom = std::pair<real, uint32_t>;
  using Mat12 = Eigen::Matrix<real, 12, 12>;
  using Pair = std::pair<uint32_t, uint32_t>;

  // for Gauss-Seidel
  const bool reordered_ = false;
  // for line search
  const bool line_search_ = true;
  // for fixed vertex constriants
  real control_mag_;
  // for system matrix
  real dt_;
  // relaxation for solver
  real relaxation_;
  // # iterations
  uint32_t n_iter_;
  // vertex to tet indices multimapping
  uint32_t* v2t_ids_ = nullptr;  // indices
  uint32_t* v2t_off_ = nullptr;  // offset
  // adjacent graph from tet
  uint32_t* v2e_off_ = nullptr;
  uint32_t* edge_to_ = nullptr;
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

  ESpMat Uf_;
  ESpMat* U_ = nullptr;
  ESpMat Af_;
  std::vector<ETriplet> Af_coo_;
  ESpMat Af_add_;
  std::vector<ETriplet> Af_add_coo_;
  ESpMat Af_composed_;
  MatX Af_dense_;
  ESpMat* A_ = nullptr;
  ESpMat UtAU_diag_fix_;
  ESpMat* diag_add_ = nullptr;

  Vec3* old_verts_ = nullptr;
  Vec3* inertia_verts_ = nullptr;
  Vec3* fixed_verts_ = nullptr;
  real* tet_grads_ = nullptr;
  real** rhs_ = nullptr;
  real** lhs_ = nullptr;
  real** tmp_ = nullptr;
  real** P_ = nullptr;
  real** R_ = nullptr;

  bool output_data_ = false;

  uint32_t* closest_surf_vert_ = nullptr;
  bool* is_vert_surf_ = nullptr;
  uint32_t n_colli_pair_ = 0;
  TetSpatialHashing spatial_hashing_;

  std::vector<Pair> colli_vert_pairs_;
  std::vector<int> colli_edge_next_;
  std::vector<int> colli_v2e_;

  Eigen::SimplicialLDLT<ESpMat> solver_;

  std::vector<MGOpConfig> operations_;
  std::deque<std::function<void()>> deletion_queue_;

  // build vertex to tet indices multimapping
  void BuildV2T();
  // build adjacent graph from tet
  void BuildGraph();
  // select handles for each level
  void SelectHandles();
  // select handles considering collision
  void SelectHandlesWithCollision();
  // reorder for Gauss-Seidel
  void ReOrder();
  // compute interpolation matrx
  void ComputeUMatrices(bool tofile);
  // compute LHS matrix
  void ComputeAMatrices(bool tofile);
  // build update matrix auxiliary data
  void BuildUpdateAuxiliary();
  // build collision auxiliary data
  void BuildCollisionAuxiliary();
  // fix diag
  void FixDiag();

  // Gauss Seidel
  void PerformGSIteration(int32_t& layer, const int32_t max_iter,
                          const real tol);
  // Jacobi
  void PerformJacobiIteration(const int32_t layer, const int32_t max_iter,
                              const real tol);
  // down sample
  void DownSample(int32_t& layer);
  // direct solve
  void DirectSolve(int32_t& layer);
  // up sample
  void UpSample(int32_t& layer);

  // helper functions
  void ComputeShortestPath(uint32_t source, LenFrom* shortest);
  void ComputeShortestPathWithCollision(uint32_t source, LenFrom* shortest);
  // coloring for Gauss-Seidel
  uint32_t Coloring(uint32_t* v2e_off, uint32_t* edge_to, uint32_t* color,
                    uint32_t n_vert, uint32_t n_color);
  template <typename _type>
  void DeleteArray2D(_type** ptr, uint32_t size);

  virtual void ShowUI();
};
};  // namespace Rain