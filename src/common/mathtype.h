#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <type_traits>

#include "cudahelper.h"

namespace Rain {
// #define REAL_AS_DOUBLE

#ifdef REAL_AS_DOUBLE
using real = double;
#else
using real = float;
#endif

using Vec2f = Eigen::Vector2f;
using Vec2d = Eigen::Vector2d;
using Vec2 = Eigen::Matrix<real, 2, 1>;
using Vec2i = Eigen::Vector2i;
using Mat2f = Eigen::Matrix2f;
using Mat2d = Eigen::Matrix2d;
using Mat2 = Eigen::Matrix<real, 2, 2>;
using Mat2i = Eigen::Matrix2i;

using Vec3f = Eigen::Vector3f;
using Vec3d = Eigen::Vector3d;
using Vec3 = Eigen::Matrix<real, 3, 1>;
using Vec3i = Eigen::Vector3i;
using Mat3f = Eigen::Matrix3f;
using Mat3d = Eigen::Matrix3d;
using Mat3 = Eigen::Matrix<real, 3, 3>;
using Mat3i = Eigen::Matrix3i;

using Vec4f = Eigen::Vector4f;
using Vec4d = Eigen::Vector4d;
using Vec4 = Eigen::Matrix<real, 4, 1>;
using Vec4i = Eigen::Vector4i;
using Mat4f = Eigen::Matrix4f;
using Mat4d = Eigen::Matrix4d;
using Mat4 = Eigen::Matrix<real, 4, 4>;
using Mat4i = Eigen::Matrix4i;

using VecXf = Eigen::VectorXf;
using VecXd = Eigen::VectorXd;
using VecX = Eigen::Matrix<real, -1, 1>;
using VecXi = Eigen::VectorXi;
using MatXf = Eigen::MatrixXf;
using MatXd = Eigen::MatrixXd;
using MatX = Eigen::Matrix<real, -1, -1>;
using MatXi = Eigen::MatrixXi;

using Quat = Eigen::Quaternion<real>;
using Quatf = Eigen::Quaternion<float>;
using Quatd = Eigen::Quaternion<double>;

using Vec6f = Eigen::Matrix<float, 6, 1>;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Vec6i = Eigen::Matrix<int, 6, 1>;
using Vec6 = Eigen::Matrix<real, 6, 1>;
using Vec9f = Eigen::Matrix<float, 9, 1>;
using Vec9d = Eigen::Matrix<double, 9, 1>;
using Vec9i = Eigen::Matrix<int, 9, 1>;
using Vec9 = Eigen::Matrix<real, 9, 1>;
using Mat6f = Eigen::Matrix<float, 6, 6>;
using Mat6d = Eigen::Matrix<double, 6, 6>;
using Mat6 = Eigen::Matrix<real, 6, 6>;
using Mat6i = Eigen::Matrix<int, 6, 6>;
using Mat3x6f = Eigen::Matrix<float, 3, 6>;
using Mat3x6d = Eigen::Matrix<double, 3, 6>;
using Mat3x6 = Eigen::Matrix<real, 3, 6>;
using Mat3x6i = Eigen::Matrix<int, 3, 6>;
using Mat6x3f = Eigen::Matrix<float, 6, 3>;
using Mat6x3d = Eigen::Matrix<double, 6, 3>;
using Mat6x3 = Eigen::Matrix<real, 6, 3>;
using Mat6x3i = Eigen::Matrix<int, 6, 3>;
using Mat9 = Eigen::Matrix<real, 9, 9>;
using Mat9d = Eigen::Matrix<double, 9, 9>;
using Mat9f = Eigen::Matrix<float, 9, 9>;
using Mat12 = Eigen::Matrix<real, 12, 12>;
using Mat16 = Eigen::Matrix<real, 16, 16>;

template <int d>
using Vec = Eigen::Matrix<real, d, 1>;
template <int d>
using Mat = Eigen::Matrix<real, d, d>;

using ESpMat = Eigen::SparseMatrix<real>;
using ETriplet = Eigen::Triplet<real>;

constexpr float PI_ = 3.14159265f;
constexpr cudaDataType_t CudaRealType =
    std::is_same<real, float>() ? CUDA_R_32F : CUDA_R_64F;

using RowCol = std::pair<uint32_t, uint32_t>;
template <typename _type>
class Triplet {
 public:
  uint32_t row_;
  uint32_t col_;
  _type val_;

  bool operator<(const Triplet<_type>& r) {
    return row_ < r.row_ || (!(row_ < r.row_) && col_ < r.col_);
  }
};

struct JointTransform {
  Vec3 global_pos;
  Mat3 global_rot;
};

template <typename _type>
class SparseMatrix {
 public:
  uint32_t rows_;
  uint32_t cols_;
  uint32_t nnz_;

  // csr
  uint32_t* csr_row_ = nullptr;
  uint32_t* csr_col_ = nullptr;
  _type* csr_val_ = nullptr;
  // coo
  uint32_t* coo_row_ = nullptr;
  uint32_t* coo_col_ = nullptr;
  _type* coo_val_ = nullptr;  // can be the same as csr_val_
  // csc
  uint32_t* csc_col_ = nullptr;
  uint32_t* csc_row_ = nullptr;
  _type* csc_val_ = nullptr;

  // GPU
  // not uint32_t, but int32_t, for cusparse
  // csr
  int32_t* dcsr_row_ = nullptr;
  int32_t* dcsr_col_ = nullptr;
  _type* dcsr_val_ = nullptr;
  // coo
  int32_t* dcoo_row_ = nullptr;
  int32_t* dcoo_col_ = nullptr;
  _type* dcoo_val_ = nullptr;
  // csc
  int32_t* dcsc_col_ = nullptr;
  int32_t* dcsc_row_ = nullptr;
  _type* dcsc_val_ = nullptr;
  // den
  _type* dden_val_ = nullptr;
  cusparseSpMatDescr_t spmat_descr_;
  cusparseSpMatDescr_t trans_descr_;

  std::deque<std::function<void()>> deletion_queue_;

  void Init(uint32_t rows, uint32_t cols);
  void Destroy();
  void CsrToDevice(cusparseHandle_t cusparse_handle);
  void OutputCsr(std::ostream& out);
};

template <typename _type>
class BlockSparseMatrix2 {
 public:
  uint32_t _brow;
  uint32_t _bcol;
  uint32_t brows_;
  uint32_t bcols_;
  uint32_t rows_;
  uint32_t cols_;
  uint32_t nnz_;   // # non-zero
  uint32_t bnnz_;  // # non-zero block
  // stored as lower-tri, diag, upper-tri
  bool as_LDU_ = false;
  // stored as dense matrix, col major
  bool as_dense_ = false;

  // coo
  uint32_t* bcoo_row_ = nullptr;  // block
  uint32_t* bcoo_col_ = nullptr;  // block
  _type* bcoo_val_ = nullptr;     // can be the same as csr_val_

  // diagonal part
  // if not as_LDU_, diagonal part is duplicated from full matrix
  // diag_offset_ is the offset in bcoo
  _type* diag_val_ = nullptr;
  uint32_t* diag_boff_ = nullptr;

  // LDU
  uint32_t low_bnnz_;
  uint32_t diag_bnnz_;
  uint32_t up_bnnz_;
  uint32_t* low_bcoo_row_ = nullptr;
  uint32_t* low_bcoo_col_ = nullptr;
  _type* low_bcoo_val_ = nullptr;
  uint32_t* diag_bcoo_row_ = nullptr;
  uint32_t* diag_bcoo_col_ = nullptr;
  _type* diag_bcoo_val_ = nullptr;
  uint32_t* up_bcoo_row_ = nullptr;
  uint32_t* up_bcoo_col_ = nullptr;
  _type* up_bcoo_val_ = nullptr;

  // Gauss-Seidel
  // multiple matrices in one buffer
  uint32_t GS_n_color_;
  uint32_t* GS_c2m_off_ = nullptr;
  uint32_t* GS_low_bcoo_row_ = nullptr;
  uint32_t* GS_low_bcoo_col_ = nullptr;
  uint32_t* GS_low_boff_ = nullptr;
  uint32_t* GS_up_bcoo_row_ = nullptr;
  uint32_t* GS_up_bcoo_col_ = nullptr;
  uint32_t* GS_up_boff_ = nullptr;

  // Cholesky factorization
  size_t chol_host_buffer_size_;
  size_t chol_dev_buffer_size_;
  void* chol_host_buffer_;

  // GPU
  // not uint32_t, but int32_t, for cusparse
  _type* dden_val_ = nullptr;

  int32_t* dbcoo_row_ = nullptr;
  int32_t* dbcoo_col_ = nullptr;
  _type* dbcoo_val_ = nullptr;
  int32_t* dbcsr_row_ = nullptr;
  int32_t* dbcsr_col_ = nullptr;
  _type* dbcsr_val_ = nullptr;

  int32_t* dlow_bcoo_row_ = nullptr;
  int32_t* dlow_bcoo_col_ = nullptr;
  _type* dlow_bcoo_val_ = nullptr;
  int32_t* dlow_bcsr_row_ = nullptr;
  int32_t* dlow_bcsr_col_ = nullptr;
  _type* dlow_bcsr_val_ = nullptr;
  int32_t* ddiag_bcoo_row_ = nullptr;
  int32_t* ddiag_bcoo_col_ = nullptr;
  _type* ddiag_bcoo_val_ = nullptr;
  int32_t* ddiag_bcsr_row_ = nullptr;
  int32_t* ddiag_bcsr_col_ = nullptr;
  _type* ddiag_bcsr_val_ = nullptr;
  int32_t* dup_bcoo_row_ = nullptr;
  int32_t* dup_bcoo_col_ = nullptr;
  _type* dup_bcoo_val_ = nullptr;
  int32_t* dup_bcsr_row_ = nullptr;
  int32_t* dup_bcsr_col_ = nullptr;
  _type* dup_bcsr_val_ = nullptr;

  _type* ddiag_val_ = nullptr;
  int32_t* ddiag_boff_ = nullptr;

  int32_t* dGS_low_bcoo_row_ = nullptr;
  int32_t* dGS_low_bcoo_col_ = nullptr;
  int32_t* dGS_low_bcsr_row_ = nullptr;
  int32_t* dGS_low_bcsr_col_ = nullptr;
  int32_t* dGS_up_bcoo_row_ = nullptr;
  int32_t* dGS_up_bcoo_col_ = nullptr;
  int32_t* dGS_up_bcsr_row_ = nullptr;
  int32_t* dGS_up_bcsr_col_ = nullptr;

  // Cholesky factorization
  int32_t* dchol_info_;
  _type* dchol_val_;
  _type* dchol_fixed_;
  void* chol_dev_buffer_;

  std::deque<std::function<void()>> deletion_queue_;

  void Init(uint32_t brows, uint32_t bcols, uint32_t _brow, uint32_t _bcol,
            bool as_LDU = false, bool as_dense = false);
  void Destroy();

  void FromCooMapSym(const std::map<RowCol, std::unique_ptr<real[]>>& coo);
  void BuildGS(const uint32_t* c2h_off, const int n_color);
  void BCooSymToDevice(cusparseHandle_t cusparse_handle);
  void GSToDevice(cusparseHandle_t cusparse_handle);
  void CholToDevice(cusolverDnHandle_t cusolverDn_handle,
                    cusolverDnParams_t cusolverDn_params);

  void OutputBlockCoo(std::ostream& out);
  void OutputLowBlockCoo(std::ostream& out);
};

template <typename _type, int _brow, int _bcol>
class BlockSparseMatrix {
 public:
  using Block = Eigen::Matrix<_type, _brow, _bcol>;
  uint32_t brows_;
  uint32_t bcols_;
  uint32_t rows_;
  uint32_t cols_;
  uint32_t nnz_;   // # non-zero
  uint32_t bnnz_;  // # non-zero block
  // stored as lower-tri, diag, upper-tri
  bool as_LDU_ = false;
  // stored as dense matrix, col major
  bool as_dense_ = false;

  // coo
  uint32_t* bcoo_row_ = nullptr;  // block
  uint32_t* bcoo_col_ = nullptr;  // block
  _type* bcoo_val_ = nullptr;     // can be the same as csr_val_

  // diagonal part
  // if not as_LDU_, diagonal part is duplicated from full matrix
  // diag_offset_ is the offset in bcoo
  _type* diag_val_ = nullptr;
  uint32_t* diag_boff_ = nullptr;

  // LDU
  uint32_t low_bnnz_;
  uint32_t diag_bnnz_;
  uint32_t up_bnnz_;
  uint32_t* low_bcoo_row_ = nullptr;
  uint32_t* low_bcoo_col_ = nullptr;
  _type* low_bcoo_val_ = nullptr;
  uint32_t* diag_bcoo_row_ = nullptr;
  uint32_t* diag_bcoo_col_ = nullptr;
  _type* diag_bcoo_val_ = nullptr;
  uint32_t* up_bcoo_row_ = nullptr;
  uint32_t* up_bcoo_col_ = nullptr;
  _type* up_bcoo_val_ = nullptr;

  // Gauss-Seidel
  // multiple matrices in one buffer
  uint32_t GS_n_color_;
  uint32_t* GS_c2m_off_ = nullptr;
  uint32_t* GS_low_bcoo_row_ = nullptr;
  uint32_t* GS_low_bcoo_col_ = nullptr;
  uint32_t* GS_low_boff_ = nullptr;
  uint32_t* GS_up_bcoo_row_ = nullptr;
  uint32_t* GS_up_bcoo_col_ = nullptr;
  uint32_t* GS_up_boff_ = nullptr;

  // Cholesky factorization
  size_t chol_host_buffer_size_;
  size_t chol_dev_buffer_size_;
  void* chol_host_buffer_;
  

  // GPU
  // not uint32_t, but int32_t, for cusparse
  _type* dden_val_ = nullptr;

  int32_t* dbcoo_row_ = nullptr;
  int32_t* dbcoo_col_ = nullptr;
  _type* dbcoo_val_ = nullptr;
  int32_t* dbcsr_row_ = nullptr;
  int32_t* dbcsr_col_ = nullptr;
  _type* dbcsr_val_ = nullptr;

  int32_t* dlow_bcoo_row_ = nullptr;
  int32_t* dlow_bcoo_col_ = nullptr;
  _type* dlow_bcoo_val_ = nullptr;
  int32_t* dlow_bcsr_row_ = nullptr;
  int32_t* dlow_bcsr_col_ = nullptr;
  _type* dlow_bcsr_val_ = nullptr;
  int32_t* ddiag_bcoo_row_ = nullptr;
  int32_t* ddiag_bcoo_col_ = nullptr;
  _type* ddiag_bcoo_val_ = nullptr;
  int32_t* ddiag_bcsr_row_ = nullptr;
  int32_t* ddiag_bcsr_col_ = nullptr;
  _type* ddiag_bcsr_val_ = nullptr;
  int32_t* dup_bcoo_row_ = nullptr;
  int32_t* dup_bcoo_col_ = nullptr;
  _type* dup_bcoo_val_ = nullptr;
  int32_t* dup_bcsr_row_ = nullptr;
  int32_t* dup_bcsr_col_ = nullptr;
  _type* dup_bcsr_val_ = nullptr;

  _type* ddiag_val_ = nullptr;
  int32_t* ddiag_boff_ = nullptr;

  int32_t* dGS_low_bcoo_row_ = nullptr;
  int32_t* dGS_low_bcoo_col_ = nullptr;
  int32_t* dGS_low_bcsr_row_ = nullptr;
  int32_t* dGS_low_bcsr_col_ = nullptr;
  int32_t* dGS_up_bcoo_row_ = nullptr;
  int32_t* dGS_up_bcoo_col_ = nullptr;
  int32_t* dGS_up_bcsr_row_ = nullptr;
  int32_t* dGS_up_bcsr_col_ = nullptr;

  // Cholesky factorization
  int32_t* dchol_info_;
  _type* dchol_val_;
  _type* dchol_fixed_;
  void* chol_dev_buffer_;
  int32_t ldlt_dev_buffer_size_;
  int32_t* dldlt_info_;
  _type* dldlt_val_;
  // _type* dldlt_fixed_;
  void* ldlt_dev_buffer_;
  int32_t* dldlt_ipiv_;
  int64_t* dldlt_ipiv_64_;

  std::deque<std::function<void()>> deletion_queue_;

  void Init(uint32_t brows, uint32_t bcols, bool as_LDU = false,
            bool as_dense = false);
  void Destroy();

  void FromCooMapSym(const std::map<RowCol, Block>& coo);
  void BuildGS(const uint32_t* c2h_off, const int n_color);
  void UpdateDiag(cudaStream_t stream);
  void BCooSymToDevice(cusparseHandle_t cusparse_handle);
  void GSToDevice(cusparseHandle_t cusparse_handle);
  void CholToDevice(cusolverDnHandle_t cusolverDn_handle,
                    cusolverDnParams_t cusolverDn_params);
  void LDLTToDevice(cusolverDnHandle_t cusolverDn_handle,
                    cusolverDnParams_t cusolverDn_params);

  void OutputBlockCoo(std::ostream& out);
  void OutputLowBlockCoo(std::ostream& out);
};

using Tri = Triplet<real>;
template <int _brow, int _bcol>
using BSMat = BlockSparseMatrix<real, _brow, _bcol>;
using BSMat2 = BlockSparseMatrix2<real>;
using SpMat = SparseMatrix<real>;
};  // namespace Rain
