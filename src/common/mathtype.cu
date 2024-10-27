#include <spdlog/spdlog.h>

#include <type_traits>

#include "mathtype.h"

namespace Rain {
template class BlockSparseMatrix<real, 3, 3>;
template class BlockSparseMatrix<real, 4, 4>;
template class BlockSparseMatrix<real, 12, 12>;
template class BlockSparseMatrix<real, 16, 16>;
template class BlockSparseMatrix<real, 3, 12>;
template class BlockSparseMatrix<real, 12, 3>;
template class BlockSparseMatrix2<real>;
template class Triplet<real>;
template class SparseMatrix<real>;

template <typename _type>
void SparseMatrix<_type>::Init(uint32_t rows, uint32_t cols) {
  rows_ = rows;
  cols_ = cols;
}

template <typename _type>
void SparseMatrix<_type>::Destroy() {
  for (auto it = deletion_queue_.rbegin(); it != deletion_queue_.rend(); it++)
    (*it)();
  deletion_queue_.clear();
}

template <typename _type>
void SparseMatrix<_type>::CsrToDevice(cusparseHandle_t cusparse_handle) {
  CheckCuda(cudaMalloc((void**)&dcsr_row_, sizeof(int32_t) * (rows_ + 1)));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dcsr_row_)); });
  CheckCuda(cudaMalloc((void**)&dcsr_col_, sizeof(int32_t) * nnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dcsr_col_)); });
  CheckCuda(cudaMalloc((void**)&dcsr_val_, sizeof(_type) * nnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dcsr_val_)); });
  CheckCuda(cudaMemcpy(dcsr_row_, csr_row_, sizeof(int32_t) * (rows_ + 1),
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(dcsr_col_, csr_col_, sizeof(int32_t) * nnz_,
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(dcsr_val_, csr_val_, sizeof(_type) * nnz_,
                       cudaMemcpyHostToDevice));
  // CheckCuda(cudaMalloc((void**)&dcsc_col_, sizeof(int32_t) * (cols_ + 1)));
  // deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dcsc_col_)); });
  // CheckCuda(cudaMalloc((void**)&dcsc_row_, sizeof(int32_t) * nnz_));
  // deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dcsc_row_)); });
  // CheckCuda(cudaMalloc((void**)&dcsc_val_, sizeof(_type) * nnz_));
  // deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dcsc_val_)); });

  constexpr cudaDataType_t _RealType =
      std::is_same<_type, float>() ? CUDA_R_32F : CUDA_R_64F;

  CheckCuda(cusparseCreateCsr(&spmat_descr_, rows_, cols_, nnz_, dcsr_row_,
                              dcsr_col_, dcsr_val_, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                              _RealType));
  deletion_queue_.push_back(
      [=]() { CheckCuda(cusparseDestroySpMat(spmat_descr_)); });

  CheckCuda(cusparseCreateCsc(&trans_descr_, cols_, rows_, nnz_, dcsr_row_,
                              dcsr_col_, dcsr_val_, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                              _RealType));
  deletion_queue_.push_back(
      [=]() { CheckCuda(cusparseDestroySpMat(trans_descr_)); });
}

template <typename _type>
void SparseMatrix<_type>::OutputCsr(std::ostream& out) {
  for (auto r = 0; r < rows_; ++r) {
    for (auto i = csr_row_[r]; i < csr_row_[r + 1]; ++i) {
      const uint32_t& c = csr_col_[i];
      const _type& v = csr_val_[i];
      out << r << " " << c << " " << v << std::endl;
    }
  }
}

template <typename _type>
void BlockSparseMatrix2<_type>::Init(uint32_t brows, uint32_t bcols,
                                     uint32_t brow, uint32_t bcol, bool as_LDU,
                                     bool as_dense) {
  if (as_LDU_) assert(brows == bcols);
  brows_ = brows;
  bcols_ = bcols;
  _brow = brow;
  _bcol = bcol;
  rows_ = brows * _brow;
  cols_ = bcols * _bcol;
  as_LDU_ = as_LDU;
  as_dense_ = as_dense;
}

template <typename _type>
void BlockSparseMatrix2<_type>::Destroy() {
  for (auto it = deletion_queue_.rbegin(); it != deletion_queue_.rend(); it++)
    (*it)();
  deletion_queue_.clear();
};

template <typename _type>
void BlockSparseMatrix2<_type>::FromCooMapSym(
    const std::map<RowCol, std::unique_ptr<real[]>>& coo) {
  bnnz_ = coo.size();
  nnz_ = coo.size() * _brow * _bcol;
  low_bnnz_ = up_bnnz_ = (bnnz_ - brows_) / 2;
  diag_bnnz_ = brows_;

  bcoo_val_ = new _type[nnz_];
  deletion_queue_.push_back([=]() { delete[] bcoo_val_; });
  bcoo_row_ = new uint32_t[bnnz_];
  deletion_queue_.push_back([=]() { delete[] bcoo_row_; });
  bcoo_col_ = new uint32_t[bnnz_];
  deletion_queue_.push_back([=]() { delete[] bcoo_col_; });
  if (as_LDU_) {
    uint32_t diag_boff = low_bnnz_;
    uint32_t up_boff = low_bnnz_ + brows_;
    low_bcoo_row_ = bcoo_row_;
    low_bcoo_col_ = bcoo_col_;
    low_bcoo_val_ = bcoo_val_;
    diag_bcoo_row_ = bcoo_row_ + diag_boff;
    diag_bcoo_col_ = bcoo_col_ + diag_boff;
    diag_bcoo_val_ = bcoo_val_ + diag_boff * _brow * _bcol;
    diag_val_ = diag_bcoo_val_;
    up_bcoo_row_ = bcoo_row_ + up_boff;
    up_bcoo_col_ = bcoo_col_ + up_boff;
    up_bcoo_val_ = bcoo_val_ + up_boff * _brow * _bcol;
  } else {
    diag_val_ = new _type[brows_ * _brow * _bcol];
    deletion_queue_.push_back([=]() { delete[] diag_val_; });
  }
  diag_boff_ = new uint32_t[brows_];
  deletion_queue_.push_back([=]() { delete[] diag_boff_; });

  uint32_t idx_f = 0, idx_d = 0, idx_u = 0, idx_l = 0;
  for (const auto& iter : coo) {
    uint32_t r = iter.first.first;
    uint32_t c = iter.first.second;
    const std::unique_ptr<real[]>& v = iter.second;
    if (as_LDU_) {
      if (r == c) {  // D
        diag_bcoo_row_[idx_d] = r;
        diag_bcoo_col_[idx_d] = c;
        for (int i = 0; i < _brow; ++i)
          for (int j = 0; j < _bcol; ++j) {
            uint32_t idx = idx_d * _brow * _bcol + i * _bcol + j;
            diag_val_[idx] = v[i * _bcol + j];
          }
        diag_boff_[r] = low_bnnz_ + idx_d;
        ++idx_d;
      } else if (r < c) {  // U
        up_bcoo_row_[idx_u] = r;
        up_bcoo_col_[idx_u] = c;
        for (int i = 0; i < _brow; ++i)
          for (int j = 0; j < _bcol; ++j) {
            uint32_t idx = idx_u * _brow * _bcol + i * _bcol + j;
            up_bcoo_val_[idx] = v[i * _bcol + j];
          }
        ++idx_u;
      } else if (r > c) {  // L
        low_bcoo_row_[idx_l] = r;
        low_bcoo_col_[idx_l] = c;
        for (int i = 0; i < _brow; ++i)
          for (int j = 0; j < _bcol; ++j) {
            uint32_t idx = idx_l * _brow * _bcol + i * _bcol + j;
            low_bcoo_val_[idx] = v[i * _bcol + j];
          }
        ++idx_l;
      }
    } else {
      bcoo_row_[idx_f] = r;
      bcoo_col_[idx_f] = c;
      for (int i = 0; i < _brow; ++i)
        for (int j = 0; j < _bcol; ++j) {
          uint32_t idx = idx_f * _brow * _bcol + i * _bcol + j;
          bcoo_val_[idx] = v[i * _bcol + j];
        }
      if (r == c) {
        for (int i = 0; i < _brow; ++i)
          for (int j = 0; j < _bcol; ++j) {
            uint32_t idx = idx_d * _brow * _bcol + i * _bcol + j;
            diag_val_[idx] = v[i * _bcol + j];
          }
        diag_boff_[r] = idx_f;
        ++idx_d;
      }
      ++idx_f;
    }
  }
}

template <typename _type>
void BlockSparseMatrix2<_type>::BuildGS(const uint32_t* c2h_off,
                                        const int n_color) {
  if (!as_LDU_) {
    spdlog::error("Gauss Seidel solver need matrix saved in LDU or dense");
    exit(1);
  }
  GS_n_color_ = n_color;
  GS_c2m_off_ = new uint32_t[n_color + 1];
  deletion_queue_.push_back([=]() { delete[] GS_c2m_off_; });
  memcpy(GS_c2m_off_, c2h_off, sizeof(uint32_t) * (n_color + 1));
  GS_low_bcoo_row_ = new uint32_t[low_bnnz_];
  deletion_queue_.push_back([=]() { delete[] GS_low_bcoo_row_; });
  GS_low_bcoo_col_ = new uint32_t[low_bnnz_];
  deletion_queue_.push_back([=]() { delete[] GS_low_bcoo_col_; });
  GS_low_boff_ = new uint32_t[n_color + 1];
  deletion_queue_.push_back([=]() { delete[] GS_low_boff_; });
  GS_up_bcoo_row_ = new uint32_t[up_bnnz_];
  deletion_queue_.push_back([=]() { delete[] GS_up_bcoo_row_; });
  GS_up_bcoo_col_ = new uint32_t[up_bnnz_];
  deletion_queue_.push_back([=]() { delete[] GS_up_bcoo_col_; });
  GS_up_boff_ = new uint32_t[n_color + 1];
  deletion_queue_.push_back([=]() { delete[] GS_up_boff_; });
  int color = -1;
  uint32_t off_now = GS_c2m_off_[0];  // 0
  uint32_t off_next = GS_c2m_off_[1];
  for (auto i = 0; i < low_bnnz_; ++i) {
    uint32_t r = low_bcoo_row_[i];
    uint32_t c = low_bcoo_col_[i];
    while (r >= off_next) {
      ++color;
      GS_low_boff_[color] = i;
      off_now = GS_c2m_off_[color + 1];
      off_next = GS_c2m_off_[color + 2];
    }
    GS_low_bcoo_row_[i] = r - off_now;
    GS_low_bcoo_col_[i] = c;
  }
  while (color < n_color - 1) GS_low_boff_[++color] = low_bnnz_;

  uint32_t row_off = 0;
  uint32_t col_off = 0;
  color = -1;
  for (auto i = 0; i < up_bnnz_; ++i) {
    uint32_t r = up_bcoo_row_[i];
    uint32_t c = up_bcoo_col_[i];
    while (r >= col_off) {
      ++color;
      GS_up_boff_[color] = i;
      row_off = GS_c2m_off_[color];
      col_off = GS_c2m_off_[color + 1];
    }
    GS_up_bcoo_row_[i] = r - row_off;
    GS_up_bcoo_col_[i] = c - col_off;
  }
  while (color < n_color - 1) GS_up_boff_[++color] = up_bnnz_;

  // {  // Test Code
  //   std::cout << "GS" << std::endl;
  //   for (int i = 0; i < n_color; ++i) {
  //     std::cout << GS_low_boff_[i] << " ";
  //   }
  //   std::cout << std::endl;
  //   for (int i = 0; i < n_color; ++i) {
  //     std::cout << GS_up_boff_[i] << " ";
  //   }
  //   std::cout << std::endl;
  // }
}

template <typename _type>
void BlockSparseMatrix2<_type>::BCooSymToDevice(
    cusparseHandle_t cusparse_handle) {
  CheckCuda(cudaMalloc(&dbcoo_val_, sizeof(_type) * nnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dbcoo_val_)); });
  CheckCuda(cudaMalloc(&dbcoo_row_, sizeof(int32_t) * bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dbcoo_row_)); });
  CheckCuda(cudaMalloc(&dbcoo_col_, sizeof(int32_t) * bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dbcoo_col_)); });
  CheckCuda(cudaMalloc(&ddiag_boff_, sizeof(int32_t) * brows_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(ddiag_boff_)); });
  if (as_LDU_) {
    CheckCuda(cudaMemcpy(dbcoo_val_, bcoo_val_, sizeof(_type) * nnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(dbcoo_row_, bcoo_row_, sizeof(int32_t) * bnnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(dbcoo_col_, bcoo_col_, sizeof(int32_t) * bnnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(ddiag_boff_, diag_boff_, sizeof(int32_t) * brows_,
                         cudaMemcpyHostToDevice));
    dlow_bcoo_row_ = dbcoo_row_;
    dlow_bcoo_col_ = dbcoo_col_;
    dlow_bcoo_val_ = dbcoo_val_;
    ddiag_bcoo_row_ = dbcoo_row_ + low_bnnz_;
    ddiag_bcoo_col_ = dbcoo_col_ + low_bnnz_;
    ddiag_bcoo_val_ = dbcoo_val_ + low_bnnz_ * _brow * _bcol;
    dup_bcoo_row_ = dbcoo_row_ + low_bnnz_ + diag_bnnz_;
    dup_bcoo_col_ = dbcoo_col_ + low_bnnz_ + diag_bnnz_;
    dup_bcoo_val_ = dbcoo_val_ + (low_bnnz_ + diag_bnnz_) * _brow * _bcol;
    // L
    CheckCuda(cudaMalloc(&dlow_bcsr_row_, sizeof(int32_t) * (brows_ + 1)));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dlow_bcsr_row_)); });
    dlow_bcsr_col_ = dlow_bcoo_col_;
    dlow_bcsr_val_ = dlow_bcoo_val_;
    CheckCuda(cusparseXcoo2csr(cusparse_handle, dlow_bcoo_row_, low_bnnz_,
                               brows_, dlow_bcsr_row_,
                               CUSPARSE_INDEX_BASE_ZERO));
    // D
    CheckCuda(cudaMalloc(&ddiag_bcsr_row_, sizeof(int32_t) * (brows_ + 1)));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(ddiag_bcsr_row_)); });
    ddiag_bcsr_col_ = ddiag_bcoo_col_;
    ddiag_bcsr_val_ = ddiag_bcoo_val_;
    ddiag_val_ = ddiag_bcoo_val_;
    CheckCuda(cusparseXcoo2csr(cusparse_handle, ddiag_bcoo_row_, diag_bnnz_,
                               brows_, ddiag_bcsr_row_,
                               CUSPARSE_INDEX_BASE_ZERO));
    // U
    CheckCuda(cudaMalloc(&dup_bcsr_row_, sizeof(int32_t) * (brows_ + 1)));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dup_bcsr_row_)); });
    dup_bcsr_col_ = dup_bcoo_col_;
    dup_bcsr_val_ = dup_bcoo_val_;
    CheckCuda(cusparseXcoo2csr(cusparse_handle, dup_bcoo_row_, up_bnnz_, brows_,
                               dup_bcsr_row_, CUSPARSE_INDEX_BASE_ZERO));
  } else if (as_dense_) {
    _type* tmp = new _type[rows_ * cols_];
    memset(tmp, 0, sizeof(_type) * rows_ * cols_);
    for (auto i = 0; i < bnnz_; ++i) {
      for (auto di = 0; di < _brow; ++di) {
        for (auto dj = 0; dj < _bcol; ++dj) {
          auto off =
              rows_ * (bcoo_col_[i] * _bcol + dj) + bcoo_row_[i] * _brow + di;
          tmp[off] = bcoo_val_[_brow * _bcol * i + _bcol * di + dj];
        }
      }
    }
    CheckCuda(cudaMalloc(&dden_val_, sizeof(_type) * rows_ * cols_));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dden_val_)); });
    CheckCuda(cudaMemcpy(dden_val_, tmp, sizeof(_type) * rows_ * cols_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMalloc(&ddiag_val_, sizeof(_type) * brows_ * _brow * _bcol));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(ddiag_val_)); });
    CheckCuda(cudaMemcpy(ddiag_val_, diag_val_,
                         sizeof(_type) * brows_ * _brow * _bcol,
                         cudaMemcpyHostToDevice));
    delete[] tmp;
  } else {
    CheckCuda(cudaMemcpy(dbcoo_val_, bcoo_val_, sizeof(_type) * nnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(dbcoo_row_, bcoo_row_, sizeof(int32_t) * bnnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(dbcoo_col_, bcoo_col_, sizeof(int32_t) * bnnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMalloc(&dbcsr_row_, sizeof(int32_t) * (brows_ + 1)));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dbcsr_row_)); });
    dbcsr_col_ = dbcoo_col_;
    dbcsr_val_ = dbcoo_val_;
    CheckCuda(cusparseXcoo2csr(cusparse_handle, dbcoo_row_, bnnz_, brows_,
                               dbcsr_row_, CUSPARSE_INDEX_BASE_ZERO));

    CheckCuda(cudaMemcpy(ddiag_boff_, diag_boff_, sizeof(int32_t) * brows_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMalloc(&ddiag_val_, sizeof(_type) * brows_ * _brow * _bcol));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(ddiag_val_)); });
    CheckCuda(cudaMemcpy(ddiag_val_, diag_val_,
                         sizeof(_type) * brows_ * _brow * _bcol,
                         cudaMemcpyHostToDevice));
  }
}

template <typename _type>
void BlockSparseMatrix2<_type>::GSToDevice(cusparseHandle_t cusparse_handle) {
  // Low
  CheckCuda(cudaMalloc(&dGS_low_bcoo_row_, sizeof(int32_t) * low_bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_low_bcoo_row_)); });
  CheckCuda(cudaMemcpy(dGS_low_bcoo_row_, GS_low_bcoo_row_,
                       sizeof(int32_t) * low_bnnz_, cudaMemcpyHostToDevice));
  CheckCuda(cudaMalloc(&dGS_low_bcsr_row_,
                       sizeof(int32_t) * (brows_ + GS_n_color_ + 1)));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_low_bcsr_row_)); });
  for (int i = 0; i + 1 < GS_n_color_; ++i) {
    CheckCuda(cusparseXcoo2csr(
        cusparse_handle, dGS_low_bcoo_row_ + GS_low_boff_[i],
        GS_low_boff_[i + 1] - GS_low_boff_[i],
        GS_c2m_off_[i + 2] - GS_c2m_off_[i + 1],
        dGS_low_bcsr_row_ + GS_c2m_off_[i + 1] - GS_c2m_off_[1] + i,
        CUSPARSE_INDEX_BASE_ZERO));
  }
  CheckCuda(cudaMalloc(&dGS_low_bcoo_col_, sizeof(int32_t) * low_bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_low_bcoo_col_)); });
  CheckCuda(cudaMemcpy(dGS_low_bcoo_col_, GS_low_bcoo_col_,
                       sizeof(int32_t) * low_bnnz_, cudaMemcpyHostToDevice));
  dGS_low_bcsr_col_ = dGS_low_bcoo_col_;

  // Up
  CheckCuda(cudaMalloc(&dGS_up_bcoo_row_, sizeof(int32_t) * up_bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_up_bcoo_row_)); });
  CheckCuda(cudaMemcpy(dGS_up_bcoo_row_, GS_up_bcoo_row_,
                       sizeof(int32_t) * up_bnnz_, cudaMemcpyHostToDevice));
  CheckCuda(cudaMalloc(&dGS_up_bcsr_row_,
                       sizeof(int32_t) * (brows_ + GS_n_color_ + 1)));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_up_bcsr_row_)); });
  for (int i = 0; i + 1 < GS_n_color_; ++i) {
    CheckCuda(cusparseXcoo2csr(
        cusparse_handle, dGS_up_bcoo_row_ + GS_up_boff_[i],
        GS_up_boff_[i + 1] - GS_up_boff_[i],
        GS_c2m_off_[i + 1] - GS_c2m_off_[i],
        dGS_up_bcsr_row_ + GS_c2m_off_[i] + i, CUSPARSE_INDEX_BASE_ZERO));
  }
  CheckCuda(cudaMalloc(&dGS_up_bcoo_col_, sizeof(int32_t) * up_bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_up_bcoo_col_)); });
  CheckCuda(cudaMemcpy(dGS_up_bcoo_col_, GS_up_bcoo_col_,
                       sizeof(int32_t) * up_bnnz_, cudaMemcpyHostToDevice));
  dGS_up_bcsr_col_ = dGS_up_bcoo_col_;
}

template <typename _type>
void BlockSparseMatrix2<_type>::CholToDevice(
    cusolverDnHandle_t cusolverDn_handle,
    cusolverDnParams_t cusolverDn_params) {
  if ((!as_dense_) || (rows_ != cols_)) {
    spdlog::error("Cholesky factorization only for symmetric dense matrix");
    exit(1);
  }
  constexpr cudaDataType_t _RealType =
      std::is_same<_type, float>() ? CUDA_R_32F : CUDA_R_64F;
  chol_host_buffer_size_ = 0;
  chol_dev_buffer_size_ = 0;
  cusolverDnXpotrf_bufferSize(cusolverDn_handle, cusolverDn_params,
                              CUBLAS_FILL_MODE_LOWER, rows_, _RealType,
                              dden_val_, rows_, _RealType,
                              &chol_dev_buffer_size_, &chol_host_buffer_size_);
  CheckCuda(cudaMalloc(&dchol_info_, sizeof(int32_t)));
  CheckCuda(cudaMemset(dchol_info_, 0, sizeof(int32_t)));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dchol_info_)); });
  CheckCuda(cudaMalloc(&dchol_val_, sizeof(_type) * rows_ * cols_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dchol_val_)); });
  CheckCuda(cudaMalloc(&dchol_fixed_, sizeof(_type) * rows_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dchol_fixed_)); });
  chol_host_buffer_ = ::operator new(chol_host_buffer_size_);
  deletion_queue_.push_back([=]() { delete[] chol_host_buffer_; });
  CheckCuda(cudaMalloc(&chol_dev_buffer_, chol_dev_buffer_size_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(chol_dev_buffer_)); });
  CheckCuda(cudaMemcpy(dchol_val_, dden_val_, sizeof(_type) * rows_ * cols_,
                       cudaMemcpyDeviceToDevice));
  _type fix = _type(1e-5);
  _type* h_fix = new _type[rows_];
  for (auto i = 0; i < rows_; ++i) h_fix[i] = fix;
  CheckCuda(cudaMemcpy(dchol_fixed_, h_fix, sizeof(_type) * rows_,
                       cudaMemcpyHostToDevice));
  delete[] h_fix;
}

template <typename _type>
void BlockSparseMatrix2<_type>::OutputBlockCoo(std::ostream& out) {
  assert(bcoo_row_ && bcoo_col_ && bcoo_val_);
  for (auto i = 0; i < bnnz_; ++i) {
    uint32_t br = bcoo_row_[i];
    uint32_t bc = bcoo_col_[i];
    for (int j = 0; j < _brow; ++j) {
      for (int k = 0; k < _bcol; ++k) {
        uint32_t r = br * _brow + j;
        uint32_t c = bc * _bcol + k;
        uint32_t idx = i * _brow * _bcol + j * _bcol + k;
        out << r << " " << c << " " << bcoo_val_[idx] << std::endl;
      }
    }
  }
}

template <typename _type>
void BlockSparseMatrix2<_type>::OutputLowBlockCoo(std::ostream& out) {
  assert(bcoo_row_ && bcoo_col_ && bcoo_val_);
  assert(brows_ == bcols_);
  for (auto i = 0; i < low_bnnz_; ++i) {
    uint32_t br = low_bcoo_row_[i];
    uint32_t bc = low_bcoo_col_[i];
    for (int j = 0; j < _brow; ++j) {
      for (int k = 0; k < _bcol; ++k) {
        uint32_t r = br * _brow + j;
        uint32_t c = bc * _bcol + k;
        uint32_t idx = i * _brow * _bcol + j * _bcol + k;
        out << r << " " << c << " " << low_bcoo_val_[idx] << std::endl;
      }
    }
  }
}

template <typename _type, int _brow, int _bcol>
void BlockSparseMatrix<_type, _brow, _bcol>::Init(uint32_t brows,
                                                  uint32_t bcols, bool as_LDU,
                                                  bool as_dense) {
  if (as_LDU_) assert(brows == bcols);
  brows_ = brows;
  bcols_ = bcols;
  rows_ = brows * _brow;
  cols_ = bcols * _bcol;
  as_LDU_ = as_LDU;
  as_dense_ = as_dense;
}

template <typename _type, int _brow, int _bcol>
void BlockSparseMatrix<_type, _brow, _bcol>::Destroy() {
  for (auto it = deletion_queue_.rbegin(); it != deletion_queue_.rend(); it++)
    (*it)();
  deletion_queue_.clear();
};

template <typename _type, int _brow, int _bcol>
void BlockSparseMatrix<_type, _brow, _bcol>::FromCooMapSym(
    const std::map<RowCol, Block>& coo) {
  bnnz_ = coo.size();
  nnz_ = coo.size() * _brow * _bcol;
  low_bnnz_ = up_bnnz_ = (bnnz_ - brows_) / 2;
  diag_bnnz_ = brows_;

  bcoo_val_ = new _type[nnz_];
  deletion_queue_.push_back([=]() { delete[] bcoo_val_; });
  bcoo_row_ = new uint32_t[bnnz_];
  deletion_queue_.push_back([=]() { delete[] bcoo_row_; });
  bcoo_col_ = new uint32_t[bnnz_];
  deletion_queue_.push_back([=]() { delete[] bcoo_col_; });
  if (as_LDU_) {
    uint32_t diag_boff = low_bnnz_;
    uint32_t up_boff = low_bnnz_ + brows_;
    low_bcoo_row_ = bcoo_row_;
    low_bcoo_col_ = bcoo_col_;
    low_bcoo_val_ = bcoo_val_;
    diag_bcoo_row_ = bcoo_row_ + diag_boff;
    diag_bcoo_col_ = bcoo_col_ + diag_boff;
    diag_bcoo_val_ = bcoo_val_ + diag_boff * _brow * _bcol;
    diag_val_ = diag_bcoo_val_;
    up_bcoo_row_ = bcoo_row_ + up_boff;
    up_bcoo_col_ = bcoo_col_ + up_boff;
    up_bcoo_val_ = bcoo_val_ + up_boff * _brow * _bcol;
  } else {
    diag_val_ = new _type[brows_ * _brow * _bcol];
    deletion_queue_.push_back([=]() { delete[] diag_val_; });
  }
  diag_boff_ = new uint32_t[brows_];
  deletion_queue_.push_back([=]() { delete[] diag_boff_; });

  uint32_t idx_f = 0, idx_d = 0, idx_u = 0, idx_l = 0;
  for (auto iter : coo) {
    uint32_t r = iter.first.first;
    uint32_t c = iter.first.second;
    const Block& v = iter.second;
    if (as_LDU_) {
      if (r == c) {  // D
        diag_bcoo_row_[idx_d] = r;
        diag_bcoo_col_[idx_d] = c;
        for (int i = 0; i < _brow; ++i)
          for (int j = 0; j < _bcol; ++j) {
            uint32_t idx = idx_d * _brow * _bcol + i * _bcol + j;
            diag_val_[idx] = v(i, j);
          }
        diag_boff_[r] = low_bnnz_ + idx_d;
        ++idx_d;
      } else if (r < c) {  // U
        up_bcoo_row_[idx_u] = r;
        up_bcoo_col_[idx_u] = c;
        for (int i = 0; i < _brow; ++i)
          for (int j = 0; j < _bcol; ++j) {
            uint32_t idx = idx_u * _brow * _bcol + i * _bcol + j;
            up_bcoo_val_[idx] = v(i, j);
          }
        ++idx_u;
      } else if (r > c) {  // L
        low_bcoo_row_[idx_l] = r;
        low_bcoo_col_[idx_l] = c;
        for (int i = 0; i < _brow; ++i)
          for (int j = 0; j < _bcol; ++j) {
            uint32_t idx = idx_l * _brow * _bcol + i * _bcol + j;
            low_bcoo_val_[idx] = v(i, j);
          }
        ++idx_l;
      }
    } else {
      bcoo_row_[idx_f] = r;
      bcoo_col_[idx_f] = c;
      for (int i = 0; i < _brow; ++i)
        for (int j = 0; j < _bcol; ++j) {
          uint32_t idx = idx_f * _brow * _bcol + i * _bcol + j;
          bcoo_val_[idx] = v(i, j);
        }
      if (r == c) {
        for (int i = 0; i < _brow; ++i)
          for (int j = 0; j < _bcol; ++j) {
            uint32_t idx = idx_d * _brow * _bcol + i * _bcol + j;
            diag_val_[idx] = v(i, j);
          }
        diag_boff_[r] = idx_f;
        ++idx_d;
      }
      ++idx_f;
    }
  }
}

template <typename _type, int _brow, int _bcol>
void BlockSparseMatrix<_type, _brow, _bcol>::BuildGS(const uint32_t* c2h_off,
                                                     const int n_color) {
  if (!as_LDU_) {
    spdlog::error("Gauss Seidel solver need matrix saved in LDU or dense");
    exit(1);
  }
  GS_n_color_ = n_color;
  GS_c2m_off_ = new uint32_t[n_color + 1];
  deletion_queue_.push_back([=]() { delete[] GS_c2m_off_; });
  memcpy(GS_c2m_off_, c2h_off, sizeof(uint32_t) * (n_color + 1));
  GS_low_bcoo_row_ = new uint32_t[low_bnnz_];
  deletion_queue_.push_back([=]() { delete[] GS_low_bcoo_row_; });
  GS_low_bcoo_col_ = new uint32_t[low_bnnz_];
  deletion_queue_.push_back([=]() { delete[] GS_low_bcoo_col_; });
  GS_low_boff_ = new uint32_t[n_color + 1];
  deletion_queue_.push_back([=]() { delete[] GS_low_boff_; });
  GS_up_bcoo_row_ = new uint32_t[up_bnnz_];
  deletion_queue_.push_back([=]() { delete[] GS_up_bcoo_row_; });
  GS_up_bcoo_col_ = new uint32_t[up_bnnz_];
  deletion_queue_.push_back([=]() { delete[] GS_up_bcoo_col_; });
  GS_up_boff_ = new uint32_t[n_color + 1];
  deletion_queue_.push_back([=]() { delete[] GS_up_boff_; });
  int color = -1;
  uint32_t off_now = GS_c2m_off_[0];  // 0
  uint32_t off_next = GS_c2m_off_[1];
  for (auto i = 0; i < low_bnnz_; ++i) {
    uint32_t r = low_bcoo_row_[i];
    uint32_t c = low_bcoo_col_[i];
    while (r >= off_next) {
      ++color;
      GS_low_boff_[color] = i;
      off_now = GS_c2m_off_[color + 1];
      off_next = GS_c2m_off_[color + 2];
    }
    GS_low_bcoo_row_[i] = r - off_now;
    GS_low_bcoo_col_[i] = c;
  }
  while (color < n_color - 1) GS_low_boff_[++color] = low_bnnz_;

  uint32_t row_off = 0;
  uint32_t col_off = 0;
  color = -1;
  for (auto i = 0; i < up_bnnz_; ++i) {
    uint32_t r = up_bcoo_row_[i];
    uint32_t c = up_bcoo_col_[i];
    while (r >= col_off) {
      ++color;
      GS_up_boff_[color] = i;
      row_off = GS_c2m_off_[color];
      col_off = GS_c2m_off_[color + 1];
    }
    GS_up_bcoo_row_[i] = r - row_off;
    GS_up_bcoo_col_[i] = c - col_off;
  }
  while (color < n_color - 1) GS_up_boff_[++color] = up_bnnz_;

  // {  // Test Code
  //   std::cout << "GS" << std::endl;
  //   for (int i = 0; i < n_color; ++i) {
  //     std::cout << GS_low_boff_[i] << " ";
  //   }
  //   std::cout << std::endl;
  //   for (int i = 0; i < n_color; ++i) {
  //     std::cout << GS_up_boff_[i] << " ";
  //   }
  //   std::cout << std::endl;
  // }
}

template <typename _type, int _brow, int _bcol>
void BlockSparseMatrix<_type, _brow, _bcol>::BCooSymToDevice(
    cusparseHandle_t cusparse_handle) {
  CheckCuda(cudaMalloc(&dbcoo_val_, sizeof(_type) * nnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dbcoo_val_)); });
  CheckCuda(cudaMalloc(&dbcoo_row_, sizeof(int32_t) * bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dbcoo_row_)); });
  CheckCuda(cudaMalloc(&dbcoo_col_, sizeof(int32_t) * bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dbcoo_col_)); });
  CheckCuda(cudaMalloc(&ddiag_boff_, sizeof(int32_t) * brows_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(ddiag_boff_)); });
  if (as_LDU_) {
    CheckCuda(cudaMemcpy(dbcoo_val_, bcoo_val_, sizeof(_type) * nnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(dbcoo_row_, bcoo_row_, sizeof(int32_t) * bnnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(dbcoo_col_, bcoo_col_, sizeof(int32_t) * bnnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(ddiag_boff_, diag_boff_, sizeof(int32_t) * brows_,
                         cudaMemcpyHostToDevice));
    dlow_bcoo_row_ = dbcoo_row_;
    dlow_bcoo_col_ = dbcoo_col_;
    dlow_bcoo_val_ = dbcoo_val_;
    ddiag_bcoo_row_ = dbcoo_row_ + low_bnnz_;
    ddiag_bcoo_col_ = dbcoo_col_ + low_bnnz_;
    ddiag_bcoo_val_ = dbcoo_val_ + low_bnnz_ * _brow * _bcol;
    dup_bcoo_row_ = dbcoo_row_ + low_bnnz_ + diag_bnnz_;
    dup_bcoo_col_ = dbcoo_col_ + low_bnnz_ + diag_bnnz_;
    dup_bcoo_val_ = dbcoo_val_ + (low_bnnz_ + diag_bnnz_) * _brow * _bcol;
    // L
    CheckCuda(cudaMalloc(&dlow_bcsr_row_, sizeof(int32_t) * (brows_ + 1)));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dlow_bcsr_row_)); });
    dlow_bcsr_col_ = dlow_bcoo_col_;
    dlow_bcsr_val_ = dlow_bcoo_val_;
    CheckCuda(cusparseXcoo2csr(cusparse_handle, dlow_bcoo_row_, low_bnnz_,
                               brows_, dlow_bcsr_row_,
                               CUSPARSE_INDEX_BASE_ZERO));
    // D
    CheckCuda(cudaMalloc(&ddiag_bcsr_row_, sizeof(int32_t) * (brows_ + 1)));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(ddiag_bcsr_row_)); });
    ddiag_bcsr_col_ = ddiag_bcoo_col_;
    ddiag_bcsr_val_ = ddiag_bcoo_val_;
    ddiag_val_ = ddiag_bcoo_val_;
    CheckCuda(cusparseXcoo2csr(cusparse_handle, ddiag_bcoo_row_, diag_bnnz_,
                               brows_, ddiag_bcsr_row_,
                               CUSPARSE_INDEX_BASE_ZERO));
    // U
    CheckCuda(cudaMalloc(&dup_bcsr_row_, sizeof(int32_t) * (brows_ + 1)));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dup_bcsr_row_)); });
    dup_bcsr_col_ = dup_bcoo_col_;
    dup_bcsr_val_ = dup_bcoo_val_;
    CheckCuda(cusparseXcoo2csr(cusparse_handle, dup_bcoo_row_, up_bnnz_, brows_,
                               dup_bcsr_row_, CUSPARSE_INDEX_BASE_ZERO));
  } else if (as_dense_) {
    _type* tmp = new _type[rows_ * cols_];
    memset(tmp, 0, sizeof(_type) * rows_ * cols_);
    for (auto i = 0; i < bnnz_; ++i) {
      for (auto di = 0; di < _brow; ++di) {
        for (auto dj = 0; dj < _bcol; ++dj) {
          auto off =
              rows_ * (bcoo_col_[i] * _bcol + dj) + bcoo_row_[i] * _brow + di;
          tmp[off] = bcoo_val_[_brow * _bcol * i + _bcol * di + dj];
        }
      }
    }
    CheckCuda(cudaMalloc(&dden_val_, sizeof(_type) * rows_ * cols_));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dden_val_)); });
    CheckCuda(cudaMemcpy(dden_val_, tmp, sizeof(_type) * rows_ * cols_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMalloc(&ddiag_val_, sizeof(_type) * brows_ * _brow * _bcol));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(ddiag_val_)); });
    CheckCuda(cudaMemcpy(ddiag_val_, diag_val_,
                         sizeof(_type) * brows_ * _brow * _bcol,
                         cudaMemcpyHostToDevice));
    delete[] tmp;
  } else {
    CheckCuda(cudaMemcpy(dbcoo_val_, bcoo_val_, sizeof(_type) * nnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(dbcoo_row_, bcoo_row_, sizeof(int32_t) * bnnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(dbcoo_col_, bcoo_col_, sizeof(int32_t) * bnnz_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMalloc(&dbcsr_row_, sizeof(int32_t) * (brows_ + 1)));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dbcsr_row_)); });
    dbcsr_col_ = dbcoo_col_;
    dbcsr_val_ = dbcoo_val_;
    CheckCuda(cusparseXcoo2csr(cusparse_handle, dbcoo_row_, bnnz_, brows_,
                               dbcsr_row_, CUSPARSE_INDEX_BASE_ZERO));

    CheckCuda(cudaMemcpy(ddiag_boff_, diag_boff_, sizeof(int32_t) * brows_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMalloc(&ddiag_val_, sizeof(_type) * brows_ * _brow * _bcol));
    deletion_queue_.push_back([=]() { CheckCuda(cudaFree(ddiag_val_)); });
    CheckCuda(cudaMemcpy(ddiag_val_, diag_val_,
                         sizeof(_type) * brows_ * _brow * _bcol,
                         cudaMemcpyHostToDevice));
  }
}

template <typename _type, int _brow, int _bcol>
void BlockSparseMatrix<_type, _brow, _bcol>::GSToDevice(
    cusparseHandle_t cusparse_handle) {
  // Low
  CheckCuda(cudaMalloc(&dGS_low_bcoo_row_, sizeof(int32_t) * low_bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_low_bcoo_row_)); });
  CheckCuda(cudaMemcpy(dGS_low_bcoo_row_, GS_low_bcoo_row_,
                       sizeof(int32_t) * low_bnnz_, cudaMemcpyHostToDevice));
  CheckCuda(cudaMalloc(&dGS_low_bcsr_row_,
                       sizeof(int32_t) * (brows_ + GS_n_color_ + 1)));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_low_bcsr_row_)); });
  for (int i = 0; i + 1 < GS_n_color_; ++i) {
    CheckCuda(cusparseXcoo2csr(
        cusparse_handle, dGS_low_bcoo_row_ + GS_low_boff_[i],
        GS_low_boff_[i + 1] - GS_low_boff_[i],
        GS_c2m_off_[i + 2] - GS_c2m_off_[i + 1],
        dGS_low_bcsr_row_ + GS_c2m_off_[i + 1] - GS_c2m_off_[1] + i,
        CUSPARSE_INDEX_BASE_ZERO));
  }
  CheckCuda(cudaMalloc(&dGS_low_bcoo_col_, sizeof(int32_t) * low_bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_low_bcoo_col_)); });
  CheckCuda(cudaMemcpy(dGS_low_bcoo_col_, GS_low_bcoo_col_,
                       sizeof(int32_t) * low_bnnz_, cudaMemcpyHostToDevice));
  dGS_low_bcsr_col_ = dGS_low_bcoo_col_;

  // Up
  CheckCuda(cudaMalloc(&dGS_up_bcoo_row_, sizeof(int32_t) * up_bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_up_bcoo_row_)); });
  CheckCuda(cudaMemcpy(dGS_up_bcoo_row_, GS_up_bcoo_row_,
                       sizeof(int32_t) * up_bnnz_, cudaMemcpyHostToDevice));
  CheckCuda(cudaMalloc(&dGS_up_bcsr_row_,
                       sizeof(int32_t) * (brows_ + GS_n_color_ + 1)));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_up_bcsr_row_)); });
  for (int i = 0; i + 1 < GS_n_color_; ++i) {
    CheckCuda(cusparseXcoo2csr(
        cusparse_handle, dGS_up_bcoo_row_ + GS_up_boff_[i],
        GS_up_boff_[i + 1] - GS_up_boff_[i],
        GS_c2m_off_[i + 1] - GS_c2m_off_[i],
        dGS_up_bcsr_row_ + GS_c2m_off_[i] + i, CUSPARSE_INDEX_BASE_ZERO));
  }
  CheckCuda(cudaMalloc(&dGS_up_bcoo_col_, sizeof(int32_t) * up_bnnz_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dGS_up_bcoo_col_)); });
  CheckCuda(cudaMemcpy(dGS_up_bcoo_col_, GS_up_bcoo_col_,
                       sizeof(int32_t) * up_bnnz_, cudaMemcpyHostToDevice));
  dGS_up_bcsr_col_ = dGS_up_bcoo_col_;
}

template <typename _type, int _brow, int _bcol>
void BlockSparseMatrix<_type, _brow, _bcol>::CholToDevice(
    cusolverDnHandle_t cusolverDn_handle,
    cusolverDnParams_t cusolverDn_params) {
  if ((!as_dense_) || (rows_ != cols_)) {
    spdlog::error("Cholesky factorization only for symmetric dense matrix");
    exit(1);
  }
  constexpr cudaDataType_t _RealType =
      std::is_same<_type, float>() ? CUDA_R_32F : CUDA_R_64F;
  chol_host_buffer_size_ = 0;
  chol_dev_buffer_size_ = 0;
  CheckCuda(cusolverDnXpotrf_bufferSize(
      cusolverDn_handle, cusolverDn_params, CUBLAS_FILL_MODE_LOWER, rows_,
      _RealType, dden_val_, rows_, _RealType, &chol_dev_buffer_size_,
      &chol_host_buffer_size_));
  CheckCuda(cudaMalloc(&dchol_info_, sizeof(int32_t)));
  CheckCuda(cudaMemset(dchol_info_, 0, sizeof(int32_t)));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dchol_info_)); });
  CheckCuda(cudaMalloc(&dchol_val_, sizeof(_type) * rows_ * cols_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dchol_val_)); });
  CheckCuda(cudaMalloc(&dchol_fixed_, sizeof(_type) * rows_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dchol_fixed_)); });
  chol_host_buffer_ = ::operator new(chol_host_buffer_size_);
  deletion_queue_.push_back([=]() { delete[] chol_host_buffer_; });
  CheckCuda(cudaMalloc(&chol_dev_buffer_, chol_dev_buffer_size_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(chol_dev_buffer_)); });
  CheckCuda(cudaMemcpy(dchol_val_, dden_val_, sizeof(_type) * rows_ * cols_,
                       cudaMemcpyDeviceToDevice));
  _type fix = _type(1e-5);
  _type* h_fix = new _type[rows_];
  for (auto i = 0; i < rows_; ++i) h_fix[i] = fix;
  CheckCuda(cudaMemcpy(dchol_fixed_, h_fix, sizeof(_type) * rows_,
                       cudaMemcpyHostToDevice));
  delete[] h_fix;
}

template <typename _type, int _brow, int _bcol>
void BlockSparseMatrix<_type, _brow, _bcol>::LDLTToDevice(
    cusolverDnHandle_t cusolverDn_handle,
    cusolverDnParams_t cusolverDn_params) {
  if ((!as_dense_) || (rows_ != cols_)) {
    spdlog::error("LDLT factorization only for symmetric dense matrix");
    exit(1);
  }
  ldlt_dev_buffer_size_ = 0;
#ifdef REAL_AS_DOUBLE
  CheckCuda(cusolverDnDsytrf_bufferSize(cusolverDn_handle, rows_, dden_val_,
                                        rows_, &ldlt_dev_buffer_size_));
#else
  CheckCuda(cusolverDnSsytrf_bufferSize(cusolverDn_handle, rows_, dden_val_,
                                        rows_, &ldlt_dev_buffer_size_));
#endif
  CheckCuda(cudaMalloc(&dldlt_info_, sizeof(int32_t)));
  CheckCuda(cudaMemset(dldlt_info_, 0, sizeof(int32_t)));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dldlt_info_)); });
  CheckCuda(cudaMalloc(&dldlt_val_, sizeof(_type) * rows_ * cols_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dldlt_val_)); });
  CheckCuda(cudaMemcpy(dldlt_val_, dden_val_, sizeof(_type) * rows_ * cols_,
                       cudaMemcpyDeviceToDevice));
  CheckCuda(cudaMalloc(&ldlt_dev_buffer_, sizeof(real) * ldlt_dev_buffer_size_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(ldlt_dev_buffer_)); });
  CheckCuda(cudaMalloc(&dldlt_ipiv_, sizeof(int32_t) * rows_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dldlt_ipiv_)); });
  CheckCuda(cudaMalloc(&dldlt_ipiv_64_, sizeof(int64_t) * rows_));
  deletion_queue_.push_back([=]() { CheckCuda(cudaFree(dldlt_ipiv_64_)); });
}

template <typename _type, int _brow, int _bcol>
void BlockSparseMatrix<_type, _brow, _bcol>::OutputBlockCoo(std::ostream& out) {
  assert(bcoo_row_ && bcoo_col_ && bcoo_val_);
  for (auto i = 0; i < bnnz_; ++i) {
    uint32_t br = bcoo_row_[i];
    uint32_t bc = bcoo_col_[i];
    for (int j = 0; j < _brow; ++j) {
      for (int k = 0; k < _bcol; ++k) {
        uint32_t r = br * _brow + j;
        uint32_t c = bc * _bcol + k;
        uint32_t idx = i * _brow * _bcol + j * _bcol + k;
        out << r << " " << c << " " << bcoo_val_[idx] << std::endl;
      }
    }
  }
}

template <typename _type, int _brow, int _bcol>
void BlockSparseMatrix<_type, _brow, _bcol>::OutputLowBlockCoo(
    std::ostream& out) {
  assert(bcoo_row_ && bcoo_col_ && bcoo_val_);
  assert(brows_ == bcols_);
  for (auto i = 0; i < low_bnnz_; ++i) {
    uint32_t br = low_bcoo_row_[i];
    uint32_t bc = low_bcoo_col_[i];
    for (int j = 0; j < _brow; ++j) {
      for (int k = 0; k < _bcol; ++k) {
        uint32_t r = br * _brow + j;
        uint32_t c = bc * _bcol + k;
        uint32_t idx = i * _brow * _bcol + j * _bcol + k;
        out << r << " " << c << " " << low_bcoo_val_[idx] << std::endl;
      }
    }
  }
}

template <typename _type, int _brow, int _bcol>
__global__ void UpdateDiagFromDense(_type* diag, const _type* dense,
                                    const int32_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int s = _brow * _bcol;
  int b = i / s;
  if (b >= n) return;
  int r = (i % s) / _bcol;
  int c = (i % s) % _bcol;
  diag[i] = dense[(b * _bcol + c) * _bcol * n + r * _bcol + c];
}

template <typename _type, int _brow, int _bcol>
__global__ void UpdateDiagFromCoo(_type* diag, const _type* coo_val,
                                  const int32_t* diag_boff, const int32_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int s = _brow * _bcol;
  int b = i / s;
  if (b >= n) return;
  int r = (i % s) / _bcol;
  int c = (i % s) % _bcol;
  diag[i] = coo_val[diag_boff[b] * s + r * _bcol + c];
}

template <typename _type, int _brow, int _bcol>
void BlockSparseMatrix<_type, _brow, _bcol>::UpdateDiag(cudaStream_t stream) {
  // LDU matrix has diag_val_ as part of bcoo_val_, so no need to update
  if (as_LDU_)
    return;
  else if (as_dense_) {
    int n_block = (brows_ * _brow * _bcol + 63) / 64;
    UpdateDiagFromDense<_type, _brow, _bcol>
        <<<n_block, 64, 0, stream>>>(ddiag_val_, dden_val_, brows_);
  } else {
    int n_block = (brows_ * _brow * _bcol + 63) / 64;
    UpdateDiagFromCoo<_type, _brow, _bcol><<<n_block, 64, 0, stream>>>(
        ddiag_val_, dbcoo_val_, ddiag_boff_, brows_);
  }
}

};  // namespace Rain