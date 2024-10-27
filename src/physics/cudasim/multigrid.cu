#include "cudaelasticmodel.h"
#include "cudasvd.h"
#include "multigrid.h"

namespace Rain::CUDA {
__global__ void UpdateBasic(Vec3* X, Vec3* V, int32_t n_vert, Vec3 grav,
                            real damping, real dt) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  V[i] += grav * dt;
  V[i] -= damping * V[i];
  // for the letter example
  // V[i][1] = max(V[i][1], -20.);
  X[i] += V[i] * dt;
}

__global__ void UpdateAfDiagReordered(real* Af_diag_add, int32_t* h2m,
                                      bool* fixed, const real control_mag,
                                      int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  if (!fixed[v]) return;
  int32_t i = h2m[v];
  real* diag = Af_diag_add + 9 * i;
  diag[0] = control_mag;
  diag[4] = control_mag;
  diag[8] = control_mag;
}

__global__ void UpdateAfDiag(real* Af_diag_add, bool* fixed,
                             const real control_mag, int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  if (!fixed[v]) return;
  real* diag = Af_diag_add + 9 * v;
  diag[0] = control_mag;
  diag[4] = control_mag;
  diag[8] = control_mag;
}

__global__ void UpdateAfDiagWithCtrlReordered(real* Af_diag_add, int32_t* h2m,
                                              bool* fixed,
                                              const real control_mag,
                                              int32_t ctrl_vert,
                                              int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  if (!fixed[v] && (v != ctrl_vert)) return;
  int32_t i = h2m[v];
  real* diag = Af_diag_add + 9 * i;
  diag[0] = control_mag;
  diag[4] = control_mag;
  diag[8] = control_mag;
}

__global__ void UpdateAfDiagWithCtrl(real* Af_diag_add, bool* fixed,
                                     const real control_mag, int32_t ctrl_vert,
                                     int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  if (!fixed[v] && (v != ctrl_vert)) return;
  real* diag = Af_diag_add + 9 * v;
  diag[0] = control_mag;
  diag[4] = control_mag;
  diag[8] = control_mag;
}

__global__ void UpdateUtAUDiag(real* UtAU_diag_add, const real* Af_diag_add,
                               const real* diag_XXt, const int32_t* update_off,
                               int32_t n_vert) {
  // matrix multiplying in thread block
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t v = idx / 9;
  if (v >= n_vert) return;
  int32_t p = idx % 9;
  int32_t px = p / 3, py = p % 3;
#pragma unroll
  for (int t = 0; t < 16; ++t) {
    int32_t tx = t / 4, ty = t % 4;
    ::atomicAdd(
        &UtAU_diag_add[144 * update_off[v] + (px * 4 + tx) * 12 + py * 4 + ty],
        Af_diag_add[idx] * diag_XXt[16 * v + t]);
  }
}

__global__ void UpdateUltAUlDiag(real* diag_add, const real* diag_add_finer,
                                 const int32_t* update_off, int32_t n_handle) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_handle) return;
  int32_t off = threadIdx.y;
  ::atomicAdd(&diag_add[144 * update_off[i] + off],
              diag_add_finer[144 * i + off]);
}

__global__ void UpdateDenseDiag(real* den_val, const real* diag_add,
                                int32_t n_handle, int32_t rows) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_handle) return;
  int32_t off = threadIdx.y;
  int32_t dx = off / 12, dy = off % 12;
  ::atomicAdd(&den_val[(12 * i + dx) * rows + (12 * i + dy)],
              diag_add[144 * i + off]);
}

__global__ void UpdateDenseDiag(real* den_val, const real* diag_add,
                                const int32_t dof, const int32_t n_handle,
                                const int32_t rows) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_handle) return;
  int32_t off = threadIdx.y;
  int32_t dx = off / dof, dy = off % dof;
  ::atomicAdd(&den_val[(dof * i + dx) * rows + (dof * i + dy)],
              diag_add[dof * dof * i + off]);
}

__global__ void DerivativeDiagReduction(real* down_diag, const real* up_diag,
                                        const real* diag_XXt,
                                        const int32_t* update_off,
                                        const int32_t up_dof,
                                        const int32_t n_handle) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int32_t up_block = up_dof * up_dof;
  int32_t i = idx / up_block;
  if (i >= n_handle) return;
  int32_t p = idx % up_block;
  int32_t px = p % up_dof, py = p / up_dof;
#pragma unroll
  for (int t = 0; t < 16; ++t) {
    int32_t tx = t / 4, ty = t % 4;
    ::atomicAdd(&down_diag[up_block * 16 * update_off[i] +
                           (px * 4 + tx) * up_dof * 4 + py * 4 + ty],
                up_diag[idx] * diag_XXt[16 * i + t]);
  }
}

__global__ void SameDiagReduction(real* down_diag, const real* up_diag,
                                  const int32_t* update_off, const int32_t dof,
                                  const int32_t n_handle) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_handle) return;
  int32_t off = threadIdx.y;
  ::atomicAdd(&down_diag[dof * dof * update_off[i] + off],
              up_diag[dof * dof * i + off]);
}

__global__ void HessianNeoHookean(const Vec3* X, const uint32_t* tet,
                                  const Mat3* Dm_inv, const real* vol,
                                  const int32_t* t2off, real* bcoo_val,
                                  const real mu, const real lambda,
                                  const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;

  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  Eigen::Matrix<real, 4, 3> idm;
  idm.block<3, 3>(0, 0) = Dm_inv[t];
  idm(3, 0) = -idm(0, 0) - idm(1, 0) - idm(2, 0);
  idm(3, 1) = -idm(0, 1) - idm(1, 1) - idm(2, 1);
  idm(3, 2) = -idm(0, 2) - idm(1, 2) - idm(2, 2);
  Mat3 Ds;
  Ds.col(0) = X[v1] - X[v4];
  Ds.col(1) = X[v2] - X[v4];
  Ds.col(2) = X[v3] - X[v4];
  Mat3 F = Ds * Dm_inv[t];
  Vec9 pJpF;
  pJpF.segment<3>(0) = Cross(F.col(1), F.col(2));
  pJpF.segment<3>(3) = Cross(F.col(2), F.col(0));
  pJpF.segment<3>(6) = Cross(F.col(0), F.col(1));

  const real scale = lambda * (Determinant(F) - 1.0 - mu / lambda);
  Mat3 ahat, bhat, chat;
  ahat << 0, -F(2, 0), F(1, 0), F(2, 0), 0, -F(0, 0), -F(1, 0), F(0, 0), 0;
  bhat << 0, -F(2, 1), F(1, 1), F(2, 1), 0, -F(0, 1), -F(1, 1), F(0, 1), 0;
  chat << 0, -F(2, 2), F(1, 2), F(2, 2), 0, -F(0, 2), -F(1, 2), F(0, 2), 0;

  Mat9 dPdF;
  dPdF.block<3, 3>(0, 0).setZero();
  dPdF.block<3, 3>(0, 3) = -scale * chat;
  dPdF.block<3, 3>(0, 6) = scale * bhat;
  dPdF.block<3, 3>(3, 0) = scale * chat;
  dPdF.block<3, 3>(3, 3).setZero();
  dPdF.block<3, 3>(3, 6) = -scale * ahat;
  dPdF.block<3, 3>(6, 0) = -scale * bhat;
  dPdF.block<3, 3>(6, 3) = scale * ahat;
  dPdF.block<3, 3>(6, 6).setZero();
  dPdF += mu * Mat9::Identity() + lambda * pJpF * pJpF.transpose();

  const int32_t* e2off = &t2off[16 * t];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      Mat3 tH = Mat3::Zero();
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          tH += idm(i, a) * dPdF.block<3, 3>(3 * a, 3 * b) * idm(j, b) * vol[t];
        }
      }
      real* bval = &bcoo_val[9 * e2off[4 * i + j]];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          ::atomicAdd(&bval[3 * a + b], tH(a, b));
        }
      }
    }
  }
}

__global__ void HessianNeoHookeanClamped(const Vec3* X, const uint32_t* tet,
                                         const Mat3* Dm_inv, const real* vol,
                                         const int32_t* t2off, real* bcoo_val,
                                         const real mu, const real lambda,
                                         const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;

  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  Eigen::Matrix<real, 4, 3> idm;
  idm.block<3, 3>(0, 0) = Dm_inv[t];
  idm(3, 0) = -idm(0, 0) - idm(1, 0) - idm(2, 0);
  idm(3, 1) = -idm(0, 1) - idm(1, 1) - idm(2, 1);
  idm(3, 2) = -idm(0, 2) - idm(1, 2) - idm(2, 2);
  Mat3 Ds;
  Ds.col(0) = X[v1] - X[v4];
  Ds.col(1) = X[v2] - X[v4];
  Ds.col(2) = X[v3] - X[v4];
  Mat3 F = Ds * Dm_inv[t];
  Mat3 U, V;
  Vec3 S;
#ifdef REAL_AS_DOUBLE
  Mat3f Uf, Vf;
  Vec3f Sf;
  SVD(F.cast<float>(), Uf, Vf, Sf);
  U = Uf.cast<double>();
  V = Vf.cast<double>();
  S = Sf.cast<double>();
#else
  SVD(F, U, V, S);
#endif
  Vec9 eigenvalues;
  Mat9 eigenvectors;

  {
    Mat3 L = Mat3::Identity();
    L(2, 2) = Determinant(U * V.transpose());
    real detU = Determinant(U);
    real detV = Determinant(V);
    if (detU < 0. && detV > 0.) U = U * L;
    if (detU > 0. && detV < 0.) V = V * L;
    S(2) = S(2) * L(2, 2);
  }

  const real J = Determinant(F);
  eigenvalues.segment<3>(0) = S;
  eigenvalues.segment<3>(3) = -S;
  const real evScale = lambda * (J - 1.0) - mu;
  eigenvalues.segment<6>(0) *= evScale;
  eigenvalues.segment<6>(0).array() += mu;
  BuildTwistAndFlipEigenvectors(U, V, eigenvectors);

  // Compute the remaining three eigenvalues and eigenvectors
  {
    real lm = lambda / mu;
    Mat3 A;
    const real s0s0 = S(0) * S(0);
    const real s1s1 = S(1) * S(1);
    const real s2s2 = S(2) * S(2);
    A(0, 0) = 1. + lm * s1s1 * s2s2;
    A(1, 1) = 1. + lm * s0s0 * s2s2;
    A(2, 2) = 1. + lm * s0s0 * s1s1;
    const real evScale = lm * (2.0 * J - 1.0) - 1.;
    A(0, 1) = evScale * S(2);
    A(1, 0) = A(0, 1);
    A(0, 2) = evScale * S(1);
    A(2, 0) = A(0, 2);
    A(1, 2) = evScale * S(0);
    A(2, 1) = A(1, 2);

    real max_scale = A.maxCoeff();
    // real max_scale = 1e8;
    real inv_max_scale = 1 / max_scale;
    A = A * inv_max_scale;

    Mat3 U1, V1;
#ifdef REAL_AS_DOUBLE
    Mat3f U1f, V1f;
    SVD(A.cast<float>(), U1f, V1f, Sf);
    U1 = U1f.cast<double>();
    V1 = V1f.cast<double>();
    S = Sf.cast<double>();
#else
    SVD(A, U1, V1, S);
#endif
    // if (!HasNan(A) && HasNan(S)) {
    //   printf("{\n");
    //   for (int i = 0; i < 3; ++i) {
    //     printf("{");
    //     for (int j = 0; j < 3; ++j) {
    //       printf("%f", A(i, j));
    //       if (j < 2) printf(", ");
    //     }
    //     printf("}");
    //     if (i < 2) printf(",\n");
    //   }
    //   printf("}\n");

    //   printf("%f %f %f\n", S(0), S(1), S(2));
    // }

    if (U1.col(0).dot(V1.col(0)) < 0.) {
      S(0) *= -1.;
    }
    if (U1.col(1).dot(V1.col(1)) < 0.) {
      S(1) *= -1.;
    }
    if (U1.col(2).dot(V1.col(2)) < 0.) {
      S(2) *= -1.;
    }

    eigenvalues.segment<3>(6) = mu * S * max_scale;

    Eigen::Map<Mat3>(eigenvectors.data() + 54) =
        U * V1.col(0).asDiagonal() * V.transpose();
    Eigen::Map<Mat3>(eigenvectors.data() + 63) =
        U * V1.col(1).asDiagonal() * V.transpose();
    Eigen::Map<Mat3>(eigenvectors.data() + 72) =
        U * V1.col(2).asDiagonal() * V.transpose();
  }

  // Clamp the eigenvalues
#pragma unroll
  for (int i = 0; i < 9; i++) {
    if (eigenvalues(i) < 0.0) {
      eigenvalues(i) = 0.0;
    }
  }

  Mat9 dPdF =
      eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

  const int32_t* e2off = &t2off[16 * t];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      Mat3 tH = Mat3::Zero();
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          tH += idm(i, a) * dPdF.block<3, 3>(3 * a, 3 * b) * idm(j, b) * vol[t];
        }
      }
      real* bval = &bcoo_val[9 * e2off[4 * i + j]];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          ::atomicAdd(&bval[3 * a + b], tH(a, b));
        }
      }
    }
  }
}

__global__ void EnergyNeoHookean(const Vec3* X, const uint32_t* tet,
                                 const Mat3* Dm_inv, const real* vol,
                                 const real mu, const real lambda, real* out,
                                 const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;

  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  Mat3 Ds;
  Ds.col(0) = X[v1] - X[v4];
  Ds.col(1) = X[v2] - X[v4];
  Ds.col(2) = X[v3] - X[v4];
  Mat3 F = Ds * Dm_inv[t];

  real Ic = 0;
#pragma unroll
  for (int i = 0; i < 9; ++i) Ic += F.data()[i] * F.data()[i];
  real e = mu / 2 * (Ic - 3);
  real J = Determinant(F);
  real alpha = 1 + mu / lambda;
  e += lambda / 2 * (J - alpha) * (J - alpha);
  e *= vol[t];
  // not efficient, but works
  ::atomicAdd(out, e);
}

__global__ void EnergyPD(const Vec3* X, const uint32_t* tet, const Mat3* Dm_inv,
                         const real* vol, const real mu, real* out,
                         const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;

  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  Mat3 Ds;
  Ds.col(0) = X[v1] - X[v4];
  Ds.col(1) = X[v2] - X[v4];
  Ds.col(2) = X[v3] - X[v4];
  Mat3 F = Ds * Dm_inv[t];
  Mat3 R;
  GetRotation(F, R);
  real e = 0;
#pragma unroll
  for (int i = 0; i < 9; ++i)
    e += (F.data()[i] - R.data()[i]) * (F.data()[i] - R.data()[i]);
  e = e * mu * vol[t];
  ::atomicAdd(out, e);
}

__global__ void EnergyFixed(const bool* fixed, const Vec3* fixed_X,
                            const Vec3* X, const real control_mag, real* out,
                            const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  if (fixed[v]) {
    real e = control_mag / 2 * (X[v] - fixed_X[v]).squaredNorm();
    ::atomicAdd(out, e);
  }
}

__global__ void EnergyFixedWithCtrl(const bool* fixed, const Vec3* fixed_X,
                                    const Vec3* X, const real control_mag,
                                    int ctrl_vert, Vec3 ctrl_pos, real* out,
                                    const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  if (fixed[v]) {
    real e = control_mag / 2 * (X[v] - fixed_X[v]).squaredNorm();
    ::atomicAdd(out, e);
  } else if (v == ctrl_vert) {
    real e = control_mag / 2 * (X[v] - ctrl_pos).squaredNorm();
    ::atomicAdd(out, e);
  }
}

__global__ void EnergyGrav(const Vec3* X, const real* M, const Vec3 grav,
                           real* out, const uint32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  real e = M[v] * grav.dot(X[v]);
  ::atomicAdd(out, e);
}

__global__ void EnergyInertia(const Vec3* inertia_X, const Vec3* X,
                              const real* M, const real dt_inv, real* out,
                              const uint32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  real e = M[v] * dt_inv * dt_inv / 2 * (X[v] - inertia_X[v]).squaredNorm();
  ::atomicAdd(out, e);
}

__global__ void HessianPD(const Mat3* Dm_inv, const real* vol,
                          const int32_t* t2off, real* bcoo_val, const real mu,
                          const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;

  Eigen::Matrix<real, 4, 3> idm;
  idm.block<3, 3>(0, 0) = Dm_inv[t];
  idm(3, 0) = -idm(0, 0) - idm(1, 0) - idm(2, 0);
  idm(3, 1) = -idm(0, 1) - idm(1, 1) - idm(2, 1);
  idm(3, 2) = -idm(0, 2) - idm(1, 2) - idm(2, 2);

  const int32_t* e2off = &t2off[16 * t];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      real diag = 0;
#pragma unroll
      for (int a = 0; a < 3; ++a) {
        diag += 2 * mu * idm(i, a) * idm(j, a);
      }
      diag *= vol[t];
      real* bval = &bcoo_val[9 * e2off[4 * i + j]];
      ::atomicAdd(&bval[0], diag);
      ::atomicAdd(&bval[4], diag);
      ::atomicAdd(&bval[8], diag);
    }
  }
}

__global__ void InteriaHessian(const real* M, const int32_t* d2off,
                               real* bcoo_val, const real dt_inv,
                               const uint32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  real* bval = &bcoo_val[9 * d2off[v]];
  real diag = M[v] * dt_inv * dt_inv;
  bval[0] += diag;
  bval[4] += diag;
  bval[8] += diag;
}

__global__ void AfKroneckerXXtHalf(const real* Af, const real* XXt,
                                   const int32_t* half_off, real* Af_XXt,
                                   const uint32_t n_half) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= 144 * n_half) return;
  int32_t j = idx % 16;
  int32_t i = idx / 16;
  int32_t b = i / 9;
  i = i % 9;
  int32_t r = (i / 3) * 4 + (j / 4);
  int32_t c = (i % 3) * 4 + (j % 4);
  int32_t a = half_off[b];
  Af_XXt[144 * b + 12 * r + c] = Af[9 * a + i] * XXt[16 * a + j];
}

__global__ void AfDenseReduction(const real* Af_XXt, const int32_t* red_off,
                                 const int32_t* half_off, real* UtAU_dense,
                                 const int32_t n_half, const int32_t n_handle) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 144;
  if (a >= n_half) return;
  int32_t b = red_off[half_off[a]];
  int32_t r = b / n_handle;
  int32_t c = b % n_handle;
  int32_t i = idx % 144;
  ::atomicAdd(
      &UtAU_dense[(12 * c + (i % 12)) * 12 * n_handle + (12 * r + (i / 12))],
      Af_XXt[idx]);
}

__global__ void AfSparseReduction(const real* Af_XXt, const int32_t* red_off,
                                  const int32_t* half_off, real* UtAU_sparse,
                                  const int32_t n_half) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 144;
  if (a >= n_half) return;
  int32_t b = red_off[half_off[a]];
  int32_t i = idx % 144;
  ::atomicAdd(&UtAU_sparse[144 * b + i], Af_XXt[idx]);
}

__global__ void AsDenseMirror(real* A_dense, const int32_t* low_off,
                              const int32_t n_handle, const int32_t low_nnz) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 144;
  if (a >= low_nnz) return;
  int32_t i = idx % 144;
  int32_t b = low_off[a];
  int32_t r = 12 * (b / n_handle) + i / 12;
  int32_t c = 12 * (b % n_handle) + i % 12;
  A_dense[r * 12 * n_handle + c] = A_dense[c * 12 * n_handle + r];
}

__global__ void AsSparseMirror(real* A_sparse, const int32_t* low_off,
                               const int32_t* mirror_off,
                               const int32_t low_nnz) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 144;
  if (a >= low_nnz) return;
  int32_t i = idx % 144;
  int32_t j = (i % 12) * 12 + (i / 12);
  A_sparse[144 * mirror_off[a] + j] = A_sparse[144 * low_off[a] + i];
}

__global__ void AsSparseReduction(const real* A_upper, const int32_t* red_off,
                                  real* A, const int32_t upper_nnz) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 144;
  if (a >= upper_nnz) return;
  int32_t i = idx % 144;
  ::atomicAdd(&A[red_off[a] * 144 + i], A_upper[idx]);
}

__global__ void AsDenseReduction(const real* A_upper, const int32_t* red_off,
                                 real* A, const int32_t n_handle,
                                 const int32_t upper_nnz) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 144;
  if (a >= upper_nnz) return;
  int32_t i = idx % 144;
  int32_t b = red_off[a];
  int32_t r = (b / n_handle) * 12 + (i / 12);
  int32_t c = (b % n_handle) * 12 + (i % 12);
  ::atomicAdd(&A[12 * n_handle * c + r], A_upper[idx]);
}

__global__ void TetGradientPD(const Vec3* X, const uint32_t* tet,
                              const Mat3* Dm_inv, const real* vol, real* grad,
                              const real mu, const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  Mat3 Ds;
  Ds.col(0) = X[v1] - X[v4];
  Ds.col(1) = X[v2] - X[v4];
  Ds.col(2) = X[v3] - X[v4];
  Mat3 F = Ds * Dm_inv[t];
  Mat3 R;
  GetRotation(F, R);
  Eigen::Matrix<real, 4, 3> G;
  G.block<3, 3>(0, 0) = Dm_inv[t];
  G.block<1, 3>(3, 0) = -Vec3::Ones().transpose() * Dm_inv[t];
  Eigen::Matrix<real, 3, 4> T = 2 * mu * vol[t] * (R - F) * G.transpose();
#pragma unroll
  for (int i = 0; i < 12; ++i) grad[12 * t + i] = T(i % 3, i / 3);
}

__global__ void TetGradientNeoHookean(const Vec3* X, const uint32_t* tet,
                                      const Mat3* Dm_inv, const real* vol,
                                      real* grad, const real mu, const real lam,
                                      const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  Mat3 Ds;
  Ds.col(0) = X[v1] - X[v4];
  Ds.col(1) = X[v2] - X[v4];
  Ds.col(2) = X[v3] - X[v4];
  Mat3 F = Ds * Dm_inv[t];
  // Piola
  real fac = lam * (Determinant(F) - 1 - mu / lam);
  Mat3 P;
  P.col(0) = fac * Cross(F.col(1), F.col(2));
  P.col(1) = fac * Cross(F.col(2), F.col(0));
  P.col(2) = fac * Cross(F.col(0), F.col(1));
  P += mu * F;
  Eigen::Matrix<real, 4, 3> G;
  G.block<3, 3>(0, 0) = Dm_inv[t];
  G.block<1, 3>(3, 0) = -Vec3::Ones().transpose() * Dm_inv[t];
  Eigen::Matrix<real, 3, 4> T = -vol[t] * P * G.transpose();
#pragma unroll
  for (int i = 0; i < 12; ++i) grad[12 * t + i] = T(i % 3, i / 3);
}

__global__ void EnergyGradientReordered(
    const real* tet_grad, const int32_t* v2t_ids, const int32_t* v2t_off,
    const bool* fixed, const Vec3* fixed_X, const Vec3* X, const int32_t* m2h,
    real* grad, const real control_mag, const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  int32_t v = m2h[i];
  for (int32_t idx = v2t_off[v]; idx < v2t_off[v + 1]; ++idx) {
    grad[3 * i + 0] += tet_grad[v2t_ids[idx] * 3 + 0];
    grad[3 * i + 1] += tet_grad[v2t_ids[idx] * 3 + 1];
    grad[3 * i + 2] += tet_grad[v2t_ids[idx] * 3 + 2];
  }
  if (fixed[v]) {
    grad[3 * i + 0] += control_mag * (fixed_X[v](0) - X[v](0));
    grad[3 * i + 1] += control_mag * (fixed_X[v](1) - X[v](1));
    grad[3 * i + 2] += control_mag * (fixed_X[v](2) - X[v](2));
  }
}

__global__ void EnergyGradient(const real* tet_grad, const int32_t* v2t_ids,
                               const int32_t* v2t_off, const bool* fixed,
                               const Vec3* fixed_X, const Vec3* X,
                               const real* mass, real* grad,
                               const real control_mag, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  for (int32_t idx = v2t_off[v]; idx < v2t_off[v + 1]; ++idx) {
    grad[3 * v + 0] += tet_grad[v2t_ids[idx] * 3 + 0];
    grad[3 * v + 1] += tet_grad[v2t_ids[idx] * 3 + 1];
    grad[3 * v + 2] += tet_grad[v2t_ids[idx] * 3 + 2];
  }
  if (fixed[v]) {
    grad[3 * v + 0] += control_mag * (fixed_X[v](0) - X[v](0));
    grad[3 * v + 1] += control_mag * (fixed_X[v](1) - X[v](1));
    grad[3 * v + 2] += control_mag * (fixed_X[v](2) - X[v](2));
  }
  // else {
  //   grad[3 * v + 1] -= mass[v] * 9.8;
  // }
}

__global__ void EnergyGradientWithCtrlReordered(
    const real* tet_grad, const int32_t* v2t_ids, const int32_t* v2t_off,
    const bool* fixed, const Vec3* fixed_X, const Vec3* X, const int32_t* m2h,
    real* grad, const real control_mag, int ctrl_vert, Vec3 ctrl_pos,
    const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  int32_t v = m2h[i];
  for (int32_t idx = v2t_off[v]; idx < v2t_off[v + 1]; ++idx) {
    grad[3 * i + 0] += tet_grad[v2t_ids[idx] * 3 + 0];
    grad[3 * i + 1] += tet_grad[v2t_ids[idx] * 3 + 1];
    grad[3 * i + 2] += tet_grad[v2t_ids[idx] * 3 + 2];
  }
  if (fixed[v]) {
    grad[3 * i + 0] += control_mag * (fixed_X[v](0) - X[v](0));
    grad[3 * i + 1] += control_mag * (fixed_X[v](1) - X[v](1));
    grad[3 * i + 2] += control_mag * (fixed_X[v](2) - X[v](2));
  } else if (v == ctrl_vert) {
    grad[3 * i + 0] += control_mag * (ctrl_pos(0) - X[v](0));
    grad[3 * i + 1] += control_mag * (ctrl_pos(1) - X[v](1));
    grad[3 * i + 2] += control_mag * (ctrl_pos(2) - X[v](2));
  }
}

__global__ void EnergyGradientWithCtrl(const real* tet_grad,
                                       const int32_t* v2t_ids,
                                       const int32_t* v2t_off,
                                       const bool* fixed, const Vec3* fixed_X,
                                       const Vec3* X, real* grad,
                                       const real control_mag, int ctrl_vert,
                                       Vec3 ctrl_pos, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  for (int32_t idx = v2t_off[v]; idx < v2t_off[v + 1]; ++idx) {
    grad[3 * v + 0] += tet_grad[v2t_ids[idx] * 3 + 0];
    grad[3 * v + 1] += tet_grad[v2t_ids[idx] * 3 + 1];
    grad[3 * v + 2] += tet_grad[v2t_ids[idx] * 3 + 2];
  }
  if (fixed[v]) {
    grad[3 * v + 0] += control_mag * (fixed_X[v](0) - X[v](0));
    grad[3 * v + 1] += control_mag * (fixed_X[v](1) - X[v](1));
    grad[3 * v + 2] += control_mag * (fixed_X[v](2) - X[v](2));
  } else if (v == ctrl_vert) {
    grad[3 * v + 0] += control_mag * (ctrl_pos(0) - X[v](0));
    grad[3 * v + 1] += control_mag * (ctrl_pos(1) - X[v](1));
    grad[3 * v + 2] += control_mag * (ctrl_pos(2) - X[v](2));
  }
}

__global__ void InertiaGradientReordered(const Vec3* inertia_X, const Vec3* X,
                                         const int32_t* m2h, const real* M,
                                         real* grad, const real dt_inv,
                                         const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  int32_t v = m2h[i];
  real c = M[v] * dt_inv * dt_inv;
  grad[3 * i + 0] += c * (inertia_X[v](0) - X[v](0));
  grad[3 * i + 1] += c * (inertia_X[v](1) - X[v](1));
  grad[3 * i + 2] += c * (inertia_X[v](2) - X[v](2));
}

__global__ void InertiaGradient(const Vec3* inertia_X, const Vec3* X,
                                const real* M, real* grad, const real dt_inv,
                                const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  real c = M[v] * dt_inv * dt_inv;
  grad[3 * v + 0] += c * (inertia_X[v](0) - X[v](0));
  grad[3 * v + 1] += c * (inertia_X[v](1) - X[v](1));
  grad[3 * v + 2] += c * (inertia_X[v](2) - X[v](2));
}

__global__ void ColoredGSAf(real* X, const real* Af_diag,
                            const real* Af_diag_add, const real* b,
                            const int32_t base, const int32_t n_in_color) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_in_color) return;
  int32_t t = base + i;
  Solve3x3Sym(Af_diag + 9 * t, Af_diag_add + 9 * t, b + 3 * t, X + 3 * t);
}

__global__ void GSAf(real* X, const real* Af_diag, const real* Af_diag_add,
                     const real* b, const int32_t* colors, const int c,
                     const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  if (colors[i] != c) return;
  Solve3x3Sym(Af_diag + 9 * i, Af_diag_add + 9 * i, b + 3 * i, X + 3 * i);
}

__global__ void GSAfInc(real* X, const real* Af_diag, const real* Af_diag_add,
                        const real* bcsr_val, const int32_t* bcsr_row,
                        const int32_t* bcsr_col, const real* b,
                        const int32_t* colors, const int c,
                        const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  if (colors[i] != c) return;
  real tmp[3];
  tmp[0] = b[3 * i];
  tmp[1] = b[3 * i + 1];
  tmp[2] = b[3 * i + 2];
  for (int idx = bcsr_row[i]; idx < bcsr_row[i + 1]; ++idx) {
    int32_t j = bcsr_col[idx];
    if (colors[j] >= c) continue;
    const real* Aj = bcsr_val + 9 * idx;
    for (int x = 0; x < 3; ++x) {
      for (int y = 0; y < 3; ++y) {
        tmp[x] -= Aj[3 * x + y] * X[3 * j + y];
      }
    }
  }
  Solve3x3Sym(Af_diag + 9 * i, Af_diag_add + 9 * i, tmp, X + 3 * i);
}

__global__ void GSAs(real* X, const real* diag, const real* diag_add,
                     const real* B, const int32_t* colors, const int c) {
  if (colors[blockIdx.x] != c) return;
  real tol = 1e-7;
  int t = blockIdx.x;
  int off = threadIdx.x;
  const real* A = diag + 144 * t;
  const real* A_add = diag_add + 144 * t;
  real* x = X + 12 * t;
  const real* b = B + 12 * t;

  __shared__ real r[12], p[12], Ap[12];
  __shared__ volatile real temp[18];
  __shared__ real dot, alpha, beta, r_norm, old_r_norm;
  __shared__ bool flag;
  r[off] = b[off];
  p[off] = r[off];
  // wrap reduction
  temp[off] = r[off] * r[off];
  temp[off] += temp[off + 6];
  temp[off] += temp[off + 3];
  if (off == 0) {
    r_norm = temp[0] + temp[1] + temp[2];
    flag = false;
    if (r_norm < tol) flag = true;
  }
  __syncthreads();
  if (flag) return;

  for (int l = 0; l < 12; l++) {
    Ap[off] = 0;
#pragma unroll
    for (int k = 0; k < 12; k++)
      Ap[off] += (A[12 * k + off] + A_add[12 * k + off]) * p[k];
    temp[off] = p[off] * Ap[off];
    // wrap reduction
    temp[off] += temp[off + 6];
    temp[off] += temp[off + 3];
    if (off == 0) {
      dot = temp[0] + temp[1] + temp[2];
      alpha = r_norm / dot;
      if (dot < tol) flag = true;
    }
    __syncthreads();
    if (flag) return;
    x[off] += alpha * p[off];
    r[off] -= alpha * Ap[off];
    temp[off] = r[off] * r[off];
    temp[off] += temp[off + 6];
    temp[off] += temp[off + 3];
    if (off == 0) {
      old_r_norm = r_norm;
      r_norm = temp[0] + temp[1] + temp[2];
      beta = r_norm / old_r_norm;
      if (r_norm < tol) flag = true;
    }
    __syncthreads();
    if (flag) return;
    p[off] = beta * p[off] + r[off];
  }
}

__global__ void GSAsInc(real* X, const real* diag, const real* diag_add,
                        const real* bcsr_val, const int32_t* bcsr_row,
                        const int32_t* bcsr_col, const real* B,
                        const int32_t* colors, const int c) {
  if (colors[blockIdx.x] != c) return;
  real tol = 1e-7;
  int t = blockIdx.x;
  int off = threadIdx.x;
  const real* A = diag + 144 * t;
  const real* A_add = diag_add + 144 * t;
  real* x = X + 12 * t;

  __shared__ real r[12], p[12], Ap[12], b[12];
  __shared__ volatile real temp[18];
  __shared__ real dot, alpha, beta, r_norm, old_r_norm;
  __shared__ bool flag;

  b[off] = B[12 * t + off];
  for (int idx = bcsr_row[t]; idx < bcsr_row[t + 1]; ++idx) {
    int32_t j = bcsr_col[idx];
    if (colors[j] >= c) continue;
    const real* Aj = bcsr_val + 144 * idx;
#pragma unroll
    for (int jj = 0; jj < 12; ++jj) {
      b[off] -= Aj[12 * off + jj] * X[12 * j + jj];
    }
  }

  r[off] = b[off];
  p[off] = r[off];
  // wrap reduction
  temp[off] = r[off] * r[off];
  temp[off] += temp[off + 6];
  temp[off] += temp[off + 3];
  if (off == 0) {
    r_norm = temp[0] + temp[1] + temp[2];
    flag = false;
    if (r_norm < tol) flag = true;
  }
  __syncthreads();
  if (flag) return;

  for (int l = 0; l < 12; l++) {
    Ap[off] = 0;
#pragma unroll
    for (int k = 0; k < 12; k++)
      Ap[off] += (A[12 * k + off] + A_add[12 * k + off]) * p[k];
    temp[off] = p[off] * Ap[off];
    // wrap reduction
    temp[off] += temp[off + 6];
    temp[off] += temp[off + 3];
    if (off == 0) {
      dot = temp[0] + temp[1] + temp[2];
      alpha = r_norm / dot;
      if (dot < tol) flag = true;
    }
    __syncthreads();
    if (flag) return;
    x[off] += alpha * p[off];
    r[off] -= alpha * Ap[off];
    temp[off] = r[off] * r[off];
    temp[off] += temp[off + 6];
    temp[off] += temp[off + 3];
    if (off == 0) {
      old_r_norm = r_norm;
      r_norm = temp[0] + temp[1] + temp[2];
      beta = r_norm / old_r_norm;
      if (r_norm < tol) flag = true;
    }
    __syncthreads();
    if (flag) return;
    p[off] = beta * p[off] + r[off];
  }
}

__global__ void AfDiagAddMulVec(real* Y, const real* Af_diag_add, const real* X,
                                const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
#pragma unroll
  for (int32_t idx = 0; idx < 9; ++idx)
    Y[3 * i + idx / 3] += Af_diag_add[9 * i + idx] * X[3 * i + idx % 3];
}

__global__ void AsDiagAddMulVec(real* Y, const real* As_diag_add, const real* X,
                                const int32_t n_handle) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_handle) return;
#pragma unroll
  for (int32_t idx = 0; idx < 144; ++idx)
    Y[12 * i + idx / 12] += As_diag_add[144 * i + idx] * X[12 * i + idx % 12];
}

__global__ void ADiagAddMulVec(real* Y, const real* diag_add, const real* X,
                               const int32_t sb, const int32_t n_handle) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_handle) return;
  const int32_t nb = sb * sb;
#pragma unroll
  for (int32_t idx = 0; idx < nb; ++idx)
    Y[sb * i + idx / sb] += diag_add[nb * i + idx] * X[sb * i + idx % sb];
}

__global__ void ColoredGSAs(real* X, const real* diag, const real* diag_add,
                            const real* b, const int32_t base) {
  int i = blockIdx.x;
  int off = threadIdx.x;
  int t = base + i;
  const real* A = diag + 144 * t;
  const real* A_add = diag_add + 144 * t;
  real* x = X + 12 * t;
  const real eps = 1e-10;

  __shared__ real r[12], p[12], Ap[12];
  __shared__ volatile real temp[18];
  __shared__ real dot, alpha, beta, r_norm, old_r_norm;
  __shared__ bool flag;
  r[off] = b[12 * t + off];
  p[off] = r[off];
  // wrap reduction
  temp[off] = r[off] * r[off];
  temp[off] += temp[off + 6];
  temp[off] += temp[off + 3];
  if (off == 0) {
    r_norm = temp[0] + temp[1] + temp[2];
    flag = false;
    if (r_norm < eps) flag = true;
  }
  __syncthreads();
  if (flag) return;

  for (int l = 0; l < 12; l++) {
    Ap[off] = 0;
#pragma unroll
    for (int k = 0; k < 12; k++)
      Ap[off] += (A[12 * k + off] + A_add[12 * k + off]) * p[k];
    temp[off] = p[off] * Ap[off];
    // wrap reduction
    temp[off] += temp[off + 6];
    temp[off] += temp[off + 3];
    if (off == 0) {
      dot = temp[0] + temp[1] + temp[2];
      alpha = r_norm / dot;
      if (dot < eps) flag = true;
    }
    __syncthreads();
    if (flag) return;
    x[off] += alpha * p[off];
    r[off] -= alpha * Ap[off];
    temp[off] = r[off] * r[off];
    temp[off] += temp[off + 6];
    temp[off] += temp[off + 3];
    if (off == 0) {
      old_r_norm = r_norm;
      r_norm = temp[0] + temp[1] + temp[2];
      beta = r_norm / old_r_norm;
      if (r_norm < eps) flag = true;
    }
    __syncthreads();
    if (flag) return;
    p[off] = beta * p[off] + r[off];
  }
}

__global__ void JacobiAf(real* X, const real* Af_diag, const real* Af_diag_add,
                         const real* b, const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  Solve3x3Sym(Af_diag + 9 * i, Af_diag_add + 9 * i, b + 3 * i, X + 3 * i);
}

__global__ void JacobiAs(real* X, const real* diag, const real* diag_add,
                         const real* b) {
  real tol = 1e-10;
  int t = blockIdx.x;
  int o = threadIdx.x;
  const real* A = diag + 144 * t;
  const real* A_addition = diag_add + 144 * t;
  real* x = X + 12 * t;
  __shared__ real r[12], p[12], Ap[12];
  __shared__ volatile real temp[18];
  __shared__ real dot, alpha, beta, r_norm, old_r_norm;
  __shared__ bool flag;
  r[o] = b[12 * t + o];
  p[o] = r[o];
  temp[o] = r[o] * r[o];
  temp[o] += temp[o + 6];
  temp[o] += temp[o + 3];
  if (o == 0) {
    r_norm = temp[0] + temp[1] + temp[2];
    flag = false;
    if (r_norm < tol) flag = true;
  }
  __syncthreads();
  if (flag) return;

  for (int l = 0; l < 12; l++) {
    Ap[o] = 0;
#pragma unroll
    for (int k = 0; k < 12; k++)
      Ap[o] += (A[12 * k + o] + A_addition[12 * k + o]) * p[k];
    temp[o] = p[o] * Ap[o];
    temp[o] += temp[o + 6];
    temp[o] += temp[o + 3];
    if (o == 0) {
      dot = temp[0] + temp[1] + temp[2];
      alpha = r_norm / dot;
      if (dot < tol) flag = true;
    }
    __syncthreads();
    if (flag) return;
    x[o] += alpha * p[o];
    r[o] -= alpha * Ap[o];
    temp[o] = r[o] * r[o];
    temp[o] += temp[o + 6];
    temp[o] += temp[o + 3];
    if (o == 0) {
      old_r_norm = r_norm;
      r_norm = temp[0] + temp[1] + temp[2];
      beta = r_norm / old_r_norm;
      if (r_norm < tol) flag = true;
    }
    __syncthreads();
    if (flag) return;
    p[o] = beta * p[o] + r[o];
  }
}

__global__ void JacobiAs(real* X, const real* diag, const real* diag_add,
                         const real* b, const int32_t sb) {
  real tol = 1e-10;
  int t = blockIdx.x;
  int o = threadIdx.x;
  const real* A = diag + sb * sb * t;
  const real* A_addition = diag_add + sb * sb * t;
  real* x = X + sb * t;
  extern __shared__ real array[];
  // __shared__ real r[12], p[12], Ap[12];
  // __shared__ volatile real temp[18];
  real* r = array;
  real* p = r + sb;
  real* Ap = p + sb;
  real* temp = Ap + sb;
  __shared__ real dot, alpha, beta, r_norm, old_r_norm;
  __shared__ bool flag;
  r[o] = b[sb * t + o];
  p[o] = r[o];
  temp[o] = r[o] * r[o];
  __syncthreads();
  // sb is always a multiple of 3
  for (uint32_t s = blockDim.x / 2; s > 1; s >>= 1) {
    if (o < s) {
      temp[o] += temp[o + s];
    }
    __syncthreads();
  }
  if (o == 0) {
    r_norm = temp[0] + temp[1] + temp[2];
    flag = false;
    if (r_norm < tol) flag = true;
  }
  __syncthreads();
  if (flag) return;

  for (int l = 0; l < sb; l++) {
    Ap[o] = 0;
#pragma unroll
    for (int k = 0; k < sb; k++)
      Ap[o] += (A[sb * k + o] + A_addition[sb * k + o]) * p[k];
    temp[o] = p[o] * Ap[o];
    __syncthreads();
    // sb is always a multiple of 3
    for (uint32_t s = blockDim.x / 2; s > 1; s >>= 1) {
      if (o < s) {
        temp[o] += temp[o + s];
      }
      __syncthreads();
    }
    if (o == 0) {
      dot = temp[0] + temp[1] + temp[2];
      alpha = r_norm / dot;
      if (dot < tol) flag = true;
    }
    __syncthreads();
    if (flag) return;
    x[o] += alpha * p[o];
    r[o] -= alpha * Ap[o];
    temp[o] = r[o] * r[o];
    __syncthreads();
    // sb is always a multiple of 3
    for (uint32_t s = blockDim.x / 2; s > 1; s >>= 1) {
      if (o < s) {
        temp[o] += temp[o + s];
      }
      __syncthreads();
    }
    if (o == 0) {
      old_r_norm = r_norm;
      r_norm = temp[0] + temp[1] + temp[2];
      beta = r_norm / old_r_norm;
      if (r_norm < tol) flag = true;
    }
    __syncthreads();
    if (flag) return;
    p[o] = beta * p[o] + r[o];
  }
}

__global__ void ColoredGSAs(real* X, const real* diag, const real* diag_add,
                            const real* b, const int32_t sb,
                            const int32_t base) {
  int i = blockIdx.x;
  int off = threadIdx.x;
  int t = base + i;
  const real* A = diag + sb * sb * t;
  const real* A_add = diag_add + sb * sb * t;
  real* x = X + sb * t;
  const real eps = 1e-10;

  // __shared__ real r[12], p[12], Ap[12];
  // __shared__ volatile real temp[18];
  extern __shared__ real array[];
  real* r = array;
  real* p = r + sb;
  real* Ap = p + sb;
  real* temp = Ap + sb;
  __shared__ real dot, alpha, beta, r_norm, old_r_norm;
  __shared__ bool flag;
  r[off] = b[sb * t + off];
  p[off] = r[off];
  // warp reduction
  temp[off] = r[off] * r[off];
  __syncthreads();
  // sb is always a multiple of 3
  for (uint32_t s = blockDim.x / 2; s > 1; s >>= 1) {
    if (off < s) {
      temp[off] += temp[off + s];
    }
    __syncthreads();
  }
  if (off == 0) {
    r_norm = temp[0] + temp[1] + temp[2];
    flag = false;
    if (r_norm < eps) flag = true;
  }
  __syncthreads();
  if (flag) return;

  for (int l = 0; l < sb; l++) {
    Ap[off] = 0;
#pragma unroll
    for (int k = 0; k < sb; k++)
      Ap[off] += (A[sb * k + off] + A_add[sb * k + off]) * p[k];
    temp[off] = p[off] * Ap[off];
    // wrap reduction
    __syncthreads();
    // sb is always a multiple of 3
    for (uint32_t s = blockDim.x / 2; s > 1; s >>= 1) {
      if (off < s) {
        temp[off] += temp[off + s];
      }
      __syncthreads();
    }
    if (off == 0) {
      dot = temp[0] + temp[1] + temp[2];
      alpha = r_norm / dot;
      if (dot < eps) flag = true;
    }
    __syncthreads();
    if (flag) return;
    x[off] += alpha * p[off];
    r[off] -= alpha * Ap[off];
    temp[off] = r[off] * r[off];
    __syncthreads();
    // sb is always a multiple of 3
    for (uint32_t s = blockDim.x / 2; s > 1; s >>= 1) {
      if (off < s) {
        temp[off] += temp[off + s];
      }
      __syncthreads();
    }
    if (off == 0) {
      old_r_norm = r_norm;
      r_norm = temp[0] + temp[1] + temp[2];
      beta = r_norm / old_r_norm;
      if (r_norm < eps) flag = true;
    }
    __syncthreads();
    if (flag) return;
    p[off] = beta * p[off] + r[off];
  }
}

__global__ void UpdatedXReordered(Vec3* X, const real alpha, const real* dX,
                                  const int32_t* m2h, const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  int32_t v = m2h[i];
  X[v] += alpha * Vec3(dX[3 * i], dX[3 * i + 1], dX[3 * i + 2]);
}

__global__ void UpdatedX(Vec3* X, const real alpha, const real* dX,
                         const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  X[v] += alpha * Vec3(dX[3 * v], dX[3 * v + 1], dX[3 * v + 2]);
}

__global__ void UpdateVelFromPos(Vec3* V, const Vec3* X, const Vec3* old_X,
                                 const real dt_inv, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  V[v] = (X[v] - old_X[v]) * dt_inv;
}

__global__ void UpdateDiagXXt(const Vec3* X, real* diag_XXt,
                              const uint32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  Vec4 tmp = Vec4::Unit(3);
  tmp.segment<3>(0) = X[v];
  for (int x = 0; x < 4; ++x) {
    for (int y = 0; y < 4; ++y) {
      diag_XXt[16 * v + 4 * x + y] = tmp[x] * tmp[y];
    }
  }
}

__global__ void UpdateDiagXXtReordered(const Vec3* X, const int32_t* m2h,
                                       real* diag_XXt, const uint32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  Vec4 tmp = Vec4::Unit(3);
  tmp.segment<3>(0) = X[m2h[i]];
#pragma unroll
  for (int x = 0; x < 4; ++x) {
    for (int y = 0; y < 4; ++y) {
      diag_XXt[16 * i + 4 * x + y] = tmp[x] * tmp[y];
    }
  }
}

__global__ void UpdateXXt(const Vec3* X, const int32_t* bcoo_row,
                          const int32_t* bcoo_col, real* XXt,
                          const uint32_t bnnz) {
  int32_t b = blockDim.x * blockIdx.x + threadIdx.x;
  if (b >= bnnz) return;
  int32_t i = bcoo_row[b];
  int32_t j = bcoo_col[b];
  Vec4 xi = Vec4::Unit(3);
  xi.segment<3>(0) = X[i];
  Vec4 xj = Vec4::Unit(3);
  xj.segment<3>(0) = X[j];
#pragma unroll
  for (int x = 0; x < 4; ++x) {
    for (int y = 0; y < 4; ++y) {
      XXt[16 * b + 4 * x + y] = xi[x] * xj[y];
    }
  }
}

__global__ void UpdateXXtReordered(const Vec3* X, const int32_t* m2h,
                                   const int32_t* bcoo_row,
                                   const int32_t* bcoo_col, real* XXt,
                                   const uint32_t bnnz) {
  int32_t b = blockDim.x * blockIdx.x + threadIdx.x;
  if (b >= bnnz) return;
  int32_t i = bcoo_row[b];
  int32_t j = bcoo_col[b];
  Vec4 xi = Vec4::Unit(3);
  xi.segment<3>(0) = X[m2h[i]];
  Vec4 xj = Vec4::Unit(3);
  xj.segment<3>(0) = X[m2h[j]];
#pragma unroll
  for (int x = 0; x < 4; ++x) {
    for (int y = 0; y < 4; ++y) {
      XXt[16 * b + 4 * x + y] = xi[x] * xj[y];
    }
  }
}

__global__ void UpdateU(const Vec3* X, real* csr_val, const uint32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
#pragma unroll
  for (int d = 0; d < 3; ++d) {
    csr_val[12 * v + 4 * d + 0] = X[v][0];
    csr_val[12 * v + 4 * d + 1] = X[v][1];
    csr_val[12 * v + 4 * d + 2] = X[v][2];
    csr_val[12 * v + 4 * d + 3] = 1.;
  }
}

__global__ void UpdateUReordered(const Vec3* X, const int32_t* m2h,
                                 real* csr_val, const uint32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  int32_t v = m2h[i];
#pragma unroll
  for (int d = 0; d < 3; ++d) {
    csr_val[12 * i + 4 * d + 0] = X[v][0];
    csr_val[12 * i + 4 * d + 1] = X[v][1];
    csr_val[12 * i + 4 * d + 2] = X[v][2];
    csr_val[12 * i + 4 * d + 3] = 1.;
  }
}

__global__ void SkeletonGradient(real* grad, const Vec3* X,
                                 const int32_t* bone_ids,
                                 const JointTransform* trans,
                                 const Vec3* local_pos, const real ctr_mag,
                                 const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  int32_t b = bone_ids[v];
  if (b < 0) return;
  const Mat3& rot = trans[b].global_rot;
  const Vec3& pos = trans[b].global_pos;
  Vec3 gpos = rot * local_pos[v] + pos;
  grad[3 * v + 0] += ctr_mag * (gpos(0) - X[v](0));
  grad[3 * v + 1] += ctr_mag * (gpos(1) - X[v](1));
  grad[3 * v + 2] += ctr_mag * (gpos(2) - X[v](2));
}

__global__ void UpdateAfDiagSkeleton(real* Af_diag_add, const int32_t* bone_id,
                                     const real ctr_mag, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  int32_t b = bone_id[v];
  if (b < 0) return;
  real* diag = Af_diag_add + 9 * v;
  diag[0] += ctr_mag;
  diag[4] += ctr_mag;
  diag[8] += ctr_mag;
}

__global__ void MakeSelfCollision(
    real* grad, int32_t* vv_pairs, real* hessians, real* diag_add,
    const Vec3* X, const Vec3* norms, const uint32_t* tets,
    const uint32_t* vt_pairs, const int32_t* closest_surf_vert,
    const real k_penalty, const int32_t n_vert, const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  uint32_t v = vt_pairs[2 * c + 0];
  uint32_t t = vt_pairs[2 * c + 1];
  real min_dist = 1e16;
  uint32_t p = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    if (closest_surf_vert[4 * t + i] < 0) break;
    uint32_t vi = closest_surf_vert[4 * t + i];
    real dist = norms[vi].dot(X[vi] - X[v]);
    // real dist = (X[vi] - X[v]).norm();
    if (dist < min_dist) {
      min_dist = dist;
      p = vi;
    }
  }

  vv_pairs[2 * c + 0] = v;
  vv_pairs[2 * c + 1] = p;

  Vec3 n = norms[p];
  for (int x = 0; x < 3; ++x) {
    for (int y = 0; y < 3; ++y) {
      real h = k_penalty * n[x] * n[y];
      hessians[9 * c + 3 * x + y] = h;
      ::atomicAdd(&diag_add[9 * v + 3 * x + y], h);
      ::atomicAdd(&diag_add[9 * p + 3 * x + y], h);
    }
  }
  // for (int x = 0; x < 3; ++x) {
  //   hessians[16 * c + 5 * x] = k_penalty;
  //   ::atomicAdd(&diag_add[16 * v + 5 * x], k_penalty);
  //   ::atomicAdd(&diag_add[16 * p + 5 * x], k_penalty);
  // }
  Vec3 fv = k_penalty * n.dot(X[p] - X[v]) * n;
  // Vec3 fv = k_penalty * (X[p] - X[v]);
  Vec3 fp = -fv;
  for (int x = 0; x < 3; ++x) {
    ::atomicAdd(&grad[3 * v + x], fv[x]);
    ::atomicAdd(&grad[3 * p + x], fp[x]);
  }
}

__global__ void SelfCollisionFineOffReduction(
    int32_t* coarse_pairs, real* coarse_hessian, const int32_t* fine_pairs,
    const real* fine_hessian, const Vec3* pos_rest, const int32_t* handle,
    const int32_t* handle_ids, const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = fine_pairs[2 * c + 0];
  int32_t v2 = fine_pairs[2 * c + 1];
  int32_t h1 = handle[v1];
  int32_t h2 = handle[v2];
  coarse_pairs[2 * c + 0] = h1;
  coarse_pairs[2 * c + 1] = h2;
  if (h1 == h2) return;
  int32_t vh1 = handle_ids[h1];
  int32_t vh2 = handle_ids[h2];
  const real* fhessian = &fine_hessian[9 * c];
  real* chessian = &coarse_hessian[144 * c];
  Vec4 x1 = Vec4::Unit(3);
  Vec4 x2 = Vec4::Unit(3);
  x1[0] = pos_rest[v1](0);
  x1[1] = pos_rest[v1](1);
  x1[2] = pos_rest[v1](2);
  x2[0] = pos_rest[v2](0);
  x2[1] = pos_rest[v2](1);
  x2[2] = pos_rest[v2](2);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
          int r = 4 * i + x;
          int c = 4 * j + y;
          chessian[12 * r + c] = fhessian[3 * i + j] * x1[x] * x2[y];
        }
      }
    }
  }
}

__global__ void SelfCollisionFineDiagReduction(
    real* diag_add, const int32_t* fine_pairs, const real* fine_hessian,
    const Vec3* pos_rest, const int32_t* handle, const int32_t* handle_ids,
    const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = fine_pairs[2 * c + 0];
  int32_t v2 = fine_pairs[2 * c + 1];
  int32_t h1 = handle[v1];
  int32_t h2 = handle[v2];
  if (h1 != h2) return;
  int32_t vh1 = handle_ids[h1];
  int32_t vh2 = handle_ids[h2];
  const real* fhessian = &fine_hessian[9 * c];
  Vec4 x1 = Vec4::Unit(3);
  Vec4 x2 = Vec4::Unit(3);
  x1[0] = pos_rest[v1](0);
  x1[1] = pos_rest[v1](1);
  x1[2] = pos_rest[v1](2);
  x2[0] = pos_rest[v2](0);
  x2[1] = pos_rest[v2](1);
  x2[2] = pos_rest[v2](2);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
          int r = 4 * i + x;
          int c = 4 * j + y;
          real tmp = fhessian[3 * i + j] * x1[x] * x2[y] +
                     fhessian[3 * j + i] * x1[y] * x2[x];
          ::atomicAdd(&diag_add[144 * h1 + 12 * r + c], -tmp);
        }
      }
    }
  }
}

__global__ void SelfCollisionCoarseOffReduction(
    int32_t* coarse_pairs, real* coarse_hessian, const int32_t* fine_pairs,
    const real* fine_hessian, const int32_t* handle, const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = fine_pairs[2 * c + 0];
  int32_t v2 = fine_pairs[2 * c + 1];
  int32_t h1 = handle[v1];
  int32_t h2 = handle[v2];
  coarse_pairs[2 * c + 0] = h1;
  coarse_pairs[2 * c + 1] = h2;
  if ((v1 == v2) || (h1 == h2)) return;
  int32_t i = threadIdx.y;
  const real* fhessian = &fine_hessian[144 * c];
  real* chessian = &coarse_hessian[144 * c];
  chessian[i] = fhessian[i];
}

__global__ void SelfCollisionCoarseDiagReduction(real* diag_add,
                                                const int32_t* fine_pairs,
                                                const real* fine_hessian,
                                                const int32_t* handle,
                                                const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = fine_pairs[2 * c + 0];
  int32_t v2 = fine_pairs[2 * c + 1];
  int32_t h1 = handle[v1];
  int32_t h2 = handle[v2];
  if ((v1 == v2) || (h1 != h2)) return;
  const real* fhessian = &fine_hessian[144 * c];
  int32_t i = threadIdx.y;
  int32_t x = i / 12, y = i % 12;
  ::atomicAdd(&diag_add[144 * h1 + i], -fhessian[i] - fhessian[12 * y + x]);
}

__global__ void SelfCollisionCoarseDenseHessian(real* den,
                                                     const int32_t* pairs,
                                                     const real* hessian,
                                                     const int32_t dim,
                                                     const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[144 * c];
  int32_t i = threadIdx.y;
  int32_t x = i / 12, y = i % 12;
  int32_t row = 12 * v1 + x;
  int32_t col = 12 * v2 + y;
  ::atomicAdd(&den[row + dim * col], -h[12 * x + y]);
  ::atomicAdd(&den[col + dim * row], -h[12 * x + y]);
}


__global__ void SelfCollisionFineOffGSAP(
    real* AP, const int32_t* pairs, const real* hessian, const int32_t* color,
    const int32_t co, const real* P, const real alpha, const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[9 * c];
  const real* p1 = &P[3 * v1];
  const real* p2 = &P[3 * v2];
  if (color[v1] == co) {
    real ap2[3];
    for (int i = 0; i < 3; ++i) {
      ap2[i] = 0.;
      for (int j = 0; j < 3; ++j) {
        ap2[i] -= h[3 * j + i] * p1[j];
      }
    }
    ::atomicAdd(&AP[3 * v2 + 0], alpha * ap2[0]);
    ::atomicAdd(&AP[3 * v2 + 1], alpha * ap2[1]);
    ::atomicAdd(&AP[3 * v2 + 2], alpha * ap2[2]);
  }
  if (color[v2] == co) {
    real ap1[3];
    for (int i = 0; i < 3; ++i) {
      ap1[i] = 0.;
      for (int j = 0; j < 3; ++j) {
        ap1[i] -= h[3 * i + j] * p2[j];
      }
    }
    ::atomicAdd(&AP[3 * v1 + 0], alpha * ap1[0]);
    ::atomicAdd(&AP[3 * v1 + 1], alpha * ap1[1]);
    ::atomicAdd(&AP[3 * v1 + 2], alpha * ap1[2]);
  }
}

__global__ void SelfCollisionCoarseOffGSAP(
    real* AP, const int32_t* pairs, const real* hessian, const int32_t* color,
    const int32_t co, const real* P, const real alpha, const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[144 * c];
  int32_t i = threadIdx.y;
  const real* p1 = &P[12 * v1];
  const real* p2 = &P[12 * v2];
  if (color[v1] == co) {
    real ap2 = 0.;
    for (int j = 0; j < 12; ++j) {
      ap2 -= h[12 * j + i] * p1[j];
    }
    ::atomicAdd(&AP[12 * v2 + i], alpha * ap2);
  }
  if (color[v2] == co) {
    real ap1 = 0.;
    for (int j = 0; j < 12; ++j) {
      ap1 -= h[12 * i + j] * p2[j];
    }
    ::atomicAdd(&AP[12 * v1 + i], alpha * ap1);
  }
}

__global__ void SelfCollisionFineOffAP(real* AP, const int32_t* pairs,
                                            const real* hessian, const real* P,
                                            const real alpha,
                                            const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[9 * c];
  const real* p1 = &P[3 * v1];
  const real* p2 = &P[3 * v2];
  real ap1[3], ap2[3];
  for (int i = 0; i < 3; ++i) {
    ap1[i] = 0.;
    ap2[i] = 0.;
    for (int j = 0; j < 3; ++j) {
      ap1[i] -= h[3 * i + j] * p2[j];
      ap2[i] -= h[3 * j + i] * p1[j];
    }
  }

  ::atomicAdd(&AP[3 * v1 + 0], alpha * ap1[0]);
  ::atomicAdd(&AP[3 * v1 + 1], alpha * ap1[1]);
  ::atomicAdd(&AP[3 * v1 + 2], alpha * ap1[2]);
  ::atomicAdd(&AP[3 * v2 + 0], alpha * ap2[0]);
  ::atomicAdd(&AP[3 * v2 + 1], alpha * ap2[1]);
  ::atomicAdd(&AP[3 * v2 + 2], alpha * ap2[2]);
}

__global__ void SelfCollisionCoarseOffAP(real* AP, const int32_t* pairs,
                                              const real* hessian,
                                              const real* P, const real alpha,
                                              const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[144 * c];
  int32_t i = threadIdx.y;
  const real* p1 = &P[12 * v1];
  const real* p2 = &P[12 * v2];
  real ap1 = 0., ap2 = 0.;
  for (int j = 0; j < 12; ++j) {
    ap1 -= h[12 * i + j] * p2[j];
    ap2 -= h[12 * j + i] * p1[j];
  }
  ::atomicAdd(&AP[12 * v1 + i], alpha * ap1);
  ::atomicAdd(&AP[12 * v2 + i], alpha * ap2);
}

};  // namespace Rain::CUDA