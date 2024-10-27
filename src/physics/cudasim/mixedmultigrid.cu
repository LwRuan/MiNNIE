#include <curand.h>
#include <curand_kernel.h>

#include "cudaelasticmodel.h"
#include "cudasvd.h"
#include "mixedmultigrid.h"

namespace Rain::CUDA {
__global__ void HessianMixedNeoHookeanClamped(
    const Vec3* X, const real* P, const uint32_t* tet, const Mat3* Dm_inv,
    const real* vol, const int32_t* t2off, real* bcoo_val,
    const int32_t* normal_sign, const real mu, const real lambda_inv,
    const real p_smooth, const real p_scale, const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  const uint32_t* ctet = &tet[4 * t];
  real p_ave = (P[v1] + P[v2] + P[v3] + P[v4]) / 4.;
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
  {
    Mat3 L = Mat3::Identity();
    L(2, 2) = Determinant(U * V.transpose());
    real detU = Determinant(U);
    real detV = Determinant(V);
    if (detU < 0. && detV > 0.) U = U * L;
    if (detU > 0. && detV < 0.) V = V * L;
    S(2) = S(2) * L(2, 2);
  }

  Vec9 eigenvalues;
  Mat9 eigenvectors;

  BuildTwistAndFlipEigenvectors(U, V, eigenvectors);
  eigenvalues.segment<3>(0) = -S;
  eigenvalues.segment<3>(3) = S;
  eigenvalues.segment<6>(0) *= mu;
  eigenvalues.segment<6>(0).array() += mu;

  Mat3 tA = Mat3::Zero();
  tA << 0., S(2), S(1), S(2), 0., S(0), S(1), S(0), 0.;
  S.setZero();
  Mat3 U1, V1;
#ifdef REAL_AS_DOUBLE
  Mat3f U1f, V1f;
  SVD(tA.cast<float>(), U1f, V1f, Sf);
  U1 = U1f.cast<double>();
  V1 = V1f.cast<double>();
  S = Sf.cast<double>();
#else
  SVD(tA, U1, V1, S);
#endif
  if (U1.col(0).dot(V1.col(0)) < 0.) {
    S(0) *= -1.;
  }
  if (U1.col(1).dot(V1.col(1)) < 0.) {
    S(1) *= -1.;
  }
  if (U1.col(2).dot(V1.col(2)) < 0.) {
    S(2) *= -1.;
  }

  eigenvalues.segment<3>(6) = mu * Vec3::Ones() - (mu - p_ave) * S;
  // eigenvalues.segment<3>(6) = mu * Vec3::Ones() - mu * S;
  Eigen::Map<Mat3>(eigenvectors.data() + 54) =
      U * U1.col(0).asDiagonal() * V.transpose();
  Eigen::Map<Mat3>(eigenvectors.data() + 63) =
      U * U1.col(1).asDiagonal() * V.transpose();
  Eigen::Map<Mat3>(eigenvectors.data() + 72) =
      U * U1.col(2).asDiagonal() * V.transpose();

  // Vec3 peigs = p_ave * S;

  // Clamp the eigenvalues
#pragma unroll
  for (int i = 0; i < 9; i++) {
    if (eigenvalues(i) < 0.0) {
      eigenvalues(i) = 0.0;
    }
  }
  // for (int i = 0; i < 3; ++i) {
  //   if (eigenvalues(6 + i) < -peigs(i)) {
  //     eigenvalues(6 + i) = -peigs(i);
  //   }
  // }

  Mat9 dPdF =
      eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

  real pk = p_smooth * vol[t] / 80. / mu;
  real pm = vol[t] * lambda_inv / 20.;

  const int32_t* e2off = &t2off[16 * t];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    Vec3 dxi1 = X[ctet[(i + 1) % 4]] - X[ctet[(i + 3) % 4]];
    Vec3 dxi2 = X[ctet[(i + 2) % 4]] - X[ctet[(i + 3) % 4]];
    Vec3 ni = Cross(dxi1, dxi2) / 6. * normal_sign[4 * t + i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      Vec3 dxj1 = X[ctet[(j + 1) % 4]] - X[ctet[(j + 3) % 4]];
      Vec3 dxj2 = X[ctet[(j + 2) % 4]] - X[ctet[(j + 3) % 4]];
      Vec3 nj = Cross(dxj1, dxj2) / 6. * normal_sign[4 * t + j];
      Mat4 tH = Mat4::Zero();
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          tH.block<3, 3>(0, 0) +=
              idm(i, a) * dPdF.block<3, 3>(3 * a, 3 * b) * idm(j, b) * vol[t];
        }
      }
      tH.block<1, 3>(3, 0) += p_scale * nj.transpose() / 4.;
      tH.block<3, 1>(0, 3) += p_scale * ni / 4.;
      if (i == j)
        tH(3, 3) -= (3 * pk + 2 * pm) * p_scale * p_scale;
      else
        tH(3, 3) -= (pm - pk) * p_scale * p_scale;

      real* bval = &bcoo_val[16 * e2off[4 * i + j]];
#pragma unroll
      for (int a = 0; a < 4; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          ::atomicAdd(&bval[4 * a + b], tH(a, b));
        }
      }
    }
  }
}

__global__ void HessianMixedStVKClamped(
    const Vec3* X, const real* P, const uint32_t* tet, const Mat3* Dm_inv,
    const real* vol, const int32_t* t2off, real* bcoo_val,
    const int32_t* normal_sign, const real mu, const real lambda_inv,
    const real p_smooth, const real p_scale, const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  const uint32_t* ctet = &tet[4 * t];
  real p_ave = (P[v1] + P[v2] + P[v3] + P[v4]) / 4.;
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
  {
    Mat3 L = Mat3::Identity();
    L(2, 2) = Determinant(U * V.transpose());
    real detU = Determinant(U);
    real detV = Determinant(V);
    if (detU < 0. && detV > 0.) U = U * L;
    if (detU > 0. && detV < 0.) V = V * L;
    S(2) = S(2) * L(2, 2);
  }

  Vec9 eigenvalues;
  Mat9 eigenvectors;

  BuildTwistAndFlipEigenvectors(U, V, eigenvectors);
  eigenvalues[0] = -mu + p_ave + mu * (S[1] * S[1] + S[2] * S[2] - S[1] * S[2]);
  eigenvalues[1] = -mu + p_ave + mu * (S[0] * S[0] + S[2] * S[2] - S[0] * S[2]);
  eigenvalues[2] = -mu + p_ave + mu * (S[0] * S[0] + S[1] * S[1] - S[0] * S[1]);
  eigenvalues[3] = -mu + p_ave + mu * (S[1] * S[1] + S[2] * S[2] + S[1] * S[2]);
  eigenvalues[4] = -mu + p_ave + mu * (S[0] * S[0] + S[2] * S[2] + S[0] * S[2]);
  eigenvalues[5] = -mu + p_ave + mu * (S[0] * S[0] + S[1] * S[1] + S[0] * S[1]);

  BuildScaleEigenvectors(U, V, eigenvectors);
  eigenvalues[6] = -mu + p_ave + 3. * mu * (S[0] * S[0]);
  eigenvalues[7] = -mu + p_ave + 3. * mu * (S[1] * S[1]);
  eigenvalues[8] = -mu + p_ave + 3. * mu * (S[2] * S[2]);

  // Clamp the eigenvalues
#pragma unroll
  for (int i = 0; i < 9; i++) {
    if (eigenvalues(i) < 0.0) {
      eigenvalues(i) = 0.0;
    }
  }

  Mat9 dPdF =
      eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

  real pk = p_smooth * vol[t] / 80. / mu;
  real pm = vol[t] * lambda_inv / 20.;

  Eigen::Matrix<real, 3, 4> tmpg = F * idm.transpose();

  const int32_t* e2off = &t2off[16 * t];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      Mat4 tH = Mat4::Zero();
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          tH.block<3, 3>(0, 0) +=
              idm(i, a) * dPdF.block<3, 3>(3 * a, 3 * b) * idm(j, b) * vol[t];
        }
      }

      tH.block<1, 3>(3, 0) = p_scale * tmpg.col(j).transpose() / 4. * vol[t];
      tH.block<3, 1>(0, 3) = p_scale * tmpg.col(i) / 4. * vol[t];

      if (i == j)
        tH(3, 3) -= (3 * pk + 2 * pm) * p_scale * p_scale;
      else
        tH(3, 3) -= (pm - pk) * p_scale * p_scale;

      real* bval = &bcoo_val[16 * e2off[4 * i + j]];
#pragma unroll
      for (int a = 0; a < 4; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          ::atomicAdd(&bval[4 * a + b], tH(a, b));
        }
      }
    }
  }
}

__global__ void HessianMixedCorotationClamped(
    const Vec3* X, const real* P, const uint32_t* tet, const Mat3* Dm_inv,
    const real* vol, const int32_t* t2off, real* bcoo_val,
    const int32_t* normal_sign, const real mu, const real lambda_inv,
    const real p_smooth, const real p_scale, const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  const uint32_t* ctet = &tet[4 * t];
  real p_ave = (P[v1] + P[v2] + P[v3] + P[v4]) / 4.;
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
  {
    Mat3 L = Mat3::Identity();
    L(2, 2) = Determinant(U * V.transpose());
    real detU = Determinant(U);
    real detV = Determinant(V);
    if (detU < 0. && detV > 0.) U = U * L;
    if (detU > 0. && detV < 0.) V = V * L;
    S(2) = S(2) * L(2, 2);
  }

  Mat9 dPdF = 2 * mu * Mat9::Identity();

  real pk = p_smooth * vol[t] / 80. / mu;
  real pm = vol[t] * lambda_inv / 20.;

  Eigen::Matrix<real, 3, 4> tmpg = U * V.transpose() * idm.transpose();

  const int32_t* e2off = &t2off[16 * t];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      Mat4 tH = Mat4::Zero();
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          tH.block<3, 3>(0, 0) +=
              idm(i, a) * dPdF.block<3, 3>(3 * a, 3 * b) * idm(j, b) * vol[t];
        }
      }

      tH.block<1, 3>(3, 0) = p_scale * tmpg.col(j).transpose() / 4. * vol[t];
      tH.block<3, 1>(0, 3) = p_scale * tmpg.col(i) / 4. * vol[t];

      if (i == j)
        tH(3, 3) -= (3 * pk + 2 * pm) * p_scale * p_scale;
      else
        tH(3, 3) -= (pm - pk) * p_scale * p_scale;

      real* bval = &bcoo_val[16 * e2off[4 * i + j]];
#pragma unroll
      for (int a = 0; a < 4; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          ::atomicAdd(&bval[4 * a + b], tH(a, b));
        }
      }
    }
  }
}

__global__ void HessianMixedNeoHookeanLogClamped(
    const Vec3* X, const real* P, const uint32_t* tet, const Mat3* Dm_inv,
    const real* vol, const int32_t* t2off, real* bcoo_val,
    const int32_t* normal_sign, const real mu, const real lambda_inv,
    const real p_smooth, const real p_scale, const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  const uint32_t* ctet = &tet[4 * t];
  real p_ave = (P[v1] + P[v2] + P[v3] + P[v4]) / 4.;
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
  {
    Mat3 L = Mat3::Identity();
    L(2, 2) = Determinant(U * V.transpose());
    real detU = Determinant(U);
    real detV = Determinant(V);
    if (detU < 0. && detV > 0.) U = U * L;
    if (detU > 0. && detV < 0.) V = V * L;
    S(2) = S(2) * L(2, 2);
  }

  Vec9 eigenvalues;
  Mat9 eigenvectors;

  BuildTwistAndFlipEigenvectors(U, V, eigenvectors);
  eigenvalues.segment<3>(0) = -S;
  eigenvalues.segment<3>(3) = S;
  eigenvalues.segment<6>(0) *= mu;
  eigenvalues.segment<6>(0).array() += mu;

  Mat3 tA = Mat3::Zero();
  tA << 0., S(2), S(1), S(2), 0., S(0), S(1), S(0), 0.;
  S.setZero();
  Mat3 U1, V1;
#ifdef REAL_AS_DOUBLE
  Mat3f U1f, V1f;
  SVD(tA.cast<float>(), U1f, V1f, Sf);
  U1 = U1f.cast<double>();
  V1 = V1f.cast<double>();
  S = Sf.cast<double>();
#else
  SVD(tA, U1, V1, S);
#endif
  if (U1.col(0).dot(V1.col(0)) < 0.) {
    S(0) *= -1.;
  }
  if (U1.col(1).dot(V1.col(1)) < 0.) {
    S(1) *= -1.;
  }
  if (U1.col(2).dot(V1.col(2)) < 0.) {
    S(2) *= -1.;
  }

  eigenvalues.segment<3>(6) = mu * Vec3::Ones() - (mu - p_ave) * S;
  // eigenvalues.segment<3>(6) = mu * Vec3::Ones() - mu * S;
  Eigen::Map<Mat3>(eigenvectors.data() + 54) =
      U * U1.col(0).asDiagonal() * V.transpose();
  Eigen::Map<Mat3>(eigenvectors.data() + 63) =
      U * U1.col(1).asDiagonal() * V.transpose();
  Eigen::Map<Mat3>(eigenvectors.data() + 72) =
      U * U1.col(2).asDiagonal() * V.transpose();

  // Vec3 peigs = p_ave * S;

  // Clamp the eigenvalues
#pragma unroll
  for (int i = 0; i < 9; i++) {
    if (eigenvalues(i) < 0.0) {
      eigenvalues(i) = 0.0;
    }
  }
  // for (int i = 0; i < 3; ++i) {
  //   if (eigenvalues(6 + i) < -peigs(i)) {
  //     eigenvalues(6 + i) = -peigs(i);
  //   }
  // }

  Mat9 dPdF =
      eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

  real pk = p_smooth * vol[t] / 80. / mu;
  real pm = vol[t] * lambda_inv / 20.;

  const int32_t* e2off = &t2off[16 * t];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    Vec3 dxi1 = X[ctet[(i + 1) % 4]] - X[ctet[(i + 3) % 4]];
    Vec3 dxi2 = X[ctet[(i + 2) % 4]] - X[ctet[(i + 3) % 4]];
    Vec3 ni = Cross(dxi1, dxi2) / 6. * normal_sign[4 * t + i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      Vec3 dxj1 = X[ctet[(j + 1) % 4]] - X[ctet[(j + 3) % 4]];
      Vec3 dxj2 = X[ctet[(j + 2) % 4]] - X[ctet[(j + 3) % 4]];
      Vec3 nj = Cross(dxj1, dxj2) / 6. * normal_sign[4 * t + j];
      Mat4 tH = Mat4::Zero();
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          tH.block<3, 3>(0, 0) +=
              idm(i, a) * dPdF.block<3, 3>(3 * a, 3 * b) * idm(j, b) * vol[t];
        }
      }
      tH.block<1, 3>(3, 0) += p_scale * nj.transpose() / 4. / Determinant(F);
      tH.block<3, 1>(0, 3) += p_scale * ni / 4. / Determinant(F);
      // tH.block<1, 3>(3, 0) += p_scale * nj.transpose() / 4.;
      // tH.block<3, 1>(0, 3) += p_scale * ni / 4.;
      if (i == j)
        tH(3, 3) -= (3 * pk + 2 * pm) * p_scale * p_scale;
      else
        tH(3, 3) -= (pm - pk) * p_scale * p_scale;

      real* bval = &bcoo_val[16 * e2off[4 * i + j]];
#pragma unroll
      for (int a = 0; a < 4; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          ::atomicAdd(&bval[4 * a + b], tH(a, b));
        }
      }
    }
  }
}

__global__ void HessianMixedNeoHookean(
    const Vec3* X, const real* P, const uint32_t* tet, const Mat3* Dm_inv,
    const real* vol, const int32_t* t2off, real* bcoo_val,
    const int32_t* normal_sign, const real mu, const real lambda_inv,
    const real p_smooth, const real p_scale, const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  const uint32_t* ctet = &tet[4 * t];
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

  Mat3 ahat, bhat, chat;
  ahat << 0., -F(2, 0), F(1, 0), F(2, 0), 0., -F(0, 0), -F(1, 0), F(0, 0), 0.;
  bhat << 0., -F(2, 1), F(1, 1), F(2, 1), 0., -F(0, 1), -F(1, 1), F(0, 1), 0.;
  chat << 0., -F(2, 2), F(1, 2), F(2, 2), 0., -F(0, 2), -F(1, 2), F(0, 2), 0.;
  Mat9 dPdF = Mat9::Zero();
  real p_ave = (P[v1] + P[v2] + P[v3] + P[v4]) / 4.;
  real fac = mu - p_ave;
  dPdF.block<3, 3>(0, 3) = fac * chat;
  dPdF.block<3, 3>(0, 6) = -fac * bhat;
  dPdF.block<3, 3>(3, 0) = -fac * chat;
  dPdF.block<3, 3>(3, 6) = fac * ahat;
  dPdF.block<3, 3>(6, 0) = fac * bhat;
  dPdF.block<3, 3>(6, 3) = -fac * ahat;
  dPdF += mu * Mat9::Identity();

  real pk = p_smooth * vol[t] / 80. / mu;
  real pm = vol[t] * lambda_inv / 20.;

  const int32_t* e2off = &t2off[16 * t];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    Vec3 dxi1 = X[ctet[(i + 1) % 4]] - X[ctet[(i + 3) % 4]];
    Vec3 dxi2 = X[ctet[(i + 2) % 4]] - X[ctet[(i + 3) % 4]];
    Vec3 ni = Cross(dxi1, dxi2) / 6. * normal_sign[4 * t + i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      Vec3 dxj1 = X[ctet[(j + 1) % 4]] - X[ctet[(j + 3) % 4]];
      Vec3 dxj2 = X[ctet[(j + 2) % 4]] - X[ctet[(j + 3) % 4]];
      Vec3 nj = Cross(dxj1, dxj2) / 6. * normal_sign[4 * t + j];
      Mat4 tH = Mat4::Zero();
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          tH.block<3, 3>(0, 0) +=
              idm(i, a) * dPdF.block<3, 3>(3 * a, 3 * b) * idm(j, b) * vol[t];
        }
      }
      tH.block<1, 3>(3, 0) += p_scale * nj.transpose() / 4.;
      tH.block<3, 1>(0, 3) += p_scale * ni / 4.;
      if (i == j)
        tH(3, 3) -= (3 * pk + 2 * pm) * p_scale * p_scale;
      else
        tH(3, 3) -= (pm - pk) * p_scale * p_scale;

      real* bval = &bcoo_val[16 * e2off[4 * i + j]];
#pragma unroll
      for (int a = 0; a < 4; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          ::atomicAdd(&bval[4 * a + b], tH(a, b));
        }
      }
    }
  }
}

__global__ void InertiaHessianMixed(const real* M, const int32_t* d2off,
                                    real* bcoo_val, const real dt_inv,
                                    const uint32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  real* bval = &bcoo_val[16 * d2off[v]];
  real diag = M[v] * dt_inv * dt_inv;
  bval[0] += diag;
  bval[5] += diag;
  bval[10] += diag;
}

__global__ void AfKroneckerXXtHalfMixed(const real* Af, const real* XXt,
                                        const int32_t* half_off, real* Af_XXt,
                                        const uint32_t n_half) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= 256 * n_half) return;
  int32_t j = idx % 16;
  int32_t i = idx / 16;
  int32_t b = i / 16;
  i = i % 16;
  int32_t r = (i / 4) * 4 + (j / 4);
  int32_t c = (i % 4) * 4 + (j % 4);
  int32_t a = half_off[b];
  Af_XXt[256 * b + 16 * r + c] = Af[16 * a + i] * XXt[16 * a + j];
}

__global__ void AfDenseReductionMixed(const real* Af_XXt,
                                      const int32_t* red_off,
                                      const int32_t* half_off, real* UtAU_dense,
                                      const int32_t n_half,
                                      const int32_t n_handle) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 256;
  if (a >= n_half) return;
  int32_t b = red_off[half_off[a]];
  int32_t r = b / n_handle;
  int32_t c = b % n_handle;
  int32_t i = idx % 256;
  ::atomicAdd(
      &UtAU_dense[(16 * c + (i % 16)) * 16 * n_handle + (16 * r + (i / 16))],
      Af_XXt[idx]);
}

__global__ void AfDenseReductionShurMixed(
    const real* Af_XXt, const int32_t* red_off, const int32_t* half_off,
    real* UtAU_dense, const int32_t n_half, const int32_t n_handle) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 256;
  if (a >= n_half) return;
  int32_t b = red_off[half_off[a]];
  int32_t r = b / n_handle;
  int32_t c = b % n_handle;
  int32_t i = idx % 256;
  int32_t br = i / 16;
  int32_t bc = i % 16;
  int32_t x = 12 * r + br;
  int32_t y = 12 * c + bc;
  if (br < 12) {
    if (bc >= 12) {  // G^T
      y = 12 * n_handle + 4 * c + bc - 12;
    }
  } else {
    if (bc < 12) {  // G
      x = 12 * n_handle + 4 * r + br - 12;
    } else {  // -C
      x = 12 * n_handle + 4 * r + br - 12;
      y = 12 * n_handle + 4 * c + bc - 12;
    }
  }
  ::atomicAdd(&UtAU_dense[x + 16 * n_handle * y], Af_XXt[idx]);
}

__global__ void AsDenseMirrorMixed(real* A_dense, const int32_t* low_off,
                                   const int32_t n_handle,
                                   const int32_t low_nnz) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 256;
  if (a >= low_nnz) return;
  int32_t i = idx % 256;
  int32_t b = low_off[a];
  int32_t r = 16 * (b / n_handle) + i / 16;
  int32_t c = 16 * (b % n_handle) + i % 16;
  A_dense[r * 16 * n_handle + c] = A_dense[c * 16 * n_handle + r];
}

__global__ void AsDenseMirrorShurMixed(real* A_dense, const int32_t* low_off,
                                       const int32_t n_handle,
                                       const int32_t low_nnz) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 256;
  if (a >= low_nnz) return;
  int32_t i = idx % 256;
  int32_t b = low_off[a];
  int32_t br = i / 16;
  int32_t bc = i % 16;
  int32_t r = b / n_handle;
  int32_t c = b % n_handle;
  int32_t x = 12 * r + br;
  int32_t y = 12 * c + bc;
  if (br < 12) {
    if (bc >= 12) {  // G^T
      y = 12 * n_handle + 4 * c + bc - 12;
    }
  } else {
    if (bc < 12) {  // G
      x = 12 * n_handle + 4 * r + br - 12;
    } else {  // -C
      x = 12 * n_handle + 4 * r + br - 12;
      y = 12 * n_handle + 4 * c + bc - 12;
    }
  }
  A_dense[x * 16 * n_handle + y] = A_dense[y * 16 * n_handle + x];
}

__global__ void AfSparseReductionMixed(const real* Af_XXt,
                                       const int32_t* red_off,
                                       const int32_t* half_off,
                                       real* UtAU_sparse,
                                       const int32_t n_half) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 256;
  if (a >= n_half) return;
  int32_t b = red_off[half_off[a]];
  int32_t i = idx % 256;
  ::atomicAdd(&UtAU_sparse[256 * b + i], Af_XXt[idx]);
}

__global__ void AsSparseMirrorMixed(real* A_sparse, const int32_t* low_off,
                                    const int32_t* mirror_off,
                                    const int32_t low_nnz) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 256;
  if (a >= low_nnz) return;
  int32_t i = idx % 256;
  int32_t j = (i % 16) * 16 + (i / 16);
  A_sparse[256 * mirror_off[a] + j] = A_sparse[256 * low_off[a] + i];
}

__global__ void AsSparseReductionMixed(const real* A_upper,
                                       const int32_t* red_off, real* A,
                                       const int32_t upper_nnz) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 256;
  if (a >= upper_nnz) return;
  int32_t i = idx % 256;
  ::atomicAdd(&A[red_off[a] * 256 + i], A_upper[idx]);
}

__global__ void AsDenseReductionMixed(const real* A_upper,
                                      const int32_t* red_off, real* A,
                                      const int32_t n_handle,
                                      const int32_t upper_nnz) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 256;
  if (a >= upper_nnz) return;
  int32_t i = idx % 256;
  int32_t b = red_off[a];
  int32_t r = (b / n_handle) * 16 + (i / 16);
  int32_t c = (b % n_handle) * 16 + (i % 16);
  ::atomicAdd(&A[16 * n_handle * c + r], A_upper[idx]);
}

__global__ void AsDenseReductionShurMixed(const real* A_upper,
                                          const int32_t* red_off, real* A,
                                          const int32_t n_handle,
                                          const int32_t upper_nnz) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t a = idx / 256;
  if (a >= upper_nnz) return;
  int32_t i = idx % 256;
  int32_t b = red_off[a];
  int32_t br = i / 16;
  int32_t bc = i % 16;
  int32_t r = b / n_handle;
  int32_t c = b % n_handle;
  int32_t x = 12 * r + br;
  int32_t y = 12 * c + bc;
  if (br < 12) {
    if (bc >= 12) {
      y = 12 * n_handle + 4 * c + bc - 12;
    }
  } else {
    if (bc < 12) {
      x = 12 * n_handle + 4 * r + br - 12;
    } else {
      x = 12 * n_handle + 4 * r + br - 12;
      y = 12 * n_handle + 4 * c + bc - 12;
    }
  }
  ::atomicAdd(&A[16 * n_handle * y + x], A_upper[idx]);
}

__global__ void TetGradientMixedNeoHookean(
    const Vec3* X, const uint32_t* tet, const real* pressure,
    const int32_t* normal_sign, const Mat3* Dm_inv, const real* vol, real* grad,
    real* p_grad, const real mu, const real lam_inv, const real p_smooth,
    const real p_scale, const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t* ctet = &tet[4 * t];
  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  Mat3 Ds;
  Ds.col(0) = X[v1] - X[v4];
  Ds.col(1) = X[v2] - X[v4];
  Ds.col(2) = X[v3] - X[v4];
  real cvol = Determinant(Ds) / 6.;
  Mat3 F = Ds * Dm_inv[t];
  real fac = -mu;
  Mat3 P;
  P.col(0) = fac * Cross(F.col(1), F.col(2));
  P.col(1) = fac * Cross(F.col(2), F.col(0));
  P.col(2) = fac * Cross(F.col(0), F.col(1));
  P += mu * F;
  Eigen::Matrix<real, 4, 3> G;
  G.block<3, 3>(0, 0) = Dm_inv[t];
  G.block<1, 3>(3, 0) = -Vec3::Ones().transpose() * Dm_inv[t];
  Eigen::Matrix<real, 3, 4> T = -vol[t] * P * G.transpose();
  // -G^Tp
  real p_ave = (pressure[v1] + pressure[v2] + pressure[v3] + pressure[v4]) / 4.;
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    Vec3 dx1 = X[ctet[(a + 1) % 4]] - X[ctet[(a + 3) % 4]];
    Vec3 dx2 = X[ctet[(a + 2) % 4]] - X[ctet[(a + 3) % 4]];
    Vec3 n = Cross(dx1, dx2) / 6. * normal_sign[4 * t + a];
    T.col(a) -= n * p_ave;
  }
#pragma unroll
  for (int i = 0; i < 12; ++i) grad[12 * t + i] = T(i % 3, i / 3);
  // Cp - phi
  real k = p_smooth * vol[t] / 80. / mu;
  real m = vol[t] * lam_inv / 20.;
  p_grad[4 * t + 0] +=
      k * (3 * pressure[v1] - pressure[v2] - pressure[v3] - pressure[v4]) +
      m * (2 * pressure[v1] + pressure[v2] + pressure[v3] + pressure[v4]);
  p_grad[4 * t + 1] +=
      k * (-pressure[v1] + 3 * pressure[v2] - pressure[v3] - pressure[v4]) +
      m * (pressure[v1] + 2 * pressure[v2] + pressure[v3] + pressure[v4]);
  p_grad[4 * t + 2] +=
      k * (-pressure[v1] - pressure[v2] + 3 * pressure[v3] - pressure[v4]) +
      m * (pressure[v1] + pressure[v2] + 2 * pressure[v3] + pressure[v4]);
  p_grad[4 * t + 3] +=
      k * (-pressure[v1] - pressure[v2] - pressure[v3] + 3 * pressure[v4]) +
      m * (pressure[v1] + pressure[v2] + pressure[v3] + 2 * pressure[v4]);
  real phi = cvol - vol[t];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p_grad[4 * t + i] -= phi / 4.;
    p_grad[4 * t + i] *= p_scale;
  }
}

__global__ void TetGradientMixedNeoHookeanLog(
    const Vec3* X, const uint32_t* tet, const real* pressure,
    const int32_t* normal_sign, const Mat3* Dm_inv, const real* vol, real* grad,
    real* p_grad, const real mu, const real lam_inv, const real p_smooth,
    const real p_scale, const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t* ctet = &tet[4 * t];
  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  Mat3 Ds;
  Ds.col(0) = X[v1] - X[v4];
  Ds.col(1) = X[v2] - X[v4];
  Ds.col(2) = X[v3] - X[v4];
  real cvol = Determinant(Ds) / 6.;
  Mat3 F = Ds * Dm_inv[t];
  real fac = -mu;
  Mat3 P;
  P.col(0) = fac * Cross(F.col(1), F.col(2));
  P.col(1) = fac * Cross(F.col(2), F.col(0));
  P.col(2) = fac * Cross(F.col(0), F.col(1));
  P += mu * F;
  Eigen::Matrix<real, 4, 3> G;
  G.block<3, 3>(0, 0) = Dm_inv[t];
  G.block<1, 3>(3, 0) = -Vec3::Ones().transpose() * Dm_inv[t];
  Eigen::Matrix<real, 3, 4> T = -vol[t] * P * G.transpose();
  // -G^Tp
  real p_ave = (pressure[v1] + pressure[v2] + pressure[v3] + pressure[v4]) / 4.;
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    Vec3 dx1 = X[ctet[(a + 1) % 4]] - X[ctet[(a + 3) % 4]];
    Vec3 dx2 = X[ctet[(a + 2) % 4]] - X[ctet[(a + 3) % 4]];
    Vec3 n = Cross(dx1, dx2) / 6. * normal_sign[4 * t + a];
    T.col(a) -= n * p_ave / Determinant(F);
    // T.col(a) -= n * p_ave;
  }
#pragma unroll
  for (int i = 0; i < 12; ++i) grad[12 * t + i] = T(i % 3, i / 3);
  // Cp - phi
  real k = p_smooth * vol[t] / 80. / mu;
  real m = vol[t] * lam_inv / 20.;
  p_grad[4 * t + 0] +=
      k * (3 * pressure[v1] - pressure[v2] - pressure[v3] - pressure[v4]) +
      m * (2 * pressure[v1] + pressure[v2] + pressure[v3] + pressure[v4]);
  p_grad[4 * t + 1] +=
      k * (-pressure[v1] + 3 * pressure[v2] - pressure[v3] - pressure[v4]) +
      m * (pressure[v1] + 2 * pressure[v2] + pressure[v3] + pressure[v4]);
  p_grad[4 * t + 2] +=
      k * (-pressure[v1] - pressure[v2] + 3 * pressure[v3] - pressure[v4]) +
      m * (pressure[v1] + pressure[v2] + 2 * pressure[v3] + pressure[v4]);
  p_grad[4 * t + 3] +=
      k * (-pressure[v1] - pressure[v2] - pressure[v3] + 3 * pressure[v4]) +
      m * (pressure[v1] + pressure[v2] + pressure[v3] + 2 * pressure[v4]);
  real phi = log(max(cvol / vol[t], 1e-4)) * vol[t];
  // real phi = cvol - vol[t];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p_grad[4 * t + i] -= phi / 4.;
    p_grad[4 * t + i] *= p_scale;
  }
}

__global__ void TetGradientMixedCorotation(
    const Vec3* X, const uint32_t* tet, const real* pressure,
    const int32_t* normal_sign, const Mat3* Dm_inv, const real* vol, real* grad,
    real* p_grad, const real mu, const real lam_inv, const real p_smooth,
    const real p_scale, const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t* ctet = &tet[4 * t];
  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
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
  {
    Mat3 L = Mat3::Identity();
    L(2, 2) = Determinant(U * V.transpose());
    real detU = Determinant(U);
    real detV = Determinant(V);
    if (detU < 0. && detV > 0.) U = U * L;
    if (detU > 0. && detV < 0.) V = V * L;
    S(2) = S(2) * L(2, 2);
  }

  Mat3 P = 2 * mu * (F - U * V.transpose());
  Eigen::Matrix<real, 4, 3> G;
  G.block<3, 3>(0, 0) = Dm_inv[t];
  G.block<1, 3>(3, 0) = -Vec3::Ones().transpose() * Dm_inv[t];
  Eigen::Matrix<real, 3, 4> T = -vol[t] * P * G.transpose();

  real p_ave = (pressure[v1] + pressure[v2] + pressure[v3] + pressure[v4]) / 4.;
  T -= p_ave * vol[t] * (U * V.transpose()) * G.transpose();

#pragma unroll
  for (int i = 0; i < 12; ++i) grad[12 * t + i] = T(i % 3, i / 3);

  // Cp - phi
  real k = p_smooth * vol[t] / 80. / mu;
  real m = vol[t] * lam_inv / 20.;
  p_grad[4 * t + 0] +=
      k * (3 * pressure[v1] - pressure[v2] - pressure[v3] - pressure[v4]) +
      m * (2 * pressure[v1] + pressure[v2] + pressure[v3] + pressure[v4]);
  p_grad[4 * t + 1] +=
      k * (-pressure[v1] + 3 * pressure[v2] - pressure[v3] - pressure[v4]) +
      m * (pressure[v1] + 2 * pressure[v2] + pressure[v3] + pressure[v4]);
  p_grad[4 * t + 2] +=
      k * (-pressure[v1] - pressure[v2] + 3 * pressure[v3] - pressure[v4]) +
      m * (pressure[v1] + pressure[v2] + 2 * pressure[v3] + pressure[v4]);
  p_grad[4 * t + 3] +=
      k * (-pressure[v1] - pressure[v2] - pressure[v3] + 3 * pressure[v4]) +
      m * (pressure[v1] + pressure[v2] + pressure[v3] + 2 * pressure[v4]);

  real phi = (S(0) + S(1) + S(2) - 3.) * vol[t] / 4.;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p_grad[4 * t + i] -= phi;
    p_grad[4 * t + i] *= p_scale;
  }
}

__global__ void TetGradientMixedStVK(const Vec3* X, const uint32_t* tet,
                                     const real* pressure,
                                     const int32_t* normal_sign,
                                     const Mat3* Dm_inv, const real* vol,
                                     real* grad, real* p_grad, const real mu,
                                     const real lam_inv, const real p_smooth,
                                     const real p_scale, const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t* ctet = &tet[4 * t];
  const uint32_t& v1 = tet[4 * t];
  const uint32_t& v2 = tet[4 * t + 1];
  const uint32_t& v3 = tet[4 * t + 2];
  const uint32_t& v4 = tet[4 * t + 3];
  Mat3 Ds;
  Ds.col(0) = X[v1] - X[v4];
  Ds.col(1) = X[v2] - X[v4];
  Ds.col(2) = X[v3] - X[v4];
  Mat3 F = Ds * Dm_inv[t];
  // if (Determinant(F) < 1e-8) {
  //   printf("Warning: Singular F detected in StVK gradient computation\n");
  // }
  Mat3 E = 0.5 * (F.transpose() * F - Mat3::Identity());
  Mat3 P = 2. * mu * F * E;
  Eigen::Matrix<real, 4, 3> G;
  G.block<3, 3>(0, 0) = Dm_inv[t];
  G.block<1, 3>(3, 0) = -Vec3::Ones().transpose() * Dm_inv[t];
  Eigen::Matrix<real, 3, 4> T = -vol[t] * P * G.transpose();

  // -G^Tp
  real p_ave = (pressure[v1] + pressure[v2] + pressure[v3] + pressure[v4]) / 4.;
  T -= p_ave * vol[t] * F * G.transpose();

#pragma unroll
  for (int i = 0; i < 12; ++i) grad[12 * t + i] = T(i % 3, i / 3);

  // Cp - phi
  real k = p_smooth * vol[t] / 80. / mu;
  real m = vol[t] * lam_inv / 20.;
  p_grad[4 * t + 0] +=
      k * (3 * pressure[v1] - pressure[v2] - pressure[v3] - pressure[v4]) +
      m * (2 * pressure[v1] + pressure[v2] + pressure[v3] + pressure[v4]);
  p_grad[4 * t + 1] +=
      k * (-pressure[v1] + 3 * pressure[v2] - pressure[v3] - pressure[v4]) +
      m * (pressure[v1] + 2 * pressure[v2] + pressure[v3] + pressure[v4]);
  p_grad[4 * t + 2] +=
      k * (-pressure[v1] - pressure[v2] + 3 * pressure[v3] - pressure[v4]) +
      m * (pressure[v1] + pressure[v2] + 2 * pressure[v3] + pressure[v4]);
  p_grad[4 * t + 3] +=
      k * (-pressure[v1] - pressure[v2] - pressure[v3] + 3 * pressure[v4]) +
      m * (pressure[v1] + pressure[v2] + pressure[v3] + 2 * pressure[v4]);

  real phi = E.trace() * vol[t] / 4.;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p_grad[4 * t + i] -= phi;
    p_grad[4 * t + i] *= p_scale;
  }
}

__global__ void MixedEnergyGradient(
    const real* tet_grad, const real* tet_p_grad, const int32_t* v2t_ids,
    const int32_t* v2t_off, const bool* fixed, const Vec3* fixed_X,
    const Vec3* X, const real* mass, real* grad, const real control_mag,
    const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  for (int32_t idx = v2t_off[v]; idx < v2t_off[v + 1]; ++idx) {
    grad[4 * v + 0] += tet_grad[v2t_ids[idx] * 3 + 0];
    grad[4 * v + 1] += tet_grad[v2t_ids[idx] * 3 + 1];
    grad[4 * v + 2] += tet_grad[v2t_ids[idx] * 3 + 2];
    grad[4 * v + 3] += tet_p_grad[v2t_ids[idx]];
  }
  if (fixed[v]) {
    grad[4 * v + 0] += control_mag * (fixed_X[v](0) - X[v](0));
    grad[4 * v + 1] += control_mag * (fixed_X[v](1) - X[v](1));
    grad[4 * v + 2] += control_mag * (fixed_X[v](2) - X[v](2));
  }
}

__global__ void MixedEnergyGradientWithCtrl(
    const real* tet_grad, const real* tet_p_grad, const int32_t* v2t_ids,
    const int32_t* v2t_off, const bool* fixed, const Vec3* fixed_X,
    const Vec3* X, real* grad, const real control_mag, int ctrl_vert,
    Vec3 ctrl_pos, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  for (int32_t idx = v2t_off[v]; idx < v2t_off[v + 1]; ++idx) {
    grad[4 * v + 0] += tet_grad[v2t_ids[idx] * 3 + 0];
    grad[4 * v + 1] += tet_grad[v2t_ids[idx] * 3 + 1];
    grad[4 * v + 2] += tet_grad[v2t_ids[idx] * 3 + 2];
    grad[4 * v + 3] += tet_p_grad[v2t_ids[idx]];
  }
  if (fixed[v]) {
    grad[4 * v + 0] += control_mag * (fixed_X[v](0) - X[v](0));
    grad[4 * v + 1] += control_mag * (fixed_X[v](1) - X[v](1));
    grad[4 * v + 2] += control_mag * (fixed_X[v](2) - X[v](2));
  } else if (v == ctrl_vert) {
    grad[4 * v + 0] += control_mag * (ctrl_pos(0) - X[v](0));
    grad[4 * v + 1] += control_mag * (ctrl_pos(1) - X[v](1));
    grad[4 * v + 2] += control_mag * (ctrl_pos(2) - X[v](2));
  }
}

__global__ void SkeletonGradientMixed(real* grad, const Vec3* X,
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
  grad[4 * v + 0] += ctr_mag * (gpos(0) - X[v](0));
  grad[4 * v + 1] += ctr_mag * (gpos(1) - X[v](1));
  grad[4 * v + 2] += ctr_mag * (gpos(2) - X[v](2));
}

__global__ void InertiaGradientMixed(const Vec3* inertia_X, const Vec3* X,
                                     const real* M, real* grad,
                                     const real dt_inv, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  real c = M[v] * dt_inv * dt_inv;
  grad[4 * v + 0] += c * (inertia_X[v](0) - X[v](0));
  grad[4 * v + 1] += c * (inertia_X[v](1) - X[v](1));
  grad[4 * v + 2] += c * (inertia_X[v](2) - X[v](2));
}

__global__ void UpdateAfDiagWithCtrlMixed(real* Af_diag_add, bool* fixed,
                                          const real control_mag,
                                          int32_t ctrl_vert, int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  if (!fixed[v] && (v != ctrl_vert)) return;
  real* diag = Af_diag_add + 16 * v;
  diag[0] += control_mag;
  diag[5] += control_mag;
  diag[10] += control_mag;
}

__global__ void UpdateAfDiagMixed(real* Af_diag_add, bool* fixed,
                                  const real control_mag, int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  if (!fixed[v]) return;
  real* diag = Af_diag_add + 16 * v;
  diag[0] += control_mag;
  diag[5] += control_mag;
  diag[10] += control_mag;
}

__global__ void UpdateAfDiagSkeletonMixed(real* Af_diag_add,
                                          const int32_t* bone_id,
                                          const real ctr_mag,
                                          const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  int32_t b = bone_id[v];
  if (b < 0) return;
  real* diag = Af_diag_add + 16 * v;
  diag[0] += ctr_mag;
  diag[5] += ctr_mag;
  diag[10] += ctr_mag;
}

__global__ void UpdateUtAUDiagMixed(real* UtAU_diag_add,
                                    const real* Af_diag_add,
                                    const real* diag_XXt,
                                    const int32_t* update_off, int32_t n_vert) {
  // matrix multiplying in thread block
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t v = idx / 16;
  if (v >= n_vert) return;
  int32_t p = idx % 16;
  int32_t px = p / 4, py = p % 4;
#pragma unroll
  for (int t = 0; t < 16; ++t) {
    int32_t tx = t / 4, ty = t % 4;
    ::atomicAdd(
        &UtAU_diag_add[256 * update_off[v] + (px * 4 + tx) * 16 + py * 4 + ty],
        Af_diag_add[idx] * diag_XXt[16 * v + t]);
  }
}

__global__ void UpdateUltAUlDiagMixed(real* diag_add,
                                      const real* diag_add_finer,
                                      const int32_t* update_off,
                                      int32_t n_handle) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_handle) return;
  int32_t off = threadIdx.y;
  ::atomicAdd(&diag_add[256 * update_off[i] + off],
              diag_add_finer[256 * i + off]);
}

__global__ void UpdateDenseDiagMixed(real* den_val, const real* diag_add,
                                     int32_t n_handle, int32_t rows) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_handle) return;
  int32_t off = threadIdx.y;
  int32_t dx = off / 16, dy = off % 16;
  ::atomicAdd(&den_val[(16 * i + dx) * rows + (16 * i + dy)],
              diag_add[256 * i + off]);
}

__global__ void UpdateDenseDiagShurMixed(real* den_val, const real* diag_add,
                                         int32_t n_handle, int32_t rows) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_handle) return;
  int32_t off = threadIdx.y;
  int32_t dx = off / 16, dy = off % 16;
  int32_t x = 12 * i + dx, y = 12 * i + dy;
  if (dx < 12) {
    if (dy > 12) {
      y = 12 * n_handle + 4 * i + dy - 12;
    }
  } else {
    if (dy < 12) {
      x = 12 * n_handle + 4 * i + dx - 12;
    } else {
      x = 12 * n_handle + 4 * i + dx - 12;
      y = 12 * n_handle + 4 * i + dy - 12;
    }
  }
  ::atomicAdd(&den_val[x + y * rows], diag_add[256 * i + off]);
}

__global__ void CoarseAOS2SOAMixed(real* P, const real* B, const int32_t dim) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= dim) return;
  int32_t n = dim / 16;
  int32_t i = idx / 16;
  int32_t j = idx % 16;
  if (j < 12) {
    P[12 * i + j] = B[idx];
  } else {
    P[12 * n + 4 * i + j - 12] = B[idx];
  }
}

__global__ void CoarseSOA2AOSMixed(real* P, const real* B, const int32_t dim) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= dim) return;
  int32_t n = dim / 16;
  int32_t offset = 12 * n;
  if (idx < offset) {
    int32_t i = idx / 12;
    int32_t j = idx % 12;
    P[16 * i + j] = B[idx];
  } else {
    int32_t i = (idx - offset) / 4;
    int32_t j = (idx - offset) % 4;
    P[16 * i + j + 12] = B[idx];
  }
}

__global__ void GSAfMixed(real* X, const real* Af_diag, const real* Af_diag_add,
                          const real* b, const int32_t* colors, const int c,
                          const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  if (colors[i] != c) return;
  Solve4x4Sym(Af_diag + 16 * i, Af_diag_add + 16 * i, b + 4 * i, X + 4 * i);
}

__global__ void GSAfIncMixed(real* X, const real* Af_diag,
                             const real* Af_diag_add, const real* bcsr_val,
                             const int32_t* bcsr_row, const int32_t* bcsr_col,
                             const real* b, const int32_t* colors, const int c,
                             const real relax, const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  if (colors[i] != c) return;
  // compute rhs
  real tmp[4];
  tmp[0] = b[4 * i + 0];
  tmp[1] = b[4 * i + 1];
  tmp[2] = b[4 * i + 2];
  tmp[3] = b[4 * i + 3];
  for (int32_t idx = bcsr_row[i]; idx < bcsr_row[i + 1]; ++idx) {
    int32_t j = bcsr_col[idx];
    if (colors[j] >= c) continue;
    const real* Aj = &bcsr_val[16 * idx];
    for (int x = 0; x < 4; ++x) {
      for (int y = 0; y < 4; ++y) {
        tmp[x] -= Aj[4 * x + y] * X[4 * j + y];
      }
    }
  }
  tmp[0] *= relax;
  tmp[1] *= relax;
  tmp[2] *= relax;
  tmp[3] *= relax;
  Solve4x4Sym(Af_diag + 16 * i, Af_diag_add + 16 * i, tmp, X + 4 * i);
}

__global__ void CellVankaJacobiMixed(real* X, const real* Af_diag,
                                     const real* Af_diag_add,
                                     const real* bcsr_val,
                                     const int32_t* bcsr_row,
                                     const int32_t* bcsr_col, const real* B,
                                     const uint32_t* tet, const real relax) {
  int t = blockIdx.x;
  int off = threadIdx.y;
  __shared__ real A[256], x[16], b[16];
  x[off] = 0;
  for (int i = 0; i < 16; ++i) A[16 * off + i] = 0;
  // construct matrix
  int v1 = tet[4 * t + off / 4];
  for (int i = 0; i < 4; ++i) {
    int v2 = tet[4 * t + i];
    int idx = bcsr_row[v1];
    while (bcsr_col[idx] != v2) ++idx;
    const real* Aj = &bcsr_val[16 * idx];
    for (int j = 0; j < 4; ++j) {
      int col = 4 * i + j;
      A[16 * off + col] = Aj[4 * (off % 4) + j];
    }
  }
  // add diag
  for (int i = 0; i < 4; ++i) {
    const real* add = &Af_diag_add[16 * v1];
    int col = 4 * (off / 4) + i;
    A[16 * off + col] += add[4 * (off % 4) + i];
  }
  // construct rhs
  b[off] = B[4 * v1 + off % 4];
  b[off] *= relax;
  real tol = 1e-7;
  real shift = 0.;

  __shared__ real r1[16], r2[16], y[16], w[16], v[16], w2[16];
  __shared__ volatile real tmp[24];
  __shared__ real alfa, beta, beta1, oldb, dbar, oldeps, epsln, gbar, phi,
      phibar, delta, gamma, cs, sn;
  __shared__ bool flag;

  y[off] = b[off];
  // preconditioner
  if (off < 4) {
    Solve3x3Sym(A + 68 * off, b + 4 * off, 16, y + 4 * off);
    y[4 * off + 3] = -b[4 * off + 3] / A[16 * (4 * off + 3) + 4 * off + 3];
  }
  __syncthreads();
  w[off] = 0.;
  w2[off] = 0.;
  r1[off] = b[off];
  r2[off] = b[off];

  tmp[off] = y[off] * b[off];
  tmp[off] += tmp[off + 8];
  tmp[off] += tmp[off + 4];
  tmp[off] += tmp[off + 2];
  tmp[off] += tmp[off + 1];
  if (off == 0) {
    beta1 = sqrt(tmp[0]);
    oldb = 0.;
    beta = beta1;
    dbar = 0.;
    epsln = 0.;
    phibar = beta1;
    cs = -1.;
    sn = 0.;
  }
  __syncthreads();

  if (beta < tol) {
    return;
  }

  for (int iter = 0; iter < 16; ++iter) {
    v[off] = y[off] / beta;
#pragma unroll
    for (int k = 0; k < 16; k++) y[off] += A[16 * off + k] * v[k];
    y[off] -= shift * v[off];
    if (iter > 0) y[off] -= beta / oldb * r1[off];
    tmp[off] = v[off] * y[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      alfa = tmp[0];
      oldb = beta;
    }
    __syncthreads();
    y[off] -= alfa / beta * r2[off];
    r1[off] = r2[off];
    r2[off] = y[off];
    // preconditioner
    if (off < 4) {
      Solve3x3Sym(A + 68 * off, r2 + 4 * off, 16, y + 4 * off);
      y[4 * off + 3] = -r2[4 * off + 3] / A[16 * (4 * off + 3) + 4 * off + 3];
    }
    __syncthreads();
    tmp[off] = r2[off] * y[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      beta = sqrt(tmp[0]);
      oldeps = epsln;
      delta = cs * dbar + sn * alfa;
      gbar = sn * dbar - cs * alfa;
      epsln = sn * beta;
      dbar = -cs * beta;
      gamma = sqrt(gbar * gbar + beta * beta);
      gamma = max(gamma, 1e-10);
      cs = gbar / gamma;
      sn = beta / gamma;
      phi = cs * phibar;
      phibar = sn * phibar;
    }
    __syncthreads();
    real w1 = w2[off];
    w2[off] = w[off];
    w[off] = (v[off] - oldeps * w1 - delta * w2[off]) / gamma;
    x[off] += phi * w[off];
    // if (abs(phibar) / beta1 < tol) {
    //   X[4 * v1 + off % 4] += x[off];
    //   return;
    // }
  }
  ::atomicAdd(&X[4 * v1 + off % 4], x[off]);
}

__global__ void CellVankaAfMixed(real* X, const real* Af_diag,
                                 const real* Af_diag_add, const real* bcsr_val,
                                 const int32_t* bcsr_row,
                                 const int32_t* bcsr_col, const real* B,
                                 const uint32_t* tet, const uint32_t* colors,
                                 const int c, const real relax,
                                 int32_t* updated) {
  int t = blockIdx.x;
  if (colors[t] != c) return;
  int off = threadIdx.y;
  __shared__ real A[256], x[16], b[16];
  x[off] = 0;
  for (int i = 0; i < 16; ++i) A[16 * off + i] = 0;
  // construct matrix
  int v1 = tet[4 * t + off / 4];
  for (int i = 0; i < 4; ++i) {
    int v2 = tet[4 * t + i];
    int idx = bcsr_row[v1];
    while (bcsr_col[idx] != v2) ++idx;
    const real* Aj = &bcsr_val[16 * idx];
    for (int j = 0; j < 4; ++j) {
      int col = 4 * i + j;
      A[16 * off + col] = Aj[4 * (off % 4) + j];
    }
  }
  // for (int i = 0; i < 4; ++i) {
  //   const real* diag = &Af_diag[16 * v1];
  //   int col = 4 * (off / 4) + i;
  //   A[16 * off + col] += diag[4 * (off % 4) + i];
  // }
  // add diag
  for (int i = 0; i < 4; ++i) {
    const real* add = &Af_diag_add[16 * v1];
    int col = 4 * (off / 4) + i;
    A[16 * off + col] += add[4 * (off % 4) + i];
  }
  // construct rhs
  b[off] = B[4 * v1 + off % 4];
  for (int32_t idx = bcsr_row[v1]; idx < bcsr_row[v1 + 1]; ++idx) {
    int32_t v2 = bcsr_col[idx];
    // if (!updated[v2]) continue;
    const real* Aj = &bcsr_val[16 * idx];
    for (int i = 0; i < 4; ++i) {
      b[off] -= Aj[4 * (off % 4) + i] * X[4 * v2 + i];
    }
  }
  for (int i = 0; i < 4; ++i) {
    const real* add = &Af_diag_add[16 * v1];
    b[off] -= add[4 * (off % 4) + i] * X[4 * v1 + i];
  }
  b[off] *= relax;

  // if (off < 4) {
  //   Solve4x4Sym(A + 68 * off, b + 4 * off, 16, x + 4 * off);
  // }
  // __syncthreads();
  // X[4 * v1 + off % 4] += x[off];
  // solve
  real tol = 1e-7;
  real shift = 0.;

  __shared__ real r1[16], r2[16], y[16], w[16], v[16], w2[16];
  __shared__ volatile real tmp[24];
  __shared__ real alfa, beta, beta1, oldb, dbar, oldeps, epsln, gbar, phi,
      phibar, delta, gamma, cs, sn;
  __shared__ bool flag;

  y[off] = b[off];
  // preconditioner
  if (off < 4) {
    Solve3x3Sym(A + 68 * off, b + 4 * off, 16, y + 4 * off);
    y[4 * off + 3] = -b[4 * off + 3] / A[16 * (4 * off + 3) + 4 * off + 3];
  }
  __syncthreads();
  w[off] = 0.;
  w2[off] = 0.;
  r1[off] = b[off];
  r2[off] = b[off];

  tmp[off] = y[off] * b[off];
  tmp[off] += tmp[off + 8];
  tmp[off] += tmp[off + 4];
  tmp[off] += tmp[off + 2];
  tmp[off] += tmp[off + 1];
  if (off == 0) {
    beta1 = sqrt(tmp[0]);
    oldb = 0.;
    beta = beta1;
    dbar = 0.;
    epsln = 0.;
    phibar = beta1;
    cs = -1.;
    sn = 0.;
  }
  __syncthreads();

  if (beta < tol) {
    return;
  }

  for (int iter = 0; iter < 16; ++iter) {
    v[off] = y[off] / beta;
#pragma unroll
    for (int k = 0; k < 16; k++) y[off] += A[16 * off + k] * v[k];
    y[off] -= shift * v[off];
    if (iter > 0) y[off] -= beta / oldb * r1[off];
    tmp[off] = v[off] * y[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      alfa = tmp[0];
      oldb = beta;
    }
    __syncthreads();
    y[off] -= alfa / beta * r2[off];
    r1[off] = r2[off];
    r2[off] = y[off];
    // preconditioner
    if (off < 4) {
      Solve3x3Sym(A + 68 * off, r2 + 4 * off, 16, y + 4 * off);
      y[4 * off + 3] = -r2[4 * off + 3] / A[16 * (4 * off + 3) + 4 * off + 3];
    }
    __syncthreads();
    tmp[off] = r2[off] * y[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      beta = sqrt(tmp[0]);
      oldeps = epsln;
      delta = cs * dbar + sn * alfa;
      gbar = sn * dbar - cs * alfa;
      epsln = sn * beta;
      dbar = -cs * beta;
      gamma = sqrt(gbar * gbar + beta * beta);
      gamma = max(gamma, 1e-10);
      cs = gbar / gamma;
      sn = beta / gamma;
      phi = cs * phibar;
      phibar = sn * phibar;
    }
    __syncthreads();
    real w1 = w2[off];
    w2[off] = w[off];
    w[off] = (v[off] - oldeps * w1 - delta * w2[off]) / gamma;
    x[off] += phi * w[off];
    // if (abs(phibar) / beta1 < tol) {
    //   X[4 * v1 + off % 4] += x[off];
    //   return;
    // }
  }
  X[4 * v1 + off % 4] += x[off];
}

__global__ void GSAfDecMixed(real* X, const real* Af_diag,
                             const real* Af_diag_add, const real* bcsr_val,
                             const int32_t* bcsr_row, const int32_t* bcsr_col,
                             const real* b, const int32_t* colors, const int c,
                             const real relax, const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  if (colors[i] != c) return;
  // compute rhs
  real tmp[4];
  tmp[0] = b[4 * i + 0];
  tmp[1] = b[4 * i + 1];
  tmp[2] = b[4 * i + 2];
  tmp[3] = b[4 * i + 3];
  for (int32_t idx = bcsr_row[i]; idx < bcsr_row[i + 1]; ++idx) {
    int32_t j = bcsr_col[idx];
    if (colors[j] <= c) continue;
    const real* Aj = &bcsr_val[16 * idx];
    for (int x = 0; x < 4; ++x) {
      for (int y = 0; y < 4; ++y) {
        tmp[x] -= Aj[4 * x + y] * X[4 * j + y];
      }
    }
  }
  tmp[0] *= relax;
  tmp[1] *= relax;
  tmp[2] *= relax;
  tmp[3] *= relax;
  Solve4x4Sym(Af_diag + 16 * i, Af_diag_add + 16 * i, tmp, X + 4 * i);
}

__global__ void GSAsMixedCG(real* X, const real* diag, const real* diag_add,
                            const real* b, const int32_t* colors, const int c) {
  if (colors[blockIdx.x] != c) return;
  real tol = 1e-10;
  int t = blockIdx.x;
  int off = threadIdx.x;
  const real* A = diag + 256 * t;
  const real* A_add = diag_add + 256 * t;
  real* x = X + 16 * t;
  __shared__ real r[16], p[16], Ap[16];
  __shared__ volatile real tmp[24];
  __shared__ real dot, alpha, beta, r_norm, old_r_norm;
  __shared__ bool flag;
  r[off] = b[16 * t + off];
  p[off] = r[off];
  tmp[off] = r[off] * r[off];
  tmp[off] += tmp[off + 8];
  tmp[off] += tmp[off + 4];
  tmp[off] += tmp[off + 2];
  tmp[off] += tmp[off + 1];
  if (off == 0) {
    r_norm = tmp[0];
    flag = false;
    if (r_norm < tol) flag = true;
  }
  __syncthreads();
  if (flag) return;

  for (int l = 0; l < 16; ++l) {
#pragma unroll
    for (int k = 0; k < 16; k++)
      Ap[off] += (A[16 * off + k] + A_add[16 * off + k]) * p[k];
    tmp[off] = p[off] * Ap[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      dot = tmp[0];
      alpha = r_norm / dot;
      if (dot < tol) flag = true;
    }
    __syncthreads();
    if (flag) return;
    x[off] += alpha * p[off];
    r[off] -= alpha * Ap[off];
    tmp[off] = r[off] * r[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      old_r_norm = r_norm;
      r_norm = tmp[0];
      beta = r_norm / old_r_norm;
      if (r_norm < tol) flag = true;
    }
    __syncthreads();
    if (flag) return;
    p[off] = r[off] + beta * p[off];
  }
}

__global__ void GSAsMixed(real* X, const real* diag, const real* diag_add,
                          const real* B, const int32_t* colors, const int c) {
  if (colors[blockIdx.x] != c) return;
  real tol = 1e-7;
  real shift = 0.;
  int t = blockIdx.x;
  int off = threadIdx.x;
  const real* A = diag + 256 * t;
  const real* A_add = diag_add + 256 * t;
  real* x = X + 16 * t;
  const real* b = B + 16 * t;
  // if (t == 72 && off == 0) {
  //   printf("A: %f %f %f %f\n", A[0], A_add[0], A[255], A_add[255]);
  // }
  __shared__ real r1[16], r2[16], y[16], w[16], v[16], w2[16];
  __shared__ volatile real tmp[24];
  __shared__ real alfa, beta, beta1, oldb, dbar, oldeps, epsln, gbar, phi,
      phibar, delta, gamma, cs, sn;
  __shared__ bool flag;

  y[off] = b[off];
  // preconditioner
  if (off < 4) {
    Solve4x4Sym(A + 68 * off, A_add + 68 * off, b + 4 * off, 16, y + 4 * off);
  }
  __syncthreads();
  if (off >= 12) y[off] = -y[off];
  __syncthreads();
  w[off] = 0.;
  w2[off] = 0.;
  r1[off] = b[off];
  r2[off] = b[off];

  tmp[off] = y[off] * b[off];
  tmp[off] += tmp[off + 8];
  tmp[off] += tmp[off + 4];
  tmp[off] += tmp[off + 2];
  tmp[off] += tmp[off + 1];
  if (off == 0) {
    beta1 = sqrt(tmp[0]);
    oldb = 0.;
    beta = beta1;
    dbar = 0.;
    epsln = 0.;
    phibar = beta1;
    cs = -1.;
    sn = 0.;
  }
  __syncthreads();

  if (beta < tol) {
    return;
  }

  for (int iter = 0; iter < 16; ++iter) {
    v[off] = y[off] / beta;
#pragma unroll
    for (int k = 0; k < 16; k++)
      y[off] += (A[16 * off + k] + A_add[16 * off + k]) * v[k];
    y[off] -= shift * v[off];
    if (iter > 0) y[off] -= beta / oldb * r1[off];
    tmp[off] = v[off] * y[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      alfa = tmp[0];
      oldb = beta;
    }
    __syncthreads();
    y[off] -= alfa / beta * r2[off];
    r1[off] = r2[off];
    r2[off] = y[off];
    // preconditioner
    if (off < 4) {
      Solve4x4Sym(A + 68 * off, A_add + 68 * off, r2 + 4 * off, 16,
                  y + 4 * off);
    }
    __syncthreads();
    if (off >= 12) y[off] = -y[off];
    __syncthreads();
    tmp[off] = r2[off] * y[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      beta = sqrt(tmp[0]);
      oldeps = epsln;
      delta = cs * dbar + sn * alfa;
      gbar = sn * dbar - cs * alfa;
      epsln = sn * beta;
      dbar = -cs * beta;
      gamma = sqrt(gbar * gbar + beta * beta);
      gamma = max(gamma, 1e-10);
      cs = gbar / gamma;
      sn = beta / gamma;
      phi = cs * phibar;
      phibar = sn * phibar;
    }
    __syncthreads();
    real w1 = w2[off];
    w2[off] = w[off];
    w[off] = (v[off] - oldeps * w1 - delta * w2[off]) / gamma;
    x[off] += phi * w[off];
    if (abs(phibar) / beta1 < tol) {
      return;
    }
  }
}

__global__ void GSAsIncMixed(real* X, const real* diag, const real* diag_add,
                             const real* bcsr_val, const int32_t* bcsr_row,
                             const int32_t* bcsr_col, const real* B,
                             const int32_t* colors, const int c,
                             const real relax) {
  if (colors[blockIdx.x] != c) return;
  real tol = 1e-7;
  real shift = 0.;
  int t = blockIdx.x;
  int off = threadIdx.x;
  const real* A = diag + 256 * t;
  const real* A_add = diag_add + 256 * t;
  real* x = X + 16 * t;

  __shared__ real r1[16], r2[16], y[16], w[16], v[16], w2[16], b[16];
  __shared__ volatile real tmp[24];
  __shared__ real alfa, beta, beta1, oldb, dbar, oldeps, epsln, gbar, phi,
      phibar, delta, gamma, cs, sn;
  __shared__ bool flag;

  b[off] = B[16 * t + off];
  for (int32_t idx = bcsr_row[t]; idx < bcsr_row[t + 1]; ++idx) {
    int32_t j = bcsr_col[idx];
    if (colors[j] >= c) continue;
    const real* Aj = &bcsr_val[256 * idx];
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
      b[off] -= Aj[16 * off + jj] * X[16 * j + jj];
    }
  }
  b[off] *= relax;

  y[off] = b[off];
  // preconditioner
  if (off < 4) {
    Solve4x4Sym(A + 68 * off, A_add + 68 * off, b + 4 * off, 16, y + 4 * off);
  }
  __syncthreads();
  if (off >= 12) y[off] = -y[off];
  __syncthreads();
  w[off] = 0.;
  w2[off] = 0.;
  r1[off] = b[off];
  r2[off] = b[off];

  tmp[off] = y[off] * b[off];
  tmp[off] += tmp[off + 8];
  tmp[off] += tmp[off + 4];
  tmp[off] += tmp[off + 2];
  tmp[off] += tmp[off + 1];
  if (off == 0) {
    beta1 = sqrt(tmp[0]);
    oldb = 0.;
    beta = beta1;
    dbar = 0.;
    epsln = 0.;
    phibar = beta1;
    cs = -1.;
    sn = 0.;
  }
  __syncthreads();

  if (beta < tol) {
    return;
  }

  for (int iter = 0; iter < 16; ++iter) {
    v[off] = y[off] / beta;
#pragma unroll
    for (int k = 0; k < 16; k++)
      y[off] += (A[16 * off + k] + A_add[16 * off + k]) * v[k];
    y[off] -= shift * v[off];
    if (iter > 0) y[off] -= beta / oldb * r1[off];
    tmp[off] = v[off] * y[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      alfa = tmp[0];
      oldb = beta;
    }
    __syncthreads();
    y[off] -= alfa / beta * r2[off];
    r1[off] = r2[off];
    r2[off] = y[off];
    // preconditioner
    if (off < 4) {
      Solve4x4Sym(A + 68 * off, A_add + 68 * off, r2 + 4 * off, 16,
                  y + 4 * off);
    }
    __syncthreads();
    if (off >= 12) y[off] = -y[off];
    __syncthreads();
    tmp[off] = r2[off] * y[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      beta = sqrt(tmp[0]);
      oldeps = epsln;
      delta = cs * dbar + sn * alfa;
      gbar = sn * dbar - cs * alfa;
      epsln = sn * beta;
      dbar = -cs * beta;
      gamma = sqrt(gbar * gbar + beta * beta);
      gamma = max(gamma, 1e-10);
      cs = gbar / gamma;
      sn = beta / gamma;
      phi = cs * phibar;
      phibar = sn * phibar;
    }
    __syncthreads();
    real w1 = w2[off];
    w2[off] = w[off];
    w[off] = (v[off] - oldeps * w1 - delta * w2[off]) / gamma;
    x[off] += phi * w[off];
    if (abs(phibar) / beta1 < tol) {
      return;
    }
  }
}

__global__ void GSAsDecMixed(real* X, const real* diag, const real* diag_add,
                             const real* bcsr_val, const int32_t* bcsr_row,
                             const int32_t* bcsr_col, const real* B,
                             const int32_t* colors, const int c,
                             const real relax) {
  if (colors[blockIdx.x] != c) return;
  real tol = 1e-7;
  real shift = 0.;
  int t = blockIdx.x;
  int off = threadIdx.x;
  const real* A = diag + 256 * t;
  const real* A_add = diag_add + 256 * t;
  real* x = X + 16 * t;

  __shared__ real r1[16], r2[16], y[16], w[16], v[16], w2[16], b[16];
  __shared__ volatile real tmp[24];
  __shared__ real alfa, beta, beta1, oldb, dbar, oldeps, epsln, gbar, phi,
      phibar, delta, gamma, cs, sn;
  __shared__ bool flag;

  b[off] = B[16 * t + off];
  for (int32_t idx = bcsr_row[t]; idx < bcsr_row[t + 1]; ++idx) {
    int32_t j = bcsr_col[idx];
    if (colors[j] <= c) continue;
    const real* Aj = &bcsr_val[256 * idx];
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
      b[off] -= Aj[16 * off + jj] * X[16 * j + jj];
    }
  }
  b[off] *= relax;

  y[off] = b[off];
  // preconditioner
  if (off < 4) {
    Solve4x4Sym(A + 68 * off, A_add + 68 * off, b + 4 * off, 16, y + 4 * off);
  }
  __syncthreads();
  if (off >= 12) y[off] = -y[off];
  __syncthreads();
  w[off] = 0.;
  w2[off] = 0.;
  r1[off] = b[off];
  r2[off] = b[off];

  tmp[off] = y[off] * b[off];
  tmp[off] += tmp[off + 8];
  tmp[off] += tmp[off + 4];
  tmp[off] += tmp[off + 2];
  tmp[off] += tmp[off + 1];
  if (off == 0) {
    beta1 = sqrt(tmp[0]);
    oldb = 0.;
    beta = beta1;
    dbar = 0.;
    epsln = 0.;
    phibar = beta1;
    cs = -1.;
    sn = 0.;
  }
  __syncthreads();

  if (beta < tol) {
    return;
  }

  for (int iter = 0; iter < 16; ++iter) {
    v[off] = y[off] / beta;
#pragma unroll
    for (int k = 0; k < 16; k++)
      y[off] += (A[16 * off + k] + A_add[16 * off + k]) * v[k];
    y[off] -= shift * v[off];
    if (iter > 0) y[off] -= beta / oldb * r1[off];
    tmp[off] = v[off] * y[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      alfa = tmp[0];
      oldb = beta;
    }
    __syncthreads();
    y[off] -= alfa / beta * r2[off];
    r1[off] = r2[off];
    r2[off] = y[off];
    // preconditioner
    if (off < 4) {
      Solve4x4Sym(A + 68 * off, A_add + 68 * off, r2 + 4 * off, 16,
                  y + 4 * off);
    }
    __syncthreads();
    if (off >= 12) y[off] = -y[off];
    __syncthreads();
    tmp[off] = r2[off] * y[off];
    tmp[off] += tmp[off + 8];
    tmp[off] += tmp[off + 4];
    tmp[off] += tmp[off + 2];
    tmp[off] += tmp[off + 1];
    if (off == 0) {
      beta = sqrt(tmp[0]);
      oldeps = epsln;
      delta = cs * dbar + sn * alfa;
      gbar = sn * dbar - cs * alfa;
      epsln = sn * beta;
      dbar = -cs * beta;
      gamma = sqrt(gbar * gbar + beta * beta);
      gamma = max(gamma, 1e-10);
      cs = gbar / gamma;
      sn = beta / gamma;
      phi = cs * phibar;
      phibar = sn * phibar;
    }
    __syncthreads();
    real w1 = w2[off];
    w2[off] = w[off];
    w[off] = (v[off] - oldeps * w1 - delta * w2[off]) / gamma;
    x[off] += phi * w[off];
    if (abs(phibar) / beta1 < tol) {
      return;
    }
  }
}

__global__ void hVankaAfMixed(real* X, const real* Af_diag,
                              const real* Af_diag_add, const real* bcsr_val,
                              const int32_t* bcsr_row, const int32_t* bcsr_col,
                              const real* b, const int32_t* colors, const int c,
                              const real relax, const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  if (colors[i] != c) {
    // real d = -Af_diag[16 * i + 15] - Af_diag_add[16 * i + 15];
    // real tmp[3];
    // for (int32_t idx = bcsr_row[i]; idx < bcsr_row[i + 1]; ++idx) {
    //   int32_t j = bcsr_col[idx];
    //   const real* g = &bcsr_val[16 * idx + 12];
    //   Solve3x3Sym(Af_diag + 16 * j, Af_diag_add + 16 * j, g, 4, tmp);
    //   d += g[0] * tmp[0] + g[1] * tmp[1] + g[2] * tmp[2];
    // }
    // X[4 * i + 3] -= .2 * b[4 * i + 3] / d;
    // Solve3x3Sym(Af_diag + 16 * i, Af_diag_add + 16 * i, b + 4 * i, 4, tmp);
    // X[4 * i + 0] += .2 * tmp[0];
    // X[4 * i + 1] += .2 * tmp[1];
    // X[4 * i + 2] += .2 * tmp[2];
  } else {
    real tmp[4];
    tmp[0] = b[4 * i + 0];
    tmp[1] = b[4 * i + 1];
    tmp[2] = b[4 * i + 2];
    tmp[3] = b[4 * i + 3];
    for (int32_t idx = bcsr_row[i]; idx < bcsr_row[i + 1]; ++idx) {
      int32_t j = bcsr_col[idx];
      if (colors[j] > c) continue;
      const real* Aj = &bcsr_val[16 * idx];
      for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
          tmp[x] -= Aj[4 * x + y] * X[4 * j + y];
        }
      }
    }
    tmp[0] *= relax;
    tmp[1] *= relax;
    tmp[2] *= relax;
    tmp[3] *= relax;
    Solve4x4Sym(Af_diag + 16 * i, Af_diag_add + 16 * i, tmp, X + 4 * i);
  }
}

__global__ void RestrictedVankaAfMixed(
    real* X, const real* Af_diag, const real* Af_diag_add, const real* bcsr_val,
    const int32_t* bcsr_row, const int32_t* bcsr_col, const real* b,
    const int32_t* colors, const int co, const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  real d = -Af_diag[16 * i + 15] - Af_diag_add[16 * i + 15];
  real r = -b[4 * i + 3];
  real tmp[3];
  for (int32_t idx = bcsr_row[i]; idx < bcsr_row[i + 1]; ++idx) {
    int32_t j = bcsr_col[idx];
    const real* g = &bcsr_val[16 * idx + 12];
    Solve3x3Sym(Af_diag + 16 * j, Af_diag_add + 16 * j, g, 4, tmp);
    d += g[0] * tmp[0] + g[1] * tmp[1] + g[2] * tmp[2];
    Solve3x3Sym(Af_diag + 16 * j, Af_diag_add + 16 * j, b + 4 * j, 4, tmp);
    r += g[0] * tmp[0] + g[1] * tmp[1] + g[2] * tmp[2];
  }
  real delta_p = r / d;
  X[4 * i + 3] += delta_p;
  real tmp2[3];
  tmp2[0] = b[4 * i + 0] - delta_p * Af_diag[16 * i + 12];
  tmp2[1] = b[4 * i + 1] - delta_p * Af_diag[16 * i + 13];
  tmp2[2] = b[4 * i + 2] - delta_p * Af_diag[16 * i + 14];
  Solve3x3Sym(Af_diag + 16 * i, Af_diag_add + 16 * i, tmp2, 4, tmp);
  X[4 * i + 0] += tmp[0];
  X[4 * i + 1] += tmp[1];
  X[4 * i + 2] += tmp[2];
}

__global__ void InexactUzawaXAfMixed(real* X, const real* Af_diag,
                                     const real* Af_diag_add, const real* b,
                                     const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  Solve3x3Sym(Af_diag + 16 * i, Af_diag_add + 16 * i, b + 4 * i, 4, X + 4 * i);
}

__global__ void InexactUzawaGSXAfMixed(real* X, const real* Af_diag,
                                       const real* Af_diag_add,
                                       const real* bcsr_val,
                                       const int32_t* bcsr_row,
                                       const int32_t* bcsr_col, const real* b,
                                       const int32_t* colors, const int c,
                                       const real relax, const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  if (colors[i] != c) return;
  real tmp[4];
  tmp[0] = b[4 * i + 0];
  tmp[1] = b[4 * i + 1];
  tmp[2] = b[4 * i + 2];
  for (int32_t idx = bcsr_row[i]; idx < bcsr_row[i + 1]; ++idx) {
    int32_t j = bcsr_col[idx];
    const real* Aj = &bcsr_val[16 * idx];
    for (int x = 0; x < 3; ++x) {
      for (int y = 0; y < 4; ++y) {
        tmp[x] -= Aj[4 * x + y] * X[4 * j + y];
      }
    }
  }
  tmp[0] *= relax;
  tmp[1] *= relax;
  tmp[2] *= relax;
  Solve3x3Sym(Af_diag + 16 * i, Af_diag_add + 16 * i, tmp, 4, X + 4 * i);
}

__global__ void InexactUzawaXAsMixed(real* X, const real* diag,
                                     const real* diag_add, const real* b) {
  // TODO
}

__global__ void InexactUzawaPAfMixed(real* P, const real* Af_diag,
                                     const real* Af_diag_add,
                                     const real* bcsr_val,
                                     const int32_t* bcsr_row,
                                     const int32_t* bcsr_col, const real* b,
                                     const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  real d = -Af_diag[16 * i + 15] - Af_diag_add[16 * i + 15];
  for (int32_t idx = bcsr_row[i]; idx < bcsr_row[i + 1]; ++idx) {
    int32_t j = bcsr_col[idx];
    const real* g = &bcsr_val[16 * idx + 12];
    real tmp[3];
    Solve3x3Sym(Af_diag + 16 * j, Af_diag_add + 16 * j, g, 4, tmp);
    d += g[0] * tmp[0] + g[1] * tmp[1] + g[2] * tmp[2];
  }
  P[4 * i + 3] = -b[4 * i + 3] / d;
}

__global__ void KaczmarzIteration(real* P, const real* Af_diag_add,
                                  const real* bcsr_val, const int32_t* bcsr_row,
                                  const int32_t* bcsr_col, const real* B,
                                  const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  real A[16], tmp[4];
  for (int idx = 0; idx < 16; ++idx) A[idx] = 0;
  for (int idx = bcsr_row[i]; idx < bcsr_row[i + 1]; ++idx) {
    int j = bcsr_col[idx];
    const real* Aj = &bcsr_val[16 * idx];
    if (j == i) {
      const real* add = &Af_diag_add[16 * i];
      for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
          for (int z = 0; z < 4; ++z) {
            A[4 * x + y] += (Aj[4 * x + z] + add[4 * x + z]) *
                            (Aj[4 * y + z] + add[4 * y + z]);
          }
        }
      }
    } else {
      for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
          for (int z = 0; z < 4; ++z) {
            A[4 * x + y] += Aj[4 * x + z] * Aj[4 * y + z];
          }
        }
      }
    }
  }
  // Solve4x4Sym(A, &B[4 * i], 4, tmp);
  P[4 * i + 0] += B[4 * i + 0] / A[0];
  P[4 * i + 1] += B[4 * i + 1] / A[5];
  P[4 * i + 2] += B[4 * i + 2] / A[10];
  P[4 * i + 3] += B[4 * i + 3] / A[15];
}

__global__ void AfDiagMulVecMixed(real* Y, const real* Af_diag_add,
                                  const real* X, const real alpha,
                                  const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t b = i / 4;
  int32_t off = i % 4;
  if (b >= n_vert) return;
#pragma unroll
  for (int32_t idx = 0; idx < 4; ++idx)
    Y[i] += alpha * Af_diag_add[16 * b + 4 * off + idx] * X[4 * b + idx];
}

__global__ void AsDiagMulVecMixed(real* Y, const real* As_diag_add,
                                  const real* X, const real alpha,
                                  const int32_t n_handle) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t b = i / 16;
  int32_t off = i % 16;
  if (b >= n_handle) return;
#pragma unroll
  for (int32_t idx = 0; idx < 16; ++idx)
    Y[i] += alpha * As_diag_add[256 * b + 16 * off + idx] * X[16 * b + idx];
}

__global__ void UpdatePosPressureMixed(Vec3* X, real* p, const real alpha,
                                       const real* dX, const real p_scale,
                                       const int32_t n_vert) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  X[i] += alpha * Vec3(dX[4 * i + 0], dX[4 * i + 1], dX[4 * i + 2]);
  p[i] += alpha * p_scale * dX[4 * i + 3];
}

__global__ void ComputeDistortionEnergyMixed(const Vec3* X, const uint32_t* tet,
                                             const real* vol,
                                             const Mat3* Dm_inv, real* E,
                                             const real mu,
                                             const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t* ctet = &tet[4 * t];
  Mat3 Ds;
  Ds.col(0) = X[ctet[0]] - X[ctet[3]];
  Ds.col(1) = X[ctet[1]] - X[ctet[3]];
  Ds.col(2) = X[ctet[2]] - X[ctet[3]];
  Mat3 F = Ds * Dm_inv[t];
  real Ic = 0;
#pragma unroll
  for (int i = 0; i < 9; ++i) Ic += F.data()[i] * F.data()[i];
  real e = mu / 2 * (Ic - 3) - mu * (Determinant(F) - 1);
  ::atomicAdd(E, e * vol[t]);
}

__global__ void MakeSelfCollisionMixed(
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
      hessians[16 * c + 4 * x + y] = h;
      ::atomicAdd(&diag_add[16 * v + 4 * x + y], h);
      ::atomicAdd(&diag_add[16 * p + 4 * x + y], h);
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
    ::atomicAdd(&grad[4 * v + x], fv[x]);
    ::atomicAdd(&grad[4 * p + x], fp[x]);
  }
}

__global__ void SelfCollisionFineOffReductionMixed(
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
  const real* fhessian = &fine_hessian[16 * c];
  real* chessian = &coarse_hessian[256 * c];
  Vec4 x1 = Vec4::Unit(3);
  Vec4 x2 = Vec4::Unit(3);
  x1[0] = pos_rest[v1](0) - pos_rest[vh1](0);
  x1[1] = pos_rest[v1](1) - pos_rest[vh1](1);
  x1[2] = pos_rest[v1](2) - pos_rest[vh1](2);
  x2[0] = pos_rest[v2](0) - pos_rest[vh2](0);
  x2[1] = pos_rest[v2](1) - pos_rest[vh2](1);
  x2[2] = pos_rest[v2](2) - pos_rest[vh2](2);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
          int r = 4 * i + x;
          int c = 4 * j + y;
          chessian[16 * r + c] = fhessian[4 * i + j] * x1[x] * x2[y];
        }
      }
    }
  }
}

__global__ void SelfCollisionFineDiagReductionMixed(
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
  const real* fhessian = &fine_hessian[16 * c];
  Vec4 x1 = Vec4::Unit(3);
  Vec4 x2 = Vec4::Unit(3);
  x1[0] = pos_rest[v1](0) - pos_rest[vh1](0);
  x1[1] = pos_rest[v1](1) - pos_rest[vh1](1);
  x1[2] = pos_rest[v1](2) - pos_rest[vh1](2);
  x2[0] = pos_rest[v2](0) - pos_rest[vh2](0);
  x2[1] = pos_rest[v2](1) - pos_rest[vh2](1);
  x2[2] = pos_rest[v2](2) - pos_rest[vh2](2);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
          int r = 4 * i + x;
          int c = 4 * j + y;
          real tmp = fhessian[4 * i + j] * x1[x] * x2[y] +
                     fhessian[4 * j + i] * x1[y] * x2[x];
          ::atomicAdd(&diag_add[256 * h1 + 16 * r + c], -tmp);
        }
      }
    }
  }
}

__global__ void SelfCollisionCoarseOffReductionMixed(
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
  const real* fhessian = &fine_hessian[256 * c];
  real* chessian = &coarse_hessian[256 * c];
  chessian[i] = fhessian[i];
}

__global__ void SelfCollisionCoarseDiagReductionMixed(real* diag_add,
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
  const real* fhessian = &fine_hessian[256 * c];
  int32_t i = threadIdx.y;
  int32_t x = i / 16, y = i % 16;
  ::atomicAdd(&diag_add[256 * h1 + i], -fhessian[i] - fhessian[16 * y + x]);
}

__global__ void SelfCollisionFineOffAPMixed(real* AP, const int32_t* pairs,
                                            const real* hessian, const real* P,
                                            const real alpha,
                                            const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[16 * c];
  const real* p1 = &P[4 * v1];
  const real* p2 = &P[4 * v2];
  real ap1[4], ap2[4];
  for (int i = 0; i < 4; ++i) {
    ap1[i] = 0.;
    ap2[i] = 0.;
    for (int j = 0; j < 4; ++j) {
      ap1[i] -= h[4 * i + j] * p2[j];
      ap2[i] -= h[4 * j + i] * p1[j];
    }
  }

  ::atomicAdd(&AP[4 * v1 + 0], alpha * ap1[0]);
  ::atomicAdd(&AP[4 * v1 + 1], alpha * ap1[1]);
  ::atomicAdd(&AP[4 * v1 + 2], alpha * ap1[2]);
  ::atomicAdd(&AP[4 * v1 + 3], alpha * ap1[3]);
  ::atomicAdd(&AP[4 * v2 + 0], alpha * ap2[0]);
  ::atomicAdd(&AP[4 * v2 + 1], alpha * ap2[1]);
  ::atomicAdd(&AP[4 * v2 + 2], alpha * ap2[2]);
  ::atomicAdd(&AP[4 * v2 + 3], alpha * ap2[3]);
}

__global__ void SelfCollisionFineOffGSAPMixed(
    real* AP, const int32_t* pairs, const real* hessian, const int32_t* color,
    const int32_t co, const real* P, const real alpha, const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[16 * c];
  const real* p1 = &P[4 * v1];
  const real* p2 = &P[4 * v2];
  if (color[v1] == co) {
    real ap2[4];
    for (int i = 0; i < 4; ++i) {
      ap2[i] = 0.;
      for (int j = 0; j < 4; ++j) {
        ap2[i] -= h[4 * j + i] * p1[j];
      }
    }
    ::atomicAdd(&AP[4 * v2 + 0], alpha * ap2[0]);
    ::atomicAdd(&AP[4 * v2 + 1], alpha * ap2[1]);
    ::atomicAdd(&AP[4 * v2 + 2], alpha * ap2[2]);
    ::atomicAdd(&AP[4 * v2 + 3], alpha * ap2[3]);
  }
  if (color[v2] == co) {
    real ap1[4];
    for (int i = 0; i < 4; ++i) {
      ap1[i] = 0.;
      for (int j = 0; j < 4; ++j) {
        ap1[i] -= h[4 * i + j] * p2[j];
      }
    }
    ::atomicAdd(&AP[4 * v1 + 0], alpha * ap1[0]);
    ::atomicAdd(&AP[4 * v1 + 1], alpha * ap1[1]);
    ::atomicAdd(&AP[4 * v1 + 2], alpha * ap1[2]);
    ::atomicAdd(&AP[4 * v1 + 3], alpha * ap1[3]);
  }
}

__global__ void SelfCollisionCoarseOffAPMixed(real* AP, const int32_t* pairs,
                                              const real* hessian,
                                              const real* P, const real alpha,
                                              const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[256 * c];
  int32_t i = threadIdx.y;
  const real* p1 = &P[16 * v1];
  const real* p2 = &P[16 * v2];
  real ap1 = 0., ap2 = 0.;
  for (int j = 0; j < 16; ++j) {
    ap1 -= h[16 * i + j] * p2[j];
    ap2 -= h[16 * j + i] * p1[j];
  }
  ::atomicAdd(&AP[16 * v1 + i], alpha * ap1);
  ::atomicAdd(&AP[16 * v2 + i], alpha * ap2);
}

__global__ void SelfCollisionCoarseOffGSAPMixed(
    real* AP, const int32_t* pairs, const real* hessian, const int32_t* color,
    const int32_t co, const real* P, const real alpha, const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[256 * c];
  int32_t i = threadIdx.y;
  const real* p1 = &P[16 * v1];
  const real* p2 = &P[16 * v2];
  if (color[v1] == co) {
    real ap2 = 0.;
    for (int j = 0; j < 16; ++j) {
      ap2 -= h[16 * j + i] * p1[j];
    }
    ::atomicAdd(&AP[16 * v2 + i], alpha * ap2);
  }
  if (color[v2] == co) {
    real ap1 = 0.;
    for (int j = 0; j < 16; ++j) {
      ap1 -= h[16 * i + j] * p2[j];
    }
    ::atomicAdd(&AP[16 * v1 + i], alpha * ap1);
  }
}

__global__ void SelfCollisionFineDenseHessianMixed(real* den,
                                                   const int32_t* pairs,
                                                   const real* hessian,
                                                   const int32_t dim,
                                                   const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[16 * c];
  int32_t i = threadIdx.y;
  int32_t row = 4 * v1 + i / 4;
  int32_t col = 4 * v2 + i % 4;
  ::atomicAdd(&den[row + dim * col], -h[i]);
  ::atomicAdd(&den[col + dim * row], -h[i]);
}

__global__ void SelfCollisionCoarseDenseHessianMixed(real* den,
                                                     const int32_t* pairs,
                                                     const real* hessian,
                                                     const int32_t dim,
                                                     const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[256 * c];
  int32_t i = threadIdx.y;
  int32_t x = i / 12, y = i % 12;
  int32_t row = 16 * v1 + x;
  int32_t col = 16 * v2 + y;
  ::atomicAdd(&den[row + dim * col], -h[16 * x + y]);
  ::atomicAdd(&den[col + dim * row], -h[16 * x + y]);
}

__global__ void SelfCollisionCoarseDenseHessianShurMixed(
    real* den, const int32_t* pairs, const real* hessian, const int32_t dim,
    const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = pairs[2 * c + 0];
  int32_t v2 = pairs[2 * c + 1];
  if (v1 == v2) return;
  const real* h = &hessian[256 * c];
  int32_t i = threadIdx.y;
  int32_t x = i / 12, y = i % 12;
  int32_t row = 12 * v1 + x;
  int32_t col = 12 * v2 + y;
  ::atomicAdd(&den[row + dim * col], -h[16 * x + y]);
  ::atomicAdd(&den[col + dim * row], -h[16 * x + y]);
}

__global__ void Int32toInt64(const int32_t* in, int64_t* out, int32_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = (int64_t)in[idx];
}
}  // namespace Rain::CUDA