#include "neohookeanlog.h"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

namespace Rain {
void NeoHookeanLog::SetLame(real young, real poisson) {
  ElasticModel::SetLame(young, poisson);
  lambda_ = lambda_ + mu_;
  lambda_inv_ = 2. * (1. + poisson_) * (1. - 2. * poisson_) / young_;
  alpha_ = 1 + mu_ / lambda_;
}

void NeoHookeanLog::GetPiola(const Mat3& F, Mat3& P) {
  real fac = lambda_ * (F.determinant() - alpha_);
  P.col(0) = fac * F.col(1).cross(F.col(2));
  P.col(1) = fac * F.col(2).cross(F.col(0));
  P.col(2) = fac * F.col(0).cross(F.col(1));
  P += mu_ * F;
};

void NeoHookeanLog::GetMixedPiola(const Mat3& F, Mat3& P) {
  real fac = -mu_;
  P.col(0) = fac * F.col(1).cross(F.col(2));
  P.col(1) = fac * F.col(2).cross(F.col(0));
  P.col(2) = fac * F.col(0).cross(F.col(1));
  P += mu_ * F;
};

static void BuildTwistAndFlipEigenvectors(const Mat3& U, const Mat3& V,
                                          Mat9& Q) {
  static const real scale = 1.0 / std::sqrt(2.0);
  const Mat3 sV = scale * V;

  using M3 = Eigen::Matrix<real, 3, 3, Eigen::ColMajor>;

  M3 A;
  A << sV(0, 2) * U(0, 1), sV(1, 2) * U(0, 1), sV(2, 2) * U(0, 1),
      sV(0, 2) * U(1, 1), sV(1, 2) * U(1, 1), sV(2, 2) * U(1, 1),
      sV(0, 2) * U(2, 1), sV(1, 2) * U(2, 1), sV(2, 2) * U(2, 1);

  M3 B;
  B << sV(0, 1) * U(0, 2), sV(1, 1) * U(0, 2), sV(2, 1) * U(0, 2),
      sV(0, 1) * U(1, 2), sV(1, 1) * U(1, 2), sV(2, 1) * U(1, 2),
      sV(0, 1) * U(2, 2), sV(1, 1) * U(2, 2), sV(2, 1) * U(2, 2);

  M3 C;
  C << sV(0, 2) * U(0, 0), sV(1, 2) * U(0, 0), sV(2, 2) * U(0, 0),
      sV(0, 2) * U(1, 0), sV(1, 2) * U(1, 0), sV(2, 2) * U(1, 0),
      sV(0, 2) * U(2, 0), sV(1, 2) * U(2, 0), sV(2, 2) * U(2, 0);

  M3 D;
  D << sV(0, 0) * U(0, 2), sV(1, 0) * U(0, 2), sV(2, 0) * U(0, 2),
      sV(0, 0) * U(1, 2), sV(1, 0) * U(1, 2), sV(2, 0) * U(1, 2),
      sV(0, 0) * U(2, 2), sV(1, 0) * U(2, 2), sV(2, 0) * U(2, 2);

  M3 E;
  E << sV(0, 1) * U(0, 0), sV(1, 1) * U(0, 0), sV(2, 1) * U(0, 0),
      sV(0, 1) * U(1, 0), sV(1, 1) * U(1, 0), sV(2, 1) * U(1, 0),
      sV(0, 1) * U(2, 0), sV(1, 1) * U(2, 0), sV(2, 1) * U(2, 0);

  M3 F;
  F << sV(0, 0) * U(0, 1), sV(1, 0) * U(0, 1), sV(2, 0) * U(0, 1),
      sV(0, 0) * U(1, 1), sV(1, 0) * U(1, 1), sV(2, 0) * U(1, 1),
      sV(0, 0) * U(2, 1), sV(1, 0) * U(2, 1), sV(2, 0) * U(2, 1);

  // Twist eigenvectors
  Eigen::Map<M3>(Q.data()) = B - A;
  Eigen::Map<M3>(Q.data() + 9) = D - C;
  Eigen::Map<M3>(Q.data() + 18) = F - E;

  // Flip eigenvectors
  Eigen::Map<M3>(Q.data() + 27) = A + B;
  Eigen::Map<M3>(Q.data() + 36) = C + D;
  Eigen::Map<M3>(Q.data() + 45) = E + F;
}

void NeoHookeanLog::GetdPdF(const Mat3& F, Mat9& dPdF) {
  Vec9 eigenvalues;
  Mat9 eigenvectors;

  const Eigen::JacobiSVD<Mat3, Eigen::NoQRPreconditioner> svd(
      F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 U = svd.matrixU();
  Mat3 V = svd.matrixV();
  Vec3 S = svd.singularValues();
  {
    Mat3 L = Mat3::Identity();
    L(2, 2) = (U * V.transpose()).determinant();
    real detU = U.determinant();
    real detV = V.determinant();
    if (detU < 0. && detV > 0.) U = U * L;
    if (detU > 0. && detV < 0.) V = V * L;
    S(2) = S(2) * L(2, 2);
  }

  const real J = F.determinant();
  eigenvalues.segment<3>(0) = S;
  eigenvalues.segment<3>(3) = -S;
  const real evScale = lambda_ * (J - 1.0) - mu_;
  eigenvalues.segment<6>(0) *= evScale;
  eigenvalues.segment<6>(0).array() += mu_;

  BuildTwistAndFlipEigenvectors(U, V, eigenvectors);

  // Compute the remaining three eigenvalues and eigenvectors
  {
    Mat3 A;
    const real s0s0 = S(0) * S(0);
    const real s1s1 = S(1) * S(1);
    const real s2s2 = S(2) * S(2);
    A(0, 0) = mu_ + lambda_ * s1s1 * s2s2;
    A(1, 1) = mu_ + lambda_ * s0s0 * s2s2;
    A(2, 2) = mu_ + lambda_ * s0s0 * s1s1;
    const real evScale = lambda_ * (2.0 * J - 1.0) - mu_;
    A(0, 1) = evScale * S(2);
    A(1, 0) = A(0, 1);
    A(0, 2) = evScale * S(1);
    A(2, 0) = A(0, 2);
    A(1, 2) = evScale * S(0);
    A(2, 1) = A(1, 2);

    const Eigen::SelfAdjointEigenSolver<Mat3> Aeigs(A);
    eigenvalues.segment<3>(6) = Aeigs.eigenvalues();

    Eigen::Map<Mat3>(eigenvectors.data() + 54) =
        U * Aeigs.eigenvectors().col(0).asDiagonal() * V.transpose();
    Eigen::Map<Mat3>(eigenvectors.data() + 63) =
        U * Aeigs.eigenvectors().col(1).asDiagonal() * V.transpose();
    Eigen::Map<Mat3>(eigenvectors.data() + 72) =
        U * Aeigs.eigenvectors().col(2).asDiagonal() * V.transpose();
  }

  // Clamp the eigenvalues
  for (int i = 0; i < 9; i++) {
    if (eigenvalues(i) < 0.0) {
      eigenvalues(i) = 0.0;
    }
  }

  dPdF = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}

void NeoHookeanLog::GetMixeddPdF(const Mat3& F, Mat9& dPdF) {
  Vec9 eigenvalues;
  Mat9 eigenvectors;

  const Eigen::JacobiSVD<Mat3, Eigen::NoQRPreconditioner> svd(
      F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 U = svd.matrixU();
  Mat3 V = svd.matrixV();
  Vec3 S = svd.singularValues();
  {
    Mat3 L = Mat3::Identity();
    L(2, 2) = (U * V.transpose()).determinant();
    real detU = U.determinant();
    real detV = V.determinant();
    if (detU < 0. && detV > 0.) U = U * L;
    if (detU > 0. && detV < 0.) V = V * L;
    S(2) = S(2) * L(2, 2);
  }

  // Twist And Flip
  BuildTwistAndFlipEigenvectors(U, V, eigenvectors);
  eigenvalues.segment<3>(0) = -S;
  eigenvalues.segment<3>(3) = S;
  eigenvalues.segment<6>(0) *= mu_;
  eigenvalues.segment<6>(0).array() += mu_;

  Mat3 tA = Mat3::Zero();
  tA << 0., S(2), S(1), S(2), 0., S(0), S(1), S(0), 0.;
  const Eigen::SelfAdjointEigenSolver<Mat3> Aeigs(tA);
  eigenvalues.segment<3>(6) = mu_ * (Vec3::Ones() - Aeigs.eigenvalues());
  Eigen::Map<Mat3>(eigenvectors.data() + 54) =
      U * Aeigs.eigenvectors().col(0).asDiagonal() * V.transpose();
  Eigen::Map<Mat3>(eigenvectors.data() + 63) =
      U * Aeigs.eigenvectors().col(1).asDiagonal() * V.transpose();
  Eigen::Map<Mat3>(eigenvectors.data() + 72) =
      U * Aeigs.eigenvectors().col(2).asDiagonal() * V.transpose();

  // Clamp the eigenvalues
  for (int i = 0; i < 9; i++) {
    if (eigenvalues(i) < 0.0) {
      eigenvalues(i) = 0.0;
    }
  }

  dPdF = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}

void NeoHookeanLog::GetdPdFSVD(const Mat3& F, const Mat3& U, const Mat3& V,
                            const Vec3& S, Mat9& dPdF) {
  Vec9 eigenvalues;
  Mat9 eigenvectors;

  const real J = F.determinant();
  eigenvalues.segment<3>(0) = S;
  eigenvalues.segment<3>(3) = -S;
  const real evScale = lambda_ * (J - 1.0) - mu_;
  eigenvalues.segment<6>(0) *= evScale;
  eigenvalues.segment<6>(0).array() += mu_;

  BuildTwistAndFlipEigenvectors(U, V, eigenvectors);

  // Compute the remaining three eigenvalues and eigenvectors
  {
    Mat3 A;
    const real s0s0 = S(0) * S(0);
    const real s1s1 = S(1) * S(1);
    const real s2s2 = S(2) * S(2);
    A(0, 0) = mu_ + lambda_ * s1s1 * s2s2;
    A(1, 1) = mu_ + lambda_ * s0s0 * s2s2;
    A(2, 2) = mu_ + lambda_ * s0s0 * s1s1;
    const real evScale = lambda_ * (2.0 * J - 1.0) - mu_;
    A(0, 1) = evScale * S(2);
    A(1, 0) = A(0, 1);
    A(0, 2) = evScale * S(1);
    A(2, 0) = A(0, 2);
    A(1, 2) = evScale * S(0);
    A(2, 1) = A(1, 2);

    const Eigen::SelfAdjointEigenSolver<Mat3> Aeigs(A);
    eigenvalues.segment<3>(6) = Aeigs.eigenvalues();

    Eigen::Map<Mat3>(eigenvectors.data() + 54) =
        U * Aeigs.eigenvectors().col(0).asDiagonal() * V.transpose();
    Eigen::Map<Mat3>(eigenvectors.data() + 63) =
        U * Aeigs.eigenvectors().col(1).asDiagonal() * V.transpose();
    Eigen::Map<Mat3>(eigenvectors.data() + 72) =
        U * Aeigs.eigenvectors().col(2).asDiagonal() * V.transpose();
  }

  // Clamp the eigenvalues
  for (int i = 0; i < 9; i++) {
    if (eigenvalues(i) < 0.0) {
      eigenvalues(i) = 0.0;
    }
  }

  dPdF = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}

void NeoHookeanLog::GetMixeddPdFSVD(const Mat3& F, const Mat3& U, const Mat3& V,
                                 const Vec3& S, Mat9& dPdF) {
  Vec9 eigenvalues;
  Mat9 eigenvectors;

  // Twist And Flip
  BuildTwistAndFlipEigenvectors(U, V, eigenvectors);
  eigenvalues.segment<3>(0) = -S;
  eigenvalues.segment<3>(3) = S;
  eigenvalues.segment<6>(0) *= mu_;
  eigenvalues.segment<6>(0).array() += mu_;

  Mat3 tA = Mat3::Zero();
  tA << 0., S(2), S(1), S(2), 0., S(0), S(1), S(0), 0.;
  const Eigen::SelfAdjointEigenSolver<Mat3> Aeigs(tA);
  eigenvalues.segment<3>(6) = mu_ * (Vec3::Ones() - Aeigs.eigenvalues());
  Eigen::Map<Mat3>(eigenvectors.data() + 54) =
      U * Aeigs.eigenvectors().col(0).asDiagonal() * V.transpose();
  Eigen::Map<Mat3>(eigenvectors.data() + 63) =
      U * Aeigs.eigenvectors().col(1).asDiagonal() * V.transpose();
  Eigen::Map<Mat3>(eigenvectors.data() + 72) =
      U * Aeigs.eigenvectors().col(2).asDiagonal() * V.transpose();

  // Clamp the eigenvalues
  for (int i = 0; i < 9; i++) {
    if (eigenvalues(i) < 0.0) {
      eigenvalues(i) = 0.0;
    }
  }

  dPdF = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}
};  // namespace Rain