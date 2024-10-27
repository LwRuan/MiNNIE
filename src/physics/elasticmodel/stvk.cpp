#include "stvk.h"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

namespace Rain {
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

static void BuildScaleEigenvectors(const Mat3& U, const Mat3& V,
                                          Mat9& Q) {
  using M3 = Eigen::Matrix<real, 3, 3, Eigen::ColMajor>;

  M3 A;
  A << U(0, 0) * V(0, 0), U(0, 0) * V(1, 0), U(0, 0) * V(2, 0),
       U(1, 0) * V(0, 0), U(1, 0) * V(1, 0), U(1, 0) * V(2, 0),
       U(2, 0) * V(0, 0), U(2, 0) * V(1, 0), U(2, 0) * V(2, 0);
  M3 B;
  B << U(0, 1) * V(0, 1), U(0, 1) * V(1, 1), U(0, 1) * V(2, 1),
       U(1, 1) * V(0, 1), U(1, 1) * V(1, 1), U(1, 1) * V(2, 1),
       U(2, 1) * V(0, 1), U(2, 1) * V(1, 1), U(2, 1) * V(2, 1);
  M3 C;
  C << U(0, 2) * V(0, 2), U(0, 2) * V(1, 2), U(0, 2) * V(2, 2),
       U(1, 2) * V(0, 2), U(1, 2) * V(1, 2), U(1, 2) * V(2, 2),
       U(2, 2) * V(0, 2), U(2, 2) * V(1, 2), U(2, 2) * V(2, 2);
  
  // Scale eigenvectors
  Eigen::Map<M3>(Q.data() + 54) = A;
  Eigen::Map<M3>(Q.data() + 63) = B;
  Eigen::Map<M3>(Q.data() + 72) = C;
}

void StVK::GetPiola(const Mat3& F, Mat3& P) {
  Mat3 E = 0.5 * (F.transpose() * F - Mat3::Identity());
  P = F * (2 * mu_ * E + lambda_ * E.trace() * Mat3::Identity());
}

void StVK::GetMixedPiola(const Mat3& F, Mat3& P) {
  Mat3 E = 0.5 * (F.transpose() * F - Mat3::Identity());
  P = 2 * mu_ * F * E;
}

void StVK::GetMixeddPdF(const Mat3& F, Mat9& dPdF) {
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
  eigenvalues[0] = -mu_ + mu_ * (S[1] * S[1] + S[2] * S[2] - S[1] * S[2]);
  eigenvalues[1] = -mu_ + mu_ * (S[0] * S[0] + S[2] * S[2] - S[0] * S[2]);
  eigenvalues[2] = -mu_ + mu_ * (S[0] * S[0] + S[1] * S[1] - S[0] * S[1]);
  eigenvalues[3] = -mu_ + mu_ * (S[1] * S[1] + S[2] * S[2] + S[1] * S[2]);
  eigenvalues[4] = -mu_ + mu_ * (S[0] * S[0] + S[2] * S[2] + S[0] * S[2]);
  eigenvalues[5] = -mu_ + mu_ * (S[0] * S[0] + S[1] * S[1] + S[0] * S[1]);

  // Scale
  BuildScaleEigenvectors(U, V, eigenvectors);
  eigenvalues[6] = -mu_ + 3. * mu_ * (S[0] * S[0]);
  eigenvalues[7] = -mu_ + 3. * mu_ * (S[1] * S[1]);
  eigenvalues[8] = -mu_ + 3. * mu_ * (S[2] * S[2]);

  // Clamp the eigenvalues
  for (int i = 0; i < 9; i++) {
    if (eigenvalues[i] < 0.0) {
      eigenvalues[i] = 0.0;
    }
  }

  dPdF = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}
};  // namespace Rain