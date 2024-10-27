#include "corotation.h"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

namespace Rain {
void Corotation::GetMixedPiola(const Mat3& F, Mat3& P) {
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

  P = 2 * mu_ * (F - U * V.transpose());
}

void Corotation::GetMixeddPdF(const Mat3& F, Mat9& dPdF) {
  dPdF = 2 * mu_ * Mat9::Identity();
}
};  // namespace Rain