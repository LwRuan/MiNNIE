#include "cudaelasticmodel.h"

namespace Rain::CUDA {
__device__ void GetPiolaStVK(const Mat3& F, real mu, real lam, Mat3& P) {
  Mat3 E = 0.5 * (F.transpose() * F - Mat3::Identity());
  P = F * (2 * mu * E + lam * E.trace() * Mat3::Identity());
}

__device__ void BuildScaleEigenvectors(const Mat3& U, const Mat3& V, Mat9& Q) {
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

__device__ void BuildTwistAndFlipEigenvectors(const Mat3& U, const Mat3& V,
                                              Mat9& Q) {
  const real scale = 1.0 / std::sqrt(2.0);
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
};  // namespace Rain::CUDA