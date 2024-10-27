#pragma once

#include <string>

#include "enumtype.h"
#include "mathtype.h"

namespace Rain {
class ElasticModel {
 public:
  ElasticModelType type_;
  real young_;
  real poisson_;
  real mu_;
  real lambda_;
  real lambda_inv_;
  // bool require_svd_;
  virtual void SetLame(real young, real poisson) {
    young_ = young;
    poisson_ = poisson;
    mu_ = young / 2.0 / (1 + poisson);
    lambda_ = young * poisson / (1 + poisson) / (1 - 2 * poisson);
    lambda_inv_ = (1 + poisson) * (1 - 2 * poisson) / poisson / young;
  }
  virtual void GetPiola(const Mat3& F, Mat3& P){};
  virtual void GetPiolaSVD(const Mat3& F, const Mat3& U, const Mat3& V, real s1,
                           real s2, real s3, Mat3& P){};
  virtual void GetPiolaR(const Mat3& F, const Mat3& R, Mat3& P){};
  virtual void GetdPdF(const Mat3& F, Mat9& dPdF){};
  virtual void GetdPdFSVD(const Mat3& F, const Mat3& U, const Mat3& V, real s1,
                          real s2, real s3, Mat9& dPdF){};
  virtual void GetMixedPiola(const Mat3& F, Mat3& P){};
  virtual void GetMixedPiolaSVD(const Mat3& F, const Mat3& U, const Mat3& V, real s1,
                           real s2, real s3, Mat3& P){};
  virtual void GetMixedPiolaR(const Mat3& F, const Mat3& R, Mat3& P){};
  virtual void GetMixeddPdF(const Mat3& F, Mat9& dPdF){};
  virtual void GetMixeddPdFSVD(const Mat3& F, const Mat3& U, const Mat3& V, real s1,
                          real s2, real s3, Mat9& dPdF){};
  virtual std::string GetName() = 0;
};
};  // namespace Rain