#pragma once

#include "elasticmodel.h"

namespace Rain {
class NeoHookeanLog : public ElasticModel {
 public:
  real alpha_;
  NeoHookeanLog() {
    type_ = ElasticModelType::NeoHookeanLog;
  }
  virtual void GetPiola(const Mat3& F, Mat3& P);
  virtual void GetdPdFSVD(const Mat3& F, const Mat3& U, const Mat3& V, const Vec3& S, Mat9& dPdF);
  virtual void GetdPdF(const Mat3& F, Mat9& dPdF);
  virtual void GetMixedPiola(const Mat3& F, Mat3& P);
  virtual void GetMixeddPdFSVD(const Mat3& F, const Mat3& U, const Mat3& V, const Vec3& S, Mat9& dPdF);
  virtual void GetMixeddPdF(const Mat3& F, Mat9& dPdF);
  virtual void SetLame(real young, real poisson);
  virtual std::string GetName() {
    return "NeoHookeanLog";
  }
};
};  // namespace Rain