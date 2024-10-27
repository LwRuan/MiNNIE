#pragma once

#include "elasticmodel.h"

namespace Rain {
class PD : public ElasticModel {
 public:
  PD() {
    type_ = ElasticModelType::PD;
  }
  virtual void GetPiola(const Mat3& F, Mat3& P);
  virtual void GetPiolaR(const Mat3& F, const Mat3& R, Mat3& P);
  virtual void GetdPdF(const Mat3& F, Mat9& dPdF);
  virtual void GetMixedPiola(const Mat3& F, Mat3& P);
  virtual void GetMixedPiolaR(const Mat3& F, const Mat3& R, Mat3& P);
  virtual void GetMixeddPdF(const Mat3& F, Mat9& dPdF);
  virtual std::string GetName() { return "PD"; }
};
};  // namespace Rain