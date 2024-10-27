#pragma once

#include "elasticmodel.h"

namespace Rain {
class StVK : public ElasticModel {
 public:
  StVK() {
    type_ = ElasticModelType::StVK;
  }
  virtual void GetPiola(const Mat3& F, Mat3& P);
  virtual std::string GetName() {
    return "StVK";
  }
  virtual void GetMixedPiola(const Mat3& F, Mat3& P);
  virtual void GetMixeddPdF(const Mat3& F, Mat9& dPdF);
};
};  // namespace Rain