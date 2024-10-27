#pragma once

#include "cudahelper.h"
#include "cudamath.h"
#include "mathtype.h"

namespace Rain {
namespace CUDA {
__device__ void GetPiolaStVK(const Mat3& F, real mu, real lam, Mat3& P);
__device__ void BuildTwistAndFlipEigenvectors(const Mat3& U, const Mat3& V,
                                              Mat9& Q);
__device__ void BuildScaleEigenvectors(const Mat3& U, const Mat3& V, Mat9& Q);
};  // namespace CUDA
};  // namespace Rain