// all the math functions not defined in Eigen/Core
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "mathtype.h"

namespace Rain {
namespace CUDA {
__device__ void Cross(const Vec3& a, const Vec3& b, Vec3& out);
__device__ Vec3 Cross(const Vec3& a, const Vec3& b);

__device__ void Solve3x3Sym(const real* A, const real* A_a, const real* b,
                            real* x);

__device__ void Solve3x3Sym(const real* A, const real* b, const int32_t lda,
                            real* x);

__device__ void Solve3x3Sym(const real* A, const real* A_a, const real* b,
                            const int32_t lda, real* x);

__device__ void Solve4x4Sym(const real* A, const real* A_a, const real* b,
                            real* x);

__device__ void Solve4x4SymMINRES(const real* A, const real* A_a, const real* b,
                                  real* x);

__device__ void Solve4x4SymCG(const real* A, const real* A_a, const real* b,
                              real* x);

__device__ void Solve4x4Sym(const real* A, const real* A_a, const real* b,
                            const int32_t lda, real* x);

__device__ void Solve4x4Sym(const real* A, const real* b, const int32_t lda,
                            real* x);

__device__ void GetRotation(const Mat3& F, Mat3& R);

__device__ real Determinant(const Mat3& A);

__device__ Vec3 atomicAdd(Vec3& target, const Vec3& val);

__device__ bool HasNan(const Mat3& A);

__device__ bool HasNan(const Vec3& V);

__global__ void CheckNan(const real* X, const int32_t size, int32_t* info);

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val);
#endif
};  // namespace CUDA

__host__ int32_t CheckNan(const real* X, const int32_t size);
};  // namespace Rain
