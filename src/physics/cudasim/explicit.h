#pragma once

#include "cudaelasticmodel.h"
#include "cudahelper.h"
#include "cudamath.h"
#include "mathtype.h"

namespace Rain {
namespace CUDA {
void TestCUDAEigen();
void TestSVD();
__global__ void TestCUDAEigenEntry();
__global__ void TestSVDEntry(Mat3f A);
__global__ void TestExternalBuffer(Vec3* X, real dt);
__global__ void ClearArray(Vec3* arr, uint32_t num);
__global__ void ComputeForceExplicitStVK(uint32_t* indices, Vec3* X,
                                         Mat3* Dm_inv, real* volumes, Vec3* F,
                                         uint32_t n_tet, uint32_t n_vert,
                                         real mu, real lam);
__global__ void ComputeForceExplicitNeoHookean(uint32_t* indices, Vec3* X,
                                               Mat3* Dm_inv, real* volumes,
                                               Vec3* F, uint32_t n_tet,
                                               uint32_t n_vert, real mu,
                                               real lam);
__global__ void UpdateSimpletic(Vec3* X, Vec3* V, bool* fixed, Vec3* F, real* M,
                                uint32_t n_vert, Vec3 grav, real damping,
                                real dt);
__global__ void UpdateFaceNormals(Vec3* N, Vec3* X, uint32_t* faces,
                                  uint32_t n_face);
__global__ void UpdateVertNormals(Vec3* N, uint32_t n_vert);

__global__ void PrintArraySeq(Vec3* arr, uint32_t num);

__global__ void ComputeTetVolume(const Vec3* X, const uint32_t* tet, real* vol,
                                 const uint32_t n_tet);
};  // namespace CUDA
};  // namespace Rain