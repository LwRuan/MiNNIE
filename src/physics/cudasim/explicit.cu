#include <cstdio>

#include "cudasvd.h"
#include "explicit.h"

namespace Rain::CUDA {
void TestCUDAEigen() { TestCUDAEigenEntry<<<1, 1>>>(); }
void TestSVD() {
  Mat3f A{{766084.062500, 1777374.875000, 67576440.000000},
          {1777374.875000, 3710950.000000, 191023568.000000},
          {67576440.000000, 191023568.000000, 4866236928.000000}};
  TestSVDEntry<<<1, 1>>>(A);
}

__global__ void TestCUDAEigenEntry() {
  printf("CUDA Eigen test begin\n");
  Vec3 a(1, 0, 0);
  Vec3 b(0, 1, 0);
  Vec3 c = Cross(a, b);
  printf("a = %f %f %f\n", a[0], a[1], a[2]);
  printf("b = %f %f %f\n", b[0], b[1], b[2]);
  printf("a x b = %f %f %f\n", c[0], c[1], c[2]);
  printf("CUDA Eigen test end\n");
}

__global__ void TestSVDEntry(Mat3f A) {
  Mat3f U, V;
  Vec3f S;
  SVD(A, U, V, S);
  printf("A:\n%f %f %f\n%f %f %f\n%f %f %f\n", A(0, 0), A(0, 1), A(0, 2),
         A(1, 0), A(1, 1), A(1, 2), A(2, 0), A(2, 1), A(2, 2));
  printf("U:\n%f %f %f\n%f %f %f\n%f %f %f\n", U(0, 0), U(0, 1), U(0, 2),
         U(1, 0), U(1, 1), U(1, 2), U(2, 0), U(2, 1), U(2, 2));
  printf("V:\n%f %f %f\n%f %f %f\n%f %f %f\n", V(0, 0), V(0, 1), V(0, 2),
         V(1, 0), V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2));
  printf("S:\n%f 0 0\n0 %f 0\n0 0 %f\n", S(0), S(1), S(2));
  U = U * S.asDiagonal() * V.transpose();
  printf("USVt:\n%f %f %f\n%f %f %f\n%f %f %f\n", U(0, 0), U(0, 1), U(0, 2),
         U(1, 0), U(1, 1), U(1, 2), U(2, 0), U(2, 1), U(2, 2));
}

__global__ void ClearArray(Vec3* arr, uint32_t num) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < num) arr[i] = Vec3::Zero();
}

__global__ void TestExternalBuffer(Vec3* X, real dt) {
  X[0] += Vec3::Ones() * dt * 1e-1;
}

__global__ void ComputeForceExplicitStVK(uint32_t* indices, Vec3* X,
                                         Mat3* Dm_inv, real* volumes, Vec3* F,
                                         uint32_t n_tet, uint32_t n_vert,
                                         real mu, real lam) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t& v1 = indices[4 * t];
  const uint32_t& v2 = indices[4 * t + 1];
  const uint32_t& v3 = indices[4 * t + 2];
  const uint32_t& v4 = indices[4 * t + 3];
  Mat3 Ds;
  Ds.col(0) = X[v1] - X[v4];
  Ds.col(1) = X[v2] - X[v4];
  Ds.col(2) = X[v3] - X[v4];
  Mat3 Piola;
  GetPiolaStVK(Ds * Dm_inv[t], mu, lam, Piola);
  Mat3 H = -volumes[t] * Piola * Dm_inv[t].transpose();
  atomicAdd(F[v1], H.col(0));
  atomicAdd(F[v2], H.col(1));
  atomicAdd(F[v3], H.col(2));
  atomicAdd(F[v4], -H.col(0) - H.col(1) - H.col(2));
}

__global__ void ComputeForceExplicitNeoHookean(uint32_t* indices, Vec3* X,
                                               Mat3* Dm_inv, real* volumes,
                                               Vec3* F, uint32_t n_tet,
                                               uint32_t n_vert, real mu,
                                               real lam) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t& v1 = indices[4 * t];
  const uint32_t& v2 = indices[4 * t + 1];
  const uint32_t& v3 = indices[4 * t + 2];
  const uint32_t& v4 = indices[4 * t + 3];
  Mat3 Ds;
  Ds.col(0) = X[v1] - X[v4];
  Ds.col(1) = X[v2] - X[v4];
  Ds.col(2) = X[v3] - X[v4];
  Mat3 DF = Ds * Dm_inv[t];
  // Piola
  real fac = lam * (Determinant(DF) - 1 - mu / lam);
  Mat3 P;
  P.col(0) = fac * Cross(DF.col(1), DF.col(2));
  P.col(1) = fac * Cross(DF.col(2), DF.col(0));
  P.col(2) = fac * Cross(DF.col(0), DF.col(1));
  P += mu * DF;
  Mat3 H = -volumes[t] * P * Dm_inv[t].transpose();
  atomicAdd(F[v1], H.col(0));
  atomicAdd(F[v2], H.col(1));
  atomicAdd(F[v3], H.col(2));
  atomicAdd(F[v4], -H.col(0) - H.col(1) - H.col(2));
}

__global__ void UpdateSimpletic(Vec3* X, Vec3* V, bool* fixed, Vec3* F, real* M,
                                uint32_t n_vert, Vec3 grav, real damping,
                                real dt) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if ((i >= n_vert) || fixed[i]) return;
  V[i] += (F[i] / M[i] + grav) * dt;
  V[i] -= damping * V[i];
  X[i] += V[i] * dt;
}

__global__ void UpdateFaceNormals(Vec3* N, Vec3* X, uint32_t* faces,
                                  uint32_t n_face) {
  uint32_t f = blockDim.x * blockIdx.x + threadIdx.x;
  if (f >= n_face) return;
  const uint32_t& v1 = faces[3 * f];
  const uint32_t& v2 = faces[3 * f + 1];
  const uint32_t& v3 = faces[3 * f + 2];
  Vec3 dx1 = X[v2] - X[v1];
  Vec3 dx2 = X[v3] - X[v1];
  Vec3 normal = Cross(dx1, dx2).normalized();
  atomicAdd(N[v1], normal);
  atomicAdd(N[v2], normal);
  atomicAdd(N[v3], normal);
}
__global__ void UpdateVertNormals(Vec3* N, uint32_t n_vert) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_vert) return;
  N[i] = N[i].normalized();
}

__global__ void PrintArraySeq(Vec3* arr, uint32_t num) {
  for (uint32_t i = 0; i < num; ++i) {
    printf("(%f, %f, %f) ", arr[i][0], arr[i][1], arr[i][2]);
  }
  printf("\n");
}

__global__ void ComputeTetVolume(const Vec3* X, const uint32_t* tet, real* vol,
                                 const uint32_t n_tet) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t* ctet = &tet[4 * t];
  Mat3 Ds;
  Ds.col(0) = X[ctet[0]] - X[ctet[3]];
  Ds.col(1) = X[ctet[1]] - X[ctet[3]];
  Ds.col(2) = X[ctet[2]] - X[ctet[3]];
  vol[t] = Determinant(Ds) / 6.;
}
};  // namespace Rain::CUDA