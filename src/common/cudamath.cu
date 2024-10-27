#include "cudamath.h"

namespace Rain::CUDA {
__device__ void Cross(const Vec3& a, const Vec3& b, Vec3& out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ Vec3 Cross(const Vec3& a, const Vec3& b) {
  return Vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
              a[0] * b[1] - a[1] * b[0]);
}

__device__ void Solve3x3Sym(const real* A, const real* A_a, const real* b,
                            real* x) {
  // A + A_a is symmetric
  // a0 a1 a2
  // a3 a4 a5
  // a6 a7 a8
  real tA[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) tA[i] = A[i] + A_a[i];
  real det = tA[0] * (tA[4] * tA[8] - tA[5] * tA[7]) +
             tA[1] * (tA[5] * tA[6] - tA[3] * tA[8]) +
             tA[2] * (tA[3] * tA[7] - tA[4] * tA[6]);
  real inv_det = 1 / det;
  real adjA[6];
  // upper right part
  // a0 a1 a2
  // a1 a3 a4
  // a2 a4 a5
  adjA[0] = (tA[4] * tA[8] - tA[5] * tA[7]);
  adjA[1] = -(tA[1] * tA[8] - tA[2] * tA[7]);
  adjA[2] = (tA[1] * tA[5] - tA[2] * tA[4]);
  adjA[3] = (tA[0] * tA[8] - tA[2] * tA[6]);
  adjA[4] = -(tA[0] * tA[7] - tA[1] * tA[6]);
  adjA[5] = (tA[0] * tA[4] - tA[1] * tA[3]);
  x[0] = (adjA[0] * b[0] + adjA[1] * b[1] + adjA[2] * b[2]) * inv_det;
  x[1] = (adjA[1] * b[0] + adjA[3] * b[1] + adjA[4] * b[2]) * inv_det;
  x[2] = (adjA[2] * b[0] + adjA[4] * b[1] + adjA[5] * b[2]) * inv_det;
}

__device__ void Solve3x3Sym(const real* A, const real* A_a, const real* b,
                            const int32_t lda, real* x) {
  real tA[9];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      tA[i * 3 + j] = (A[i * lda + j] + A_a[i * lda + j]);
    }
  }
  real det = tA[0] * (tA[4] * tA[8] - tA[5] * tA[7]) +
             tA[1] * (tA[5] * tA[6] - tA[3] * tA[8]) +
             tA[2] * (tA[3] * tA[7] - tA[4] * tA[6]);
  real inv_det = 1 / det;
  real adjA[6];
  // upper right part
  // a0 a1 a2
  // a1 a3 a4
  // a2 a4 a5
  adjA[0] = (tA[4] * tA[8] - tA[5] * tA[7]);
  adjA[1] = -(tA[1] * tA[8] - tA[2] * tA[7]);
  adjA[2] = (tA[1] * tA[5] - tA[2] * tA[4]);
  adjA[3] = (tA[0] * tA[8] - tA[2] * tA[6]);
  adjA[4] = -(tA[0] * tA[7] - tA[1] * tA[6]);
  adjA[5] = (tA[0] * tA[4] - tA[1] * tA[3]);
  x[0] = (adjA[0] * b[0] + adjA[1] * b[1] + adjA[2] * b[2]) * inv_det;
  x[1] = (adjA[1] * b[0] + adjA[3] * b[1] + adjA[4] * b[2]) * inv_det;
  x[2] = (adjA[2] * b[0] + adjA[4] * b[1] + adjA[5] * b[2]) * inv_det;
}

__device__ void Solve3x3Sym(const real* A, const real* b, const int32_t lda,
                            real* x) {
  real tA[9];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      tA[i * 3 + j] = A[i * lda + j];
    }
  }
  real det = tA[0] * (tA[4] * tA[8] - tA[5] * tA[7]) +
             tA[1] * (tA[5] * tA[6] - tA[3] * tA[8]) +
             tA[2] * (tA[3] * tA[7] - tA[4] * tA[6]);
  real inv_det = 1 / det;
  real adjA[6];
  // upper right part
  // a0 a1 a2
  // a1 a3 a4
  // a2 a4 a5
  adjA[0] = (tA[4] * tA[8] - tA[5] * tA[7]);
  adjA[1] = -(tA[1] * tA[8] - tA[2] * tA[7]);
  adjA[2] = (tA[1] * tA[5] - tA[2] * tA[4]);
  adjA[3] = (tA[0] * tA[8] - tA[2] * tA[6]);
  adjA[4] = -(tA[0] * tA[7] - tA[1] * tA[6]);
  adjA[5] = (tA[0] * tA[4] - tA[1] * tA[3]);
  x[0] = (adjA[0] * b[0] + adjA[1] * b[1] + adjA[2] * b[2]) * inv_det;
  x[1] = (adjA[1] * b[0] + adjA[3] * b[1] + adjA[4] * b[2]) * inv_det;
  x[2] = (adjA[2] * b[0] + adjA[4] * b[1] + adjA[5] * b[2]) * inv_det;
}

__device__ void Solve4x4Sym(const real* A, const real* A_a, const real* b,
                            real* x) {
  // A + A_a is symmetric
  // a0  a1  a2  a3
  // a4  a5  a6  a7
  // a8  a9  a10 a11
  // a12 a13 a14 a15
  real tA[16];
  real fac = 1.;
  if ((A[0] + A_a[0]) > 1e7) fac = 1e-2;
#pragma unroll
  for (int i = 0; i < 16; ++i) tA[i] = (A[i] + A_a[i]) * fac;
  real det = tA[0] * (tA[5] * (tA[10] * tA[15] - tA[11] * tA[14]) -
                      tA[6] * (tA[9] * tA[15] - tA[11] * tA[13]) +
                      tA[7] * (tA[9] * tA[14] - tA[10] * tA[13])) -
             tA[1] * (tA[4] * (tA[10] * tA[15] - tA[11] * tA[14]) -
                      tA[6] * (tA[8] * tA[15] - tA[11] * tA[12]) +
                      tA[7] * (tA[8] * tA[14] - tA[10] * tA[12])) +
             tA[2] * (tA[4] * (tA[9] * tA[15] - tA[11] * tA[13]) -
                      tA[5] * (tA[8] * tA[15] - tA[11] * tA[12]) +
                      tA[7] * (tA[8] * tA[13] - tA[9] * tA[12])) -
             tA[3] * (tA[4] * (tA[9] * tA[14] - tA[10] * tA[13]) -
                      tA[5] * (tA[8] * tA[14] - tA[10] * tA[12]) +
                      tA[6] * (tA[8] * tA[13] - tA[9] * tA[12]));
  real inv_det = 1 / det;
  real adjA[10];
  // upper right part
  // a0 a1 a2 a3
  // a1 a4 a5 a6
  // a2 a5 a7 a8
  // a3 a6 a8 a9
  adjA[0] = (tA[5] * (tA[10] * tA[15] - tA[11] * tA[14]) -
             tA[6] * (tA[9] * tA[15] - tA[11] * tA[13]) +
             tA[7] * (tA[9] * tA[14] - tA[10] * tA[13]));
  adjA[1] = -(tA[4] * (tA[10] * tA[15] - tA[11] * tA[14]) -
              tA[6] * (tA[8] * tA[15] - tA[11] * tA[12]) +
              tA[7] * (tA[8] * tA[14] - tA[10] * tA[12]));
  adjA[2] = (tA[4] * (tA[9] * tA[15] - tA[11] * tA[13]) -
             tA[5] * (tA[8] * tA[15] - tA[11] * tA[12]) +
             tA[7] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[3] = -(tA[4] * (tA[9] * tA[14] - tA[10] * tA[13]) -
              tA[5] * (tA[8] * tA[14] - tA[10] * tA[12]) +
              tA[6] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[4] = (tA[0] * (tA[10] * tA[15] - tA[11] * tA[14]) -
             tA[2] * (tA[8] * tA[15] - tA[11] * tA[12]) +
             tA[3] * (tA[8] * tA[14] - tA[10] * tA[12]));
  adjA[5] = -(tA[0] * (tA[9] * tA[15] - tA[11] * tA[13]) -
              tA[1] * (tA[8] * tA[15] - tA[11] * tA[12]) +
              tA[3] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[6] = (tA[0] * (tA[9] * tA[14] - tA[10] * tA[13]) -
             tA[1] * (tA[8] * tA[14] - tA[10] * tA[12]) +
             tA[2] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[7] = (tA[0] * (tA[5] * tA[15] - tA[7] * tA[13]) -
             tA[1] * (tA[4] * tA[15] - tA[7] * tA[12]) +
             tA[3] * (tA[4] * tA[13] - tA[5] * tA[12]));
  adjA[8] = -(tA[0] * (tA[5] * tA[14] - tA[6] * tA[13]) -
              tA[1] * (tA[4] * tA[14] - tA[6] * tA[12]) +
              tA[2] * (tA[4] * tA[13] - tA[5] * tA[12]));
  adjA[9] = (tA[0] * (tA[5] * tA[10] - tA[6] * tA[9]) -
             tA[1] * (tA[4] * tA[10] - tA[6] * tA[8]) +
             tA[2] * (tA[4] * tA[9] - tA[5] * tA[8]));
  x[0] = fac *
         (adjA[0] * b[0] + adjA[1] * b[1] + adjA[2] * b[2] + adjA[3] * b[3]) *
         inv_det;
  x[1] = fac *
         (adjA[1] * b[0] + adjA[4] * b[1] + adjA[5] * b[2] + adjA[6] * b[3]) *
         inv_det;
  x[2] = fac *
         (adjA[2] * b[0] + adjA[5] * b[1] + adjA[7] * b[2] + adjA[8] * b[3]) *
         inv_det;
  x[3] = fac *
         (adjA[3] * b[0] + adjA[6] * b[1] + adjA[8] * b[2] + adjA[9] * b[3]) *
         inv_det;
}

__device__ void Solve4x4Sym(const real* A, const real* b, const int32_t lda,
                            real* x) {
  // A + A_a is symmetric
  // a0  a1  a2  a3
  // a4  a5  a6  a7
  // a8  a9  a10 a11
  // a12 a13 a14 a15
  real tA[16];
  real fac = 1.;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      tA[i * 4 + j] = A[i * lda + j] * fac;
    }
  }

  real det = tA[0] * (tA[5] * (tA[10] * tA[15] - tA[11] * tA[14]) -
                      tA[6] * (tA[9] * tA[15] - tA[11] * tA[13]) +
                      tA[7] * (tA[9] * tA[14] - tA[10] * tA[13])) -
             tA[1] * (tA[4] * (tA[10] * tA[15] - tA[11] * tA[14]) -
                      tA[6] * (tA[8] * tA[15] - tA[11] * tA[12]) +
                      tA[7] * (tA[8] * tA[14] - tA[10] * tA[12])) +
             tA[2] * (tA[4] * (tA[9] * tA[15] - tA[11] * tA[13]) -
                      tA[5] * (tA[8] * tA[15] - tA[11] * tA[12]) +
                      tA[7] * (tA[8] * tA[13] - tA[9] * tA[12])) -
             tA[3] * (tA[4] * (tA[9] * tA[14] - tA[10] * tA[13]) -
                      tA[5] * (tA[8] * tA[14] - tA[10] * tA[12]) +
                      tA[6] * (tA[8] * tA[13] - tA[9] * tA[12]));
  real inv_det = 1 / det;
  real adjA[10];
  // upper right part
  // a0 a1 a2 a3
  // a1 a4 a5 a6
  // a2 a5 a7 a8
  // a3 a6 a8 a9
  adjA[0] = (tA[5] * (tA[10] * tA[15] - tA[11] * tA[14]) -
             tA[6] * (tA[9] * tA[15] - tA[11] * tA[13]) +
             tA[7] * (tA[9] * tA[14] - tA[10] * tA[13]));
  adjA[1] = -(tA[4] * (tA[10] * tA[15] - tA[11] * tA[14]) -
              tA[6] * (tA[8] * tA[15] - tA[11] * tA[12]) +
              tA[7] * (tA[8] * tA[14] - tA[10] * tA[12]));
  adjA[2] = (tA[4] * (tA[9] * tA[15] - tA[11] * tA[13]) -
             tA[5] * (tA[8] * tA[15] - tA[11] * tA[12]) +
             tA[7] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[3] = -(tA[4] * (tA[9] * tA[14] - tA[10] * tA[13]) -
              tA[5] * (tA[8] * tA[14] - tA[10] * tA[12]) +
              tA[6] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[4] = (tA[0] * (tA[10] * tA[15] - tA[11] * tA[14]) -
             tA[2] * (tA[8] * tA[15] - tA[11] * tA[12]) +
             tA[3] * (tA[8] * tA[14] - tA[10] * tA[12]));
  adjA[5] = -(tA[0] * (tA[9] * tA[15] - tA[11] * tA[13]) -
              tA[1] * (tA[8] * tA[15] - tA[11] * tA[12]) +
              tA[3] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[6] = (tA[0] * (tA[9] * tA[14] - tA[10] * tA[13]) -
             tA[1] * (tA[8] * tA[14] - tA[10] * tA[12]) +
             tA[2] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[7] = (tA[0] * (tA[5] * tA[15] - tA[7] * tA[13]) -
             tA[1] * (tA[4] * tA[15] - tA[7] * tA[12]) +
             tA[3] * (tA[4] * tA[13] - tA[5] * tA[12]));
  adjA[8] = -(tA[0] * (tA[5] * tA[14] - tA[6] * tA[13]) -
              tA[1] * (tA[4] * tA[14] - tA[6] * tA[12]) +
              tA[2] * (tA[4] * tA[13] - tA[5] * tA[12]));
  adjA[9] = (tA[0] * (tA[5] * tA[10] - tA[6] * tA[9]) -
             tA[1] * (tA[4] * tA[10] - tA[6] * tA[8]) +
             tA[2] * (tA[4] * tA[9] - tA[5] * tA[8]));

  x[0] = fac *
         (adjA[0] * b[0] + adjA[1] * b[1] + adjA[2] * b[2] + adjA[3] * b[3]) *
         inv_det;
  x[1] = fac *
         (adjA[1] * b[0] + adjA[4] * b[1] + adjA[5] * b[2] + adjA[6] * b[3]) *
         inv_det;
  x[2] = fac *
         (adjA[2] * b[0] + adjA[5] * b[1] + adjA[7] * b[2] + adjA[8] * b[3]) *
         inv_det;
  x[3] = fac *
         (adjA[3] * b[0] + adjA[6] * b[1] + adjA[8] * b[2] + adjA[9] * b[3]) *
         inv_det;
}

__device__ void Solve4x4Sym(const real* A, const real* A_a, const real* b,
                            const int32_t lda, real* x) {
  // A + A_a is symmetric
  // a0  a1  a2  a3
  // a4  a5  a6  a7
  // a8  a9  a10 a11
  // a12 a13 a14 a15
  real tA[16];
  real fac = 1.;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      tA[i * 4 + j] = (A[i * lda + j] + A_a[i * lda + j]) * fac;
    }
  }

  real det = tA[0] * (tA[5] * (tA[10] * tA[15] - tA[11] * tA[14]) -
                      tA[6] * (tA[9] * tA[15] - tA[11] * tA[13]) +
                      tA[7] * (tA[9] * tA[14] - tA[10] * tA[13])) -
             tA[1] * (tA[4] * (tA[10] * tA[15] - tA[11] * tA[14]) -
                      tA[6] * (tA[8] * tA[15] - tA[11] * tA[12]) +
                      tA[7] * (tA[8] * tA[14] - tA[10] * tA[12])) +
             tA[2] * (tA[4] * (tA[9] * tA[15] - tA[11] * tA[13]) -
                      tA[5] * (tA[8] * tA[15] - tA[11] * tA[12]) +
                      tA[7] * (tA[8] * tA[13] - tA[9] * tA[12])) -
             tA[3] * (tA[4] * (tA[9] * tA[14] - tA[10] * tA[13]) -
                      tA[5] * (tA[8] * tA[14] - tA[10] * tA[12]) +
                      tA[6] * (tA[8] * tA[13] - tA[9] * tA[12]));
  real inv_det = 1 / det;
  real adjA[10];
  // upper right part
  // a0 a1 a2 a3
  // a1 a4 a5 a6
  // a2 a5 a7 a8
  // a3 a6 a8 a9
  adjA[0] = (tA[5] * (tA[10] * tA[15] - tA[11] * tA[14]) -
             tA[6] * (tA[9] * tA[15] - tA[11] * tA[13]) +
             tA[7] * (tA[9] * tA[14] - tA[10] * tA[13]));
  adjA[1] = -(tA[4] * (tA[10] * tA[15] - tA[11] * tA[14]) -
              tA[6] * (tA[8] * tA[15] - tA[11] * tA[12]) +
              tA[7] * (tA[8] * tA[14] - tA[10] * tA[12]));
  adjA[2] = (tA[4] * (tA[9] * tA[15] - tA[11] * tA[13]) -
             tA[5] * (tA[8] * tA[15] - tA[11] * tA[12]) +
             tA[7] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[3] = -(tA[4] * (tA[9] * tA[14] - tA[10] * tA[13]) -
              tA[5] * (tA[8] * tA[14] - tA[10] * tA[12]) +
              tA[6] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[4] = (tA[0] * (tA[10] * tA[15] - tA[11] * tA[14]) -
             tA[2] * (tA[8] * tA[15] - tA[11] * tA[12]) +
             tA[3] * (tA[8] * tA[14] - tA[10] * tA[12]));
  adjA[5] = -(tA[0] * (tA[9] * tA[15] - tA[11] * tA[13]) -
              tA[1] * (tA[8] * tA[15] - tA[11] * tA[12]) +
              tA[3] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[6] = (tA[0] * (tA[9] * tA[14] - tA[10] * tA[13]) -
             tA[1] * (tA[8] * tA[14] - tA[10] * tA[12]) +
             tA[2] * (tA[8] * tA[13] - tA[9] * tA[12]));
  adjA[7] = (tA[0] * (tA[5] * tA[15] - tA[7] * tA[13]) -
             tA[1] * (tA[4] * tA[15] - tA[7] * tA[12]) +
             tA[3] * (tA[4] * tA[13] - tA[5] * tA[12]));
  adjA[8] = -(tA[0] * (tA[5] * tA[14] - tA[6] * tA[13]) -
              tA[1] * (tA[4] * tA[14] - tA[6] * tA[12]) +
              tA[2] * (tA[4] * tA[13] - tA[5] * tA[12]));
  adjA[9] = (tA[0] * (tA[5] * tA[10] - tA[6] * tA[9]) -
             tA[1] * (tA[4] * tA[10] - tA[6] * tA[8]) +
             tA[2] * (tA[4] * tA[9] - tA[5] * tA[8]));

  x[0] = fac *
         (adjA[0] * b[0] + adjA[1] * b[1] + adjA[2] * b[2] + adjA[3] * b[3]) *
         inv_det;
  x[1] = fac *
         (adjA[1] * b[0] + adjA[4] * b[1] + adjA[5] * b[2] + adjA[6] * b[3]) *
         inv_det;
  x[2] = fac *
         (adjA[2] * b[0] + adjA[5] * b[1] + adjA[7] * b[2] + adjA[8] * b[3]) *
         inv_det;
  x[3] = fac *
         (adjA[3] * b[0] + adjA[6] * b[1] + adjA[8] * b[2] + adjA[9] * b[3]) *
         inv_det;
}

__device__ void Solve4x4SymMINRES(const real* A, const real* A_a, const real* b,
                                  real* x) {
  real tol = 1e-7;
  real y[4], w[4], w1[4], w2[4], v[4], r1[4], r2[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[i] = 0.;
    y[i] = b[i];
    w[i] = 0.;
    w2[i] = 0.;
    r1[i] = b[i];
    r2[i] = b[i];
  }
  real beta1 = sqrt(b[0] * y[0] + b[1] * y[1] + b[2] * y[2] + b[3] * y[3]);
  if (beta1 < tol) return;
  real oldb = 0., beta = beta1, dbar = 0., epsln = 0.;
  real phibar = beta1;
  real cs = -1., sn = 0.;

  for (int iter = 0; iter < 4; ++iter) {
    v[0] = y[0] / beta;
    v[1] = y[1] / beta;
    v[2] = y[2] / beta;
    v[3] = y[3] / beta;
    y[0] = 0.;
    y[1] = 0.;
    y[2] = 0.;
    y[3] = 0.;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      y[i / 4] += (A[i] + A_a[i]) * v[i % 4];
    }
    if (iter > 0) {
      real fac = -beta / oldb;
      y[0] += fac * r1[0];
      y[1] += fac * r1[1];
      y[2] += fac * r1[2];
      y[3] += fac * r1[3];
    }
    real alfa = v[0] * y[0] + v[1] * y[1] + v[2] * y[2] + v[3] * y[3];
    {
      real fac = -alfa / beta;
      y[0] += fac * r2[0];
      y[1] += fac * r2[1];
      y[2] += fac * r2[2];
      y[3] += fac * r2[3];
    }
    r1[0] = r2[0];
    r1[1] = r2[1];
    r1[2] = r2[2];
    r1[3] = r2[3];
    r2[0] = y[0];
    r2[1] = y[1];
    r2[2] = y[2];
    r2[3] = y[3];
    oldb = beta;
    beta = sqrt(r2[0] * y[0] + r2[1] * y[1] + r2[2] * y[2] + r2[3] * y[3]);

    real oldeps = epsln;
    real delta = cs * dbar + sn * alfa;
    real gbar = sn * dbar - cs * alfa;
    epsln = sn * beta;
    dbar = -cs * beta;

    real gamma = sqrt(gbar * gbar + beta * beta);
    gamma = max(gamma, 1e-15);
    cs = gbar / gamma;
    sn = beta / gamma;
    real phi = cs * phibar;
    phibar = sn * phibar;

    w1[0] = w2[0];
    w1[1] = w2[1];
    w1[2] = w2[2];
    w1[3] = w2[3];
    w2[0] = w[0];
    w2[1] = w[1];
    w2[2] = w[2];
    w2[3] = w[3];
    w[0] = (v[0] - oldeps * w1[0] - delta * w2[0]) / gamma;
    w[1] = (v[1] - oldeps * w1[1] - delta * w2[1]) / gamma;
    w[2] = (v[2] - oldeps * w1[2] - delta * w2[2]) / gamma;
    w[3] = (v[3] - oldeps * w1[3] - delta * w2[3]) / gamma;
    x[0] += phi * w[0];
    x[1] += phi * w[1];
    x[2] += phi * w[2];
    x[3] += phi * w[3];
    if (phibar / beta1 < tol) break;
  }
}

__device__ void Solve4x4SymCG(const real* A, const real* A_a, const real* b,
                              real* x) {
  // A + A_a is symmetric
  Mat4 tA;
#pragma unroll
  for (int i = 0; i < 16; ++i) tA.data()[i] = A[i] + A_a[i];
  Vec4 tb;
#pragma unroll
  for (int i = 0; i < 4; ++i) tb.data()[i] = b[i];
  Vec4 tx;
  tx.setZero();
  Vec4 r = tb - tA * tx;
  Vec4 p = r;
  real rsold = r.dot(r);
  for (int i = 0; i < 4; ++i) {
    Vec4 Ap = tA * p;
    real alpha = rsold / p.dot(Ap);
    tx += alpha * p;
    r -= alpha * Ap;
    real rsnew = r.dot(r);
    p = r + (rsnew / rsold) * p;
    rsold = rsnew;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = tx.data()[i];
}

__device__ void GetRotation(const Mat3& F, Mat3& R) {
  Mat3 C = F.transpose() * F;
  Mat3 C2 = C * C;
  real det = Determinant(F);
  real I_C = C(0, 0) + C(1, 1) + C(2, 2);
  real I_C2 = I_C * I_C;
  real II_C = real(0.5) * (I_C2 - C2(0, 0) - C2(1, 1) - C2(2, 2));
  real III_C = det * det;
  real k = I_C2 - 3 * II_C;

  Mat3 U_inv = Mat3::Zero();
  if (k < real(1e-7)) {
    real lambda_inv = 1.0 / sqrt(I_C / 3);
    U_inv(0, 0) = lambda_inv;
    U_inv(1, 1) = lambda_inv;
    U_inv(2, 2) = lambda_inv;
  } else {
    real l = I_C * (I_C2 - real(4.5) * II_C) + real(13.5) * III_C;
    real k_root = sqrt(k);
    real value = l / (k * k_root);
    if (value < -1.0) value = -1.0;
    if (value > 1.0) value = 1.0;
    real phi = acos(value);
    real lambda2 = (I_C + 2 * k_root * cos(phi / 3)) / 3;
    real lambda = sqrt(lambda2);

    real III_U = sqrt(III_C);
    if (det < 0) III_U = -III_U;
    real I_U = lambda + sqrt(-lambda2 + I_C + 2 * III_U / lambda);
    real II_U = (I_U * I_U - I_C) / 2;

    real inv_rate = 1 / (I_U * II_U - III_U);
    real factor = I_U * III_U * inv_rate;
    Mat3 U = factor * Mat3::Identity();
    factor = (I_U * I_U - II_U) * inv_rate;
    U += factor * C - inv_rate * C2;

    inv_rate = 1 / III_U;
    factor = II_U * inv_rate;
    U_inv(0, 0) = factor;
    U_inv(1, 1) = factor;
    U_inv(2, 2) = factor;
    factor = -I_U * inv_rate;
    U_inv += factor * U + inv_rate * C;
  }

  R = F * U_inv;
}

__device__ real Determinant(const Mat3& A) {
  return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) -
         A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
         A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
}

__device__ Vec3 atomicAdd(Vec3& target, const Vec3& val) {
  Vec3 ret;
  ret[0] = ::atomicAdd(&target.data()[0], val[0]);
  ret[1] = ::atomicAdd(&target.data()[1], val[1]);
  ret[2] = ::atomicAdd(&target.data()[2], val[2]);
  return ret;
}

__device__ bool HasNan(const Mat3& A) {
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    if (isnan(A.data()[i])) return true;
  }
  return false;
}

__device__ bool HasNan(const Vec3& V) {
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    if (isnan(V[i])) return true;
  }
  return false;
}

__global__ void CheckNan(const real* X, const int32_t size, int32_t* info) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  if (isnan(X[i])) {
    ::atomicAdd(info, 1);
  }
}

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif
};  // namespace Rain::CUDA

namespace Rain {
int32_t CheckNan(const real* X, const int32_t size) {
  int32_t* dnan_info;
  CheckCuda(cudaMalloc(&dnan_info, sizeof(int32_t)));
  CheckCuda(cudaMemset(dnan_info, 0, sizeof(int32_t)));
  CUDA::CheckNan<<<(size + 63) / 64, 64>>>(X, size, dnan_info);
  int32_t nan_info;
  CheckCuda(cudaMemcpy(&nan_info, dnan_info, sizeof(int32_t),
                       cudaMemcpyDeviceToHost));
  CheckCuda(cudaFree(dnan_info));
  return nan_info;
}

};  // namespace Rain