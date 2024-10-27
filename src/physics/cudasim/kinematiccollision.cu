#include "kinematiccollision.h"

namespace Rain::CUDA {
__global__ void KinematicCollisionSphereReordered(
    const Vec3* X, real* grad, real* A_diag, int32_t* h2m, const real k_penalty,
    const Vec3 center, const real r, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  int32_t i = h2m[v];
  const Vec3& pos = X[v];
  Vec3 d = pos - center;
  real dist = d.norm();
  if (dist < r) {  // collision
    d = d / dist;
    Vec3 f = k_penalty * (r - dist) * d;
    grad[3 * i + 0] += f.x();
    grad[3 * i + 1] += f.y();
    grad[3 * i + 2] += f.z();
    Mat3 h = k_penalty * d * d.transpose();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[9 * i + a] += h.data()[a];
    }
  }
}

__global__ void KinematicCollisionSphere(const Vec3* X, real* grad,
                                         real* A_diag, const real k_penalty,
                                         const Vec3 center, const real r,
                                         const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 d = pos - center;
  real dist = d.norm();
  if (dist < r) {  // collision
    d = d / dist;
    Vec3 f = k_penalty * (r - dist) * d;
    grad[3 * v + 0] += f.x();
    grad[3 * v + 1] += f.y();
    grad[3 * v + 2] += f.z();
    Mat3 h = k_penalty * d * d.transpose();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[9 * v + a] += h.data()[a];
    }
  }
}

__global__ void KinematicCollisionSphereEnergy(const Vec3* X, real* E,
                                               const real k_penalty,
                                               const Vec3 center, const real r,
                                               const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 d = pos - center;
  real dist = d.norm();
  if (dist < r) {  // collision
    real e = k_penalty * (r - dist) * (r - dist);
    ::atomicAdd(E, e);
  }
}

__global__ void KinematicCollisionSphereMixed(
    const Vec3* X, real* grad, real* A_diag, const real k_penalty,
    const Vec3 center, const real r, const int32_t sign, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 d = pos - center;
  real dist = d.norm();
  if ((r - dist) * sign > 0) {  // collision
    d = d / dist;
    Vec3 f = k_penalty * (r - dist) * d;
    grad[4 * v + 0] += f.x();
    grad[4 * v + 1] += f.y();
    grad[4 * v + 2] += f.z();
    Mat3 h = k_penalty * d * d.transpose();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[16 * v + 4 * (a / 3) + (a % 3)] += h(a / 3, a % 3);
    }
  }
}

__global__ void KinematicCollisionTorusReordered(
    const Vec3* X, real* grad, real* A_diag, int32_t* h2m, const real k_penalty,
    const Vec3 center, const Vec3 dir, const real a, const real r,
    const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  int32_t i = h2m[v];
  const Vec3& pos = X[v];
  Vec3 p = X[v] - center;
  real y = p.dot(dir);
  Vec3 dp = p - y * dir;
  real x = dp.norm();
  real dist = sqrt((x - a) * (x - a) + y * y);
  if (dist < r) {  // collision
    Vec3 d = (x - a) * dp.normalized() + y * dir;
    d = d / dist;
    Vec3 f = k_penalty * (r - dist) * d;
    grad[3 * i + 0] += f.x();
    grad[3 * i + 1] += f.y();
    grad[3 * i + 2] += f.z();
    Mat3 h = k_penalty * d * d.transpose();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[9 * i + a] += h.data()[a];
    }
  }
}

__global__ void KinematicCollisionTorus(const Vec3* X, real* grad, real* A_diag,
                                        const real k_penalty, const Vec3 center,
                                        const Vec3 dir, const real a,
                                        const real r, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 p = X[v] - center;
  real y = p.dot(dir);
  Vec3 dp = p - y * dir;
  real x = dp.norm();
  real dist = sqrt((x - a) * (x - a) + y * y);
  if (dist < r) {  // collision
    Vec3 d = (x - a) * dp.normalized() + y * dir;
    d = d / dist;
    Vec3 f = k_penalty * (r - dist) * d;
    grad[3 * v + 0] += f.x();
    grad[3 * v + 1] += f.y();
    grad[3 * v + 2] += f.z();
    Mat3 h = k_penalty * d * d.transpose();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[9 * v + a] += h.data()[a];
    }
  }
}

__global__ void KinematicCollisionTorusEnergy(const Vec3* X, real* E,
                                              const real k_penalty,
                                              const Vec3 center, const Vec3 dir,
                                              const real a, const real r,
                                              const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 p = X[v] - center;
  real y = p.dot(dir);
  Vec3 dp = p - y * dir;
  real x = dp.norm();
  real dist = sqrt((x - a) * (x - a) + y * y);
  if (dist < r) {  // collision
    real e = k_penalty * (r - dist) * (r - dist);
    ::atomicAdd(E, e);
  }
}

__global__ void KinematicCollisionTorusMixed(const Vec3* X, real* grad,
                                             real* A_diag, const real k_penalty,
                                             const Vec3 center, const Vec3 dir,
                                             const real a, const real r,
                                             const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 p = X[v] - center;
  real y = p.dot(dir);
  Vec3 dp = p - y * dir;
  real x = dp.norm();
  real dist = sqrt((x - a) * (x - a) + y * y);
  if (dist < r) {  // collision
    Vec3 d = (x - a) * dp.normalized() + y * dir;
    d = d / dist;
    Vec3 f = k_penalty * (r - dist) * d;
    grad[4 * v + 0] += f.x();
    grad[4 * v + 1] += f.y();
    grad[4 * v + 2] += f.z();
    Mat3 h = k_penalty * d * d.transpose();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[16 * v + 4 * (a / 3) + (a % 3)] += h(a / 3, a % 3);
    }
  }
}

__global__ void KinematicCollisionCylinderReordered(
    const Vec3* X, real* grad, real* A_diag, int32_t* h2m, const real k_penalty,
    const Vec3 center, const Vec3 dir, const real r, const real h,
    const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  int32_t i = h2m[v];
  const Vec3& pos = X[v];
  Vec3 pz = (pos - center).dot(dir) * dir;
  Vec3 pr = pos - center - pz;
  real dist = pr.norm();
//   if (dist < r) {
//     pr = pr / dist;
//     Vec3 f = k_penalty * (r - dist) * pr;
//     grad[3 * i + 0] += f.x();
//     grad[3 * i + 1] += f.y();
//     grad[3 * i + 2] += f.z();
//     Mat3 h = k_penalty * pr * pr.transpose();
// #pragma unroll
//     for (int a = 0; a < 9; ++a) {
//       A_diag[9 * i + a] += h.data()[a];
//     }
//   }
  real er = dist - r;
  real eh1 = pz.dot(dir) - h / 2;
  real eh2 = pz.dot(dir) + h / 2;
  if (er < 0 && eh1 < 0 && eh2 > 0) {
    Vec3 f = Vec3::Zero();
    Mat3 H = Mat3::Zero();
    if (-er < -eh1 && -er < eh2) {
      pr = pr / dist;
      f = k_penalty * (r - dist) * pr;
      H = k_penalty * pr * pr.transpose();
    } else if (-eh1 < -er && -eh1 < eh2) {
      f = -k_penalty * eh1 * dir;
      H = k_penalty * dir * dir.transpose();
    } else {
      f = -k_penalty * eh2 * dir;
      H = k_penalty * dir * dir.transpose();
    }
    grad[3 * i + 0] += f.x();
    grad[3 * i + 1] += f.y();
    grad[3 * i + 2] += f.z();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[9 * i + a] += H.data()[a];
    }
  }
}

__global__ void KinematicCollisionCylinder(const Vec3* X, real* grad,
                                           real* A_diag, const real k_penalty,
                                           const Vec3 center, const Vec3 dir,
                                           const real r, const real h,
                                           const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 pz = (pos - center).dot(dir) * dir;
  Vec3 pr = pos - center - pz;
  real dist = pr.norm();
//   if (dist < r) {
//     pr = pr / dist;
//     Vec3 f = k_penalty * (r - dist) * pr;
//     grad[3 * v + 0] += f.x();
//     grad[3 * v + 1] += f.y();
//     grad[3 * v + 2] += f.z();
//     Mat3 h = k_penalty * pr * pr.transpose();
// #pragma unroll
//     for (int a = 0; a < 9; ++a) {
//       A_diag[9 * v + a] += h.data()[a];
//     }
//   }
  real er = dist - r;
  real eh1 = pz.dot(dir) - h / 2;
  real eh2 = pz.dot(dir) + h / 2;
  if (er < 0 && eh1 < 0 && eh2 > 0) {
    Vec3 f = Vec3::Zero();
    Mat3 H = Mat3::Zero();
    if (-er < -eh1 && -er < eh2) {
      pr = pr / dist;
      f = k_penalty * (r - dist) * pr;
      H = k_penalty * pr * pr.transpose();
    } else if (-eh1 < -er && -eh1 < eh2) {
      f = -k_penalty * eh1 * dir;
      H = k_penalty * dir * dir.transpose();
    } else {
      f = -k_penalty * eh2 * dir;
      H = k_penalty * dir * dir.transpose();
    }
    grad[3 * v + 0] += f.x();
    grad[3 * v + 1] += f.y();
    grad[3 * v + 2] += f.z();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[9 * v + a] += H.data()[a];
    }
  }
}

__global__ void KinematicCollisionCylinderEnergy(
    const Vec3* X, real* E, const real k_penalty, const Vec3 center,
    const Vec3 dir, const real r, const real h, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 pz = (pos - center).dot(dir) * dir;
  Vec3 pr = pos - center - pz;
  real dist = pr.norm();
  // if (dist < r) {
  //   real e = k_penalty * (r - dist) * (r - dist);
  //   ::atomicAdd(E, e);
  // }
  real er = dist - r;
  real eh1 = pz.dot(dir) - h / 2;
  real eh2 = pz.dot(dir) + h / 2;
  if (er < 0 && eh1 < 0 && eh2 > 0) {
    if (-er < -eh1 && -er < eh2) {
      real e = k_penalty * (r - dist) * (r - dist);
      ::atomicAdd(E, e);
    } else if (-eh1 < -er && -eh1 < eh2) {
      real e = k_penalty * eh1 * eh1;
      ::atomicAdd(E, e);
    } else {
      real e = k_penalty * eh2 * eh2;
      ::atomicAdd(E, e);
    }
  }
}

__global__ void KinematicCollisionCylinderMixed(
    const Vec3* X, real* grad, real* A_diag, const real k_penalty,
    const Vec3 center, const Vec3 dir, const real r, const real h,
    const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 pz = (pos - center).dot(dir) * dir;
  Vec3 pr = pos - center - pz;
  real dist = pr.norm();
  //     if (dist < r) {
  //       pr = pr / dist;
  //       Vec3 f = k_penalty * (r - dist) * pr;
  //       grad[4 * v + 0] += f.x();
  //       grad[4 * v + 1] += f.y();
  //       grad[4 * v + 2] += f.z();
  //       Mat3 h = k_penalty * pr * pr.transpose();
  // #pragma unroll
  //       for (int a = 0; a < 9; ++a) {
  //         A_diag[16 * v + 4 * (a / 3) + (a % 3)] += h(a / 3, a % 3);
  //       }
  //     }
  real er = dist - r;
  real eh1 = pz.dot(dir) - h / 2;
  real eh2 = pz.dot(dir) + h / 2;
  if (er < 0 && eh1 < 0 && eh2 > 0) {
    Vec3 f = Vec3::Zero();
    Mat3 H = Mat3::Zero();
    if (-er < -eh1 && -er < eh2) {
      pr = pr / dist;
      f = k_penalty * (r - dist) * pr;
      H = k_penalty * pr * pr.transpose();
    } else if (-eh1 < -er && -eh1 < eh2) {
      f = -k_penalty * eh1 * dir;
      H = k_penalty * dir * dir.transpose();
    } else {
      f = -k_penalty * eh2 * dir;
      H = k_penalty * dir * dir.transpose();
    }
    grad[4 * v + 0] += f.x();
    grad[4 * v + 1] += f.y();
    grad[4 * v + 2] += f.z();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[16 * v + 4 * (a / 3) + (a % 3)] += H(a / 3, a % 3);
    }
  }
}

__global__ void KinematicCollisionPlaneReordered(
    const Vec3* X, real* grad, real* A_diag, int32_t* h2m, const real k_penalty,
    const Vec3 center, const Vec3 norm, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  int32_t i = h2m[v];
  const Vec3& pos = X[v];
  Vec3 d = pos - center;
  real dist = d.dot(norm);
  if (dist < 0) {  // collision
    Vec3 f = -k_penalty * dist * norm;
    grad[3 * i + 0] += f.x();
    grad[3 * i + 1] += f.y();
    grad[3 * i + 2] += f.z();
    Mat3 h = k_penalty * norm * norm.transpose();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[9 * i + a] += h.data()[a];
    }
  }
}

__global__ void KinematicCollisionPlane(const Vec3* X, real* grad, real* A_diag,
                                        const real k_penalty, const Vec3 center,
                                        const Vec3 norm, const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 d = pos - center;
  real dist = d.dot(norm);
  if (dist < 0) {  // collision
    Vec3 f = -k_penalty * dist * norm;
    grad[3 * v + 0] += f.x();
    grad[3 * v + 1] += f.y();
    grad[3 * v + 2] += f.z();
    Mat3 h = k_penalty * norm * norm.transpose();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[9 * v + a] += h.data()[a];
    }
  }
}

__global__ void KinematicCollisionPlaneEnergy(const Vec3* X, real* E,
                                              const real k_penalty,
                                              const Vec3 center,
                                              const Vec3 norm,
                                              const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 d = pos - center;
  real dist = d.dot(norm);
  if (dist < 0) {  // collision
    real e = k_penalty * dist * dist;
    ::atomicAdd(E, e);
  }
}

__global__ void KinematicCollisionPlaneMixed(const Vec3* X, real* grad,
                                             real* A_diag, const real k_penalty,
                                             const Vec3 center, const Vec3 norm,
                                             const int32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  const Vec3& pos = X[v];
  Vec3 d = pos - center;
  real dist = d.dot(norm);
  if (dist < 0) {  // collision
    Vec3 f = -k_penalty * dist * norm;
    grad[4 * v + 0] += f.x();
    grad[4 * v + 1] += f.y();
    grad[4 * v + 2] += f.z();
    Mat3 h = k_penalty * norm * norm.transpose();
#pragma unroll
    for (int a = 0; a < 9; ++a) {
      A_diag[16 * v + 4 * (a / 3) + (a % 3)] += h(a / 3, a % 3);
    }
  }
}
};  // namespace Rain::CUDA