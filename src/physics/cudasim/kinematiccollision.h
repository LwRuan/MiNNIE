#pragma once

#include "cudahelper.h"
#include "cudamath.h"
#include "mathtype.h"

namespace Rain {
namespace CUDA {

__global__ void KinematicCollisionSphereReordered(
    const Vec3* X, real* grad, real* A_diag, int32_t* h2m, const real k_penalty,
    const Vec3 center, const real r, const int32_t n_vert);

__global__ void KinematicCollisionSphere(const Vec3* X, real* grad,
                                         real* A_diag, const real k_penalty,
                                         const Vec3 center, const real r,
                                         const int32_t n_vert);

__global__ void KinematicCollisionSphereEnergy(const Vec3* X, real* E,
                                               const real k_penalty,
                                               const Vec3 center, const real r,
                                               const int32_t n_vert);

__global__ void KinematicCollisionSphereMixed(
    const Vec3* X, real* grad, real* A_diag, const real k_penalty,
    const Vec3 center, const real r, const int32_t sign, const int32_t n_vert);

__global__ void KinematicCollisionTorusReordered(
    const Vec3* X, real* grad, real* A_diag, int32_t* h2m, const real k_penalty,
    const Vec3 center, const Vec3 dir, const real a, const real r,
    const int32_t n_vert);

__global__ void KinematicCollisionTorus(const Vec3* X, real* grad, real* A_diag,
                                        const real k_penalty, const Vec3 center,
                                        const Vec3 dir, const real a,
                                        const real r, const int32_t n_vert);

__global__ void KinematicCollisionTorusEnergy(const Vec3* X, real* E,
                                              const real k_penalty,
                                              const Vec3 center, const Vec3 dir,
                                              const real a, const real r,
                                              const int32_t n_vert);

__global__ void KinematicCollisionTorusMixed(const Vec3* X, real* grad,
                                             real* A_diag, const real k_penalty,
                                             const Vec3 center, const Vec3 dir,
                                             const real a, const real r,
                                             const int32_t n_vert);

__global__ void KinematicCollisionCylinderReordered(
    const Vec3* X, real* grad, real* A_diag, int32_t* h2m, const real k_penalty,
    const Vec3 center, const Vec3 dir, const real r, const real h,
    const int32_t n_vert);

__global__ void KinematicCollisionCylinder(const Vec3* X, real* grad,
                                           real* A_diag, const real k_penalty,
                                           const Vec3 center, const Vec3 dir,
                                           const real r, const real h,
                                           const int32_t n_vert);

__global__ void KinematicCollisionCylinderEnergy(
    const Vec3* X, real* E, const real k_penalty, const Vec3 center,
    const Vec3 dir, const real r, const real h, const int32_t n_vert);

__global__ void KinematicCollisionCylinderMixed(
    const Vec3* X, real* grad, real* A_diag, const real k_penalty,
    const Vec3 center, const Vec3 dir, const real r, const real h,
    const int32_t n_vert);

__global__ void KinematicCollisionPlaneReordered(
    const Vec3* X, real* grad, real* A_diag, int32_t* h2m, const real k_penalty,
    const Vec3 center, const Vec3 norm, const int32_t n_vert);

__global__ void KinematicCollisionPlane(const Vec3* X, real* grad, real* A_diag,
                                        const real k_penalty, const Vec3 center,
                                        const Vec3 norm, const int32_t n_vert);

__global__ void KinematicCollisionPlaneEnergy(const Vec3* X, real* E,
                                              const real k_penalty,
                                              const Vec3 center,
                                              const Vec3 norm,
                                              const int32_t n_vert);

__global__ void KinematicCollisionPlaneMixed(const Vec3* X, real* grad,
                                             real* A_diag, const real k_penalty,
                                             const Vec3 center, const Vec3 norm,
                                             const int32_t n_vert);
};  // namespace CUDA
};  // namespace Rain