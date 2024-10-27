#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include "cudahelper.h"
#include "cudamath.h"
#include "mathtype.h"

namespace Rain {
namespace CUDA {
__global__ void SelfCollisionReordered(
    Vec3* verts, const Vec3* norms, const uint32_t* tets,
    const uint32_t* colli_pairs, const int32_t* closest_surf_vert, real* grad,
    uint32_t* colli_verts, Mat3* colli_hessian, real* A_diag, int32_t* h2m,
    const real k_penalty, const int32_t n_vert, const int32_t n_colli);

__global__ void SelfCollision(Vec3* verts, const Vec3* norms,
                              const uint32_t* tets, const uint32_t* colli_pairs,
                              const int32_t* closest_surf_vert, real* grad,
                              uint32_t* colli_verts, Mat3* colli_hessian,
                              real* A_diag, const real k_penalty,
                              const int32_t n_vert, const int32_t n_colli);

// X is the reset configuration!!!
__global__ void SelfCollisionFineOffReduction(const uint32_t* colli_verts,
                                              const Mat3* colli_hessian,
                                              const Vec3* X, const int32_t* f2c,
                                              uint32_t* coarse_colli_graph,
                                              real* coarse_colli_hessians,
                                              const int32_t n_colli,
                                              const uint32_t n_handle);

__global__ void SelfCollisionFineDiagReduction(
    const uint32_t* colli_verts, const Mat3* colli_hessian, const Vec3* X,
    const int32_t* f2c, real* diag_add, const int32_t n_colli,
    const uint32_t n_handle);

// TODO
__global__ void SelfCollisionCoarseReduction();

__global__ void SelfCollisionOffDiagAPReordered(uint32_t* colli_verts,
                                                Mat3* colli_hessian, real* AP,
                                                const real alpha, const real* P,
                                                int32_t* h2m,
                                                const int32_t n_colli);

__global__ void SelfCollisionOffDiagAP(uint32_t* colli_verts,
                                       Mat3* colli_hessian, real* AP,
                                       const real alpha, const real* P,
                                       const int32_t n_colli);

__global__ void BuildSelfCollisionGraph(const uint32_t* colli_verts,
                                        int32_t* v2e, int32_t* next_edge,
                                        int32_t* edge_to,
                                        const uint32_t n_colli);

__global__ void BuildSelfCollisionGraph(const int32_t* colli_verts,
                                        int32_t* v2e, int32_t* next_edge,
                                        int32_t* edge_to,
                                        const uint32_t n_colli);

__global__ void ComputeDegrees(const uint32_t* v2e_off, const uint32_t* edge_to,
                               const int32_t* colli_v2e,
                               const int32_t* colli_next_edge,
                               const int32_t* colli_edge_to, int32_t* degrees,
                               const uint32_t n_vert);

__global__ void GetMaxDegrees(const int32_t* degrees, int32_t* max_degree,
                              const uint32_t n_vert);

__global__ void InitRandState(curandState* states, const int32_t n);

__global__ void VivacePass1(int32_t* psize, const int32_t* degree,
                            const int32_t shrink, const int32_t maxp,
                            const int32_t minp, const int32_t n);

__global__ void VivacePass2(int32_t* color, curandState* states,
                            const int32_t* has_color, const bool* palette,
                            const int32_t* psize, const int32_t maxp,
                            const int32_t n);

__global__ void VivacePass3(int32_t* has_color, bool* palette,
                            const int32_t* color, const int32_t* v2e_off,
                            const int32_t* edge_to, const int32_t maxp,
                            const int32_t n);

__global__ void VivacePass3WithSelfCollision(
    int32_t* has_color, bool* palette, const int32_t* color,
    const int32_t* v2e_off, const int32_t* edge_to, const int32_t* c_v2e,
    const int32_t* c_next_edge, const int32_t* c_edge_to, const int32_t maxp,
    const int32_t n);

__global__ void VivacePass4(bool* palette, int32_t* psize,
                            const int32_t* has_color, const int32_t maxp,
                            const int32_t n);

__global__ void VivacePassStuck(bool* palette, int32_t* psize,
                                const int32_t* has_color, const int32_t maxp,
                                const int32_t n);

};  // namespace CUDA
};  // namespace Rain