#include "selfcollision.h"

namespace Rain::CUDA {
__global__ void SelfCollisionReordered(
    Vec3* verts, const Vec3* norms, const uint32_t* tets,
    const uint32_t* colli_pairs, const int32_t* closest_surf_vert, real* grad,
    uint32_t* colli_verts, Mat3* colli_hessian, real* A_diag, int32_t* h2m,
    const real k_penalty, const int32_t n_vert, const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  uint32_t v = colli_pairs[2 * c + 0];
  uint32_t t = colli_pairs[2 * c + 1];
  real min_dist = 1e16;
  uint32_t p = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    if (closest_surf_vert[4 * t + i] < 0) break;
    uint32_t vi = closest_surf_vert[4 * t + i];
    real dist = norms[vi].dot(verts[vi] - verts[v]);
    if (dist < min_dist) {
      min_dist = dist;
      p = vi;
    }
  }
  Vec3 n = norms[p];
  Mat3 nnt = n * n.transpose();

  colli_verts[2 * c + 0] = v;
  colli_verts[2 * c + 1] = p;
  colli_hessian[c] = k_penalty * nnt;

  uint32_t idv = h2m[v];
  uint32_t idp = h2m[p];
#pragma unroll
  for (uint32_t x = 0; x < 9; ++x) {
    ::atomicAdd(&A_diag[9 * idv + x], k_penalty * nnt(x / 3, x % 3));
    ::atomicAdd(&A_diag[9 * idp + x], k_penalty * nnt(x / 3, x % 3));
  }
  Vec3 fv = k_penalty * n.dot(verts[p] - verts[v]) * n;
  Vec3 fp = -k_penalty * n.dot(verts[p] - verts[v]) * n;
#pragma unroll
  for (uint32_t x = 0; x < 3; ++x) {
    ::atomicAdd(&grad[3 * idv + x], fv(x));
    ::atomicAdd(&grad[3 * idp + x], fp(x));
  }
}

__global__ void SelfCollision(Vec3* verts, const Vec3* norms,
                              const uint32_t* tets, const uint32_t* colli_pairs,
                              const int32_t* closest_surf_vert, real* grad,
                              uint32_t* colli_verts, Mat3* colli_hessian,
                              real* A_diag, const real k_penalty,
                              const int32_t n_vert, const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  uint32_t v = colli_pairs[2 * c + 0];
  uint32_t t = colli_pairs[2 * c + 1];
  real min_dist = 1e16;
  uint32_t p = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    if (closest_surf_vert[4 * t + i] < 0) break;
    uint32_t vi = closest_surf_vert[4 * t + i];
    real dist = norms[vi].dot(verts[vi] - verts[v]);
    if (dist < min_dist) {
      min_dist = dist;
      p = vi;
    }
  }
  Vec3 n = norms[p];
  Mat3 nnt = n * n.transpose();

  colli_verts[2 * c + 0] = v;
  colli_verts[2 * c + 1] = p;
  colli_hessian[c] = k_penalty * nnt;

#pragma unroll
  for (uint32_t x = 0; x < 9; ++x) {
    ::atomicAdd(&A_diag[9 * v + x], k_penalty * nnt(x / 3, x % 3));
    ::atomicAdd(&A_diag[9 * p + x], k_penalty * nnt(x / 3, x % 3));
  }
  Vec3 fv = k_penalty * n.dot(verts[p] - verts[v]) * n;
  Vec3 fp = -k_penalty * n.dot(verts[p] - verts[v]) * n;
#pragma unroll
  for (uint32_t x = 0; x < 3; ++x) {
    ::atomicAdd(&grad[3 * v + x], fv(x));
    ::atomicAdd(&grad[3 * p + x], fp(x));
  }
}

__global__ void SelfCollisionFineOffReduction(const uint32_t* colli_verts,
                                              const Mat3* colli_hessian,
                                              const Vec3* X, const int32_t* f2c,
                                              uint32_t* coarse_colli_graph,
                                              real* coarse_colli_hessians,
                                              const int32_t n_colli,
                                              const uint32_t n_handle) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  uint32_t v1 = colli_verts[2 * c + 0];
  uint32_t v2 = colli_verts[2 * c + 1];
  int32_t h1 = f2c[v1];
  int32_t h2 = f2c[v2];
  if (h1 == h2) return;
  const Mat3& hessian = colli_hessian[c];

  const Vec4& x1{X[v1](0), X[v1](1), X[v1](2), 1};
  const Vec4& x2{X[v2](0), X[v2](1), X[v2](2), 1};
  Mat4 xxt = x1 * x2.transpose();

  Mat12 H = Mat12::Zero();
#pragma unroll
  for (uint32_t i = 0; i < 3; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 3; ++j) {
      H.block<4, 4>(4 * i, 4 * j) = hessian(i, j) * xxt;
    }
  }

  ::atomicAdd(&coarse_colli_graph[h1 * n_handle + h2], 1);
  ::atomicAdd(&coarse_colli_graph[h2 * n_handle + h1], 1);

#pragma unroll
  for (uint32_t i = 0; i < 12; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 12; ++j) {
      ::atomicAdd(
          &coarse_colli_hessians[12 * n_handle * (12 * h2 + j) + 12 * h1 + i],
          -H(i, j));
      ::atomicAdd(
          &coarse_colli_hessians[12 * n_handle * (12 * h1 + i) + 12 * h2 + j],
          -H(i, j));
    }
  }
}

__global__ void SelfCollisionFineDiagReduction(
    const uint32_t* colli_verts, const Mat3* colli_hessian, const Vec3* X,
    const int32_t* f2c, real* diag_add, const int32_t n_colli,
    const uint32_t n_handle) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  uint32_t v1 = colli_verts[2 * c + 0];
  uint32_t v2 = colli_verts[2 * c + 1];
  int32_t h1 = f2c[v1];
  int32_t h2 = f2c[v2];
  if (h1 != h2) return;
  const Mat3& hessian = colli_hessian[c];

  const Vec4& x1{X[v1](0), X[v1](1), X[v1](2), 1};
  const Vec4& x2{X[v2](0), X[v2](1), X[v2](2), 1};
  Mat4 xxt = x1 * x2.transpose();

  Mat12 H = Mat12::Zero();
#pragma unroll
  for (uint32_t i = 0; i < 3; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 3; ++j) {
      H.block<4, 4>(4 * i, 4 * j) =
          hessian(i, j) * xxt + hessian(j, i) * xxt.transpose();
    }
  }

#pragma unroll
  for (uint32_t i = 0; i < 12; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 12; ++j) {
      ::atomicAdd(&diag_add[144 * h1 + 12 * i + j], -H(i, j));
    }
  }
}

__global__ void SelfCollisionOffDiagAPReordered(uint32_t* colli_verts,
                                                Mat3* colli_hessian, real* AP,
                                                const real alpha, const real* P,
                                                int32_t* h2m,
                                                const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  uint32_t v1 = colli_verts[2 * c + 0];
  uint32_t v2 = colli_verts[2 * c + 1];
  const Mat3& H = colli_hessian[c];
  uint32_t idx1 = h2m[v1];
  uint32_t idx2 = h2m[v2];

  Vec3 p1{P[3 * idx1 + 0], P[3 * idx1 + 1], P[3 * idx1 + 2]};
  Vec3 p2{P[3 * idx2 + 0], P[3 * idx2 + 1], P[3 * idx2 + 2]};

  Vec3 hp1 = H * p1;
  Vec3 hp2 = H * p2;

  ::atomicAdd(&AP[3 * idx1 + 0], -alpha * hp2[0]);
  ::atomicAdd(&AP[3 * idx1 + 1], -alpha * hp2[1]);
  ::atomicAdd(&AP[3 * idx1 + 2], -alpha * hp2[2]);

  ::atomicAdd(&AP[3 * idx2 + 0], -alpha * hp1[0]);
  ::atomicAdd(&AP[3 * idx2 + 1], -alpha * hp1[1]);
  ::atomicAdd(&AP[3 * idx2 + 2], -alpha * hp1[2]);
}

__global__ void SelfCollisionOffDiagAP(uint32_t* colli_verts,
                                       Mat3* colli_hessian, real* AP,
                                       const real alpha, const real* P,
                                       const int32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  uint32_t v1 = colli_verts[2 * c + 0];
  uint32_t v2 = colli_verts[2 * c + 1];
  const Mat3& H = colli_hessian[c];

  Vec3 p1{P[3 * v1 + 0], P[3 * v1 + 1], P[3 * v1 + 2]};
  Vec3 p2{P[3 * v2 + 0], P[3 * v2 + 1], P[3 * v2 + 2]};

  Vec3 hp1 = H * p1;
  Vec3 hp2 = H * p2;

  ::atomicAdd(&AP[3 * v1 + 0], -alpha * hp2[0]);
  ::atomicAdd(&AP[3 * v1 + 1], -alpha * hp2[1]);
  ::atomicAdd(&AP[3 * v1 + 2], -alpha * hp2[2]);

  ::atomicAdd(&AP[3 * v2 + 0], -alpha * hp1[0]);
  ::atomicAdd(&AP[3 * v2 + 1], -alpha * hp1[1]);
  ::atomicAdd(&AP[3 * v2 + 2], -alpha * hp1[2]);
}

__global__ void BuildSelfCollisionGraph(const uint32_t* colli_verts,
                                        int32_t* v2e, int32_t* next_edge,
                                        int32_t* edge_to,
                                        const uint32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  uint32_t v1 = colli_verts[2 * c + 0];
  uint32_t v2 = colli_verts[2 * c + 1];
  if (v1 == v2) return;

  next_edge[2 * c + 0] = atomicExch(&v2e[v1], 2 * c + 0);
  edge_to[2 * c + 0] = v2;
  next_edge[2 * c + 1] = atomicExch(&v2e[v2], 2 * c + 1);
  edge_to[2 * c + 1] = v1;
}

__global__ void BuildSelfCollisionGraph(const int32_t* colli_verts,
                                        int32_t* v2e, int32_t* next_edge,
                                        int32_t* edge_to,
                                        const uint32_t n_colli) {
  int32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n_colli) return;
  int32_t v1 = colli_verts[2 * c + 0];
  int32_t v2 = colli_verts[2 * c + 1];
  if (v1 == v2) return;

  next_edge[2 * c + 0] = atomicExch(&v2e[v1], 2 * c + 0);
  edge_to[2 * c + 0] = v2;
  next_edge[2 * c + 1] = atomicExch(&v2e[v2], 2 * c + 1);
  edge_to[2 * c + 1] = v1;
}

__global__ void ComputeDegrees(const uint32_t* v2e_off, const uint32_t* edge_to,
                               const int32_t* colli_v2e,
                               const int32_t* colli_next_edge,
                               const int32_t* colli_edge_to, int32_t* degrees,
                               const uint32_t n_vert) {
  int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  uint32_t d = v2e_off[v + 1] - v2e_off[v];
  int32_t e = colli_v2e[v];
  while (e != -1) {
    d += 1;
    e = colli_next_edge[e];
  }
  degrees[v] = d;
}

__global__ void GetMaxDegrees(const int32_t* degrees, int32_t* max_degree,
                              const uint32_t n_vert) {
  static const int blockSize = 1024;
  int idx = threadIdx.x;
  int m = 0;
  for (int i = idx; i < n_vert; i += blockSize) m = max(m, degrees[i]);
  __shared__ int r[blockSize];
  r[idx] = m;
  __syncthreads();
  for (int size = blockSize / 2; size > 0; size /= 2) {  // uniform
    if (idx < size) r[idx] = max(r[idx], r[idx + size]);
    __syncthreads();
  }
  if (idx == 0) *max_degree = r[0];
}

__global__ void InitRandState(curandState* states, const int32_t n) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  curand_init(114514, idx, 0, &states[idx]);
}

__global__ void VivacePass1(int32_t* psize, const int32_t* degree,
                            const int32_t shrink, const int32_t maxp,
                            const int32_t minp, const int32_t n) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  psize[idx] = degree[idx] / shrink + 1;
  psize[idx] = max(psize[idx], minp);
  // psize[idx] = maxp / 2;
}

__global__ void VivacePass2(int32_t* color, curandState* states,
                            const int32_t* has_color, const bool* palette,
                            const int32_t* psize, const int32_t maxp,
                            const int32_t n) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  if (has_color[idx]) return;
  const bool* pal = &palette[maxp * idx];
  int32_t ps = psize[idx];
  int cnt = 0;
  for (int i = 0; i < ps; ++i) {
    if (!pal[i]) ++cnt;
  }
  int32_t rn = curand(&states[idx]) % cnt;
  for (int i = 0; i < ps; ++i) {
    if (!pal[i]) {
      if (rn > 0) {
        --rn;
      } else {
        color[idx] = i;
        break;
      }
    }
  }
}

__global__ void VivacePass3(int32_t* has_color, bool* palette,
                            const int32_t* color, const int32_t* v2e_off,
                            const int32_t* edge_to, const int32_t maxp,
                            const int32_t n) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  if (has_color[idx]) return;
  int32_t c = color[idx];
  bool conflict = false;
  for (int32_t e = v2e_off[idx]; e < v2e_off[idx + 1]; ++e) {
    int32_t v = edge_to[e];
    if (color[v] == c) {
      conflict = true;
      break;
    }
  }
  if (!conflict) {
    has_color[idx] = 1;
    for (int32_t e = v2e_off[idx]; e < v2e_off[idx + 1]; ++e) {
      int32_t v = edge_to[e];
      palette[maxp * v + c] = true;
    }
  } else {
    // Hungarian heuristic
    for (int32_t e = v2e_off[idx]; e < v2e_off[idx + 1]; ++e) {
      int32_t v = edge_to[e];
      if (color[v] == c) {
        if (idx > v)
          has_color[idx] = 1;
        else
          has_color[idx] = 0;
      }
    }
    if (has_color[idx]) {
      for (int32_t e = v2e_off[idx]; e < v2e_off[idx + 1]; ++e) {
        int32_t v = edge_to[e];
        palette[maxp * v + c] = true;
      }
    }
  }
}

__global__ void VivacePass3WithSelfCollision(
    int32_t* has_color, bool* palette, const int32_t* color,
    const int32_t* v2e_off, const int32_t* edge_to, const int32_t* c_v2e,
    const int32_t* c_next_edge, const int32_t* c_edge_to, const int32_t maxp,
    const int32_t n) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  if (has_color[idx]) return;
  int32_t c = color[idx];
  bool conflict = false;
  for (int32_t e = v2e_off[idx]; e < v2e_off[idx + 1]; ++e) {
    int32_t v = edge_to[e];
    if (color[v] == c) {
      conflict = true;
      break;
    }
  }
  for (int32_t e = c_v2e[idx]; e != -1; e = c_next_edge[e]) {
    int32_t v = c_edge_to[e];
    if (color[v] == c) {
      conflict = true;
      break;
    }
  }
  if (!conflict) {
    has_color[idx] = 1;
    for (int32_t e = v2e_off[idx]; e < v2e_off[idx + 1]; ++e) {
      int32_t v = edge_to[e];
      palette[maxp * v + c] = true;
    }
    for (int32_t e = c_v2e[idx]; e != -1; e = c_next_edge[e]) {
      int32_t v = c_edge_to[e];
      palette[maxp * v + c] = true;
    }
  } else {
    // Hungarian heuristic
    for (int32_t e = v2e_off[idx]; e < v2e_off[idx + 1]; ++e) {
      int32_t v = edge_to[e];
      if (color[v] == c) {
        if (idx > v)
          has_color[idx] = 1;
        else
          has_color[idx] = 0;
      }
    }
    for (int32_t e = c_v2e[idx]; e != -1; e = c_next_edge[e]) {
      int32_t v = c_edge_to[e];
      if (color[v] == c) {
        if (idx > v)
          has_color[idx] = 1;
        else
          has_color[idx] = 0;
      }
    }
    if (has_color[idx]) {
      for (int32_t e = v2e_off[idx]; e < v2e_off[idx + 1]; ++e) {
        int32_t v = edge_to[e];
        palette[maxp * v + c] = true;
      }
      for (int32_t e = c_v2e[idx]; e != -1; e = c_next_edge[e]) {
        int32_t v = c_edge_to[e];
        palette[maxp * v + c] = true;
      }
    }
  }
}

__global__ void VivacePass4(bool* palette, int32_t* psize,
                            const int32_t* has_color, const int32_t maxp,
                            const int32_t n) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  if (has_color[idx]) return;
  bool* pal = &palette[maxp * idx];
  int32_t cnt = 0;
  for (int32_t i = 0; i < psize[idx]; ++i) {
    if (!pal[i]) ++cnt;
  }
  if (cnt == 0) {
    pal[psize[idx]] = false;
    psize[idx] += 1;
  }
}

__global__ void VivacePassStuck(bool* palette, int32_t* psize,
                                const int32_t* has_color, const int32_t maxp,
                                const int32_t n) {
  int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  if (has_color[idx]) return;
  bool* pal = &palette[maxp * idx];
  pal[psize[idx]] = false;
  psize[idx] += 1;
}
};  // namespace Rain::CUDA