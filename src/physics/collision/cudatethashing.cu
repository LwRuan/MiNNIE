#include <spdlog/spdlog.h>

#include <fstream>

#include "cudamath.h"
#include "cudatethashing.h"

namespace Rain {
static const int red_block_size = 512;
static const int sweep_block_size = 64;
namespace CUDA {

static struct VecMin {
  __device__ __host__ Vec3 operator()(const Vec3& a, const Vec3& b) const {
    return Vec3(min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]));
  }
};

static struct VecMax {
  __device__ __host__ Vec3 operator()(const Vec3& a, const Vec3& b) const {
    return Vec3(max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]));
  }
};

static __device__ inline uint32_t HashMap(uint32_t x, uint32_t y, uint32_t z,
                                          uint32_t n) {
  return ((x * 73856093u) ^ (y * 19349663u) ^ (z * 83492791u)) % n;
}

static __device__ inline bool SameSide(const Vec3& v1, const Vec3& v2,
                                       const Vec3& v3, const Vec3& p1,
                                       const Vec3& p2) {
  Vec3 n = Cross(v2 - v1, v3 - v1);
  real d1 = n.dot(p1 - v1);
  real d2 = n.dot(p2 - v1);
  return (std::signbit(d1) == std::signbit(d2));
}

static __device__ inline bool VertInTet(const Vec3& p, const Vec3& v1,
                                        const Vec3& v2, const Vec3& v3,
                                        const Vec3& v4) {
  return SameSide(v1, v2, v3, v4, p) && SameSide(v2, v3, v4, v1, p) &&
         SameSide(v3, v4, v1, v2, p) && SameSide(v4, v1, v2, v3, p);
}

__global__ void UpdateAABB(const Vec3* verts, const uint32_t* tets,
                           Vec3* AABB_min_, Vec3* AABB_max_, real* AABB_size_,
                           uint32_t n_tet, uint32_t n_vert) {
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  Vec3 bbox_min = 1e16 * Vec3::Ones();
  Vec3 bbox_max = -1e16 * Vec3::Ones();
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t idx = tets[4 * t + i];
    bbox_min = bbox_min.cwiseMin(verts[idx]);
    bbox_max = bbox_max.cwiseMax(verts[idx]);
  }
  AABB_min_[t] = bbox_min;
  AABB_max_[t] = bbox_max;
  AABB_size_[t] = (bbox_max - bbox_min).maxCoeff();
}

__global__ void GetSceneAABB(const Vec3* AABBs, Vec3* grid_min, Vec3* grid_max,
                             real* max_ele_size, uint32_t n_tet) {
  int idx = threadIdx.x;
  Vec3 bbox_min = 1e16 * Vec3::Ones();
  Vec3 bbox_max = -1e16 * Vec3::Ones();
  real max_size = 0;

  for (int i = idx; i < n_tet; i += red_block_size) {
    bbox_min = bbox_min.cwiseMin(AABBs[2 * i + 0]);
    bbox_max = bbox_max.cwiseMax(AABBs[2 * i + 1]);
    Vec3 delta = AABBs[2 * i + 1] - AABBs[2 * i + 0];
    max_size = max(delta.maxCoeff(), max_size);
  }
  __shared__ Vec3 red_min[red_block_size];
  __shared__ Vec3 red_max[red_block_size];
  __shared__ real red_ele_size[red_block_size];
  red_min[idx] = bbox_min;
  red_max[idx] = bbox_max;
  red_ele_size[idx] = max_size;
  __syncthreads();
  for (int size = red_block_size / 2; size > 0; size /= 2) {  // uniform
    if (idx < size) {
      red_min[idx] = red_min[idx].cwiseMin(red_min[idx + size]);
      red_max[idx] = red_max[idx].cwiseMax(red_max[idx + size]);
      red_ele_size[idx] = max(red_ele_size[idx], red_ele_size[idx + size]);
    }
    __syncthreads();
  }

  if (idx == 0) {
    *grid_min = red_min[0];
    *grid_max = red_max[0];
    if (max_ele_size) *max_ele_size = red_ele_size[0];
  }
}

__global__ void FillHash(const Vec3* verts, const Vec3* grid_min,
                         const Vec3* grid_max, const real* grid_size,
                         uint32_t* hashing, uint32_t n_hash, uint32_t n_vert) {
  int v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  // if (verts[v].x() < grid_min->x() || verts[v].x() > grid_max->x() ||
  //     verts[v].y() < grid_min->y() || verts[v].y() > grid_max->y() ||
  //     verts[v].z() < grid_min->z() || verts[v].z() > grid_max->z()) {
  //   ::atomicAdd(&hashing[0], 1);
  //   return;
  // }
  Vec3i cell = ((verts[v] - *grid_min) / (*grid_size)).cast<int>();
  uint32_t key = HashMap(cell.x(), cell.y(), cell.z(), n_hash);
  // if (key == 0) key = 1;
  ::atomicAdd(&hashing[key], 1);
}

__global__ void PrefixSumSweepOne(uint32_t* arr, uint32_t size) {
  int t = threadIdx.x;
  int range = (size + sweep_block_size - 1) / sweep_block_size;
  int offset = t * range;
  for (int i = offset + 1; i < min(size, offset + range); ++i) {
    arr[i] += arr[i - 1];
  }
}

__global__ void PrefixSumSweepTwo(uint32_t* arr, uint32_t size) {
  int range = (size + sweep_block_size - 1) / sweep_block_size;
  for (int i = 2 * range - 1; i < size; i += range) {
    arr[i] += arr[i - range];
  }
}

__global__ void PrefixSumSweepThree(uint32_t* arr, uint32_t size) {
  int t = threadIdx.x;
  int range = (size + sweep_block_size - 1) / sweep_block_size;
  if (t == 0) return;
  int offset = t * range;
  for (int i = offset; i < min(size, offset + range - 1); ++i) {
    arr[i] += arr[offset - 1];
  }
}

__global__ void FillHashIds(const Vec3* verts, const Vec3* grid_min,
                            const Vec3* grid_max, const real* grid_size,
                            uint32_t* hashing, uint32_t* vert_ids,
                            uint32_t n_vert, uint32_t n_hash) {
  int v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v >= n_vert) return;
  // if (verts[v].x() < grid_min->x() || verts[v].x() > grid_max->x() ||
  //     verts[v].y() < grid_min->y() || verts[v].y() > grid_max->y() ||
  //     verts[v].z() < grid_min->z() || verts[v].z() > grid_max->z()) {
  //   int slot = atomicSub(&hashing[0], 1);
  //   vert_ids[slot - 1] = v;
  //   return;
  // }
  Vec3i cell = ((verts[v] - *grid_min) / (*grid_size)).cast<int>();
  uint32_t key = HashMap(cell.x(), cell.y(), cell.z(), n_hash);
  // if (key == 0) key = 1;
  int slot = atomicSub(&hashing[key], 1);
  vert_ids[slot - 1] = v;
}

__global__ void VertTetCollision(
    const Vec3* verts, const uint32_t* tets, const Vec3* AABB_min,
    const Vec3* AABB_max, const uint32_t* hashing, const uint32_t* vert_ids,
    const bool* tet_sign, const bool* is_vert_surf, const Vec3* grid_min,
    const Vec3* grid_max, const real* grid_size, uint32_t* colli_pairs,
    uint32_t* n_colli, uint32_t n_vert, uint32_t n_tet, uint32_t n_hash) {
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n_tet) return;
  const uint32_t* tet = &tets[4 * t];
  Vec3i low_cell = ((AABB_min[t] - *grid_min) / (*grid_size)).cast<int>();
  Vec3i up_cell = ((AABB_max[t] - *grid_min) / (*grid_size)).cast<int>();
  // Vec3i lengrid = ((*grid_max - *grid_min) / (*grid_size)).cast<int>();
  // if (low_cell.x() < 0) low_cell.x() = 0;
  // if (low_cell.y() < 0) low_cell.y() = 0;
  // if (low_cell.z() < 0) low_cell.z() = 0;
  // if (up_cell.x() >= lengrid.x()) up_cell.x() = lengrid.x() - 1;
  // if (up_cell.y() >= lengrid.y()) up_cell.y() = lengrid.y() - 1;
  // if (up_cell.z() >= lengrid.z()) up_cell.z() = lengrid.z() - 1;

  if (!(is_vert_surf[tet[0]] | is_vert_surf[tet[1]] | is_vert_surf[tet[2]] |
        is_vert_surf[tet[3]]))
    return;

  // check inversion
  // Vec3 dv0 = verts[tet[0]] - verts[tet[3]];
  // Vec3 dv1 = verts[tet[1]] - verts[tet[3]];
  // Vec3 dv2 = verts[tet[2]] - verts[tet[3]];
  // real volume = dv0.dot(Cross(dv1, dv2));
  // if (volume * tet_sign[t] < 0) return;

  uint32_t n_unique_vert = 0;
  uint32_t unique_verts[400];
  for (uint32_t idx = low_cell(0); idx <= up_cell(0); ++idx) {
    for (uint32_t idy = low_cell(1); idy <= up_cell(1); ++idy) {
      for (uint32_t idz = low_cell(2); idz <= up_cell(2); ++idz) {
        uint32_t key = HashMap(idx, idy, idz, n_hash);
        // if (key == 0) key = 1;
        uint32_t b = hashing[key];
        uint32_t e = hashing[key + 1];
        for (uint32_t i = b; i < e; ++i) {
          uint32_t v = vert_ids[i];
          // if (!is_vert_surf[v]) continue;
          if ((v != tet[0]) && (v != tet[1]) && (v != tet[2]) &&
              (v != tet[3]) &&
              (VertInTet(verts[v], verts[tet[0]], verts[tet[1]], verts[tet[2]],
                         verts[tet[3]]))) {
            bool flag = false;
            for (int k = 0; k < n_unique_vert; ++k) {
              if (unique_verts[k] == v) flag = true;
            }
            if (!flag) {
              unique_verts[n_unique_vert++] = v;
              uint32_t slot = ::atomicAdd(n_colli, 1);
              colli_pairs[slot * 2 + 0] = v;
              colli_pairs[slot * 2 + 1] = t;
            }
          }
        }
      }
    }
  }
  if (n_unique_vert >= 200) printf("error at VertTetCollision\n");
}
};  // namespace CUDA

void CudaTetHashing::Init(Vec3* dverts, bool* tet_sign, bool* is_vert_surf,
                          uint32_t* dtets, uint32_t n_vert, uint32_t n_tet,
                          uint32_t n_hash, uint32_t max_n_colli) {
  dverts_ = dverts;
  dtets_ = dtets;
  dtet_sign_ = tet_sign;
  dis_vert_surf_ = is_vert_surf;
  n_vert_ = n_vert;
  n_tet_ = n_tet;
  n_hash_ = n_hash;
  max_n_colli_ = max_n_colli;

  CheckCuda(cudaMalloc((void**)&dhashing_, sizeof(uint32_t) * (n_hash + 1)));
  CheckCuda(cudaMalloc((void**)&dvert_ids_, sizeof(uint32_t) * n_vert));
  CheckCuda(cudaMalloc(&dAABB_max_, sizeof(Vec3) * n_tet_));
  CheckCuda(cudaMalloc(&dAABB_min_, sizeof(Vec3) * n_tet_));
  CheckCuda(cudaMalloc(&dAABB_size_, sizeof(real) * n_tet_));
  CheckCuda(cudaMalloc((void**)&dn_colli_, sizeof(uint32_t)));
  CheckCuda(
      cudaMalloc((void**)&dcolli_pairs_, sizeof(uint32_t) * max_n_colli_ * 2));

  vert_blocks_ =
      (n_vert_ + vert_threads_per_block_ - 1) / vert_threads_per_block_;
  tet_blocks_ = (n_tet_ + tet_threads_per_block_ - 1) / tet_threads_per_block_;

  CUDA::UpdateAABB<<<tet_blocks_, tet_threads_per_block_>>>(
      dverts_, dtets_, dAABB_min_, dAABB_max_, dAABB_size_, n_tet_, n_vert_);
  CheckCuda(cudaMalloc((void**)&dgrid_min_, sizeof(Vec3)));
  CheckCuda(cudaMalloc((void**)&dgrid_max_, sizeof(Vec3)));
  CheckCuda(cudaMalloc((void**)&dgrid_size_, sizeof(real)));

  thrust::device_ptr<Vec3> AABB_min_ptr(dAABB_min_);
  thrust::device_ptr<Vec3> AABB_max_ptr(dAABB_max_);
  thrust::device_ptr<real> AABB_size_ptr(dAABB_size_);

  CUDA::VecMin min_op;
  CUDA::VecMax max_op;
  Vec3 tmin = 1e16 * Vec3::Ones();
  Vec3 tmax = -1e16 * Vec3::Ones();
  grid_min_ = thrust::reduce(AABB_min_ptr, AABB_min_ptr + n_tet_, tmin, min_op);
  grid_max_ = thrust::reduce(AABB_max_ptr, AABB_max_ptr + n_tet_, tmax, max_op);
  grid_size_ = thrust::reduce(AABB_size_ptr, AABB_size_ptr + n_tet_, 0.0f,
                              thrust::maximum<real>());

  grid_size_ = grid_size_ / 2;
  int32_t min_grid_res = 64;
  int32_t max_grid_res = 256;
  real min_grid_size = (grid_max_ - grid_min_).maxCoeff() / max_grid_res;
  real max_grid_size = (grid_max_ - grid_min_).minCoeff() / min_grid_res;
  if (grid_size_ < min_grid_size) grid_size_ = min_grid_size;
  if (grid_size_ > max_grid_size) grid_size_ = max_grid_size;
  CheckCuda(cudaMemcpy(dgrid_size_, &grid_size_, sizeof(real),
                       cudaMemcpyHostToDevice));
  // std::cout << grid_min_.transpose() << ", " << grid_max_.transpose() << ", "
  //           << grid_size_ << std::endl;
  Vec3i res = ((grid_max_ - grid_min_) / grid_size_).cast<int>();
  spdlog::info("hashing grid dims: {} {} {}", res(0), res(1), res(2));
}

void CudaTetHashing::Destroy() {
  if (dhashing_) CheckCuda(cudaFree(dhashing_));
  if (dvert_ids_) CheckCuda(cudaFree(dvert_ids_));
  if (dAABB_min_) CheckCuda(cudaFree(dAABB_min_));
  if (dAABB_max_) CheckCuda(cudaFree(dAABB_max_));
  if (dAABB_size_) CheckCuda(cudaFree(dAABB_size_));
  if (dgrid_min_) CheckCuda(cudaFree(dgrid_min_));
  if (dgrid_max_) CheckCuda(cudaFree(dgrid_max_));
  if (dgrid_size_) CheckCuda(cudaFree(dgrid_size_));
  if (dn_colli_) CheckCuda(cudaFree(dn_colli_));
  if (dcolli_pairs_) CheckCuda(cudaFree(dcolli_pairs_));
}

static void PrefixSum(cudaStream_t stream, uint32_t* arr, uint32_t size) {
  CUDA::PrefixSumSweepOne<<<1, sweep_block_size, 0, stream>>>(arr, size);
  CUDA::PrefixSumSweepTwo<<<1, 1, 0, stream>>>(arr, size);
  CUDA::PrefixSumSweepThree<<<1, sweep_block_size, 0, stream>>>(arr, size);
}

void CudaTetHashing::Hashing(cudaStream_t stream) {
  // setup grid
  CUDA::UpdateAABB<<<tet_blocks_, tet_threads_per_block_, 0, stream>>>(
      dverts_, dtets_, dAABB_min_, dAABB_max_, dAABB_size_, n_tet_, n_vert_);

  thrust::device_ptr<Vec3> AABB_min_ptr(dAABB_min_);
  thrust::device_ptr<Vec3> AABB_max_ptr(dAABB_max_);
  CUDA::VecMin min_op;
  CUDA::VecMax max_op;
  Vec3 tmin = 1e16 * Vec3::Ones();
  Vec3 tmax = -1e16 * Vec3::Ones();
  grid_min_ = thrust::reduce(AABB_min_ptr, AABB_min_ptr + n_tet_, tmin, min_op);
  grid_max_ = thrust::reduce(AABB_max_ptr, AABB_max_ptr + n_tet_, tmax, max_op);
  ////////////////////////
  // for the letter example
  // grid_min_ = Vec3(-10, 0, -10);
  // grid_max_ = Vec3(10, 20, 10);
  // grid_size_ = 0.2;
  // CheckCuda(cudaMemcpy(dgrid_size_, &grid_size_, sizeof(real),
  //                      cudaMemcpyHostToDevice));
  ////////////////////////
  CheckCuda(
      cudaMemcpy(dgrid_min_, &grid_min_, sizeof(Vec3), cudaMemcpyHostToDevice));
  CheckCuda(
      cudaMemcpy(dgrid_max_, &grid_max_, sizeof(Vec3), cudaMemcpyHostToDevice));

  Vec3 bbox_size = grid_max_ - grid_min_;
  grid_min_ -= 0.01 * bbox_size;
  grid_max_ += 0.01 * bbox_size;
  bbox_size = grid_max_ - grid_min_;

  CheckCuda(cudaMemset(dhashing_, 0, sizeof(uint32_t) * (n_hash_ + 1)));
  CUDA::FillHash<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
      dverts_, dgrid_min_, dgrid_max_, dgrid_size_, dhashing_, n_hash_,
      n_vert_);
  // PrefixSum(stream, dhashing_, n_hash_ + 1);
  thrust::device_ptr<uint32_t> hashing_ptr(dhashing_);
  thrust::inclusive_scan(hashing_ptr, hashing_ptr + n_hash_ + 1, hashing_ptr);

  CUDA::FillHashIds<<<vert_blocks_, vert_threads_per_block_, 0, stream>>>(
      dverts_, dgrid_min_, dgrid_max_, dgrid_size_, dhashing_, dvert_ids_,
      n_vert_, n_hash_);
  // {
  //   std::ofstream file("ids.txt");
  //   uint32_t* tmp = new uint32_t[n_vert_];
  //   CheckCuda(cudaMemcpy(tmp, dvert_ids_, sizeof(uint32_t) * n_vert_,
  //                        cudaMemcpyDeviceToHost));
  //   for (int i = 0; i < n_vert_; ++i) {
  //     file << tmp[i] << " ";
  //   }
  //   file << std::endl;
  //   delete[] tmp;
  // }
  // {
  //   std::ofstream file("hash.txt");
  //   uint32_t* tmp = new uint32_t[n_hash_ + 1];
  //   CheckCuda(cudaMemcpy(tmp, dhashing_, sizeof(uint32_t) * (n_hash_ + 1),
  //                        cudaMemcpyDeviceToHost));
  //   for (int i = 0; i < n_hash_ + 1; ++i) {
  //     file << tmp[i] << " ";
  //   }
  //   file << std::endl;
  //   delete[] tmp;
  // }
}

void CudaTetHashing::GetVertTetCollisionList(cudaStream_t stream) {
  CheckCuda(cudaMemset(dn_colli_, 0, sizeof(uint32_t)));
  CUDA::VertTetCollision<<<tet_blocks_, tet_threads_per_block_, 0, stream>>>(
      dverts_, dtets_, dAABB_min_, dAABB_max_, dhashing_, dvert_ids_,
      dtet_sign_, dis_vert_surf_, dgrid_min_, dgrid_max_, dgrid_size_,
      dcolli_pairs_, dn_colli_, n_vert_, n_tet_, n_hash_);
  CheckCuda(cudaMemcpy(&n_colli_, dn_colli_, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  // {
  //   uint32_t* tmp = new uint32_t[n_colli_ * 2];
  //   CheckCuda(cudaMemcpy(tmp, dcolli_pairs_, sizeof(uint32_t) * n_colli_ * 2,
  //                        cudaMemcpyDeviceToHost));
  //   for (int i = 0; i < n_colli_ * 2; ++i) {
  //     std::cout << tmp[i] << " ";
  //   }
  //   std::cout << std::endl;
  //   delete[] tmp;
  // }
}
};  // namespace Rain