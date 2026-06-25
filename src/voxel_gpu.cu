// GPU voxel-grid downsample: a sort-based version (thrust) and an
// atomic-hash version (single pass, no sort).
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <cstdio>
#include <vector>

#include "flashicp.hpp"

namespace {

// Matches voxel_key() in flashicp.hpp.
__host__ __device__ inline unsigned long long key_of(const flashicp::Point& p,
                                                     float leaf) {
  const long long B = 0x1FFFFF;
  long long ix = ((long long)floorf(p.x / leaf) + (1 << 20)) & B;
  long long iy = ((long long)floorf(p.y / leaf) + (1 << 20)) & B;
  long long iz = ((long long)floorf(p.z / leaf) + (1 << 20)) & B;
  return (unsigned long long)((ix << 42) | (iy << 21) | iz);
}

struct KeyFromPoint {
  float leaf;
  __host__ __device__ long long operator()(const flashicp::Point& p) const {
    return (long long)key_of(p, leaf);
  }
};
struct PointToSum {
  __host__ __device__ float4 operator()(const flashicp::Point& p) const {
    return make_float4(p.x, p.y, p.z, 1.0f);
  }
};
struct AddSum {
  __host__ __device__ float4 operator()(const float4& a, const float4& b) const {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
};
struct SumToCentroid {
  __host__ __device__ flashicp::Point operator()(const float4& s) const {
    return {s.x / s.w, s.y / s.w, s.z / s.w};
  }
};

constexpr unsigned long long EMPTY = 0xFFFFFFFFFFFFFFFFULL;  // keys are < 2^63

__device__ inline unsigned long long mix64(unsigned long long k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

__global__ void insert_kernel(const flashicp::Point* pts, int n, float leaf,
                              unsigned long long* keys, float* sx, float* sy,
                              float* sz, int* cnt, unsigned long long mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  flashicp::Point p = pts[i];
  unsigned long long key = key_of(p, leaf);
  unsigned long long slot = mix64(key) & mask;
  while (true) {
    unsigned long long prev = atomicCAS(&keys[slot], EMPTY, key);
    if (prev == EMPTY || prev == key) {
      atomicAdd(&sx[slot], p.x);
      atomicAdd(&sy[slot], p.y);
      atomicAdd(&sz[slot], p.z);
      atomicAdd(&cnt[slot], 1);
      return;
    }
    slot = (slot + 1) & mask;
  }
}

__global__ void compact_kernel(const float* sx, const float* sy, const float* sz,
                               const int* cnt, int cap, flashicp::Point* out,
                               int* out_n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  int c = cnt[i];
  if (c > 0) {
    int idx = atomicAdd(out_n, 1);
    out[idx] = {sx[i] / c, sy[i] / c, sz[i] / c};
  }
}

size_t next_pow2(size_t x) {
  size_t p = 1;
  while (p < x) p <<= 1;
  return p;
}

struct Scratch {
  size_t cap = 0, npts = 0;
  flashicp::Point* d_pts = nullptr;
  unsigned long long* d_keys = nullptr;
  float *d_sx = nullptr, *d_sy = nullptr, *d_sz = nullptr;
  int* d_cnt = nullptr;
  flashicp::Point* d_out = nullptr;
  int* d_outn = nullptr;
};
Scratch g;

void ensure(size_t n, size_t cap) {
  if (n > g.npts) {
    cudaFree(g.d_pts);
    cudaFree(g.d_out);
    cudaMalloc(&g.d_pts, n * sizeof(flashicp::Point));
    cudaMalloc(&g.d_out, n * sizeof(flashicp::Point));
    if (!g.d_outn) cudaMalloc(&g.d_outn, sizeof(int));
    g.npts = n;
  }
  if (cap > g.cap) {
    cudaFree(g.d_keys);
    cudaFree(g.d_sx);
    cudaFree(g.d_sy);
    cudaFree(g.d_sz);
    cudaFree(g.d_cnt);
    cudaMalloc(&g.d_keys, cap * sizeof(unsigned long long));
    cudaMalloc(&g.d_sx, cap * sizeof(float));
    cudaMalloc(&g.d_sy, cap * sizeof(float));
    cudaMalloc(&g.d_sz, cap * sizeof(float));
    cudaMalloc(&g.d_cnt, cap * sizeof(int));
    g.cap = cap;
  }
}

}  // namespace

namespace flashicp {

std::vector<Point> voxel_downsample_gpu(const std::vector<Point>& in, float leaf) {
  const size_t n = in.size();
  thrust::device_vector<Point> d_pts(in.begin(), in.end());
  thrust::device_vector<long long> d_keys(n);
  thrust::transform(d_pts.begin(), d_pts.end(), d_keys.begin(), KeyFromPoint{leaf});
  thrust::device_vector<float4> d_vals(n);
  thrust::transform(d_pts.begin(), d_pts.end(), d_vals.begin(), PointToSum{});
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());
  thrust::device_vector<long long> d_okeys(n);
  thrust::device_vector<float4> d_osums(n);
  auto end = thrust::reduce_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(),
                                   d_okeys.begin(), d_osums.begin(),
                                   thrust::equal_to<long long>(), AddSum{});
  const size_t m = end.first - d_okeys.begin();
  thrust::device_vector<Point> d_out(m);
  thrust::transform(d_osums.begin(), d_osums.begin() + m, d_out.begin(),
                    SumToCentroid{});
  std::vector<Point> out(m);
  thrust::copy(d_out.begin(), d_out.end(), out.begin());
  return out;
}

std::vector<Point> voxel_downsample_gpu_hash(const std::vector<Point>& in,
                                             float leaf) {
  const size_t n = in.size();
  const size_t cap = next_pow2(n) * 2;
  const unsigned long long mask = cap - 1;
  ensure(n, cap);

  cudaMemcpy(g.d_pts, in.data(), n * sizeof(Point), cudaMemcpyHostToDevice);
  cudaMemset(g.d_keys, 0xFF, cap * sizeof(unsigned long long));
  cudaMemset(g.d_sx, 0, cap * sizeof(float));
  cudaMemset(g.d_sy, 0, cap * sizeof(float));
  cudaMemset(g.d_sz, 0, cap * sizeof(float));
  cudaMemset(g.d_cnt, 0, cap * sizeof(int));
  cudaMemset(g.d_outn, 0, sizeof(int));

  const int T = 256;
  insert_kernel<<<(int)((n + T - 1) / T), T>>>(g.d_pts, (int)n, leaf, g.d_keys,
                                               g.d_sx, g.d_sy, g.d_sz, g.d_cnt,
                                               mask);
  compact_kernel<<<(int)((cap + T - 1) / T), T>>>(g.d_sx, g.d_sy, g.d_sz, g.d_cnt,
                                                  (int)cap, g.d_out, g.d_outn);
  int m = 0;
  cudaMemcpy(&m, g.d_outn, sizeof(int), cudaMemcpyDeviceToHost);
  std::vector<Point> out(m);
  cudaMemcpy(out.data(), g.d_out, m * sizeof(Point), cudaMemcpyDeviceToHost);
  return out;
}

}  // namespace flashicp
