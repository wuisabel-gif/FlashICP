// M2 — GPU correspondence search.
//
// Build a fixed-radius spatial hash grid over the target cloud (cell edge =
// search radius), then let every source point probe its own cell plus the 26
// neighbours (3x3x3) for its nearest target point. With cell edge == radius,
// any target within `radius` of a source is guaranteed to sit in one of those
// 27 cells, so the result matches the brute-force CPU baseline exactly for
// every source point whose true nearest neighbour is within the radius.
//
// This is the heavy per-point stage: unlike the voxel downsample, each query
// does real work (dozens of distance evaluations), which is where the GPU is
// expected to pull well ahead of a branchy CPU k-d tree.
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cfloat>
#include <vector>

#include "flashicp.hpp"

namespace {

// Teschner et al. spatial hash — recognisable, cheap, decorrelates cell coords.
__host__ __device__ inline unsigned int hash_cell(int x, int y, int z,
                                                  unsigned int mask) {
  unsigned int h = (unsigned int)(x * 73856093) ^ (unsigned int)(y * 19349663) ^
                   (unsigned int)(z * 83492791);
  return h & mask;
}
__host__ __device__ inline int cell_of(float v, float c) {
  return (int)floorf(v / c);
}

// Hash each target point to a bucket.
__global__ void cellid_kernel(const flashicp::Point* pts, int n, float cell,
                              unsigned int mask, unsigned int* cellid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  flashicp::Point p = pts[i];
  cellid[i] = hash_cell(cell_of(p.x, cell), cell_of(p.y, cell),
                        cell_of(p.z, cell), mask);
}

// Given cellid[] sorted ascending, record each bucket's [start,end) range.
__global__ void ranges_kernel(const unsigned int* cellid, int n, int* start,
                              int* end) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned int c = cellid[i];
  if (i == 0 || cellid[i - 1] != c) start[c] = i;
  if (i == n - 1 || cellid[i + 1] != c) end[c] = i + 1;
}

// One source point per thread: probe the 27 neighbouring cells for the nearest
// target point within `cell` (== radius).
__global__ void query_kernel(const flashicp::Point* src, int ns,
                             const flashicp::Point* tgt, const int* order,
                             const int* start, const int* end, float cell,
                             unsigned int mask, int* out_idx, float* out_d2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ns) return;
  flashicp::Point s = src[i];
  int cx = cell_of(s.x, cell), cy = cell_of(s.y, cell), cz = cell_of(s.z, cell);
  float best = cell * cell;  // radius^2
  int bi = -1;
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        unsigned int b = hash_cell(cx + dx, cy + dy, cz + dz, mask);
        int a = start[b];
        if (a < 0) continue;  // empty bucket
        int e = end[b];
        for (int k = a; k < e; ++k) {
          int j = order[k];
          flashicp::Point t = tgt[j];
          float ex = s.x - t.x, ey = s.y - t.y, ez = s.z - t.z;
          float d2 = ex * ex + ey * ey + ez * ez;
          if (d2 < best) { best = d2; bi = j; }
        }
      }
  out_idx[i] = bi;
  out_d2[i] = bi < 0 ? -1.0f : best;
}

size_t next_pow2(size_t x) {
  size_t p = 1;
  while (p < x) p <<= 1;
  return p;
}

}  // namespace

namespace flashicp {

// ponytail: per-call allocation for the first cut — mirror voxel_gpu's
// persistent Scratch here once M2 lands in the real per-keyframe loop.
std::vector<Corr> correspond_gpu(const std::vector<Point>& src,
                                 const std::vector<Point>& tgt, float radius) {
  const int ns = (int)src.size(), nt = (int)tgt.size();
  std::vector<Corr> out(ns);
  if (ns == 0 || nt == 0) return out;

  const float cell = radius;  // cell edge == radius => 27-cell search is exact
  const size_t cap = next_pow2((size_t)nt) * 2;
  const unsigned int mask = (unsigned int)(cap - 1);
  const int T = 256;

  thrust::device_vector<Point> d_tgt(tgt.begin(), tgt.end());
  thrust::device_vector<Point> d_src(src.begin(), src.end());
  thrust::device_vector<unsigned int> d_cellid(nt);
  thrust::device_vector<int> d_order(nt);
  thrust::sequence(d_order.begin(), d_order.end());

  cellid_kernel<<<(nt + T - 1) / T, T>>>(
      thrust::raw_pointer_cast(d_tgt.data()), nt, cell, mask,
      thrust::raw_pointer_cast(d_cellid.data()));
  // Group target indices by bucket so each bucket is a contiguous range.
  thrust::sort_by_key(d_cellid.begin(), d_cellid.end(), d_order.begin());

  thrust::device_vector<int> d_start(cap, -1), d_end(cap, -1);
  ranges_kernel<<<(nt + T - 1) / T, T>>>(
      thrust::raw_pointer_cast(d_cellid.data()), nt,
      thrust::raw_pointer_cast(d_start.data()),
      thrust::raw_pointer_cast(d_end.data()));

  thrust::device_vector<int> d_idx(ns);
  thrust::device_vector<float> d_d2(ns);
  query_kernel<<<(ns + T - 1) / T, T>>>(
      thrust::raw_pointer_cast(d_src.data()), ns,
      thrust::raw_pointer_cast(d_tgt.data()),
      thrust::raw_pointer_cast(d_order.data()),
      thrust::raw_pointer_cast(d_start.data()),
      thrust::raw_pointer_cast(d_end.data()), cell, mask,
      thrust::raw_pointer_cast(d_idx.data()),
      thrust::raw_pointer_cast(d_d2.data()));

  std::vector<int> h_idx(ns);
  std::vector<float> h_d2(ns);
  thrust::copy(d_idx.begin(), d_idx.end(), h_idx.begin());
  thrust::copy(d_d2.begin(), d_d2.end(), h_d2.begin());
  for (int i = 0; i < ns; ++i) out[i] = {h_idx[i], h_d2[i]};
  return out;
}

}  // namespace flashicp
