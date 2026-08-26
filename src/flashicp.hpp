// Point type, cloud IO, and the CPU voxel-downsample baseline.
#pragma once
#include "../include/flashicp/registration.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <unordered_map>

namespace flashicp {

using Point = PointXYZ;
static_assert(sizeof(Point) == sizeof(float) * 3,
              "PointXYZ must remain a packed xyz triplet for the binary format");

// Reads files written by tools/dump_cloud.py: int32 n, then n*(float x,y,z).
inline std::vector<Point> load_cloud(const char* path) {
  std::vector<Point> pts;
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    return pts;
  }
  int32_t n = 0;
  if (std::fread(&n, sizeof(n), 1, f) != 1 || n < 0) {
    std::fclose(f);
    return pts;
  }
  if (std::fseek(f, 0, SEEK_END) != 0) {
    std::fclose(f);
    return pts;
  }
  const long file_size = std::ftell(f);
  const uint64_t required = sizeof(n) +
                            static_cast<uint64_t>(n) * sizeof(Point);
  if (file_size < 0 || static_cast<uint64_t>(file_size) < required ||
      std::fseek(f, static_cast<long>(sizeof(n)), SEEK_SET) != 0) {
    std::fprintf(stderr, "invalid or truncated cloud file %s\n", path);
    std::fclose(f);
    return pts;
  }
  pts.resize(static_cast<size_t>(n));
  size_t got = std::fread(pts.data(), sizeof(Point), pts.size(), f);
  if (got != pts.size()) {
    std::fprintf(stderr, "short read from %s: got %zu of %d points\n", path, got, n);
    pts.resize(got);
  }
  std::fclose(f);
  return pts;
}

// 21 bits per axis (+/-1M voxels); the result stays below 2^63.
inline int64_t voxel_key(int ix, int iy, int iz) {
  const int64_t B = 0x1FFFFF;
  const int64_t OX = (static_cast<int64_t>(ix) + (1 << 20)) & B;
  const int64_t OY = (static_cast<int64_t>(iy) + (1 << 20)) & B;
  const int64_t OZ = (static_cast<int64_t>(iz) + (1 << 20)) & B;
  return (OX << 42) | (OY << 21) | OZ;
}

inline int floor_div(float v, float leaf) {
  if (!std::isfinite(v) || !std::isfinite(leaf) || leaf <= 0.0f) return 0;
  const double q = std::floor(static_cast<double>(v) / leaf);
  if (q >= std::numeric_limits<int>::max()) return std::numeric_limits<int>::max();
  if (q <= std::numeric_limits<int>::min()) return std::numeric_limits<int>::min();
  return static_cast<int>(q);
}

inline std::vector<Point> voxel_downsample_cpu(const std::vector<Point>& in,
                                               float leaf) {
  if (!std::isfinite(leaf) || leaf <= 0.0f) return {};
  struct Acc {
    double sx = 0, sy = 0, sz = 0;
    int n = 0;
  };
  std::unordered_map<int64_t, Acc> grid;
  grid.reserve(in.size());
  for (const Point& p : in) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    int64_t k = voxel_key(floor_div(p.x, leaf), floor_div(p.y, leaf),
                          floor_div(p.z, leaf));
    Acc& a = grid[k];
    a.sx += p.x;
    a.sy += p.y;
    a.sz += p.z;
    a.n += 1;
  }
  std::vector<Point> out;
  out.reserve(grid.size());
  for (const auto& kv : grid) {
    const Acc& a = kv.second;
    out.push_back({static_cast<float>(a.sx / a.n), static_cast<float>(a.sy / a.n),
                   static_cast<float>(a.sz / a.n)});
  }
  return out;
}

// A correspondence: source point i -> nearest target index, squared distance.
struct Corr {
  int idx;    // index into target cloud, -1 if none within the search radius
  float d2;   // squared distance to that target point
};

// CPU baseline for M2 correspondence: brute-force nearest target per source
// point. O(src * tgt) — the ground truth the GPU grid is checked against.
// `radius` caps the match distance (<=0 means unbounded); ICP rejects far pairs.
inline std::vector<Corr> correspond_cpu(const std::vector<Point>& src,
                                        const std::vector<Point>& tgt,
                                        float radius) {
  const float r2 = radius > 0 ? radius * radius
                              : std::numeric_limits<float>::max();
  std::vector<Corr> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    const Point s = src[i];
    if (!std::isfinite(s.x) || !std::isfinite(s.y) || !std::isfinite(s.z)) {
      out[i] = {-1, -1.0f};
      continue;
    }
    float best = r2;
    int bi = -1;
    for (size_t j = 0; j < tgt.size(); ++j) {
      if (!std::isfinite(tgt[j].x) || !std::isfinite(tgt[j].y) ||
          !std::isfinite(tgt[j].z)) continue;
      const float dx = s.x - tgt[j].x, dy = s.y - tgt[j].y, dz = s.z - tgt[j].z;
      const float d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) { best = d2; bi = static_cast<int>(j); }
    }
    out[i] = {bi, bi < 0 ? -1.0f : best};
  }
  return out;
}

// Exact CPU counterpart to the CUDA fixed-radius grid. Every target within the
// radius lies in one of the 27 neighbouring cells, while the distance check and
// lowest-index tie break preserve correspond_cpu's result. The brute-force
// function above remains available as the simple correctness oracle.
inline std::vector<Corr> correspond_cpu_grid(const std::vector<Point>& src,
                                             const std::vector<Point>& tgt,
                                             float radius) {
  if (!std::isfinite(radius) || radius <= 0.0f) return correspond_cpu(src, tgt, radius);
  const float radius2 = radius * radius;
  std::unordered_map<int64_t, std::vector<std::size_t>> grid;
  grid.reserve(tgt.size());
  for (std::size_t j = 0; j < tgt.size(); ++j) {
    if (!std::isfinite(tgt[j].x) || !std::isfinite(tgt[j].y) || !std::isfinite(tgt[j].z)) continue;
    grid[voxel_key(floor_div(tgt[j].x, radius), floor_div(tgt[j].y, radius),
                   floor_div(tgt[j].z, radius))].push_back(j);
  }
  std::vector<Corr> output(src.size(), {-1, -1.0f});
  for (std::size_t i = 0; i < src.size(); ++i) {
    if (!std::isfinite(src[i].x) || !std::isfinite(src[i].y) || !std::isfinite(src[i].z)) continue;
    const int cx = floor_div(src[i].x, radius);
    const int cy = floor_div(src[i].y, radius);
    const int cz = floor_div(src[i].z, radius);
    float best = radius2;
    int best_index = -1;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          const auto it = grid.find(voxel_key(cx + dx, cy + dy, cz + dz));
          if (it == grid.end()) continue;
          for (const std::size_t j : it->second) {
            const float ex = src[i].x - tgt[j].x;
            const float ey = src[i].y - tgt[j].y;
            const float ez = src[i].z - tgt[j].z;
            const float d2 = ex * ex + ey * ey + ez * ez;
            if (d2 < best || (d2 == best &&
                              (best_index < 0 || j < static_cast<std::size_t>(best_index)))) {
              best = d2;
              best_index = static_cast<int>(j);
            }
          }
        }
      }
    }
    output[i] = {best_index, best_index < 0 ? -1.0f : best};
  }
  return output;
}

}  // namespace flashicp
