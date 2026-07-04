// Point type, cloud IO, and the CPU voxel-downsample baseline.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <unordered_map>

namespace flashicp {

struct Point {
  float x, y, z;
};

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
  const int64_t OX = (ix + (1 << 20)) & B;
  const int64_t OY = (iy + (1 << 20)) & B;
  const int64_t OZ = (iz + (1 << 20)) & B;
  return (OX << 42) | (OY << 21) | OZ;
}

inline int floor_div(float v, float leaf) {
  return static_cast<int>(std::floor(v / leaf));
}

inline std::vector<Point> voxel_downsample_cpu(const std::vector<Point>& in,
                                               float leaf) {
  struct Acc {
    double sx = 0, sy = 0, sz = 0;
    int n = 0;
  };
  std::unordered_map<int64_t, Acc> grid;
  grid.reserve(in.size());
  for (const Point& p : in) {
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
    float best = r2;
    int bi = -1;
    for (size_t j = 0; j < tgt.size(); ++j) {
      const float dx = s.x - tgt[j].x, dy = s.y - tgt[j].y, dz = s.z - tgt[j].z;
      const float d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) { best = d2; bi = static_cast<int>(j); }
    }
    out[i] = {bi, bi < 0 ? -1.0f : best};
  }
  return out;
}

}  // namespace flashicp
