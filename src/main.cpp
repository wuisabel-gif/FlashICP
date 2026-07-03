// Benchmark CLI:  flashicp bench cloud.bin [leaf] [iters]
// Times the CPU voxel-downsample baseline against the CUDA versions (when built
// with USE_CUDA) and checks the GPU output against the CPU result.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "flashicp.hpp"

#ifdef FLASHICP_CUDA
namespace flashicp {
std::vector<Point> voxel_downsample_gpu(const std::vector<Point>&, float);
std::vector<Point> voxel_downsample_gpu_hash(const std::vector<Point>&, float);
std::vector<Corr> correspond_gpu(const std::vector<Point>&,
                                 const std::vector<Point>&, float);
}
#endif

using flashicp::Point;
using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Order-independent agreement check: same voxel count and matching centroid.
static void compare(const std::vector<Point>& a, const std::vector<Point>& b) {
  auto mean = [](const std::vector<Point>& v) {
    double sx = 0, sy = 0, sz = 0;
    for (auto& p : v) { sx += p.x; sy += p.y; sz += p.z; }
    size_t n = v.empty() ? 1 : v.size();
    return Point{float(sx / n), float(sy / n), float(sz / n)};
  };
  Point ma = mean(a), mb = mean(b);
  double dmean = std::abs(ma.x - mb.x) + std::abs(ma.y - mb.y) + std::abs(ma.z - mb.z);
  long dn = (long)a.size() - (long)b.size();
  std::printf("  check: cpu=%zu gpu=%zu voxels (dn=%ld), centroid L1 diff=%.6g\n",
              a.size(), b.size(), dn, dmean);
  std::printf("  %s\n", (std::abs(dn) <= 1 && dmean < 1e-3) ? "PASS (matches CPU)"
                                                            : "WARN (review)");
}

// Correctness for correspondence: compare matched squared-distances (robust to
// ties, where two equidistant targets are both "correct"). Reports agreement
// among source points the CPU matched within the radius.
static void compare_corr(const std::vector<flashicp::Corr>& cpu,
                         const std::vector<flashicp::Corr>& gpu) {
  size_t matched = 0, agree = 0;
  double sum_d = 0;
  for (size_t i = 0; i < cpu.size(); ++i) {
    if (cpu[i].idx < 0) continue;  // no CPU match within radius; skip
    ++matched;
    sum_d += std::sqrt(cpu[i].d2);
    if (gpu[i].idx >= 0 && std::abs(gpu[i].d2 - cpu[i].d2) < 1e-6f) ++agree;
  }
  double pct = matched ? 100.0 * agree / matched : 100.0;
  double mean = matched ? sum_d / matched : 0.0;
  std::printf("  check: %zu/%zu matched within radius agree (%.2f%%), "
              "mean NN dist=%.4f m\n", agree, matched, pct, mean);
  std::printf("  %s\n", (agree == matched) ? "PASS (matches CPU)"
                                           : "WARN (review radius)");
}

static int run_corr(int argc, char** argv) {
  const char* pa = argv[2];
  const char* pb = argc > 3 ? argv[3] : argv[2];
  float radius = argc > 4 ? std::atof(argv[4]) : 0.20f;
  int iters = argc > 5 ? std::atoi(argv[5]) : 20;

  std::vector<Point> src = flashicp::load_cloud(pa);
  std::vector<Point> tgt = flashicp::load_cloud(pb);
  if (src.empty() || tgt.empty()) {
    std::printf("no points loaded (src=%zu tgt=%zu)\n", src.size(), tgt.size());
    return 1;
  }
  std::printf("correspondence: src=%zu tgt=%zu points, radius=%.3f m, iters=%d\n",
              src.size(), tgt.size(), radius, iters);

  std::vector<flashicp::Corr> cpu = flashicp::correspond_cpu(src, tgt, radius);
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) cpu = flashicp::correspond_cpu(src, tgt, radius);
  double cpu_ms = ms_since(t0) / iters;
  std::printf("CPU correspondence (brute force): %.3f ms\n", cpu_ms);

#ifdef FLASHICP_CUDA
  std::vector<flashicp::Corr> gpu = flashicp::correspond_gpu(src, tgt, radius);
  auto t1 = Clock::now();
  for (int i = 0; i < iters; ++i) gpu = flashicp::correspond_gpu(src, tgt, radius);
  double gpu_ms = ms_since(t1) / iters;
  std::printf("GPU correspondence (hash grid): %.3f ms  (%.1fx vs CPU)\n", gpu_ms,
              cpu_ms / gpu_ms);
  compare_corr(cpu, gpu);
#else
  std::printf("(built without CUDA; CPU baseline only -- rebuild with -DUSE_CUDA=ON)\n");
#endif
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 3 && std::strcmp(argv[1], "corr") == 0) return run_corr(argc, argv);
  if (argc < 3 || std::strcmp(argv[1], "bench") != 0) {
    std::printf("usage: flashicp bench cloud.bin [leaf=0.05] [iters=20]\n");
    std::printf("       flashicp corr  src.bin [tgt.bin] [radius=0.20] [iters=20]\n");
    return 1;
  }
  const char* path = argv[2];
  float leaf = argc > 3 ? std::atof(argv[3]) : 0.05f;
  int iters = argc > 4 ? std::atoi(argv[4]) : 20;

  std::vector<Point> cloud = flashicp::load_cloud(path);
  if (cloud.empty()) {
    std::printf("no points loaded from %s\n", path);
    return 1;
  }
  std::printf("loaded %zu points, leaf=%.3f m, iters=%d\n", cloud.size(), leaf, iters);

  std::vector<Point> cpu_out = flashicp::voxel_downsample_cpu(cloud, leaf);
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i)
    cpu_out = flashicp::voxel_downsample_cpu(cloud, leaf);
  double cpu_ms = ms_since(t0) / iters;
  std::printf("CPU voxel downsample: %.3f ms  -> %zu voxels\n", cpu_ms, cpu_out.size());

#ifdef FLASHICP_CUDA
  std::vector<Point> gpu_out = flashicp::voxel_downsample_gpu(cloud, leaf);
  auto t1 = Clock::now();
  for (int i = 0; i < iters; ++i)
    gpu_out = flashicp::voxel_downsample_gpu(cloud, leaf);
  double gpu_ms = ms_since(t1) / iters;
  std::printf("GPU voxel (sort): %.3f ms  -> %zu voxels  (%.1fx vs CPU)\n", gpu_ms,
              gpu_out.size(), cpu_ms / gpu_ms);
  compare(cpu_out, gpu_out);

  std::vector<Point> hash_out = flashicp::voxel_downsample_gpu_hash(cloud, leaf);
  auto t2 = Clock::now();
  for (int i = 0; i < iters; ++i)
    hash_out = flashicp::voxel_downsample_gpu_hash(cloud, leaf);
  double hash_ms = ms_since(t2) / iters;
  std::printf("GPU voxel (hash): %.3f ms  -> %zu voxels  (%.1fx vs CPU)\n", hash_ms,
              hash_out.size(), cpu_ms / hash_ms);
  compare(cpu_out, hash_out);
#else
  std::printf("(built without CUDA; CPU baseline only -- rebuild with -DUSE_CUDA=ON)\n");
#endif
  return 0;
}
