#include "flashicp.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

int main() {
  using flashicp::Point;

  const std::vector<Point> points{{0.01f, 0.01f, 0.01f},
                                  {0.02f, 0.02f, 0.02f},
                                  {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f}};
  assert(flashicp::voxel_downsample_cpu(points, 0.0f).empty());
  const auto downsampled = flashicp::voxel_downsample_cpu(points, 0.1f);
  assert(downsampled.size() == 1);

  const std::vector<Point> source{{0.0f, 0.0f, 0.0f},
                                  {std::numeric_limits<float>::infinity(), 0.0f, 0.0f}};
  const std::vector<Point> target{{0.1f, 0.0f, 0.0f},
                                  {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f}};
  const auto correspondences = flashicp::correspond_cpu(source, target, 0.5f);
  assert(correspondences.size() == source.size());
  assert(correspondences[0].idx == 0);
  assert(correspondences[1].idx == -1);
  const auto hashed = flashicp::correspond_cpu_grid(
      std::vector<Point>{{0.0f, 0.0f, 0.0f}, {0.2f, 0.0f, 0.0f}},
      std::vector<Point>{{0.1f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}, 0.5f);
  const auto brute = flashicp::correspond_cpu(
      std::vector<Point>{{0.0f, 0.0f, 0.0f}, {0.2f, 0.0f, 0.0f}},
      std::vector<Point>{{0.1f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}, 0.5f);
  assert(hashed.size() == brute.size());
  for (std::size_t i = 0; i < hashed.size(); ++i) {
    assert(hashed[i].idx == brute[i].idx);
    assert(std::abs(hashed[i].d2 - brute[i].d2) < 1.0e-7f);
  }

  std::printf("test_primitives: PASS\n");
  return 0;
}
