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

  std::printf("test_primitives: PASS\n");
  return 0;
}
