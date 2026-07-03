// Self-check for the CPU correspondence baseline (the ground truth the GPU
// hash grid is verified against). Compile standalone, no CUDA needed:
//   c++ -std=c++17 -Isrc tools/test_corr.cpp -o /tmp/test_corr && /tmp/test_corr
#include <cassert>
#include <cmath>
#include <cstdio>

#include "flashicp.hpp"

using flashicp::Point;

int main() {
  std::vector<Point> tgt = {{0, 0, 0}, {1, 0, 0}, {5, 5, 5}};
  std::vector<Point> src = {{0.1f, 0, 0}, {0.9f, 0, 0}, {100, 0, 0}};

  // Unbounded: each source resolves to its true nearest target.
  auto c = flashicp::correspond_cpu(src, tgt, 0.0f);
  assert(c[0].idx == 0);                       // 0.1 -> origin
  assert(c[1].idx == 1);                       // 0.9 -> (1,0,0)
  assert(c[2].idx == 2);                       // far point -> only (5,5,5) side
  assert(std::abs(c[0].d2 - 0.01f) < 1e-6f);   // 0.1^2

  // Radius rejects the far match, keeps the close ones.
  auto r = flashicp::correspond_cpu(src, tgt, 0.5f);
  assert(r[0].idx == 0 && r[1].idx == 1);
  assert(r[2].idx == -1 && r[2].d2 == -1.0f);  // nothing within 0.5 m

  std::printf("test_corr: PASS\n");
  return 0;
}
