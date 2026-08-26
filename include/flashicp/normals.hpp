#pragma once

#include "registration.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace flashicp {

// A unit surface normal. Normals returned by estimate_normals_cpu are oriented
// toward the target sensor origin whenever the point is sufficiently far from
// that origin. Point-to-plane ICP only requires a consistent normal per target
// point because both the residual and Jacobian change sign together.
struct NormalXYZ {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

using NormalCloud = std::vector<NormalXYZ>;

enum class NormalStatus {
  Success,
  InvalidInput,
  InsufficientNeighbors,
  DegenerateGeometry,
  NumericalFailure,
};

struct NormalEstimationResult {
  NormalCloud normals;
  NormalStatus status = NormalStatus::InvalidInput;
  std::size_t valid_normals = 0;
  double timing_ms = 0.0;
  std::string message;
};

// Estimate target normals with a deterministic local covariance eigensolve.
// The search uses a voxel hash when search_radius is positive. A zero radius
// selects a one-metre local search cell and is suitable for typical KITTI
// scans; callers with a known scale should set it explicitly. At least three
// non-collinear neighbours are required for each usable normal.
NormalEstimationResult estimate_normals_cpu(const PointCloud& cloud,
                                            int k_neighbors = 8,
                                            float search_radius = 0.0f);

const char* to_string(NormalStatus status);

}  // namespace flashicp
