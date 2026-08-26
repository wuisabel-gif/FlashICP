#include "flashicp/normals.hpp"

#include "flashicp.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flashicp {
namespace {

using Clock = std::chrono::steady_clock;

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

Vec3 operator-(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator+(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator*(double scale, const Vec3& a) {
  return {scale * a.x, scale * a.y, scale * a.z};
}

double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm2(const Vec3& a) { return dot(a, a); }

bool finite_point(const PointXYZ& p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

bool covariance_normal(const std::vector<Vec3>& points, Vec3& normal) {
  if (points.size() < 3) return false;
  Vec3 center;
  for (const Vec3& point : points) center = center + point;
  center = (1.0 / static_cast<double>(points.size())) * center;

  std::array<double, 9> covariance{};
  for (const Vec3& point : points) {
    const Vec3 d = point - center;
    covariance[0] += d.x * d.x;
    covariance[1] += d.x * d.y;
    covariance[2] += d.x * d.z;
    covariance[4] += d.y * d.y;
    covariance[5] += d.y * d.z;
    covariance[8] += d.z * d.z;
  }
  covariance[3] = covariance[1];
  covariance[6] = covariance[2];
  covariance[7] = covariance[5];

  std::array<double, 9> vectors{{1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0}};
  for (int sweep = 0; sweep < 32; ++sweep) {
    int p = 0;
    int q = 1;
    double largest = std::abs(covariance[1]);
    for (int row = 0; row < 3; ++row) {
      for (int col = row + 1; col < 3; ++col) {
        if (std::abs(covariance[row * 3 + col]) > largest) {
          largest = std::abs(covariance[row * 3 + col]);
          p = row;
          q = col;
        }
      }
    }
    if (largest <= 1.0e-14) break;
    const double theta = 0.5 * std::atan2(
        2.0 * covariance[p * 3 + q], covariance[q * 3 + q] - covariance[p * 3 + p]);
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double app = covariance[p * 3 + p];
    const double aqq = covariance[q * 3 + q];
    const double apq = covariance[p * 3 + q];
    covariance[p * 3 + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    covariance[q * 3 + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    covariance[p * 3 + q] = covariance[q * 3 + p] = 0.0;
    for (int k = 0; k < 3; ++k) {
      if (k == p || k == q) continue;
      const double akp = covariance[k * 3 + p];
      const double akq = covariance[k * 3 + q];
      covariance[k * 3 + p] = covariance[p * 3 + k] = c * akp - s * akq;
      covariance[k * 3 + q] = covariance[q * 3 + k] = s * akp + c * akq;
    }
    for (int k = 0; k < 3; ++k) {
      const double vkp = vectors[k * 3 + p];
      const double vkq = vectors[k * 3 + q];
      vectors[k * 3 + p] = c * vkp - s * vkq;
      vectors[k * 3 + q] = s * vkp + c * vkq;
    }
  }

  int smallest = 0;
  int middle = 1;
  int largest = 2;
  if (covariance[smallest * 3 + smallest] > covariance[middle * 3 + middle])
    std::swap(smallest, middle);
  if (covariance[middle * 3 + middle] > covariance[largest * 3 + largest])
    std::swap(middle, largest);
  if (covariance[smallest * 3 + smallest] > covariance[middle * 3 + middle])
    std::swap(smallest, middle);
  const double spread = covariance[largest * 3 + largest];
  if (!std::isfinite(spread) || spread <= 1.0e-20 ||
      covariance[middle * 3 + middle] <= spread * 1.0e-10) {
    return false;
  }
  normal = {vectors[0 * 3 + smallest], vectors[1 * 3 + smallest],
            vectors[2 * 3 + smallest]};
  const double length = std::sqrt(norm2(normal));
  if (!std::isfinite(length) || length <= 1.0e-12) return false;
  normal = (1.0 / length) * normal;
  return true;
}

struct Candidate {
  double d2 = 0.0;
  std::size_t index = 0;
};

}  // namespace

NormalEstimationResult estimate_normals_cpu(const PointCloud& cloud,
                                            int k_neighbors,
                                            float search_radius) {
  const Clock::time_point started = Clock::now();
  NormalEstimationResult result;
  result.normals.resize(cloud.size(),
                        {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN()});
  if (cloud.empty()) {
    result.status = NormalStatus::InvalidInput;
    result.message = "cloud must be nonempty";
  } else if (k_neighbors < 3) {
    result.status = NormalStatus::InvalidInput;
    result.message = "k_neighbors must be at least 3";
  } else if (!std::isfinite(search_radius) || search_radius < 0.0f) {
    result.status = NormalStatus::InvalidInput;
    result.message = "search_radius must be finite and nonnegative";
  } else {
    const std::size_t available_neighbors = cloud.size() - 1;
    const int neighbor_count = available_neighbors > static_cast<std::size_t>(std::numeric_limits<int>::max())
                                   ? k_neighbors
                                   : std::min(k_neighbors, static_cast<int>(available_neighbors));
    if (neighbor_count < 3) {
      result.status = NormalStatus::InsufficientNeighbors;
      result.message = "cloud has fewer than three neighbours per point";
      result.timing_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
      return result;
    }
    for (const PointXYZ& point : cloud) {
      if (!finite_point(point)) {
        result.status = NormalStatus::InvalidInput;
        result.message = "cloud contains a nonfinite point";
        result.timing_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        return result;
      }
    }

    // A cell width of one metre is a useful default for KITTI. Explicit radii
    // cap candidate distance and make the scale choice reproducible.
    const double cell_width = search_radius > 0.0f ? search_radius : 1.0;
    std::unordered_map<int64_t, std::vector<std::size_t>> grid;
    grid.reserve(cloud.size());
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      const PointXYZ& p = cloud[i];
      grid[voxel_key(floor_div(p.x, static_cast<float>(cell_width)),
                     floor_div(p.y, static_cast<float>(cell_width)),
                     floor_div(p.z, static_cast<float>(cell_width)))]
          .push_back(i);
    }
    const double radius2 = search_radius > 0.0f
                               ? static_cast<double>(search_radius) * search_radius
                               : std::numeric_limits<double>::infinity();
    std::size_t failed = 0;
    std::size_t insufficient_neighbors = 0;
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      const PointXYZ& point = cloud[i];
      const int cx = floor_div(point.x, static_cast<float>(cell_width));
      const int cy = floor_div(point.y, static_cast<float>(cell_width));
      const int cz = floor_div(point.z, static_cast<float>(cell_width));
      std::vector<Candidate> candidates;
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            const auto it = grid.find(voxel_key(cx + dx, cy + dy, cz + dz));
            if (it == grid.end()) continue;
            for (const std::size_t j : it->second) {
              if (j == i) continue;
              const double ax = static_cast<double>(point.x) - cloud[j].x;
              const double ay = static_cast<double>(point.y) - cloud[j].y;
              const double az = static_cast<double>(point.z) - cloud[j].z;
              const double d2 = ax * ax + ay * ay + az * az;
              if (d2 <= radius2) candidates.push_back({d2, j});
            }
          }
        }
      }

      // Small clouds and sparse fixtures benefit from an exact fallback. Do
      // not make a large KITTI scan accidentally quadratic when local support
      // is absent: it will report an explicit normal-estimation failure.
      if (candidates.size() < static_cast<std::size_t>(neighbor_count) &&
          cloud.size() <= 4096) {
        candidates.clear();
        for (std::size_t j = 0; j < cloud.size(); ++j) {
          if (j == i) continue;
          const double ax = static_cast<double>(point.x) - cloud[j].x;
          const double ay = static_cast<double>(point.y) - cloud[j].y;
          const double az = static_cast<double>(point.z) - cloud[j].z;
          const double d2 = ax * ax + ay * ay + az * az;
          if (d2 <= radius2) candidates.push_back({d2, j});
        }
      }
      std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.d2 != b.d2) return a.d2 < b.d2;
        return a.index < b.index;
      });
      if (candidates.size() < static_cast<std::size_t>(neighbor_count)) {
        ++failed;
        ++insufficient_neighbors;
        continue;
      }
      std::vector<Vec3> neighborhood;
      neighborhood.reserve(static_cast<std::size_t>(neighbor_count) + 1);
      neighborhood.push_back({point.x, point.y, point.z});
      for (int n = 0; n < neighbor_count; ++n) {
        const PointXYZ& neighbor = cloud[candidates[static_cast<std::size_t>(n)].index];
        neighborhood.push_back({neighbor.x, neighbor.y, neighbor.z});
      }
      Vec3 normal;
      if (!covariance_normal(neighborhood, normal)) {
        ++failed;
        continue;
      }
      // Orient toward the sensor origin. For points very close to the origin,
      // retain the deterministic eigensolver orientation.
      const Vec3 position{point.x, point.y, point.z};
      if (norm2(position) > 1.0e-12 && dot(normal, position) > 0.0) normal = -1.0 * normal;
      result.normals[i] = {static_cast<float>(normal.x), static_cast<float>(normal.y),
                           static_cast<float>(normal.z)};
      ++result.valid_normals;
    }
    if (result.valid_normals == cloud.size()) {
      result.status = NormalStatus::Success;
      result.message = "normals estimated";
    } else if (result.valid_normals >= 3) {
      result.status = NormalStatus::Success;
      std::ostringstream message;
      message << "normals estimated for " << result.valid_normals << " of " << cloud.size()
              << " points (" << failed << " had insufficient local geometry)";
      result.message = message.str();
    } else if (insufficient_neighbors == cloud.size()) {
      result.status = NormalStatus::InsufficientNeighbors;
      result.message = "no point has the requested number of local neighbours";
    } else {
      result.status = NormalStatus::DegenerateGeometry;
      result.message = "fewer than three target normals could be estimated";
    }
  }
  result.timing_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  return result;
}

const char* to_string(NormalStatus status) {
  switch (status) {
    case NormalStatus::Success: return "success";
    case NormalStatus::InvalidInput: return "invalid_input";
    case NormalStatus::InsufficientNeighbors: return "insufficient_neighbors";
    case NormalStatus::DegenerateGeometry: return "degenerate_geometry";
    case NormalStatus::NumericalFailure: return "numerical_failure";
  }
  return "unknown";
}

}  // namespace flashicp
