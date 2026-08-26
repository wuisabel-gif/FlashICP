#include "flashicp/registration.hpp"

#include "flashicp.hpp"
#include "registration_internal.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace flashicp {
namespace {

using Clock = std::chrono::steady_clock;

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

Vec3 operator+(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 operator-(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 operator*(double scale, const Vec3& v) {
  return {scale * v.x, scale * v.y, scale * v.z};
}
double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}
double norm2(const Vec3& v) { return dot(v, v); }

Vec3 to_vec(const PointXYZ& p) {
  return {static_cast<double>(p.x), static_cast<double>(p.y),
          static_cast<double>(p.z)};
}
PointXYZ to_point(const Vec3& v) {
  return {static_cast<float>(v.x), static_cast<float>(v.y),
          static_cast<float>(v.z)};
}

bool finite_point(const PointXYZ& p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

bool finite_options(const ICPOptions& options, std::string& message) {
  if (options.max_iterations <= 0) {
    message = "max_iterations must be positive";
    return false;
  }
  if (!std::isfinite(options.correspondence_radius) ||
      options.correspondence_radius <= 0.0f) {
    message = "correspondence_radius must be finite and positive";
    return false;
  }
  if (!std::isfinite(options.voxel_size) || options.voxel_size < 0.0f) {
    message = "voxel_size must be finite and nonnegative";
    return false;
  }
  if (!std::isfinite(options.convergence_tolerance) ||
      options.convergence_tolerance < 0.0f) {
    message = "convergence_tolerance must be finite and nonnegative";
    return false;
  }
  if (options.min_correspondences < 3) {
    message = "min_correspondences must be at least 3 for rigid 3D alignment";
    return false;
  }
  return true;
}

void finish_timing(RegistrationResult& result, Clock::time_point started) {
  result.timing.total_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

RegistrationResult invalid_result(const std::string& message,
                                  Clock::time_point started) {
  RegistrationResult result;
  result.status = RegistrationStatus::InvalidInput;
  result.message = message;
  finish_timing(result, started);
  return result;
}

// Solve the absolute rigid transform that maps p_i to q_i. The quaternion
// eigensystem is Horn's closed-form point-to-point solution. A Jacobi iteration
// keeps this implementation dependency-free while remaining stable for the
// small 4x4 symmetric matrix.

internal::SolveStatus solve_rigid_impl(
    const std::vector<PointXYZ>& source, const std::vector<PointXYZ>& target,
    const std::vector<Corr>& correspondences, Transform& transform, float& rms) {
  std::vector<Vec3> ps;
  std::vector<Vec3> qs;
  ps.reserve(correspondences.size());
  qs.reserve(correspondences.size());
  for (std::size_t i = 0; i < correspondences.size(); ++i) {
    const Corr& corr = correspondences[i];
    if (corr.idx < 0 || static_cast<std::size_t>(corr.idx) >= target.size()) continue;
    ps.push_back(to_vec(source[i]));
    qs.push_back(to_vec(target[static_cast<std::size_t>(corr.idx)]));
  }
  if (ps.size() < 3) return internal::SolveStatus::Degenerate;

  Vec3 pc, qc;
  for (std::size_t i = 0; i < ps.size(); ++i) {
    pc = pc + ps[i];
    qc = qc + qs[i];
  }
  const double inv_n = 1.0 / static_cast<double>(ps.size());
  pc = inv_n * pc;
  qc = inv_n * qc;

  // Reject coincident/collinear source or target geometry. Non-collinear
  // planar clouds are valid for rigid point-to-point registration.
  const auto is_collinear = [](const std::vector<Vec3>& points) {
    Vec3 center;
    for (const Vec3& point : points) center = center + point;
    center = (1.0 / static_cast<double>(points.size())) * center;
    std::size_t ia = 0;
    double max_from_center = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
      const double d = norm2(points[i] - center);
      if (d > max_from_center) {
        max_from_center = d;
        ia = i;
      }
    }
    if (max_from_center <= 1.0e-20) return true;
    std::size_t ib = ia;
    double baseline2 = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
      const double d = norm2(points[i] - points[ia]);
      if (d > baseline2) {
        baseline2 = d;
        ib = i;
      }
    }
    if (baseline2 <= 1.0e-20) return true;
    const Vec3 baseline = points[ib] - points[ia];
    double max_cross2 = 0.0;
    double max_radius2 = 0.0;
    for (const Vec3& point : points) {
      max_radius2 = std::max(max_radius2, norm2(point - center));
      max_cross2 = std::max(max_cross2,
                            norm2(cross(baseline, point - points[ia])));
    }
    return max_cross2 <=
           1.0e-12 * baseline2 * std::max(max_radius2, 1.0);
  };
  if (is_collinear(ps) || is_collinear(qs)) return internal::SolveStatus::Degenerate;

  std::array<double, 9> m{};
  for (std::size_t i = 0; i < ps.size(); ++i) {
    const Vec3 p = ps[i] - pc;
    const Vec3 q = qs[i] - qc;
    m[0] += p.x * q.x;
    m[1] += p.x * q.y;
    m[2] += p.x * q.z;
    m[3] += p.y * q.x;
    m[4] += p.y * q.y;
    m[5] += p.y * q.z;
    m[6] += p.z * q.x;
    m[7] += p.z * q.y;
    m[8] += p.z * q.z;
  }

  const double trace = m[0] + m[4] + m[8];
  std::array<double, 16> n{{
      trace,       m[5] - m[7], m[6] - m[2], m[1] - m[3],
      m[5] - m[7], m[0] - m[4] - m[8], m[1] + m[3], m[2] + m[6],
      m[6] - m[2], m[1] + m[3], -m[0] + m[4] - m[8], m[5] + m[7],
      m[1] - m[3], m[2] + m[6], m[5] + m[7], -m[0] - m[4] + m[8]}};

  std::array<double, 16> eigenvectors{{1.0, 0.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0, 0.0,
                                        0.0, 0.0, 0.0, 1.0}};
  for (int sweep = 0; sweep < 64; ++sweep) {
    int p = 0;
    int q = 1;
    double largest = std::abs(n[1]);
    for (int row = 0; row < 4; ++row) {
      for (int col = row + 1; col < 4; ++col) {
        if (std::abs(n[row * 4 + col]) > largest) {
          largest = std::abs(n[row * 4 + col]);
          p = row;
          q = col;
        }
      }
    }
    if (largest < 1.0e-14) break;
    const double theta =
        0.5 * std::atan2(2.0 * n[p * 4 + q], n[q * 4 + q] - n[p * 4 + p]);
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double app = n[p * 4 + p];
    const double aqq = n[q * 4 + q];
    const double apq = n[p * 4 + q];
    n[p * 4 + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    n[q * 4 + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    n[p * 4 + q] = n[q * 4 + p] = 0.0;
    for (int k = 0; k < 4; ++k) {
      if (k == p || k == q) continue;
      const double akp = n[k * 4 + p];
      const double akq = n[k * 4 + q];
      n[k * 4 + p] = n[p * 4 + k] = c * akp - s * akq;
      n[k * 4 + q] = n[q * 4 + k] = s * akp + c * akq;
    }
    for (int k = 0; k < 4; ++k) {
      const double vkp = eigenvectors[k * 4 + p];
      const double vkq = eigenvectors[k * 4 + q];
      eigenvectors[k * 4 + p] = c * vkp - s * vkq;
      eigenvectors[k * 4 + q] = s * vkp + c * vkq;
    }
  }

  int best = 0;
  for (int i = 1; i < 4; ++i) {
    if (n[i * 4 + i] > n[best * 4 + best]) best = i;
  }
  std::array<double, 4> quaternion{{eigenvectors[0 * 4 + best],
                                     eigenvectors[1 * 4 + best],
                                     eigenvectors[2 * 4 + best],
                                     eigenvectors[3 * 4 + best]}};
  double qnorm = 0.0;
  for (double value : quaternion) qnorm += value * value;
  if (!std::isfinite(qnorm) || qnorm <= 1.0e-20) {
    return internal::SolveStatus::NumericalFailure;
  }
  const double inv_qnorm = 1.0 / std::sqrt(qnorm);
  for (double& value : quaternion) value *= inv_qnorm;
  const double w = quaternion[0];
  const double x = quaternion[1];
  const double y = quaternion[2];
  const double z = quaternion[3];

  transform.rotation = {{static_cast<float>(1.0 - 2.0 * (y * y + z * z)),
                         static_cast<float>(2.0 * (x * y - z * w)),
                         static_cast<float>(2.0 * (x * z + y * w)),
                         static_cast<float>(2.0 * (x * y + z * w)),
                         static_cast<float>(1.0 - 2.0 * (x * x + z * z)),
                         static_cast<float>(2.0 * (y * z - x * w)),
                         static_cast<float>(2.0 * (x * z - y * w)),
                         static_cast<float>(2.0 * (y * z + x * w)),
                         static_cast<float>(1.0 - 2.0 * (x * x + y * y))}};
  const PointXYZ rotated_center = transform.apply(to_point(pc));
  transform.translation = {{static_cast<float>(qc.x - rotated_center.x),
                            static_cast<float>(qc.y - rotated_center.y),
                            static_cast<float>(qc.z - rotated_center.z)}};

  double sum_squared = 0.0;
  for (std::size_t i = 0; i < ps.size(); ++i) {
    const PointXYZ mapped = transform.apply(to_point(ps[i]));
    const Vec3 error = to_vec(mapped) - qs[i];
    sum_squared += norm2(error);
  }
  rms = static_cast<float>(std::sqrt(sum_squared / ps.size()));
  if (!std::isfinite(rms) || !transform.is_finite()) {
    return internal::SolveStatus::NumericalFailure;
  }
  return internal::SolveStatus::Ok;
}

float transform_step_impl(const Transform& transform) {
  const float trace = transform.rotation[0] + transform.rotation[4] + transform.rotation[8];
  const float cosine = std::max(-1.0f, std::min(1.0f, (trace - 1.0f) * 0.5f));
  const float angle = std::acos(cosine);
  const float translation = std::sqrt(transform.translation[0] * transform.translation[0] +
                                      transform.translation[1] * transform.translation[1] +
                                      transform.translation[2] * transform.translation[2]);
  return angle + translation;
}

}  // namespace

namespace internal {

SolveStatus solve_rigid(const std::vector<PointXYZ>& source,
                        const std::vector<PointXYZ>& target,
                        const std::vector<Corr>& correspondences,
                        Transform& transform, float& rms) {
  return solve_rigid_impl(source, target, correspondences, transform, rms);
}

float transform_step(const Transform& transform) {
  return transform_step_impl(transform);
}

}  // namespace internal

Transform Transform::identity() { return Transform{}; }

PointXYZ Transform::apply(const PointXYZ& point) const {
  return {rotation[0] * point.x + rotation[1] * point.y + rotation[2] * point.z + translation[0],
          rotation[3] * point.x + rotation[4] * point.y + rotation[5] * point.z + translation[1],
          rotation[6] * point.x + rotation[7] * point.y + rotation[8] * point.z + translation[2]};
}

Transform Transform::inverse() const {
  Transform result;
  result.rotation = {{rotation[0], rotation[3], rotation[6],
                      rotation[1], rotation[4], rotation[7],
                      rotation[2], rotation[5], rotation[8]}};
  result.translation = {{-(result.rotation[0] * translation[0] +
                            result.rotation[1] * translation[1] +
                            result.rotation[2] * translation[2]),
                         -(result.rotation[3] * translation[0] +
                            result.rotation[4] * translation[1] +
                            result.rotation[5] * translation[2]),
                         -(result.rotation[6] * translation[0] +
                            result.rotation[7] * translation[1] +
                            result.rotation[8] * translation[2])}};
  return result;
}

bool Transform::is_finite() const {
  for (float value : rotation) {
    if (!std::isfinite(value)) return false;
  }
  for (float value : translation) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool Transform::is_valid(float tolerance) const {
  if (!is_finite() || !std::isfinite(tolerance) || tolerance < 0.0f) return false;
  const float r00 = rotation[0], r01 = rotation[1], r02 = rotation[2];
  const float r10 = rotation[3], r11 = rotation[4], r12 = rotation[5];
  const float r20 = rotation[6], r21 = rotation[7], r22 = rotation[8];
  const float row0_norm = r00 * r00 + r01 * r01 + r02 * r02;
  const float row1_norm = r10 * r10 + r11 * r11 + r12 * r12;
  const float row2_norm = r20 * r20 + r21 * r21 + r22 * r22;
  const float determinant = r00 * (r11 * r22 - r12 * r21) -
                           r01 * (r10 * r22 - r12 * r20) +
                           r02 * (r10 * r21 - r11 * r20);
  return std::abs(row0_norm - 1.0f) <= tolerance &&
         std::abs(row1_norm - 1.0f) <= tolerance &&
         std::abs(row2_norm - 1.0f) <= tolerance &&
         std::abs(r00 * r10 + r01 * r11 + r02 * r12) <= tolerance &&
         std::abs(r00 * r20 + r01 * r21 + r02 * r22) <= tolerance &&
         std::abs(r10 * r20 + r11 * r21 + r12 * r22) <= tolerance &&
         determinant > 0.0f && std::abs(determinant - 1.0f) <= 3.0f * tolerance;
}

Transform operator*(const Transform& lhs, const Transform& rhs) {
  Transform result;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      result.rotation[row * 3 + col] =
          lhs.rotation[row * 3] * rhs.rotation[col] +
          lhs.rotation[row * 3 + 1] * rhs.rotation[3 + col] +
          lhs.rotation[row * 3 + 2] * rhs.rotation[6 + col];
    }
    result.translation[row] =
        lhs.rotation[row * 3] * rhs.translation[0] +
        lhs.rotation[row * 3 + 1] * rhs.translation[1] +
        lhs.rotation[row * 3 + 2] * rhs.translation[2] + lhs.translation[row];
  }
  return result;
}

RegistrationResult align_cpu(const PointCloud& source, const PointCloud& target,
                             const Transform& initial_guess,
                             const ICPOptions& options) {
  const Clock::time_point started = Clock::now();
  RegistrationResult result;
  result.transform = initial_guess;

  if (options.method != ICPMethod::PointToPoint) {
    result.status = RegistrationStatus::UnsupportedMethod;
    result.message = "point-to-plane is not implemented yet";
    finish_timing(result, started);
    return result;
  }
  std::string validation_message;
  if (!finite_options(options, validation_message)) {
    return invalid_result(validation_message, started);
  }
  if (!initial_guess.is_valid()) return invalid_result("initial guess is not a valid SE(3) transform", started);
  if (source.empty() || target.empty()) return invalid_result("source and target must be nonempty", started);
  for (const PointXYZ& point : source) {
    if (!finite_point(point)) return invalid_result("source contains a nonfinite point", started);
  }
  for (const PointXYZ& point : target) {
    if (!finite_point(point)) return invalid_result("target contains a nonfinite point", started);
  }

  PointCloud source_work = source;
  PointCloud target_work = target;
  const Clock::time_point preprocess_started = Clock::now();
  if (options.voxel_size > 0.0f) {
    source_work = voxel_downsample_cpu(source_work, options.voxel_size);
    target_work = voxel_downsample_cpu(target_work, options.voxel_size);
  }
  result.timing.preprocessing_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - preprocess_started).count();
  if (source_work.size() < options.min_correspondences ||
      target_work.size() < options.min_correspondences) {
    result.status = RegistrationStatus::InsufficientCorrespondences;
    result.message = "preprocessed clouds contain too few points";
    finish_timing(result, started);
    return result;
  }

  Transform current = initial_guess;
  float previous_error = std::numeric_limits<float>::infinity();
  for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
    PointCloud transformed(source_work.size());
    for (std::size_t i = 0; i < source_work.size(); ++i) {
      transformed[i] = current.apply(source_work[i]);
    }

    const Clock::time_point correspondence_started = Clock::now();
    const std::vector<Corr> correspondences =
        correspond_cpu(transformed, target_work, options.correspondence_radius);
    result.timing.correspondence_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - correspondence_started).count();
    std::size_t matched = 0;
    for (const Corr& corr : correspondences) {
      if (corr.idx >= 0) ++matched;
    }
    result.correspondences = matched;
    result.iterations = iteration + 1;
    if (matched == 0) {
      result.status = RegistrationStatus::NoCorrespondences;
      result.message = "no correspondences within radius";
      finish_timing(result, started);
      return result;
    }
    if (matched < options.min_correspondences) {
      result.status = RegistrationStatus::InsufficientCorrespondences;
      result.message = "fewer than min_correspondences matched points";
      finish_timing(result, started);
      return result;
    }

    const Clock::time_point solve_started = Clock::now();
    Transform delta;
    float error = -1.0f;
    const internal::SolveStatus solve_status =
        internal::solve_rigid(transformed, target_work, correspondences, delta, error);
    result.timing.solve_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - solve_started).count();
    if (solve_status == internal::SolveStatus::Degenerate) {
      result.status = RegistrationStatus::DegenerateGeometry;
      result.message = "correspondences do not constrain a rigid transform";
      finish_timing(result, started);
      return result;
    }
    if (solve_status == internal::SolveStatus::NumericalFailure) {
      result.status = RegistrationStatus::NumericalFailure;
      result.message = "rigid transform solve produced a nonfinite result";
      finish_timing(result, started);
      return result;
    }

    current = delta * current;
    result.transform = current;
    result.final_error = error;
    const float error_change = std::isfinite(previous_error)
                                   ? std::abs(previous_error - error)
                                   : std::numeric_limits<float>::infinity();
    previous_error = error;
    if (internal::transform_step(delta) <= options.convergence_tolerance ||
        error_change <= options.convergence_tolerance) {
      result.status = RegistrationStatus::Converged;
      result.converged = true;
      result.message = "converged";
      finish_timing(result, started);
      return result;
    }
  }

  result.transform = current;
  result.status = RegistrationStatus::MaxIterations;
  result.message = "maximum iterations reached";
  finish_timing(result, started);
  return result;
}

RegistrationResult align(const PointCloud& source, const PointCloud& target,
                         const Transform& initial_guess,
                         const ICPOptions& options) {
  return align_cpu(source, target, initial_guess, options);
}

const char* to_string(RegistrationStatus status) {
  switch (status) {
    case RegistrationStatus::Converged: return "converged";
    case RegistrationStatus::MaxIterations: return "max_iterations";
    case RegistrationStatus::InvalidInput: return "invalid_input";
    case RegistrationStatus::NoCorrespondences: return "no_correspondences";
    case RegistrationStatus::InsufficientCorrespondences: return "insufficient_correspondences";
    case RegistrationStatus::DegenerateGeometry: return "degenerate_geometry";
    case RegistrationStatus::NumericalFailure: return "numerical_failure";
    case RegistrationStatus::UnsupportedMethod: return "unsupported_method";
    case RegistrationStatus::CudaUnavailable: return "cuda_unavailable";
    case RegistrationStatus::CudaError: return "cuda_error";
  }
  return "unknown";
}

const char* to_string(ICPMethod method) {
  switch (method) {
    case ICPMethod::PointToPoint: return "point-to-point";
    case ICPMethod::PointToPlane: return "point-to-plane";
  }
  return "unknown";
}

const char* to_string(ExecutionBackend backend) {
  switch (backend) {
    case ExecutionBackend::Auto: return "auto";
    case ExecutionBackend::CPU: return "cpu";
    case ExecutionBackend::CUDA: return "cuda";
  }
  return "unknown";
}

}  // namespace flashicp
