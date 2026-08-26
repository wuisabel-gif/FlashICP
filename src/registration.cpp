#include "flashicp/registration.hpp"

#include <algorithm>
#include <cmath>

namespace flashicp {

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
  RegistrationResult result;
  result.transform = initial_guess;
  result.status = RegistrationStatus::UnsupportedMethod;
  result.message = "point-to-point registration is added in the next milestone";
  (void)source;
  (void)target;
  (void)options;
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
