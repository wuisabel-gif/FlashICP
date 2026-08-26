#include "flashicp/registration.hpp"

#ifdef FLASHICP_CUDA
#include "registration_internal.hpp"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using flashicp::ICPMethod;
using flashicp::ICPOptions;
using flashicp::PointCloud;
using flashicp::PointXYZ;
using flashicp::RegistrationStatus;
using flashicp::Transform;

namespace {

bool near(float a, float b, float tolerance) {
  return std::abs(a - b) <= tolerance;
}

bool near_point(const PointXYZ& a, const PointXYZ& b, float tolerance) {
  return near(a.x, b.x, tolerance) && near(a.y, b.y, tolerance) &&
         near(a.z, b.z, tolerance);
}

Transform z_transform(float angle, float tx, float ty, float tz) {
  Transform transform;
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  transform.rotation = {{c, -s, 0.0f, s, c, 0.0f, 0.0f, 0.0f, 1.0f}};
  transform.translation = {{tx, ty, tz}};
  return transform;
}

Transform y_rotation(float angle) {
  Transform transform;
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  transform.rotation = {{c, 0.0f, s, 0.0f, 1.0f, 0.0f, -s, 0.0f, c}};
  return transform;
}

PointCloud fixture() {
  return {{0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f},
          {0.0f, 0.0f, 2.0f}, {2.0f, 2.0f, 0.0f}, {2.0f, 0.0f, 2.0f},
          {0.0f, 2.0f, 2.0f}, {1.3f, 0.4f, 1.7f}};
}

PointCloud apply_cloud(const PointCloud& cloud, const Transform& transform) {
  PointCloud result;
  result.reserve(cloud.size());
  for (const PointXYZ& point : cloud) result.push_back(transform.apply(point));
  return result;
}

void assert_transform_matches(const Transform& actual, const Transform& expected,
                              float tolerance) {
  const PointCloud points = fixture();
  for (const PointXYZ& point : points) {
    assert(near_point(actual.apply(point), expected.apply(point), tolerance));
  }
}

void test_transform_algebra() {
  const Transform a = z_transform(0.2f, 1.0f, -0.3f, 0.5f);
  const Transform b = z_transform(-0.1f, -0.4f, 0.8f, 0.2f);
  const PointXYZ point{0.7f, -1.2f, 2.0f};
  assert(near_point((a * b).apply(point), a.apply(b.apply(point)), 1.0e-5f));
  assert(near_point(a.inverse().apply(a.apply(point)), point, 1.0e-5f));
}

void test_known_transform() {
  const PointCloud source = fixture();
  const Transform expected =
      y_rotation(0.07f) * z_transform(0.12f, 0.22f, -0.17f, 0.13f);
  const PointCloud target = apply_cloud(source, expected);

  ICPOptions options;
  options.backend = flashicp::ExecutionBackend::CPU;
  options.correspondence_radius = 0.75f;
  options.max_iterations = 40;
  options.convergence_tolerance = 1.0e-6f;
  const auto result = flashicp::align_cpu(source, target, Transform::identity(), options);

  assert(result.status == RegistrationStatus::Converged);
  assert(result.converged);
  assert(result.correspondences == source.size());
  assert(result.final_error >= 0.0f && result.final_error < 1.0e-4f);
  assert_transform_matches(result.transform, expected, 2.0e-3f);
}

void test_noise_and_outlier_rejection() {
  const PointCloud source = fixture();
  const Transform expected = z_transform(-0.08f, -0.14f, 0.12f, -0.09f);
  PointCloud target = apply_cloud(source, expected);
  const float noise[] = {0.002f, -0.003f, 0.001f, -0.001f,
                         0.003f, 0.002f, -0.002f, 0.001f};
  for (std::size_t i = 0; i < target.size(); ++i) {
    target[i].x += noise[i];
    target[i].y -= noise[i] * 0.5f;
  }
  target.push_back({100.0f, 100.0f, 100.0f});

  ICPOptions options;
  options.backend = flashicp::ExecutionBackend::CPU;
  options.correspondence_radius = 0.5f;
  options.max_iterations = 40;
  options.convergence_tolerance = 1.0e-5f;
  const auto result = flashicp::align_cpu(source, target, Transform::identity(), options);

  assert(result.status == RegistrationStatus::Converged);
  assert(result.correspondences == source.size());
  assert(result.final_error < 0.01f);
  assert_transform_matches(result.transform, expected, 0.02f);
}

void test_failures() {
  ICPOptions options;
  options.backend = flashicp::ExecutionBackend::CPU;
  options.correspondence_radius = 0.5f;

  const PointCloud source = fixture();
  const PointCloud far_target = apply_cloud(source, z_transform(0.0f, 10.0f, 0.0f, 0.0f));
  auto result = flashicp::align_cpu(source, far_target, Transform::identity(), options);
  assert(result.status == RegistrationStatus::NoCorrespondences);
  assert(!result.converged);

  PointCloud invalid = source;
  invalid[0].x = std::numeric_limits<float>::quiet_NaN();
  result = flashicp::align_cpu(invalid, source, Transform::identity(), options);
  assert(result.status == RegistrationStatus::InvalidInput);

  Transform invalid_guess = Transform::identity();
  invalid_guess.rotation[0] = 2.0f;
  result = flashicp::align_cpu(source, source, invalid_guess, options);
  assert(result.status == RegistrationStatus::InvalidInput);

  PointCloud collinear{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
                       {2.0f, 0.0f, 0.0f}, {3.0f, 0.0f, 0.0f}};
  result = flashicp::align_cpu(collinear, collinear, Transform::identity(), options);
  assert(result.status == RegistrationStatus::DegenerateGeometry);

  options.correspondence_radius = 0.0f;
  result = flashicp::align_cpu(source, source, Transform::identity(), options);
  assert(result.status == RegistrationStatus::InvalidInput);

  options.correspondence_radius = 0.5f;
  options.method = ICPMethod::PointToPlane;
  result = flashicp::align_cpu(source, source, Transform::identity(), options);
  assert(result.status == RegistrationStatus::UnsupportedMethod);
}

void test_public_dispatch() {
  ICPOptions options;
  options.backend = flashicp::ExecutionBackend::Auto;
  options.correspondence_radius = 0.75f;
  const PointCloud source = fixture();
  const Transform expected = z_transform(0.03f, 0.05f, 0.02f, -0.03f);
  const auto result = flashicp::align(source, apply_cloud(source, expected),
                                      Transform::identity(), options);
  assert(result.status == RegistrationStatus::Converged);
  assert_transform_matches(result.transform, expected, 2.0e-3f);
}

#ifdef FLASHICP_CUDA
int test_cuda_agreement() {
  ICPOptions options;
  options.backend = flashicp::ExecutionBackend::CUDA;
  options.correspondence_radius = 0.75f;
  options.max_iterations = 40;
  options.convergence_tolerance = 1.0e-6f;
  const PointCloud source = fixture();
  const Transform expected = z_transform(0.12f, 0.22f, -0.17f, 0.13f);
  const PointCloud target = apply_cloud(source, expected);
  const auto result = flashicp::align(source, target, Transform::identity(), options);
  if (result.status == RegistrationStatus::CudaUnavailable) {
    std::printf("test_registration: CUDA unavailable, skipped\n");
    return 77;
  }
  assert(result.status == RegistrationStatus::Converged);
  assert(result.final_error < 1.0e-4f);
  assert_transform_matches(result.transform, expected, 2.0e-3f);

  const auto cpu_corr = flashicp::correspond_cpu(source, target, options.correspondence_radius);
  const auto gpu_corr = flashicp::correspond_gpu(source, target, options.correspondence_radius);
  assert(cpu_corr.size() == gpu_corr.size());
  for (std::size_t i = 0; i < cpu_corr.size(); ++i) {
    assert((cpu_corr[i].idx < 0) == (gpu_corr[i].idx < 0));
    if (cpu_corr[i].idx >= 0) {
      assert(std::abs(cpu_corr[i].d2 - gpu_corr[i].d2) < 1.0e-5f);
    }
  }
  return 0;
}
#endif

}  // namespace

int main() {
  test_transform_algebra();
  test_known_transform();
  test_noise_and_outlier_rejection();
  test_failures();
  test_public_dispatch();
#ifdef FLASHICP_CUDA
  const int cuda_result = test_cuda_agreement();
  if (cuda_result != 0) return cuda_result;
#endif
  std::printf("test_registration: PASS\n");
  return 0;
}
