#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace flashicp {

struct PointXYZ {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

using PointCloud = std::vector<PointXYZ>;

// Rigid transform in row-major form. It maps a point from the source frame into
// the target frame: q = transform.apply(p). Composition a * b means apply b,
// then apply a.
struct Transform {
  std::array<float, 9> rotation{{1.0f, 0.0f, 0.0f,
                                  0.0f, 1.0f, 0.0f,
                                  0.0f, 0.0f, 1.0f}};
  std::array<float, 3> translation{{0.0f, 0.0f, 0.0f}};

  static Transform identity();
  PointXYZ apply(const PointXYZ& point) const;
  Transform inverse() const;
  bool is_finite() const;
  bool is_valid(float tolerance = 1.0e-3f) const;
};

Transform operator*(const Transform& lhs, const Transform& rhs);

// Auto uses CUDA when the library was built with CUDA and a device is available;
// otherwise it uses the CPU reference. CPU and CUDA can always be selected
// explicitly through ICPOptions::backend.
enum class ExecutionBackend {
  Auto,
  CPU,
  CUDA,
};

enum class ICPMethod {
  PointToPoint,
  PointToPlane,
};

enum class RegistrationStatus {
  Converged,
  MaxIterations,
  InvalidInput,
  NoCorrespondences,
  InsufficientCorrespondences,
  DegenerateGeometry,
  NormalEstimationFailure,
  NumericalFailure,
  UnsupportedMethod,
  CudaUnavailable,
  CudaError,
};

struct ICPOptions {
  ICPMethod method = ICPMethod::PointToPoint;
  ExecutionBackend backend = ExecutionBackend::Auto;
  int max_iterations = 20;
  float correspondence_radius = 0.20f;
  float voxel_size = 0.0f;  // <= 0 disables optional CPU preprocessing.
  float convergence_tolerance = 1.0e-5f;
  std::size_t min_correspondences = 3;
  // The CPU implementation uses an exact radius hash by default; disabling it
  // selects the O(source*target) brute-force correctness oracle.
  bool use_spatial_hash = true;
  int normal_k_neighbors = 8;  // point-to-plane target normal neighbourhood.
  float normal_search_radius = 0.0f;  // <= 0 selects the documented local default.
};

struct TimingInfo {
  double total_ms = 0.0;
  double preprocessing_ms = 0.0;
  double transform_ms = 0.0;
  double correspondence_ms = 0.0;
  double reduction_ms = 0.0;
  double solve_ms = 0.0;
  double normal_estimation_ms = 0.0;
  double jacobian_ms = 0.0;
};

struct RegistrationResult {
  Transform transform = Transform::identity();
  ExecutionBackend backend_used = ExecutionBackend::Auto;
  RegistrationStatus status = RegistrationStatus::InvalidInput;
  bool converged = false;
  int iterations = 0;
  float final_error = -1.0f;  // RMS residual in metres (point or point-to-plane).
  std::size_t correspondences = 0;
  TimingInfo timing;
  std::string message;
};

// Public registration entry point. The transform maps source-frame points into
// the target frame. Point-to-plane estimates target normals and uses a left
// incremental SE(3) update; point-to-point remains available as the baseline.
RegistrationResult align(const PointCloud& source,
                         const PointCloud& target,
                         const Transform& initial_guess,
                         const ICPOptions& options = {});

// CPU reference implementation used as the correctness oracle for CUDA.
RegistrationResult align_cpu(const PointCloud& source,
                             const PointCloud& target,
                             const Transform& initial_guess,
                             const ICPOptions& options = {});

const char* to_string(RegistrationStatus status);
const char* to_string(ICPMethod method);
const char* to_string(ExecutionBackend backend);

// Dependency-free 6x6 normal-equation representation and solver. `rhs` is the
// already-signed right-hand side (normally -J^T r), so the returned solution x
// satisfies ata*x = rhs. The solver uses pivoted double-precision elimination
// and rejects singular or poorly conditioned systems.
struct NormalEquation6 {
  std::array<double, 36> ata{};
  std::array<double, 6> rhs{};
  std::size_t correspondences = 0;
};

enum class LinearSolveStatus {
  Success,
  Singular,
  NumericalFailure,
};

LinearSolveStatus solve_normal_equation(const NormalEquation6& equation,
                                        std::array<double, 6>& solution,
                                        double relative_pivot_tolerance = 1.0e-12);

const char* to_string(LinearSolveStatus status);

}  // namespace flashicp
