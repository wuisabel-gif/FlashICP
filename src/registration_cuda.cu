#include "registration_internal.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using flashicp::Point;

struct DeviceTransform {
  float rotation[9];
  float translation[3];
};

__global__ void transform_kernel(const Point* source, int count,
                                 DeviceTransform transform, Point* output) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const Point p = source[i];
  output[i] = {
      transform.rotation[0] * p.x + transform.rotation[1] * p.y +
          transform.rotation[2] * p.z + transform.translation[0],
      transform.rotation[3] * p.x + transform.rotation[4] * p.y +
          transform.rotation[5] * p.z + transform.translation[1],
      transform.rotation[6] * p.x + transform.rotation[7] * p.y +
          transform.rotation[8] * p.z + transform.translation[2]};
}

__global__ void pair_error_kernel(const Point* source, const Point* target,
                                  const int* target_indices, int source_count,
                                  int target_count, float* errors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= source_count) return;
  const int target_index = target_indices[i];
  if (target_index < 0 || target_index >= target_count) {
    errors[i] = 0.0f;
    return;
  }
  const Point a = source[i];
  const Point b = target[target_index];
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  errors[i] = dx * dx + dy * dy + dz * dz;
}

struct DeviceBuffers {
  Point* source = nullptr;
  Point* target = nullptr;
  Point* transformed = nullptr;
  int* target_indices = nullptr;
  float* errors = nullptr;

  ~DeviceBuffers() {
    cudaFree(source);
    cudaFree(target);
    cudaFree(transformed);
    cudaFree(target_indices);
    cudaFree(errors);
  }

  bool allocate(std::size_t source_count, std::size_t target_count) {
    if (cudaMalloc(reinterpret_cast<void**>(&source),
                   source_count * sizeof(Point)) != cudaSuccess) return false;
    if (cudaMalloc(reinterpret_cast<void**>(&target),
                   target_count * sizeof(Point)) != cudaSuccess) return false;
    if (cudaMalloc(reinterpret_cast<void**>(&transformed),
                   source_count * sizeof(Point)) != cudaSuccess) return false;
    if (cudaMalloc(reinterpret_cast<void**>(&target_indices),
                   source_count * sizeof(int)) != cudaSuccess) return false;
    if (cudaMalloc(reinterpret_cast<void**>(&errors),
                   source_count * sizeof(float)) != cudaSuccess) return false;
    return true;
  }
};

bool finite_point(const Point& p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

bool validate_options(const flashicp::ICPOptions& options, std::string& message) {
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

DeviceTransform to_device_transform(const flashicp::Transform& transform) {
  DeviceTransform result{};
  for (int i = 0; i < 9; ++i) result.rotation[i] = transform.rotation[i];
  for (int i = 0; i < 3; ++i) result.translation[i] = transform.translation[i];
  return result;
}

bool check_kernel() {
  return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

void finish_timing(flashicp::RegistrationResult& result, Clock::time_point started) {
  result.timing.total_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

flashicp::RegistrationResult failure(flashicp::RegistrationStatus status,
                                     const flashicp::Transform& initial_guess,
                                     const std::string& message,
                                     Clock::time_point started) {
  flashicp::RegistrationResult result;
  result.transform = initial_guess;
  result.status = status;
  result.message = message;
  finish_timing(result, started);
  return result;
}

}  // namespace

namespace flashicp {
namespace internal {

RegistrationResult align_cuda(const PointCloud& source, const PointCloud& target,
                              const Transform& initial_guess,
                              const ICPOptions& options) {
  const Clock::time_point started = Clock::now();
  if (options.method == ICPMethod::PointToPlane) {
    // The CPU implementation remains the correctness oracle. Keep the public
    // CUDA API usable for this method while the device normal-equation kernel
    // is developed separately; backend_used makes this fallback explicit.
    RegistrationResult result = align_cpu(source, target, initial_guess, options);
    result.backend_used = ExecutionBackend::CPU;
    result.message = "point-to-plane CUDA accumulation is not enabled; CPU reference used";
    return result;
  }
  if (options.method != ICPMethod::PointToPoint) {
    return failure(RegistrationStatus::UnsupportedMethod, initial_guess,
                   "unsupported ICP method for CUDA path", started);
  }
  std::string validation_message;
  if (!validate_options(options, validation_message)) {
    return failure(RegistrationStatus::InvalidInput, initial_guess,
                   validation_message, started);
  }
  if (!initial_guess.is_valid()) {
    return failure(RegistrationStatus::InvalidInput, initial_guess,
                   "initial guess is not a valid SE(3) transform", started);
  }
  if (source.empty() || target.empty()) {
    return failure(RegistrationStatus::InvalidInput, initial_guess,
                   "source and target must be nonempty", started);
  }
  if (source.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      target.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return failure(RegistrationStatus::InvalidInput, initial_guess,
                   "cloud is too large for the CUDA implementation", started);
  }
  for (const Point& point : source) {
    if (!finite_point(point)) {
      return failure(RegistrationStatus::InvalidInput, initial_guess,
                     "source contains a nonfinite point", started);
    }
  }
  for (const Point& point : target) {
    if (!finite_point(point)) {
      return failure(RegistrationStatus::InvalidInput, initial_guess,
                     "target contains a nonfinite point", started);
    }
  }

  int device_count = 0;
  const cudaError_t device_error = cudaGetDeviceCount(&device_count);
  if (device_error == cudaErrorNoDevice || device_error == cudaErrorInsufficientDriver ||
      device_count == 0) {
    return failure(RegistrationStatus::CudaUnavailable, initial_guess,
                   "no usable CUDA device is available", started);
  }
  if (device_error != cudaSuccess) {
    return failure(RegistrationStatus::CudaError, initial_guess,
                   cudaGetErrorString(device_error), started);
  }

  PointCloud source_work = source;
  PointCloud target_work = target;
  const Clock::time_point preprocess_started = Clock::now();
  if (options.voxel_size > 0.0f) {
    source_work = voxel_downsample_cpu(source_work, options.voxel_size);
    target_work = voxel_downsample_cpu(target_work, options.voxel_size);
  }
  RegistrationResult result;
  result.transform = initial_guess;
  result.backend_used = ExecutionBackend::CUDA;
  result.timing.preprocessing_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - preprocess_started).count();
  if (source_work.size() < options.min_correspondences ||
      target_work.size() < options.min_correspondences ||
      source_work.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      target_work.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    result.status = RegistrationStatus::InsufficientCorrespondences;
    result.message = "preprocessed clouds contain too few or too many points";
    finish_timing(result, started);
    return result;
  }

  DeviceBuffers buffers;
  if (!buffers.allocate(source_work.size(), target_work.size())) {
    return failure(RegistrationStatus::CudaError, initial_guess,
                   "CUDA buffer allocation failed", started);
  }
  if (cudaMemcpy(buffers.source, source_work.data(), source_work.size() * sizeof(Point),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(buffers.target, target_work.data(), target_work.size() * sizeof(Point),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    return failure(RegistrationStatus::CudaError, initial_guess,
                   "cloud upload failed", started);
  }

  const int ns = static_cast<int>(source_work.size());
  const int nt = static_cast<int>(target_work.size());
  const int threads = 256;
  Transform current = initial_guess;
  float previous_error = std::numeric_limits<float>::infinity();

  for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
    const DeviceTransform device_transform = to_device_transform(current);
    const Clock::time_point transform_started = Clock::now();
    transform_kernel<<<(ns + threads - 1) / threads, threads>>>(
        buffers.source, ns, device_transform, buffers.transformed);
    if (!check_kernel()) {
      return failure(RegistrationStatus::CudaError, current,
                     "source transform kernel failed", started);
    }
    PointCloud transformed(source_work.size());
    if (cudaMemcpy(transformed.data(), buffers.transformed,
                   transformed.size() * sizeof(Point), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return failure(RegistrationStatus::CudaError, current,
                     "transformed cloud download failed", started);
    }
    result.timing.transform_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - transform_started).count();

    const Clock::time_point correspondence_started = Clock::now();
    const std::vector<Corr> correspondences =
        ::flashicp::correspond_gpu(transformed, target_work,
                                   options.correspondence_radius);
    result.timing.correspondence_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - correspondence_started).count();
    std::size_t matched = 0;
    std::vector<int> target_indices(correspondences.size(), -1);
    for (std::size_t i = 0; i < correspondences.size(); ++i) {
      target_indices[i] = correspondences[i].idx;
      if (correspondences[i].idx >= 0) ++matched;
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
    if (cudaMemcpy(buffers.target_indices, target_indices.data(),
                   target_indices.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
      return failure(RegistrationStatus::CudaError, current,
                     "correspondence index upload failed", started);
    }

    const Clock::time_point solve_started = Clock::now();
    Transform delta;
    float host_error = -1.0f;
    const SolveStatus solve_status =
        solve_rigid(transformed, target_work, correspondences, delta, host_error);
    result.timing.solve_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - solve_started).count();
    if (solve_status == SolveStatus::Degenerate) {
      result.status = RegistrationStatus::DegenerateGeometry;
      result.message = "correspondences do not constrain a rigid transform";
      finish_timing(result, started);
      return result;
    }
    if (solve_status == SolveStatus::NumericalFailure) {
      result.status = RegistrationStatus::NumericalFailure;
      result.message = "rigid transform solve produced a nonfinite result";
      finish_timing(result, started);
      return result;
    }

    current = delta * current;

    // Re-evaluate the accepted transform on the device. The small rigid solve
    // remains on the host for now, while residual accumulation is a CUDA stage.
    const DeviceTransform accepted_transform = to_device_transform(current);
    const Clock::time_point reduction_started = Clock::now();
    transform_kernel<<<(ns + threads - 1) / threads, threads>>>(
        buffers.source, ns, accepted_transform, buffers.transformed);
    if (!check_kernel()) {
      return failure(RegistrationStatus::CudaError, current,
                     "accepted-transform kernel failed", started);
    }
    pair_error_kernel<<<(ns + threads - 1) / threads, threads>>>(
        buffers.transformed, buffers.target, buffers.target_indices, ns, nt, buffers.errors);
    if (!check_kernel()) {
      return failure(RegistrationStatus::CudaError, current,
                     "residual kernel failed", started);
    }
    thrust::device_ptr<float> error_begin(buffers.errors);
    const float sum_squared = thrust::reduce(error_begin, error_begin + ns, 0.0f);
    if (cudaDeviceSynchronize() != cudaSuccess || !std::isfinite(sum_squared) ||
        sum_squared < 0.0f) {
      return failure(RegistrationStatus::CudaError, current,
                     "residual reduction failed", started);
    }
    result.timing.reduction_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - reduction_started).count();
    const float error = std::sqrt(sum_squared / static_cast<float>(matched));
    if (!std::isfinite(error)) {
      return failure(RegistrationStatus::NumericalFailure, current,
                     "CUDA residual is not finite", started);
    }
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

}  // namespace internal
}  // namespace flashicp
