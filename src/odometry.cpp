#include "flashicp/odometry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace flashicp {
namespace {

struct LoadedScan {
  PointCloud points;
  std::size_t frame_index = 0;
  std::string message;
  bool ok = false;
};

using ScanLoader = std::function<LoadedScan(std::size_t)>;

bool registration_success(const RegistrationResult& result) {
  return result.status == RegistrationStatus::Converged;
}

void calculate_statistics(OdometryResult& result) {
  if (result.registration_attempts == 0) return;
  double correspondence_sum = 0.0;
  double iteration_sum = 0.0;
  std::vector<double> latencies;
  latencies.reserve(result.registration_attempts);
  for (std::size_t i = 0; i < result.frames.size(); ++i) {
    const RegistrationResult& registration = result.frames[i].registration;
    if (!result.frames[i].registration_attempted) continue;
    correspondence_sum += static_cast<double>(registration.correspondences);
    iteration_sum += static_cast<double>(registration.iterations);
    latencies.push_back(registration.timing.total_ms);
  }
  result.average_correspondences = correspondence_sum / result.registration_attempts;
  result.average_iterations = iteration_sum / result.registration_attempts;
  result.mean_latency_ms =
      std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
  std::sort(latencies.begin(), latencies.end());
  const std::size_t percentile_index = static_cast<std::size_t>(
      std::ceil(0.95 * static_cast<double>(latencies.size())));
  result.p95_latency_ms = latencies[std::min(latencies.size() - 1,
                                              percentile_index == 0 ? 0 : percentile_index - 1)];
}

OdometryResult run_odometry_impl(std::size_t scan_count, const OdometryOptions& options,
                                 const ScanLoader& loader) {
  OdometryResult result;
  if (scan_count == 0) {
    result.status = OdometryStatus::EmptySequence;
    result.message = "odometry requires at least one scan";
    return result;
  }
  LoadedScan reference = loader(0);
  if (!reference.ok || reference.points.empty()) {
    OdometryFrame failed_initial;
    failed_initial.frame_index = reference.frame_index;
    failed_initial.reference_frame = reference.frame_index;
    failed_initial.success = false;
    failed_initial.registration.status = RegistrationStatus::InvalidInput;
    failed_initial.registration.message = "failed to load initial scan: " + reference.message;
    result.frames.push_back(failed_initial);
    result.failed_frames = 1;
    result.status = OdometryStatus::ScanLoadFailure;
    result.message = "failed to load initial scan: " + reference.message;
    return result;
  }
  Transform reference_pose = Transform::identity();
  Transform previous_relative = Transform::identity();
  OdometryFrame initial;
  initial.frame_index = reference.frame_index;
  initial.reference_frame = reference.frame_index;
  initial.success = true;
  initial.relative_transform = Transform::identity();
  initial.pose = reference_pose;
  initial.registration.transform = Transform::identity();
  initial.registration.backend_used = ExecutionBackend::CPU;
  initial.registration.status = RegistrationStatus::Converged;
  initial.registration.converged = true;
  initial.registration.message = "initial frame; no registration attempted";
  result.frames.push_back(initial);

  for (std::size_t i = 1; i < scan_count; ++i) {
    LoadedScan current = loader(i);
    if (!current.ok || current.points.empty()) {
      OdometryFrame frame;
      frame.frame_index = current.frame_index;
      frame.reference_frame = reference.frame_index;
      frame.success = false;
      frame.registration_attempted = false;
      frame.pose = reference_pose;
      frame.registration.transform = Transform::identity();
      frame.registration.status = RegistrationStatus::InvalidInput;
      frame.registration.message = "failed to load scan: " + current.message;
      result.frames.push_back(frame);
      ++result.failed_frames;
      result.status = OdometryStatus::ScanLoadFailure;
      if (options.failure_policy == OdometryFailurePolicy::Stop) {
        result.message = "stopped after scan load failure at frame " +
                         std::to_string(current.frame_index);
        break;
      }
      if (options.failure_policy == OdometryFailurePolicy::Hold) {
        // A failed frame cannot become a reference because it has no points.
      }
      continue;
    }

    ICPOptions icp_options = options.icp;
    icp_options.backend = options.icp.backend;
    const Transform guess = options.use_previous_relative_guess ? previous_relative
                                                                  : Transform::identity();
    RegistrationResult registration =
        align(current.points, reference.points, guess, icp_options);
    OdometryFrame frame;
    frame.frame_index = current.frame_index;
    frame.reference_frame = reference.frame_index;
    frame.registration = registration;
    frame.registration_attempted = true;
    frame.relative_transform = registration.transform;
    frame.pose = reference_pose;
    ++result.registration_attempts;
    if (registration_success(registration)) {
      frame.success = true;
      frame.pose = reference_pose * registration.transform;
      result.frames.push_back(frame);
      ++result.successful_registrations;
      reference = std::move(current);
      reference_pose = frame.pose;
      previous_relative = registration.transform;
    } else {
      frame.success = false;
      // A failed solve never contributes its untrusted relative transform to
      // the trajectory. Holding the last valid pose is explicit in the record.
      frame.pose = reference_pose;
      result.frames.push_back(frame);
      ++result.failed_registrations;
      ++result.failed_frames;
      // The previous relative motion may have caused this failure and is not
      // reused across a held/skipped frame.
      previous_relative = Transform::identity();
      if (options.failure_policy == OdometryFailurePolicy::Stop) {
        result.status = OdometryStatus::RegistrationFailure;
        result.message = "stopped after registration failure at frame " +
                         std::to_string(current.frame_index) + ": " + registration.message;
        break;
      }
      if (options.failure_policy == OdometryFailurePolicy::Hold) {
        // Continue from the current scan, but keep the world pose held. This is
        // useful for a temporary ICP failure and is visible via success=false.
        reference = std::move(current);
      }
      // Skip keeps the last successful scan as reference, leaving an explicit
      // reference_frame gap in subsequent records.
    }
  }

  if (result.status == OdometryStatus::InvalidInput) {
    result.status = result.failed_registrations == 0 ? OdometryStatus::Success
                                                      : OdometryStatus::RegistrationFailure;
    if (result.failed_registrations == 0)
      result.message = "odometry completed";
    else
      result.message = "odometry completed with held/skipped failed frames";
  }
  calculate_statistics(result);
  return result;
}

}  // namespace

OdometryResult run_odometry(const std::vector<PointCloud>& scans,
                            const OdometryOptions& options) {
  return run_odometry_impl(scans.size(), options, [&scans](std::size_t index) {
    LoadedScan result;
    result.frame_index = index;
    if (index >= scans.size()) {
      result.message = "in-memory scan index is out of range";
      return result;
    }
    result.points = scans[index];
    result.ok = true;
    result.message = "loaded in-memory scan";
    return result;
  });
}

OdometryResult run_kitti_odometry(const KittiSequence& sequence,
                                  const OdometryOptions& options) {
  return run_odometry_impl(sequence.scan_paths.size(), options,
                           [&sequence](std::size_t index) {
                             LoadedScan result;
                             if (index >= sequence.scan_paths.size()) {
                               result.message = "KITTI scan index is out of range";
                               return result;
                             }
                             const KittiScanResult loaded = sequence.load_scan(index);
                             result.frame_index = loaded.frame_index;
                             result.points = loaded.points;
                             result.ok = loaded.ok();
                             result.message = loaded.message;
                             return result;
                           });
}

const char* to_string(OdometryFailurePolicy policy) {
  switch (policy) {
    case OdometryFailurePolicy::Stop: return "stop";
    case OdometryFailurePolicy::Hold: return "hold";
    case OdometryFailurePolicy::Skip: return "skip";
  }
  return "unknown";
}

const char* to_string(OdometryStatus status) {
  switch (status) {
    case OdometryStatus::Success: return "success";
    case OdometryStatus::InvalidInput: return "invalid_input";
    case OdometryStatus::EmptySequence: return "empty_sequence";
    case OdometryStatus::ScanLoadFailure: return "scan_load_failure";
    case OdometryStatus::RegistrationFailure: return "registration_failure";
  }
  return "unknown";
}

}  // namespace flashicp
