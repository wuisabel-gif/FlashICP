#pragma once

#include "kitti.hpp"
#include "registration.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace flashicp {

enum class OdometryFailurePolicy {
  Stop,
  Hold,
  Skip,
};

struct OdometryOptions {
  ICPOptions icp;
  OdometryFailurePolicy failure_policy = OdometryFailurePolicy::Stop;
  bool use_previous_relative_guess = true;
};

enum class OdometryStatus {
  Success,
  InvalidInput,
  EmptySequence,
  ScanLoadFailure,
  RegistrationFailure,
};

struct OdometryFrame {
  std::size_t frame_index = 0;
  // For normal operation this is frame_index - 1. With skip policy it is the
  // last successful reference frame, and is recorded to make the gap explicit.
  std::size_t reference_frame = 0;
  bool registration_attempted = false;
  // True only when an ICP solve was actually run; scan-load failures remain
  // visible records without being counted as registration attempts.
  bool success = false;
  Transform relative_transform = Transform::identity();
  // Maps the current frame into the fixed world frame. The initial frame is
  // world frame 0; a successful T_reference_current is composed as
  // T_world_current = T_world_reference * T_reference_current.
  Transform pose = Transform::identity();
  RegistrationResult registration;
};

struct OdometryResult {
  std::vector<OdometryFrame> frames;
  OdometryStatus status = OdometryStatus::InvalidInput;
  std::string message;
  std::size_t registration_attempts = 0;
  std::size_t successful_registrations = 0;
  std::size_t failed_registrations = 0;
  std::size_t failed_frames = 0;
  double average_correspondences = 0.0;
  double average_iterations = 0.0;
  double mean_latency_ms = 0.0;
  double p95_latency_ms = 0.0;
};

// Run registration over in-memory scans. This is also the deterministic CPU
// test boundary; no mapping, loop closure, or graph optimization is performed.
OdometryResult run_odometry(const std::vector<PointCloud>& scans,
                            const OdometryOptions& options = {});

// Load and process a KITTI sequence lazily, retaining only the reference scan
// and one current scan at a time.
OdometryResult run_kitti_odometry(const KittiSequence& sequence,
                                  const OdometryOptions& options = {});

const char* to_string(OdometryFailurePolicy policy);
const char* to_string(OdometryStatus status);

}  // namespace flashicp
