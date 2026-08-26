#pragma once

#include "odometry.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace flashicp {

enum class EvaluationStatus {
  Success,
  NoGroundTruth,
  LengthMismatch,
  InvalidInput,
};

struct TrajectoryFrameMetric {
  std::size_t frame_index = 0;
  bool has_ground_truth = false;
  bool registration_success = false;
  double translation_error_m = -1.0;
  double rotation_error_rad = -1.0;
};

struct EvaluationResult {
  EvaluationStatus status = EvaluationStatus::InvalidInput;
  std::string message;
  std::vector<TrajectoryFrameMetric> frames;
  std::size_t evaluated_frames = 0;
  std::size_t frames_without_ground_truth = 0;
  double translation_error_mean_m = -1.0;
  double translation_error_rmse_m = -1.0;
  double translation_error_max_m = -1.0;
  double rotation_error_mean_rad = -1.0;
  double rotation_error_rmse_rad = -1.0;
  double rotation_error_max_rad = -1.0;
  // ATE is direct world-frame translation RMSE; no free similarity alignment
  // is applied. RPE is the adjacent-frame relative-pose RMSE.
  double ate_rmse_m = -1.0;
  double rpe_translation_rmse_m = -1.0;
  double rpe_rotation_rmse_rad = -1.0;
  std::size_t rpe_pairs = 0;

  std::size_t registration_attempts = 0;
  std::size_t successful_registrations = 0;
  std::size_t failed_registrations = 0;
  std::size_t failed_frames = 0;
  double average_correspondences = 0.0;
  double average_iterations = 0.0;
  double mean_latency_ms = 0.0;
  double p95_latency_ms = 0.0;
};

// Estimated and ground-truth poses both map frame coordinates into one world
// frame. The vectors are frame-indexed from zero. A length mismatch is reported
// rather than silently padding or inventing poses; metrics for the common
// prefix are still returned when one is available.
EvaluationResult evaluate_trajectory(
    const std::vector<Transform>& estimated,
    const std::vector<Transform>& ground_truth,
    const std::vector<OdometryFrame>* registration_frames = nullptr);

const char* to_string(EvaluationStatus status);

}  // namespace flashicp
