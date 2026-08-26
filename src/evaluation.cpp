#include "flashicp/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace flashicp {
namespace {

double rotation_angle(const Transform& transform) {
  const double trace = static_cast<double>(transform.rotation[0]) + transform.rotation[4] +
                        transform.rotation[8];
  const double cosine = std::max(-1.0, std::min(1.0, (trace - 1.0) * 0.5));
  return std::acos(cosine);
}

double translation_norm(const Transform& transform) {
  return std::sqrt(static_cast<double>(transform.translation[0]) * transform.translation[0] +
                   static_cast<double>(transform.translation[1]) * transform.translation[1] +
                   static_cast<double>(transform.translation[2]) * transform.translation[2]);
}

void set_registration_statistics(EvaluationResult& result,
                                const std::vector<OdometryFrame>* records) {
  if (records == nullptr) return;
  std::vector<double> latencies;
  double correspondence_sum = 0.0;
  double iteration_sum = 0.0;
  for (std::size_t i = 0; i < records->size(); ++i) {
    if (!(*records)[i].success) ++result.failed_frames;
    if (!(*records)[i].registration_attempted) continue;
    const RegistrationResult& registration = (*records)[i].registration;
    ++result.registration_attempts;
    if (registration.status == RegistrationStatus::Converged)
      ++result.successful_registrations;
    else
      ++result.failed_registrations;
    correspondence_sum += static_cast<double>(registration.correspondences);
    iteration_sum += static_cast<double>(registration.iterations);
    latencies.push_back(registration.timing.total_ms);
  }
  if (latencies.empty()) return;
  result.average_correspondences = correspondence_sum / latencies.size();
  result.average_iterations = iteration_sum / latencies.size();
  result.mean_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                           static_cast<double>(latencies.size());
  std::sort(latencies.begin(), latencies.end());
  const std::size_t index = static_cast<std::size_t>(
      std::ceil(0.95 * static_cast<double>(latencies.size())));
  result.p95_latency_ms = latencies[std::min(latencies.size() - 1, index == 0 ? 0 : index - 1)];
}

}  // namespace

EvaluationResult evaluate_trajectory(const std::vector<Transform>& estimated,
                                     const std::vector<Transform>& ground_truth,
                                     const std::vector<OdometryFrame>* registration_frames) {
  EvaluationResult result;
  set_registration_statistics(result, registration_frames);
  for (const Transform& pose : estimated) {
    if (!pose.is_valid(2.0e-2f)) {
      result.status = EvaluationStatus::InvalidInput;
      result.message = "estimated trajectory contains an invalid transform";
      return result;
    }
  }
  for (const Transform& pose : ground_truth) {
    if (!pose.is_valid(2.0e-2f)) {
      result.status = EvaluationStatus::InvalidInput;
      result.message = "ground-truth trajectory contains an invalid transform";
      return result;
    }
  }
  result.frames.reserve(estimated.size());
  const std::size_t common = std::min(estimated.size(), ground_truth.size());
  for (std::size_t i = 0; i < estimated.size(); ++i) {
    TrajectoryFrameMetric metric;
    metric.frame_index = i;
    if (registration_frames != nullptr && i < registration_frames->size())
      metric.registration_success = (*registration_frames)[i].success;
    if (i < common) {
      metric.has_ground_truth = true;
      const Transform pose_error = ground_truth[i].inverse() * estimated[i];
      metric.translation_error_m = translation_norm(pose_error);
      metric.rotation_error_rad = rotation_angle(pose_error);
      ++result.evaluated_frames;
    } else {
      ++result.frames_without_ground_truth;
    }
    result.frames.push_back(metric);
  }

  if (ground_truth.empty()) {
    result.status = EvaluationStatus::NoGroundTruth;
    result.message = "no ground-truth poses supplied; trajectory metrics are unavailable";
    return result;
  }
  if (estimated.size() != ground_truth.size()) {
    result.status = EvaluationStatus::LengthMismatch;
    result.message = "estimated and ground-truth trajectories have different lengths; " +
                     std::to_string(common) + " common frames evaluated without padding";
  } else {
    result.status = EvaluationStatus::Success;
    result.message = "trajectory metrics computed";
  }
  if (result.evaluated_frames == 0) {
    result.message = "no common estimated/ground-truth frames";
    return result;
  }

  double translation_sum = 0.0;
  double translation_square_sum = 0.0;
  double rotation_sum = 0.0;
  double rotation_square_sum = 0.0;
  for (const TrajectoryFrameMetric& metric : result.frames) {
    if (!metric.has_ground_truth) continue;
    translation_sum += metric.translation_error_m;
    translation_square_sum += metric.translation_error_m * metric.translation_error_m;
    rotation_sum += metric.rotation_error_rad;
    rotation_square_sum += metric.rotation_error_rad * metric.rotation_error_rad;
    result.translation_error_max_m = std::max(result.translation_error_max_m,
                                               metric.translation_error_m);
    result.rotation_error_max_rad = std::max(result.rotation_error_max_rad,
                                              metric.rotation_error_rad);
  }
  const double count = static_cast<double>(result.evaluated_frames);
  result.translation_error_mean_m = translation_sum / count;
  result.translation_error_rmse_m = std::sqrt(translation_square_sum / count);
  result.translation_error_max_m = std::max(0.0, result.translation_error_max_m);
  result.rotation_error_mean_rad = rotation_sum / count;
  result.rotation_error_rmse_rad = std::sqrt(rotation_square_sum / count);
  result.rotation_error_max_rad = std::max(0.0, result.rotation_error_max_rad);
  result.ate_rmse_m = result.translation_error_rmse_m;

  double rpe_translation_square_sum = 0.0;
  double rpe_rotation_square_sum = 0.0;
  const std::size_t rpe_limit = std::min(estimated.size(), ground_truth.size());
  for (std::size_t i = 1; i < rpe_limit; ++i) {
    const Transform estimated_relative = estimated[i - 1].inverse() * estimated[i];
    const Transform ground_truth_relative = ground_truth[i - 1].inverse() * ground_truth[i];
    const Transform error = ground_truth_relative.inverse() * estimated_relative;
    const double translation_error = translation_norm(error);
    const double rotation_error = rotation_angle(error);
    rpe_translation_square_sum += translation_error * translation_error;
    rpe_rotation_square_sum += rotation_error * rotation_error;
    ++result.rpe_pairs;
  }
  if (result.rpe_pairs > 0) {
    const double pairs = static_cast<double>(result.rpe_pairs);
    result.rpe_translation_rmse_m = std::sqrt(rpe_translation_square_sum / pairs);
    result.rpe_rotation_rmse_rad = std::sqrt(rpe_rotation_square_sum / pairs);
  }
  return result;
}

const char* to_string(EvaluationStatus status) {
  switch (status) {
    case EvaluationStatus::Success: return "success";
    case EvaluationStatus::NoGroundTruth: return "no_ground_truth";
    case EvaluationStatus::LengthMismatch: return "length_mismatch";
    case EvaluationStatus::InvalidInput: return "invalid_input";
  }
  return "unknown";
}

}  // namespace flashicp
