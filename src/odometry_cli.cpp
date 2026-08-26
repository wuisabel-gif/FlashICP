#include "odometry_cli.hpp"

#include "flashicp/evaluation.hpp"
#include "flashicp/kitti.hpp"
#include "flashicp/odometry.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace flashicp {
namespace {

struct Config {
  std::string dataset;
  std::string sequence;
  std::string method = "point-to-plane";
  std::string backend = "auto";
  std::string output = "trajectory.csv";
  std::string poses;
  std::string calibration;
  std::string poses_frame = "camera";
  std::string failure_policy = "stop";
  float radius = 1.0f;
  float voxel_size = 0.0f;
  float tolerance = 1.0e-5f;
  float normal_radius = 0.0f;
  int max_iterations = 20;
  int min_correspondences = 3;
  int normal_neighbors = 8;
};

void print_odometry_help() {
  std::cout << "usage: flashicp odometry --dataset kitti --sequence SEQUENCES/00 [options]\n"
            << "options:\n"
            << "  --method point-to-point|point-to-plane (default point-to-plane)\n"
            << "  --backend auto|cpu|cuda (default auto)\n"
            << "  --radius METRES              correspondence radius (default 1.0)\n"
            << "  --voxel-size METRES          optional CPU voxel preprocessing\n"
            << "  --max-iterations N           (default 20)\n"
            << "  --tolerance VALUE            transform/error convergence tolerance\n"
            << "  --min-correspondences N      (default 3)\n"
            << "  --normal-neighbors N         point-to-plane normal neighbourhood\n"
            << "  --normal-radius METRES       zero selects local one-metre cells\n"
            << "  --on-failure stop|hold|skip  (default stop)\n"
            << "  --output PATH                .csv or .json (default trajectory.csv)\n"
            << "  --poses PATH                 KITTI poses file, optional\n"
            << "  --calib PATH                 KITTI calib.txt, required for camera poses\n"
            << "  --poses-frame camera|lidar  frame of --poses (default camera)\n";
}

bool parse_float(const std::string& text, float& value) {
  try {
    std::size_t consumed = 0;
    const float parsed = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(parsed)) return false;
    value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_int(const std::string& text, int& value) {
  try {
    std::size_t consumed = 0;
    const long parsed = std::stol(text, &consumed);
    if (consumed != text.size() || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

bool next_value(int& index, int argc, char** argv, std::string& value) {
  if (index + 1 >= argc) return false;
  value = argv[++index];
  return !value.empty();
}

bool parse_config(int argc, char** argv, Config& config, std::string& error) {
  for (int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help" || argument == "-h") {
      print_odometry_help();
      return false;
    }
    std::string value;
    if (argument == "--dataset") {
      if (!next_value(i, argc, argv, value)) { error = "--dataset requires a value"; return false; }
      config.dataset = value;
    } else if (argument == "--sequence") {
      if (!next_value(i, argc, argv, value)) { error = "--sequence requires a value"; return false; }
      config.sequence = value;
    } else if (argument == "--method") {
      if (!next_value(i, argc, argv, value)) { error = "--method requires a value"; return false; }
      config.method = value;
    } else if (argument == "--backend") {
      if (!next_value(i, argc, argv, value)) { error = "--backend requires a value"; return false; }
      config.backend = value;
    } else if (argument == "--output") {
      if (!next_value(i, argc, argv, value)) { error = "--output requires a value"; return false; }
      config.output = value;
    } else if (argument == "--poses") {
      if (!next_value(i, argc, argv, value)) { error = "--poses requires a value"; return false; }
      config.poses = value;
    } else if (argument == "--calib" || argument == "--calibration") {
      if (!next_value(i, argc, argv, value)) { error = "--calib requires a value"; return false; }
      config.calibration = value;
    } else if (argument == "--poses-frame") {
      if (!next_value(i, argc, argv, value)) { error = "--poses-frame requires a value"; return false; }
      config.poses_frame = value;
    } else if (argument == "--on-failure") {
      if (!next_value(i, argc, argv, value)) { error = "--on-failure requires a value"; return false; }
      config.failure_policy = value;
    } else if (argument == "--radius") {
      if (!next_value(i, argc, argv, value) || !parse_float(value, config.radius)) { error = "--radius must be finite"; return false; }
    } else if (argument == "--voxel-size") {
      if (!next_value(i, argc, argv, value) || !parse_float(value, config.voxel_size)) { error = "--voxel-size must be finite"; return false; }
    } else if (argument == "--tolerance" || argument == "--convergence-tolerance") {
      if (!next_value(i, argc, argv, value) || !parse_float(value, config.tolerance)) { error = "--tolerance must be finite"; return false; }
    } else if (argument == "--normal-radius") {
      if (!next_value(i, argc, argv, value) || !parse_float(value, config.normal_radius)) { error = "--normal-radius must be finite"; return false; }
    } else if (argument == "--max-iterations") {
      if (!next_value(i, argc, argv, value) || !parse_int(value, config.max_iterations)) { error = "--max-iterations must be an integer"; return false; }
    } else if (argument == "--min-correspondences") {
      if (!next_value(i, argc, argv, value) || !parse_int(value, config.min_correspondences)) { error = "--min-correspondences must be an integer"; return false; }
    } else if (argument == "--normal-neighbors") {
      if (!next_value(i, argc, argv, value) || !parse_int(value, config.normal_neighbors)) { error = "--normal-neighbors must be an integer"; return false; }
    } else {
      error = "unknown odometry option: " + argument;
      return false;
    }
  }
  if (config.dataset != "kitti") { error = "--dataset must be kitti"; return false; }
  if (config.sequence.empty()) { error = "--sequence is required"; return false; }
  if (config.method == "point-to-point") {
    // accepted
  } else if (config.method == "point-to-plane") {
    // accepted
  } else { error = "--method must be point-to-point or point-to-plane"; return false; }
  if (config.backend != "auto" && config.backend != "cpu" && config.backend != "cuda") {
    error = "--backend must be auto, cpu, or cuda"; return false;
  }
  if (config.poses_frame != "camera" && config.poses_frame != "lidar") {
    error = "--poses-frame must be camera or lidar"; return false;
  }
  if (config.failure_policy != "stop" && config.failure_policy != "hold" && config.failure_policy != "skip") {
    error = "--on-failure must be stop, hold, or skip"; return false;
  }
  if (config.radius <= 0.0f || config.voxel_size < 0.0f || config.tolerance < 0.0f ||
      config.normal_radius < 0.0f || config.max_iterations <= 0 || config.min_correspondences < 3 ||
      config.normal_neighbors < 3) {
    error = "ICP numeric options are outside their valid ranges";
    return false;
  }
  return true;
}

std::string json_escape(const std::string& text) {
  std::string escaped;
  for (const char character : text) {
    if (character == '\\') escaped += "\\\\";
    else if (character == '"') escaped += "\\\"";
    else if (character == '\n') escaped += "\\n";
    else if (character == '\r') escaped += "\\r";
    else if (character == '\t') escaped += "\\t";
    else escaped += character;
  }
  return escaped;
}

void json_number(std::ostream& output, double value) {
  if (std::isfinite(value)) output << std::setprecision(10) << value;
  else output << "null";
}

void write_transform_json(std::ostream& output, const Transform& transform) {
  output << "[";
  for (std::size_t i = 0; i < transform.rotation.size(); ++i) {
    if (i) output << ",";
    json_number(output, transform.rotation[i]);
  }
  for (float value : transform.translation) {
    output << ",";
    json_number(output, value);
  }
  output << "]";
}

void write_evaluation_object(std::ostream& output, const EvaluationResult& evaluation) {
  output << "{\"status\":\"" << to_string(evaluation.status) << "\",\"message\":\""
         << json_escape(evaluation.message) << "\",\"evaluated_frames\":"
         << evaluation.evaluated_frames << ",\"total_frames\":" << evaluation.frames.size()
         << ",\"frames_without_ground_truth\":"
         << evaluation.frames_without_ground_truth << ",\"translation_error_mean_m\":";
  json_number(output, evaluation.translation_error_mean_m);
  output << ",\"translation_error_rmse_m\":";
  json_number(output, evaluation.translation_error_rmse_m);
  output << ",\"translation_error_max_m\":";
  json_number(output, evaluation.translation_error_max_m);
  output << ",\"rotation_error_mean_rad\":";
  json_number(output, evaluation.rotation_error_mean_rad);
  output << ",\"rotation_error_rmse_rad\":";
  json_number(output, evaluation.rotation_error_rmse_rad);
  output << ",\"rotation_error_max_rad\":";
  json_number(output, evaluation.rotation_error_max_rad);
  output << ",\"ate_rmse_m\":";
  json_number(output, evaluation.ate_rmse_m);
  output << ",\"rpe_translation_rmse_m\":";
  json_number(output, evaluation.rpe_translation_rmse_m);
  output << ",\"rpe_rotation_rmse_rad\":";
  json_number(output, evaluation.rpe_rotation_rmse_rad);
  output << ",\"rpe_pairs\":" << evaluation.rpe_pairs
         << ",\"registration_attempts\":" << evaluation.registration_attempts
         << ",\"successful_registrations\":" << evaluation.successful_registrations
         << ",\"failed_registrations\":" << evaluation.failed_registrations
         << ",\"failed_frames\":" << evaluation.failed_frames
         << ",\"average_correspondences\":";
  json_number(output, evaluation.average_correspondences);
  output << ",\"average_iterations\":";
  json_number(output, evaluation.average_iterations);
  output << ",\"mean_latency_ms\":";
  json_number(output, evaluation.mean_latency_ms);
  output << ",\"p95_latency_ms\":";
  json_number(output, evaluation.p95_latency_ms);
  output << ",\"frame_metrics\":[";
  for (std::size_t i = 0; i < evaluation.frames.size(); ++i) {
    if (i) output << ",";
    const TrajectoryFrameMetric& metric = evaluation.frames[i];
    output << "{\"frame\":" << metric.frame_index << ",\"has_ground_truth\":"
           << (metric.has_ground_truth ? "true" : "false")
           << ",\"registration_success\":"
           << (metric.registration_success ? "true" : "false")
           << ",\"translation_error_m\":";
    json_number(output, metric.translation_error_m);
    output << ",\"rotation_error_rad\":";
    json_number(output, metric.rotation_error_rad);
    output << "}";
  }
  output << "]}";
}

void write_summary_object(std::ostream& output, const OdometryResult& odometry) {
  output << "{\"status\":\"" << to_string(odometry.status) << "\",\"message\":\""
         << json_escape(odometry.message) << "\",\"total_frames\":" << odometry.frames.size()
         << ",\"registration_attempts\":"
         << odometry.registration_attempts << ",\"successful_registrations\":"
         << odometry.successful_registrations << ",\"failed_registrations\":"
         << odometry.failed_registrations << ",\"failed_frames\":" << odometry.failed_frames
         << ",\"average_correspondences\":";
  json_number(output, odometry.average_correspondences);
  output << ",\"average_iterations\":";
  json_number(output, odometry.average_iterations);
  output << ",\"mean_latency_ms\":";
  json_number(output, odometry.mean_latency_ms);
  output << ",\"p95_latency_ms\":";
  json_number(output, odometry.p95_latency_ms);
  output << "}";
}

bool write_json(const std::filesystem::path& path, const Config& config,
                const OdometryResult& odometry, const EvaluationResult* evaluation) {
  std::ofstream output(path);
  if (!output) return false;
  output << "{\n  \"schema_version\": 1,\n  \"dataset\": \"kitti\",\n  \"sequence\": \""
         << json_escape(config.sequence) << "\",\n  \"method\": \"" << config.method
         << "\",\n  \"backend_requested\": \"" << config.backend
         << "\",\n  \"transform_convention\": \"T_world_current maps current Velodyne frame coordinates into the fixed initial-scan world frame; each ICP result maps current source into reference target\",\n"
         << "  \"failure_policy\": \"" << config.failure_policy << "\",\n"
         << "  \"frames\": [\n";
  for (std::size_t i = 0; i < odometry.frames.size(); ++i) {
    const OdometryFrame& frame = odometry.frames[i];
    const RegistrationResult& registration = frame.registration;
    if (i) output << ",\n";
    output << "    {\"frame\":" << frame.frame_index << ",\"reference_frame\":"
           << frame.reference_frame << ",\"registration_attempted\":"
           << (frame.registration_attempted ? "true" : "false")
           << ",\"success\":" << (frame.success ? "true" : "false")
           << ",\"status\":\"" << to_string(registration.status)
           << "\",\"message\":\"" << json_escape(registration.message)
           << "\",\"backend_used\":\"" << to_string(registration.backend_used)
           << "\",\"relative_transform\":";
    write_transform_json(output, frame.relative_transform);
    output << ",\"pose\":";
    write_transform_json(output, frame.pose);
    output << ",\"final_error_m\":";
    json_number(output, registration.final_error);
    output << ",\"correspondences\":" << registration.correspondences
           << ",\"iterations\":" << registration.iterations << ",\"latency_ms\":";
    json_number(output, registration.timing.total_ms);
    output << ",\"preprocessing_ms\":";
    json_number(output, registration.timing.preprocessing_ms);
    output << ",\"normal_estimation_ms\":";
    json_number(output, registration.timing.normal_estimation_ms);
    output << ",\"correspondence_ms\":";
    json_number(output, registration.timing.correspondence_ms);
    output << ",\"solve_ms\":";
    json_number(output, registration.timing.solve_ms);
    output << "}";
  }
  output << "\n  ],\n  \"summary\": ";
  write_summary_object(output, odometry);
  output << ",\n  \"evaluation\": ";
  if (evaluation == nullptr) output << "null";
  else write_evaluation_object(output, *evaluation);
  output << "\n}\n";
  return static_cast<bool>(output);
}

void csv_transform(std::ostream& output, const Transform& transform) {
  for (float value : transform.rotation) output << "," << value;
  for (float value : transform.translation) output << "," << value;
}

bool write_csv(const std::filesystem::path& path, const Config& config,
               const OdometryResult& odometry, const EvaluationResult* evaluation) {
  std::ofstream output(path);
  if (!output) return false;
  output << "# schema_version=1\n# dataset=kitti\n# sequence=" << config.sequence
         << "\n# method=" << config.method << "\n# backend_requested=" << config.backend
         << "\n# transform_convention=T_world_current maps current Velodyne coordinates into the initial-scan world frame; relative maps source current into target reference\n"
         << "# failure_policy=" << config.failure_policy << "\n"
         << "# total_frames=" << odometry.frames.size() << "\n"
         << "frame,reference_frame,registration_attempted,success,status,message,backend_used,relative_r00,relative_r01,relative_r02,relative_r10,relative_r11,relative_r12,relative_r20,relative_r21,relative_r22,relative_tx,relative_ty,relative_tz,pose_r00,pose_r01,pose_r02,pose_r10,pose_r11,pose_r12,pose_r20,pose_r21,pose_r22,pose_tx,pose_ty,pose_tz,final_error_m,correspondences,iterations,latency_ms,preprocessing_ms,normal_estimation_ms,correspondence_ms,solve_ms,gt_translation_error_m,gt_rotation_error_rad\n";
  for (std::size_t frame_number = 0; frame_number < odometry.frames.size(); ++frame_number) {
    const OdometryFrame& frame = odometry.frames[frame_number];
    const RegistrationResult& registration = frame.registration;
    output << frame.frame_index << "," << frame.reference_frame << ","
           << (frame.registration_attempted ? 1 : 0) << "," << (frame.success ? 1 : 0)
           << "," << to_string(registration.status) << ",\"";
    std::string message = registration.message;
    std::replace(message.begin(), message.end(), '"', '\'');
    output << message << "\"," << to_string(registration.backend_used);
    csv_transform(output, frame.relative_transform);
    csv_transform(output, frame.pose);
    output << "," << registration.final_error << "," << registration.correspondences << ","
           << registration.iterations << "," << registration.timing.total_ms << ","
           << registration.timing.preprocessing_ms << "," << registration.timing.normal_estimation_ms
           << "," << registration.timing.correspondence_ms << "," << registration.timing.solve_ms;
    if (evaluation != nullptr && frame_number < evaluation->frames.size() &&
        evaluation->frames[frame_number].has_ground_truth) {
      output << "," << evaluation->frames[frame_number].translation_error_m << ","
             << evaluation->frames[frame_number].rotation_error_rad;
    } else {
      output << ",,";
    }
    output << "\n";
  }
  return static_cast<bool>(output);
}

std::filesystem::path discover_path(const Config& config, const std::string& kind) {
  if (kind == "poses") {
    const std::string sequence_name = std::filesystem::path(config.sequence).filename().string();
    return std::filesystem::path(config.sequence).parent_path().parent_path() / "poses" /
           (sequence_name + ".txt");
  }
  return std::filesystem::path(config.sequence) / "calib.txt";
}

}  // namespace

int run_odometry_cli(int argc, char** argv) {
  Config config;
  std::string error;
  if (!parse_config(argc, argv, config, error)) {
    if (!error.empty()) std::cerr << "odometry: " << error << "\n";
    return error.empty() ? 0 : 2;
  }
  const KittiSequenceResult sequence_result = load_kitti_sequence(config.sequence);
  if (!sequence_result.ok()) {
    std::cerr << "odometry: " << to_string(sequence_result.status) << ": "
              << sequence_result.message << "\n";
    return 1;
  }

  OdometryOptions options;
  options.icp.method = config.method == "point-to-plane" ? ICPMethod::PointToPlane
                                                           : ICPMethod::PointToPoint;
  options.icp.backend = config.backend == "cpu" ? ExecutionBackend::CPU
                          : config.backend == "cuda" ? ExecutionBackend::CUDA
                                                       : ExecutionBackend::Auto;
  options.icp.correspondence_radius = config.radius;
  options.icp.voxel_size = config.voxel_size;
  options.icp.max_iterations = config.max_iterations;
  options.icp.convergence_tolerance = config.tolerance;
  options.icp.min_correspondences = static_cast<std::size_t>(config.min_correspondences);
  options.icp.normal_k_neighbors = config.normal_neighbors;
  options.icp.normal_search_radius = config.normal_radius;
  options.failure_policy = config.failure_policy == "hold" ? OdometryFailurePolicy::Hold
                           : config.failure_policy == "skip" ? OdometryFailurePolicy::Skip
                                                               : OdometryFailurePolicy::Stop;

  const OdometryResult odometry = run_kitti_odometry(sequence_result.sequence, options);
  std::cout << "odometry dataset=kitti sequence=" << config.sequence
            << " frames=" << odometry.frames.size() << " method=" << config.method
            << " backend_requested=" << config.backend << " failure_policy="
            << config.failure_policy << "\n"
            << "transform convention: current Velodyne -> reference/initial-scan world\n"
            << "registration: attempts=" << odometry.registration_attempts
            << " success=" << odometry.successful_registrations
            << " failed=" << odometry.failed_registrations
            << " failed_frames=" << odometry.failed_frames
            << " mean_latency_ms=" << odometry.mean_latency_ms
            << " p95_latency_ms=" << odometry.p95_latency_ms << "\n";

  EvaluationResult evaluation;
  const EvaluationResult* evaluation_ptr = nullptr;
  std::vector<Transform> estimated;
  estimated.reserve(odometry.frames.size());
  for (const OdometryFrame& frame : odometry.frames) estimated.push_back(frame.pose);
  std::string no_ground_truth_message =
      "no ground-truth pose file supplied; trajectory metrics are unavailable";
  std::filesystem::path poses_path;
  bool explicit_poses = !config.poses.empty();
  if (explicit_poses) poses_path = config.poses;
  else {
    const std::filesystem::path candidate = discover_path(config, "poses");
    std::error_code path_error;
    if (std::filesystem::exists(candidate, path_error)) poses_path = candidate;
  }
  if (!poses_path.empty()) {
    const KittiPoseResult poses = load_kitti_poses(poses_path);
    if (!poses.ok()) {
      if (explicit_poses) {
        std::cerr << "odometry: pose load failed: " << poses.message << "\n";
        return 1;
      }
      no_ground_truth_message = poses.message;
      std::cerr << "odometry: ground truth unavailable: " << poses.message << "\n";
    } else {
      std::vector<Transform> ground_truth = poses.poses;
      if (config.poses_frame == "camera") {
        std::filesystem::path calibration_path;
        if (!config.calibration.empty()) calibration_path = config.calibration;
        else {
          const std::filesystem::path candidate = discover_path(config, "calib");
          std::error_code path_error;
          if (std::filesystem::exists(candidate, path_error)) calibration_path = candidate;
        }
        if (calibration_path.empty()) {
          const std::string message = "camera-frame KITTI poses require --calib (not mixing camera and Velodyne frames)";
          if (explicit_poses) { std::cerr << "odometry: " << message << "\n"; return 1; }
          no_ground_truth_message = message;
          std::cerr << "odometry: ground truth unavailable: " << message << "\n";
        } else {
          const KittiCalibrationResult calibration = load_kitti_calibration(calibration_path);
          if (!calibration.ok()) {
            if (explicit_poses) { std::cerr << "odometry: " << calibration.message << "\n"; return 1; }
            no_ground_truth_message = calibration.message;
            std::cerr << "odometry: ground truth unavailable: " << calibration.message << "\n";
          } else {
            for (Transform& pose : ground_truth) pose = pose * calibration.calibration.velo_to_camera;
            evaluation = evaluate_trajectory(estimated, ground_truth, &odometry.frames);
            evaluation_ptr = &evaluation;
          }
        }
      } else {
        evaluation = evaluate_trajectory(estimated, ground_truth, &odometry.frames);
        evaluation_ptr = &evaluation;
      }
    }
  }
  if (evaluation_ptr == nullptr) {
    evaluation = evaluate_trajectory(estimated, {}, &odometry.frames);
    evaluation.message = no_ground_truth_message;
    evaluation_ptr = &evaluation;
  }
  if (evaluation_ptr != nullptr) {
    std::cout << "evaluation status=" << to_string(evaluation.status)
              << " evaluated_frames=" << evaluation.evaluated_frames
              << " message=\"" << evaluation.message << "\""
              << " translation_rmse_m=" << evaluation.translation_error_rmse_m
              << " rotation_rmse_rad=" << evaluation.rotation_error_rmse_rad
              << " ate_rmse_m=" << evaluation.ate_rmse_m
              << " rpe_pairs=" << evaluation.rpe_pairs << "\n";
  }

  const std::filesystem::path output_path(config.output);
  const std::string extension = output_path.extension().string();
  bool wrote = false;
  if (extension == ".json") wrote = write_json(output_path, config, odometry, evaluation_ptr);
  else if (extension == ".csv" || extension.empty()) {
    wrote = write_csv(output_path, config, odometry, evaluation_ptr);
    if (wrote) {
      const std::filesystem::path summary_path = output_path.string() + ".summary.json";
      std::ofstream summary(summary_path);
      if (summary) {
        summary << "{\"schema_version\":1,\"dataset\":\"kitti\",\"sequence\":\""
                << json_escape(config.sequence) << "\",\"summary\":";
        write_summary_object(summary, odometry);
        summary << ",\"evaluation\":";
        write_evaluation_object(summary, *evaluation_ptr);
        summary << "}\n";
        wrote = static_cast<bool>(summary);
      } else wrote = false;
      if (wrote) std::cout << "wrote " << output_path << " and " << summary_path << "\n";
    }
  } else {
    std::cerr << "odometry: output extension must be .csv or .json\n";
    return 2;
  }
  if (!wrote) {
    std::cerr << "odometry: cannot write output " << output_path << "\n";
    return 1;
  }
  if (odometry.status != OdometryStatus::Success) return 1;
  if (evaluation_ptr != nullptr &&
      (evaluation.status == EvaluationStatus::LengthMismatch ||
       evaluation.status == EvaluationStatus::InvalidInput)) {
    return 1;
  }
  return 0;
}

}  // namespace flashicp
