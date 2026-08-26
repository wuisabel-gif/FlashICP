#include "flashicp/evaluation.hpp"
#include "flashicp/kitti.hpp"
#include "flashicp/normals.hpp"
#include "flashicp/odometry.hpp"
#include "flashicp/registration.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using flashicp::PointCloud;
using flashicp::PointXYZ;
using flashicp::Transform;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

Transform z_transform(float angle, float tx, float ty, float tz) {
  Transform transform;
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  transform.rotation = {{c, -s, 0.0f, s, c, 0.0f, 0.0f, 0.0f, 1.0f}};
  transform.translation = {{tx, ty, tz}};
  return transform;
}

PointCloud apply_cloud(const PointCloud& cloud, const Transform& transform) {
  PointCloud output;
  output.reserve(cloud.size());
  for (const PointXYZ& point : cloud) output.push_back(transform.apply(point));
  return output;
}

PointCloud plane_fixture() {
  PointCloud points;
  for (int a = -2; a <= 2; ++a) {
    for (int b = -2; b <= 2; ++b) {
      points.push_back({static_cast<float>(a), static_cast<float>(b), 0.0f});
      points.push_back({10.0f, static_cast<float>(a), static_cast<float>(b)});
      points.push_back({static_cast<float>(a), 10.0f, static_cast<float>(b)});
    }
  }
  return points;
}

void test_normals_and_point_to_plane() {
  const PointCloud source = plane_fixture();
  const auto normals = flashicp::estimate_normals_cpu(source, 8, 2.1f);
  assert(normals.status == flashicp::NormalStatus::Success);
  assert(normals.valid_normals >= 30);
  for (const auto& normal : normals.normals) {
    if (!std::isfinite(normal.x)) continue;
    assert(near(std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z), 1.0, 1.0e-4));
  }
  flashicp::ICPOptions options;
  options.backend = flashicp::ExecutionBackend::CPU;
  options.method = flashicp::ICPMethod::PointToPlane;
  options.correspondence_radius = 0.5f;
  options.max_iterations = 50;
  options.convergence_tolerance = 1.0e-7f;
  options.min_correspondences = 6;
  options.normal_k_neighbors = 8;
  options.normal_search_radius = 2.1f;
  const Transform expected = z_transform(0.012f, 0.025f, -0.018f, 0.015f);
  const auto result = flashicp::align_cpu(source, apply_cloud(source, expected),
                                           Transform::identity(), options);
  assert(result.status == flashicp::RegistrationStatus::Converged);
  assert(result.timing.normal_estimation_ms >= 0.0);
  assert(result.final_error >= 0.0f && result.final_error < 1.0e-3f);
  for (const PointXYZ& point : source) {
    const PointXYZ actual = result.transform.apply(point);
    const PointXYZ want = expected.apply(point);
    assert(near(actual.x, want.x, 3.0e-3));
    assert(near(actual.y, want.y, 3.0e-3));
    assert(near(actual.z, want.z, 3.0e-3));
  }

  PointCloud collinear;
  for (int i = 0; i < 8; ++i) collinear.push_back({static_cast<float>(i), 0.0f, 0.0f});
  const auto bad_normals = flashicp::estimate_normals_cpu(collinear, 5, 2.0f);
  assert(bad_normals.status != flashicp::NormalStatus::Success);
  const auto bad_registration = flashicp::align_cpu(collinear, collinear,
                                                     Transform::identity(), options);
  assert(bad_registration.status == flashicp::RegistrationStatus::NormalEstimationFailure);
}

void test_six_by_six_solver() {
  flashicp::NormalEquation6 equation;
  for (int i = 0; i < 6; ++i) {
    equation.ata[static_cast<std::size_t>(i * 6 + i)] = static_cast<double>(i + 1);
    equation.rhs[static_cast<std::size_t>(i)] = static_cast<double>(i + 1);
  }
  std::array<double, 6> solution{};
  assert(flashicp::solve_normal_equation(equation, solution) ==
         flashicp::LinearSolveStatus::Success);
  for (double value : solution) assert(near(value, 1.0, 1.0e-12));
  equation.ata[0] = 1.0;
  for (int i = 1; i < 6; ++i) equation.ata[static_cast<std::size_t>(i * 6 + i)] = 0.0;
  assert(flashicp::solve_normal_equation(equation, solution) ==
         flashicp::LinearSolveStatus::Singular);
}

void write_kitti_record(std::ofstream& output, float x, float y, float z, float reflectance) {
  const float values[4] = {x, y, z, reflectance};
  output.write(reinterpret_cast<const char*>(values), sizeof(values));
}

void write_pose(std::ofstream& output, const Transform& transform) {
  output << transform.rotation[0] << " " << transform.rotation[1] << " " << transform.rotation[2]
          << " " << transform.translation[0] << " " << transform.rotation[3] << " "
          << transform.rotation[4] << " " << transform.rotation[5] << " " << transform.translation[1]
          << " " << transform.rotation[6] << " " << transform.rotation[7] << " "
          << transform.rotation[8] << " " << transform.translation[2] << "\n";
}

void test_kitti_loader() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "flashicp_milestone_fixture";
  std::error_code error;
  std::filesystem::remove_all(root, error);
  std::filesystem::create_directories(root / "sequences" / "00" / "velodyne");
  {
    std::ofstream frame(root / "sequences" / "00" / "velodyne" / "10.bin", std::ios::binary);
    write_kitti_record(frame, 1.0f, 2.0f, 3.0f, 123.0f);
    write_kitti_record(frame, -1.0f, 0.5f, 4.0f, -99.0f);
  }
  {
    std::ofstream frame(root / "sequences" / "00" / "velodyne" / "2.bin", std::ios::binary);
    write_kitti_record(frame, 5.0f, 6.0f, 7.0f, 9.0f);
  }
  const auto sequence = flashicp::load_kitti_sequence((root / "sequences" / "00").string());
  assert(sequence.ok());
  assert(sequence.sequence.size() == 2);
  assert(sequence.sequence.load_scan(0).frame_index == 2);
  assert(sequence.sequence.load_scan(1).frame_index == 10);
  const auto scan = sequence.sequence.load_scan(0);
  assert(scan.ok() && scan.points.size() == 1);
  assert(near(scan.points[0].x, 5.0, 1.0e-6));
  assert(near(scan.points[0].y, 6.0, 1.0e-6));
  assert(near(scan.points[0].z, 7.0, 1.0e-6));

  const auto malformed_path = root / "bad.bin";
  {
    std::ofstream malformed(malformed_path, std::ios::binary);
    const std::uint8_t byte = 1;
    malformed.write(reinterpret_cast<const char*>(&byte), 1);
  }
  const auto malformed = flashicp::load_kitti_velodyne_scan(malformed_path);
  assert(malformed.status == flashicp::KittiStatus::MalformedFrame);
  const auto empty_path = root / "empty.bin";
  { std::ofstream empty(empty_path, std::ios::binary); }
  const auto empty = flashicp::load_kitti_velodyne_scan(empty_path);
  assert(empty.status == flashicp::KittiStatus::EmptyFrame);
  {
    std::ofstream poses(root / "poses.txt");
    write_pose(poses, Transform::identity());
    write_pose(poses, z_transform(0.1f, 1.0f, 2.0f, 3.0f));
  }
  const auto poses = flashicp::load_kitti_poses(root / "poses.txt");
  assert(poses.ok() && poses.poses.size() == 2);
  assert(near(poses.poses[1].translation[0], 1.0, 1.0e-6));
  {
    std::ofstream calibration(root / "calib.txt");
    calibration << "R0_rect: 1 0 0 0 1 0 0 0 1\n"
                << "Tr_velo_to_cam: 1 0 0 0.5 0 1 0 -0.25 0 0 1 0.1\n";
  }
  const auto calibration = flashicp::load_kitti_calibration(root / "calib.txt");
  assert(calibration.ok());
  assert(near(calibration.calibration.velo_to_camera.translation[0], 0.5, 1.0e-6));
  std::filesystem::remove_all(root, error);
}

void test_odometry_and_evaluation() {
  const PointCloud base{{0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f},
                        {0.0f, 0.0f, 2.0f}, {2.0f, 2.0f, 0.0f}, {2.0f, 0.0f, 2.0f},
                        {0.0f, 2.0f, 2.0f}, {1.3f, 0.4f, 1.7f}};
  const Transform pose1 = z_transform(0.02f, 0.08f, -0.04f, 0.03f);
  const Transform relative2 = z_transform(-0.015f, 0.06f, 0.02f, -0.01f);
  const Transform pose2 = pose1 * relative2;
  const std::vector<PointCloud> scans{base, apply_cloud(base, pose1.inverse()),
                                      apply_cloud(base, pose2.inverse())};
  flashicp::OdometryOptions options;
  options.icp.backend = flashicp::ExecutionBackend::CPU;
  options.icp.method = flashicp::ICPMethod::PointToPoint;
  options.icp.correspondence_radius = 0.5f;
  options.icp.max_iterations = 30;
  options.icp.convergence_tolerance = 1.0e-6f;
  const auto odometry = flashicp::run_odometry(scans, options);
  assert(odometry.status == flashicp::OdometryStatus::Success);
  assert(odometry.frames.size() == 3);
  for (const PointXYZ& point : base) {
    assert(near(odometry.frames[2].pose.apply(point).x, pose2.apply(point).x, 2.0e-3));
    assert(near(odometry.frames[2].pose.apply(point).y, pose2.apply(point).y, 2.0e-3));
    assert(near(odometry.frames[2].pose.apply(point).z, pose2.apply(point).z, 2.0e-3));
  }

  const std::vector<Transform> estimated{Transform::identity(), z_transform(0.1f, 1.0f, 0.0f, 0.0f)};
  const std::vector<Transform> ground_truth{Transform::identity(), z_transform(0.0f, 0.0f, 0.0f, 0.0f)};
  const auto metrics = flashicp::evaluate_trajectory(estimated, ground_truth);
  assert(metrics.status == flashicp::EvaluationStatus::Success);
  assert(metrics.evaluated_frames == 2);
  assert(near(metrics.translation_error_rmse_m, std::sqrt(0.5), 1.0e-6));
  assert(near(metrics.rotation_error_rmse_rad, 0.1 / std::sqrt(2.0), 1.0e-5));
  assert(metrics.rpe_pairs == 1);
  assert(near(metrics.rpe_translation_rmse_m, 1.0, 1.0e-6));
  const auto mismatch = flashicp::evaluate_trajectory(estimated, {Transform::identity()});
  assert(mismatch.status == flashicp::EvaluationStatus::LengthMismatch);
  const auto no_gt = flashicp::evaluate_trajectory(estimated, {});
  assert(no_gt.status == flashicp::EvaluationStatus::NoGroundTruth);
}

}  // namespace

int main() {
  test_normals_and_point_to_plane();
  test_six_by_six_solver();
  test_kitti_loader();
  test_odometry_and_evaluation();
  return 0;
}
