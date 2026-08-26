#pragma once

#include "registration.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace flashicp {

enum class KittiStatus {
  Success,
  InvalidPath,
  MissingDirectory,
  NoFrames,
  EmptyFrame,
  MalformedFrame,
  ReadError,
  NonFinitePoint,
};

struct KittiScanResult {
  PointCloud points;
  std::size_t frame_index = 0;
  std::filesystem::path path;
  KittiStatus status = KittiStatus::InvalidPath;
  std::string message;

  bool ok() const { return status == KittiStatus::Success; }
};

struct KittiSequence {
  std::filesystem::path directory;
  std::vector<std::filesystem::path> scan_paths;

  std::size_t size() const { return scan_paths.size(); }
  KittiScanResult load_scan(std::size_t index) const;
};

struct KittiSequenceResult {
  KittiSequence sequence;
  KittiStatus status = KittiStatus::InvalidPath;
  std::string message;

  bool ok() const { return status == KittiStatus::Success; }
};

// Enumerate the official KITTI Odometry sequence directory, e.g.
// /data/kitti/sequences/00. Files must be regular .bin files with numeric
// stems; ordering is by the numeric frame index rather than path text.
KittiSequenceResult load_kitti_sequence(const std::string& sequence_directory);

inline KittiSequenceResult enumerate_kitti_sequence(const std::string& sequence_directory) {
  return load_kitti_sequence(sequence_directory);
}

// Decode one official Velodyne record: little-endian float32 quadruples
// (x,y,z,reflectance). Reflectance is deliberately ignored.
KittiScanResult load_kitti_velodyne_scan(const std::filesystem::path& path,
                                         std::size_t frame_index = 0);

inline KittiScanResult load_kitti_scan(const std::filesystem::path& path,
                                       std::size_t frame_index = 0) {
  return load_kitti_velodyne_scan(path, frame_index);
}

const char* to_string(KittiStatus status);

struct KittiPoseResult {
  std::vector<Transform> poses;
  KittiStatus status = KittiStatus::InvalidPath;
  std::string message;

  bool ok() const { return status == KittiStatus::Success; }
};

// Parse poses/<sequence>.txt. Each non-empty line is a row-major 3x4 pose
// mapping the frame coordinates into the KITTI world/camera-0 trajectory frame.
KittiPoseResult load_kitti_poses(const std::filesystem::path& path);

struct KittiCalibration {
  // Homogeneous transform mapping Velodyne coordinates to rectified camera-0
  // coordinates. KITTI's R0_rect is applied before Tr_velo_to_cam.
  Transform velo_to_camera = Transform::identity();
};

struct KittiCalibrationResult {
  KittiCalibration calibration;
  KittiStatus status = KittiStatus::InvalidPath;
  std::string message;

  bool ok() const { return status == KittiStatus::Success; }
};

// Parse the KITTI sequence calib.txt. Tr_velo_to_cam (or Tr) is required;
// R0_rect is optional and defaults to identity.
KittiCalibrationResult load_kitti_calibration(const std::filesystem::path& path);

}  // namespace flashicp
