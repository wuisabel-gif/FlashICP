#include "flashicp/kitti.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

namespace flashicp {
namespace {

KittiScanResult scan_failure(const std::filesystem::path& path, std::size_t index,
                            KittiStatus status, const std::string& message) {
  KittiScanResult result;
  result.path = path;
  result.frame_index = index;
  result.status = status;
  result.message = message;
  return result;
}

bool numeric_stem(const std::filesystem::path& path, std::size_t& value) {
  const std::string stem = path.stem().string();
  if (stem.empty()) return false;
  for (const char character : stem) {
    if (character < '0' || character > '9') return false;
  }
  try {
    const unsigned long long parsed = std::stoull(stem);
    if (parsed > std::numeric_limits<std::size_t>::max()) return false;
    value = static_cast<std::size_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

float little_endian_float(const unsigned char* bytes) {
  const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                             (static_cast<std::uint32_t>(bytes[1]) << 8) |
                             (static_cast<std::uint32_t>(bytes[2]) << 16) |
                             (static_cast<std::uint32_t>(bytes[3]) << 24);
  float value = 0.0f;
  static_assert(sizeof(value) == sizeof(bits), "float must be IEEE-754 sized");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool read_values(const std::string& line, std::size_t count, std::vector<double>& values) {
  std::string payload = line;
  const std::size_t colon = payload.find(':');
  if (colon != std::string::npos) payload = payload.substr(colon + 1);
  std::stringstream stream(payload);
  values.clear();
  double value = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    if (!(stream >> value)) return false;
    values.push_back(value);
  }
  std::string extra;
  return !(stream >> extra);
}

Transform transform_from_values(const std::vector<double>& values, std::size_t offset = 0) {
  Transform transform = Transform::identity();
  transform.rotation = {{static_cast<float>(values[offset + 0]),
                         static_cast<float>(values[offset + 1]),
                         static_cast<float>(values[offset + 2]),
                         static_cast<float>(values[offset + 4]),
                         static_cast<float>(values[offset + 5]),
                         static_cast<float>(values[offset + 6]),
                         static_cast<float>(values[offset + 8]),
                         static_cast<float>(values[offset + 9]),
                         static_cast<float>(values[offset + 10])}};
  transform.translation = {{static_cast<float>(values[offset + 3]),
                            static_cast<float>(values[offset + 7]),
                            static_cast<float>(values[offset + 11])}};
  return transform;
}

}  // namespace

KittiScanResult load_kitti_velodyne_scan(const std::filesystem::path& path,
                                         std::size_t frame_index) {
  std::error_code error;
  if (!std::filesystem::exists(path, error) || error ||
      !std::filesystem::is_regular_file(path, error) || error) {
    return scan_failure(path, frame_index, KittiStatus::InvalidPath,
                        "KITTI scan does not exist or is not a regular file: " + path.string());
  }
  const std::uintmax_t byte_count = std::filesystem::file_size(path, error);
  if (error) return scan_failure(path, frame_index, KittiStatus::ReadError,
                                 "cannot stat KITTI scan: " + path.string());
  if (byte_count == 0) return scan_failure(path, frame_index, KittiStatus::EmptyFrame,
                                           "KITTI scan is empty: " + path.string());
  if (byte_count % 16 != 0) {
    return scan_failure(path, frame_index, KittiStatus::MalformedFrame,
                        "KITTI Velodyne file size is not a multiple of 16 bytes: " + path.string());
  }
  const std::uintmax_t point_count = byte_count / 16;
  if (point_count > std::numeric_limits<std::size_t>::max()) {
    return scan_failure(path, frame_index, KittiStatus::MalformedFrame,
                        "KITTI scan has too many records: " + path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) return scan_failure(path, frame_index, KittiStatus::ReadError,
                                  "cannot open KITTI scan: " + path.string());
  KittiScanResult result;
  result.path = path;
  result.frame_index = frame_index;
  result.points.reserve(static_cast<std::size_t>(point_count));
  unsigned char record[16]{};
  for (std::uintmax_t i = 0; i < point_count; ++i) {
    input.read(reinterpret_cast<char*>(record), sizeof(record));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(record))) {
      return scan_failure(path, frame_index, KittiStatus::ReadError,
                          "short read in KITTI scan: " + path.string());
    }
    const PointXYZ point{little_endian_float(record + 0), little_endian_float(record + 4),
                         little_endian_float(record + 8)};
    // record + 12 is reflectance and is intentionally ignored.
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      return scan_failure(path, frame_index, KittiStatus::NonFinitePoint,
                          "KITTI scan contains a nonfinite XYZ record at index " +
                              std::to_string(static_cast<std::size_t>(i)) + ": " + path.string());
    }
    result.points.push_back(point);
  }
  result.status = KittiStatus::Success;
  result.message = "loaded " + std::to_string(result.points.size()) + " Velodyne points";
  return result;
}

KittiScanResult KittiSequence::load_scan(std::size_t index) const {
  if (index >= scan_paths.size()) {
    return scan_failure({}, index, KittiStatus::InvalidPath,
                        "KITTI frame index is outside the enumerated sequence");
  }
  std::size_t frame_index = 0;
  if (!numeric_stem(scan_paths[index], frame_index)) {
    return scan_failure(scan_paths[index], index, KittiStatus::MalformedFrame,
                        "KITTI frame filename is not numeric: " + scan_paths[index].string());
  }
  return load_kitti_velodyne_scan(scan_paths[index], frame_index);
}

KittiSequenceResult load_kitti_sequence(const std::string& sequence_directory) {
  KittiSequenceResult result;
  const std::filesystem::path directory(sequence_directory);
  std::error_code error;
  if (sequence_directory.empty() || !std::filesystem::exists(directory, error) || error ||
      !std::filesystem::is_directory(directory, error) || error) {
    result.status = KittiStatus::MissingDirectory;
    result.message = "KITTI sequence directory does not exist: " + sequence_directory;
    return result;
  }
  const std::filesystem::path velodyne = directory / "velodyne";
  if (!std::filesystem::exists(velodyne, error) || error ||
      !std::filesystem::is_directory(velodyne, error) || error) {
    result.status = KittiStatus::MissingDirectory;
    result.message = "KITTI sequence is missing velodyne/: " + velodyne.string();
    return result;
  }
  struct NumberedPath {
    std::size_t index;
    std::filesystem::path path;
  };
  std::vector<NumberedPath> numbered;
  for (const auto& entry : std::filesystem::directory_iterator(velodyne, error)) {
    if (error) {
      result.status = KittiStatus::ReadError;
      result.message = "cannot enumerate KITTI velodyne directory: " + velodyne.string();
      return result;
    }
    if (!entry.is_regular_file(error) || error) continue;
    if (entry.path().extension() != ".bin") continue;
    std::size_t index = 0;
    if (!numeric_stem(entry.path(), index)) {
      result.status = KittiStatus::MalformedFrame;
      result.message = "KITTI .bin frame has a nonnumeric filename: " + entry.path().string();
      return result;
    }
    numbered.push_back({index, entry.path()});
  }
  if (numbered.empty()) {
    result.status = KittiStatus::NoFrames;
    result.message = "KITTI sequence contains no velodyne/*.bin frames: " + velodyne.string();
    return result;
  }
  std::sort(numbered.begin(), numbered.end(), [](const NumberedPath& a, const NumberedPath& b) {
    return a.index < b.index;
  });
  for (std::size_t i = 1; i < numbered.size(); ++i) {
    if (numbered[i - 1].index == numbered[i].index) {
      result.status = KittiStatus::MalformedFrame;
      result.message = "duplicate numeric KITTI frame index " + std::to_string(numbered[i].index);
      return result;
    }
  }
  result.sequence.directory = directory;
  for (const NumberedPath& item : numbered) result.sequence.scan_paths.push_back(item.path);
  result.status = KittiStatus::Success;
  result.message = "enumerated " + std::to_string(result.sequence.size()) + " KITTI frames";
  return result;
}

const char* to_string(KittiStatus status) {
  switch (status) {
    case KittiStatus::Success: return "success";
    case KittiStatus::InvalidPath: return "invalid_path";
    case KittiStatus::MissingDirectory: return "missing_directory";
    case KittiStatus::NoFrames: return "no_frames";
    case KittiStatus::EmptyFrame: return "empty_frame";
    case KittiStatus::MalformedFrame: return "malformed_frame";
    case KittiStatus::ReadError: return "read_error";
    case KittiStatus::NonFinitePoint: return "nonfinite_point";
  }
  return "unknown";
}

KittiPoseResult load_kitti_poses(const std::filesystem::path& path) {
  KittiPoseResult result;
  std::ifstream input(path);
  if (!input) {
    result.status = KittiStatus::InvalidPath;
    result.message = "cannot open KITTI pose file: " + path.string();
    return result;
  }
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    std::vector<double> values;
    if (!read_values(line, 12, values)) {
      result.status = KittiStatus::MalformedFrame;
      result.message = "KITTI pose line " + std::to_string(line_number) + " does not contain exactly 12 numbers";
      result.poses.clear();
      return result;
    }
    for (double value : values) {
      if (!std::isfinite(value)) {
        result.status = KittiStatus::MalformedFrame;
        result.message = "KITTI pose line contains a nonfinite value at line " +
                         std::to_string(line_number);
        result.poses.clear();
        return result;
      }
    }
    const Transform pose = transform_from_values(values);
    if (!pose.is_valid(2.0e-2f)) {
      result.status = KittiStatus::MalformedFrame;
      result.message = "KITTI pose line has an invalid rotation at line " +
                       std::to_string(line_number);
      result.poses.clear();
      return result;
    }
    result.poses.push_back(pose);
  }
  if (result.poses.empty()) {
    result.status = KittiStatus::NoFrames;
    result.message = "KITTI pose file contains no poses: " + path.string();
    return result;
  }
  result.status = KittiStatus::Success;
  result.message = "loaded " + std::to_string(result.poses.size()) + " KITTI poses";
  return result;
}

KittiCalibrationResult load_kitti_calibration(const std::filesystem::path& path) {
  KittiCalibrationResult result;
  std::ifstream input(path);
  if (!input) {
    result.status = KittiStatus::InvalidPath;
    result.message = "cannot open KITTI calibration file: " + path.string();
    return result;
  }
  std::vector<double> velo_values;
  std::vector<double> rect_values;
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("Tr_velo_to_cam:") != std::string::npos ||
        line.find("Tr:") != std::string::npos) {
      std::vector<double> values;
      if (!read_values(line, 12, values)) {
        result.status = KittiStatus::MalformedFrame;
        result.message = "KITTI calibration Tr_velo_to_cam must contain 12 numbers";
        return result;
      }
      velo_values = std::move(values);
    } else if (line.find("R0_rect:") != std::string::npos) {
      std::vector<double> values;
      if (!read_values(line, 9, values)) {
        result.status = KittiStatus::MalformedFrame;
        result.message = "KITTI calibration R0_rect must contain 9 numbers";
        return result;
      }
      rect_values = std::move(values);
    }
  }
  if (velo_values.empty()) {
    result.status = KittiStatus::MalformedFrame;
    result.message = "KITTI calibration has no Tr_velo_to_cam (or Tr) record";
    return result;
  }
  for (double value : velo_values) {
    if (!std::isfinite(value)) {
      result.status = KittiStatus::MalformedFrame;
      result.message = "KITTI calibration contains a nonfinite value";
      return result;
    }
  }
  Transform velo_to_camera = transform_from_values(velo_values);
  if (!rect_values.empty()) {
    Transform rect = Transform::identity();
    rect.rotation = {{static_cast<float>(rect_values[0]), static_cast<float>(rect_values[1]),
                      static_cast<float>(rect_values[2]), static_cast<float>(rect_values[3]),
                      static_cast<float>(rect_values[4]), static_cast<float>(rect_values[5]),
                      static_cast<float>(rect_values[6]), static_cast<float>(rect_values[7]),
                      static_cast<float>(rect_values[8])}};
    if (!rect.is_valid(2.0e-2f)) {
      result.status = KittiStatus::MalformedFrame;
      result.message = "KITTI calibration R0_rect is not a valid rotation";
      return result;
    }
    velo_to_camera = rect * velo_to_camera;
  }
  if (!velo_to_camera.is_valid(2.0e-2f)) {
    result.status = KittiStatus::MalformedFrame;
    result.message = "KITTI calibration transform is invalid";
    return result;
  }
  result.calibration.velo_to_camera = velo_to_camera;
  result.status = KittiStatus::Success;
  result.message = "loaded Velodyne-to-camera calibration";
  return result;
}

}  // namespace flashicp
