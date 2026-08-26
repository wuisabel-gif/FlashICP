# 008 — Add the KITTI Odometry loader

**Status:** Planned  
**Priority:** P0  
**Labels:** `area:data`, `area:odometry`, `priority:p0`  
**Depends on:** 002, 003

## Goal

Load official KITTI Odometry Velodyne scans without adding KITTI data to the
repository or requiring Python for the registration path.

## Scope

Add a small C++ loader or library adapter for a sequence directory containing
`velodyne/*.bin`. Each record is KITTI's little-endian `x,y,z,reflectance`
float quadruple; ignore reflectance initially while validating file size and
finite XYZ values. Sort frames numerically, not lexicographically by path text.

Support the official layout and document how to obtain/unpack data in
`tools/kitti_download.md` or equivalent. Keep scan loading separate from ground-
truth pose loading so scans can be used without labels. Handle missing, empty,
malformed, and partially readable frames with actionable errors.

If calibration is needed for a later sensor-frame result, parse it explicitly and
document the conversion. Do not silently mix Velodyne, camera, and IMU frames.

## Acceptance criteria

- A user can point the loader at `sequences/00` and enumerate frames in order.
- A known KITTI `.bin` fixture (small and synthetic, not the dataset) decodes XYZ
  correctly and ignores reflectance as documented.
- No KITTI data, credentials, or large generated outputs are committed.
- Missing/malformed files produce a clear error and nonzero CLI status.
- AUV/ZED custom binary loading remains supported independently.
