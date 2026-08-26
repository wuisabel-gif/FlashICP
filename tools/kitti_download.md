# KITTI Odometry data and layout

FlashICP does not redistribute KITTI data. Download the **KITTI Vision
Benchmark Suite — Odometry** archive from the [official KITTI odometry
page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), accepting its
terms, and unpack it outside this repository. The expected layout is:

```text
/data/kitti/
  sequences/
    00/
      calib.txt
      velodyne/
        000000.bin
        000001.bin
        ...
    01/
      ...
  poses/
    00.txt
    01.txt
    ...
```

A sequence argument points to the sequence directory, not directly to
`velodyne`:

```bash
./build/flashicp odometry \
  --dataset kitti \
  --sequence /data/kitti/sequences/00 \
  --method point-to-plane --backend cpu \
  --radius 1.0 --normal-radius 1.0 \
  --output /tmp/flashicp-kitti-00.json
```

## File formats and conventions

Each Velodyne `.bin` record is four little-endian `float32` values:

```text
x, y, z, reflectance
```

The loader checks that the file size is a positive multiple of 16 bytes, reads
all records, rejects non-finite XYZ values, and ignores reflectance. Files are
sorted by the numeric filename stem, so `2.bin` precedes `10.bin`; duplicate or
nonnumeric `.bin` stems are errors. Empty, truncated, and malformed frames are
reported with their path and frame number. No scan data is needed to build or
test FlashICP.

KITTI `poses/<sequence>.txt` contains one row-major 3x4 pose per frame. A pose
maps camera-0 frame coordinates into the KITTI world trajectory frame. A
Velodyne odometry estimate maps Velodyne coordinates into the fixed initial
Velodyne scan frame. Therefore camera labels must be supplied with the matching
`calib.txt`; FlashICP explicitly computes:

```text
T_world_velo = T_world_camera0 * T_camera0_velo
```

where `T_camera0_velo` is `R0_rect * Tr_velo_to_cam`. Use
`--poses-frame lidar` only when a pose file is already expressed in the
Velodyne frame. If labels or calibration are missing/incompatible, the command
writes registration output but does not invent trajectory metrics.

`--on-failure stop` is the default and returns nonzero after recording the
failed frame. `hold` records the failure, holds the last valid pose, and
continues; `skip` retains the last successful scan as the next registration
reference and records the resulting frame gap. Neither policy performs mapping,
loop closure, or graph optimization.
