<p align="center">
  <img src="logo.png" alt="FlashICP" width="440">
</p>

GPU-accelerated ICP (Iterative Closest Point) for underwater point clouds, written in CUDA.

FlashICP takes the registered point cloud from an AUV's stereo camera and aligns
consecutive scans on the GPU — the same point-cloud frontend a factor-graph SLAM
backend (e.g. GTSAM) relies on, but fast enough to keep up on a Jetson.

It is built and benchmarked against **real AUV rosbags** (Barracuda / ZED Mini),
so every kernel has a CPU baseline and a measured speedup on actual data — not a
synthetic benchmark.

## Why

The point-cloud → ICP step is the hot path of a visual SLAM frontend: large clouds,
per-point work, run every keyframe. That is exactly the shape a GPU is built for.
FlashICP rebuilds that path in CUDA to (a) learn the core GPU patterns — spatial
hashing, parallel reduction, memory coalescing — and (b) produce a genuinely useful,
benchmarkable result.

## Pipeline

```
rosbag point cloud  ─►  GPU preprocess  ─►  GPU correspondence  ─►  build + solve  ─►  pose
 (ZED registered)       voxel / crop /       nearest-neighbor       6x6 normal eqs     (T)
                        outlier / normals     via voxel grid         (reduction)
```

## Status

Early scaffold. Start point: GPU voxel-grid downsample with a CPU baseline + timing
harness on `zed_20260621_225845`. See `docs/jetson_runlog.md` for the build and
benchmark on the Jetson.

## Build & run

**1. Dump a cloud from a rosbag** (no ROS needed):
```bash
python3 tools/dump_cloud.py /path/to/bag.db3 --out-dir data --max 2
# -> data/cloud0.bin, data/cloud1.bin  (int32 n, then n*(float x,y,z))
```

**2. Build:**
```bash
cmake -B build -DUSE_CUDA=ON      # CUDA auto-disables if no nvcc (e.g. on a Mac)
cmake --build build
```

**3. Benchmark CPU vs GPU:**
```bash
./build/flashicp bench data/cloud0.bin 0.05 20
# loaded N points, leaf=0.050 m, iters=20
# CPU voxel downsample: X.XXX ms -> K voxels
# GPU voxel downsample: Y.YYY ms -> K voxels
# speedup: Zx
#   check: PASS (matches CPU)
```

- On a **Jetson Orin / desktop GPU**: full CPU-vs-GPU comparison.
- On a **Mac** (no CUDA): CPU baseline only — still useful to profile the C++ path.

## Layout

```
tools/dump_cloud.py   rosbag PointCloud2 -> flat x,y,z binary
src/flashicp.hpp      Point, cloud IO, CPU voxel-downsample baseline
src/voxel_gpu.cu      CUDA voxel downsample (key -> sort -> reduce)
src/main.cpp          bench CLI (timing + CPU/GPU correctness check)
CMakeLists.txt        CPU-always, CUDA-if-available build
```
