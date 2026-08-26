<p align="center">
  <img src="logo.png" alt="FlashICP" width="440">
</p>

<p align="center">
  <a href="https://github.com/wuisabel-gif/FlashICP/actions/workflows/build.yml">
    <img src="https://github.com/wuisabel-gif/FlashICP/actions/workflows/build.yml/badge.svg" alt="build status">
  </a>
</p>

CUDA-accelerated point-cloud registration primitives for autonomous systems,
written in C++17 and CUDA.

FlashICP began as an underwater point-cloud project for Barracuda / ZED data.
The current checkout contains GPU voxelization, a spatial-hash correspondence
prototype, CPU reference code, and an offline ROS bag extractor. The roadmap is
to turn those pieces into a generic point-to-point and point-to-plane ICP
frontend for AUV depth clouds and KITTI Velodyne LiDAR, then use it for LiDAR
odometry on Jetson-class NVIDIA hardware. The CPU point-to-point registration
reference is now in place; point-to-plane and the KITTI odometry workflow remain
future milestones.

The existing performance evidence is limited to the voxel stage on one real AUV
cloud. It is not yet an end-to-end ICP or LiDAR-odometry benchmark.

See [`docs/roadmap.md`](docs/roadmap.md) for the audited status, dependencies,
and copy/paste-ready issue briefs in [`docs/issues/`](docs/issues/).

<p align="center">
  <img src="Instrument.gif" alt="ICP aligning two scans: correspondence vectors shrink and the RMS error converges" width="720">
  <br>
  <sub>ICP in action — each new scan (amber) snaps onto the reference cloud (cyan); correspondence
  vectors shrink as the RMS error converges. Illustrative 2D view of the loop; on the robot it runs in CUDA on a Jetson.</sub>
</p>

## Why it matters

Underwater there is no GPS and no radio, so a robot has to work out its own motion
purely from what its cameras and sonar see, every frame, or it silently loses track
of where it is. The catch: that perception has to run on a **small, battery-powered
computer sealed inside the hull**, not a datacenter. Today that limit is why so much
subsea inspection still needs **divers or a crewed support vessel on station**, the
expensive part of the job.

Making the perception fast *on the hardware the robot already carries* is what changes
that. FlashICP's payoff is concrete:

- **Cheaper missions.** Reliable GPS-free autonomy means fewer ship-days and dive crews.
- **Longer endurance.** Every millisecond and watt saved on compute is more battery for
  the actual mission.
- **Same chip, more capability.** A faster kernel at the same power lets one Jetson run
  perception *and* mapping *and* avoidance, instead of demanding a bigger, hotter box.

The measured **3.0× speedup for the atomic-hash voxel stage** is a small, honest
step toward doing more of the robot's thinking onboard, for less. End-to-end
registration measurements are still planned.

## Why a GPU

The point-cloud → ICP step is the hot path of a visual SLAM frontend: large clouds,
per-point work, run every keyframe. That is exactly the shape a GPU is built for.
FlashICP rebuilds that path in CUDA to (a) learn the core GPU patterns (spatial
hashing, parallel reduction, memory coalescing) and (b) produce a genuinely useful,
benchmarkable result.

## Pipeline

Implemented prototype path:

```
Barracuda/ZED rosbag ─► custom x,y,z binary ─► CPU/CUDA voxelization
                                             └► CPU/CUDA correspondence prototype
```

Target path, tracked in the roadmap:

```
KITTI LiDAR / AUV cloud ─► preprocess ─► correspondence ─► point-to-plane ICP
                                                       └► SE(3) ─► LiDAR odometry
                                                                    └► evaluation / optional Rerun
```

## Status

**Implemented:** a generic CPU-safe registration API with SE(3) transforms, CPU
point-to-point ICP, CPU/CUDA voxel-grid downsampling, CPU brute-force
correspondence, a CUDA fixed-radius spatial-hash correspondence prototype, the
custom cloud format, Barracuda/ZED SQLite bag extraction, a small benchmark CLI,
and CTest coverage. The atomic-hash voxel path was measured at **3.0× versus
CPU** on one real Jetson AGX Orin ZED cloud (0.998 ms versus 2.991 ms); the
original sort path was 0.7×. Full history is in `docs/jetson_runlog.md`.

**Experimental or unverified:** CUDA correspondence is checked in but has not
yet been executed and benchmarked on the target in this checkout. CUDA
point-to-point registration is wired behind the public API, but this checkout
has no `nvcc` to compile or run it. The CPU correctness suite passes; GPU
agreement remains a conditional hardware test.

**Planned:** point-to-plane ICP, KITTI loading, sequential odometry, trajectory
metrics, stage benchmarks, profiling, and optional Rerun/ROS 2/GTSAM
integrations. These are not presented as current capabilities.

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
ctest --test-dir build --output-on-failure
```

**3. Benchmark CPU vs GPU:**
```bash
./build/flashicp bench data/cloud0.bin 0.05 50
# loaded 113301 points, leaf=0.050 m, iters=50
# CPU voxel downsample: 2.991 ms  -> 1358 voxels
# GPU voxel (sort): 4.683 ms  -> 1358 voxels  (0.6x vs CPU)
#   check: PASS (matches CPU)
# GPU voxel (hash): 0.998 ms  -> 1358 voxels  (3.0x vs CPU)
#   check: PASS (matches CPU)
```

- On a **Jetson Orin**: full CPU-vs-GPU comparison when built with CUDA.
- Other GPUs require an appropriate CUDA architecture configuration; portable
  architecture selection is tracked in the roadmap.
- On a **Mac** (no CUDA): CPU baseline only — still useful to profile the C++ path.

**4. Correspondence (prototype):** nearest-neighbor between two clouds via a GPU spatial hash grid.
```bash
./build/flashicp corr data/cloud0.bin data/cloud1.bin 0.20 20
# radius 0.20 m; each source point probes its 27 neighbor cells on the GPU,
# checked against a brute-force CPU baseline when CUDA is available.
```
The CPU baseline has a standalone self-check (no CUDA needed):
```bash
c++ -std=c++17 -Isrc tools/test_corr.cpp -o /tmp/test_corr && /tmp/test_corr  # -> PASS
```

### CPU point-to-point API

The generic CPU-safe registration boundary is available independently of CUDA:

```cpp
#include <flashicp/registration.hpp>

flashicp::ICPOptions options;
options.backend = flashicp::ExecutionBackend::CPU;
auto result = flashicp::align(source, target,
                              flashicp::Transform::identity(), options);
// result.transform maps source-frame points into the target frame.
// Check result.status before consuming the transform.
```

The current registration method is point-to-point. Point-to-plane, KITTI
odometry, and trajectory evaluation are tracked as later roadmap milestones.

## Layout

```
tools/dump_cloud.py   rosbag PointCloud2 -> flat x,y,z binary
tools/test_corr.cpp   standalone self-check for the CPU correspondence baseline
include/flashicp/      CPU-safe public PointCloud / SE(3) / registration API
src/flashicp.hpp      Point, cloud IO, CPU voxel-downsample + correspondence baselines
src/registration.cpp  CPU point-to-point ICP and public API dispatch
src/registration_cuda.cu  CUDA transform/reduction path with host rigid solve
src/voxel_gpu.cu      CUDA voxel downsample (thrust sort baseline + atomic hash)
src/corr_gpu.cu       CUDA correspondence (fixed-radius spatial hash grid, 27-cell probe)
src/main.cpp          bench/corr CLI (timing + CPU/GPU correctness check)
CMakeLists.txt        CPU-always, CUDA-if-available build
docs/roadmap.md       audited status, dependency roadmap, and issue index
docs/issues/          copy/paste-ready implementation issue briefs
```
