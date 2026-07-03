# AGENTS.md — FlashICP

FlashICP is a CUDA implementation of the point-cloud ICP frontend used in visual
SLAM, built and measured against real AUV data.

## Goal

Take two consecutive registered point clouds and compute the rigid transform that
aligns them — entirely on the GPU — fast enough to run every keyframe on a Jetson.
The output is the same relative-pose constraint a factor-graph backend (GTSAM)
consumes; FlashICP is the *frontend*, not the optimizer.

Two equal goals:
1. **Learn the core CUDA patterns** — spatial hashing, parallel reduction, atomics,
   memory coalescing, occupancy — on a problem that genuinely needs them.
2. **Produce a useful, benchmarked result** — every kernel has a CPU baseline and a
   measured speedup on real Barracuda/ZED rosbags.

## Non-goals

- Not a SLAM system. No loop closure, no graph optimization, no mapping. Just ICP.
- Not a novel algorithm. Point-to-plane ICP is textbook; the contribution is the
  GPU implementation and the honest benchmark, not new math.
- Not tied to ROS at runtime — it reads dumped clouds, so it builds and runs without
  a ROS install. (A thin ROS 2 node wrapper is a stretch goal.)

## Pipeline (and what CUDA each stage teaches)

```
preprocess ──► correspondence ──► build linear system ──► solve ──► iterate
```

1. **Preprocess** — voxel-grid downsample, range/passthrough crop, statistical
   outlier removal, normal estimation.
   *CUDA:* spatial hashing into a voxel grid, atomics for per-voxel accumulation,
   parallel reduction for centroids/normals, coalesced memory layout (SoA not AoS).

2. **Correspondence** — for each source point, find its nearest target point.
   *CUDA:* reuse the voxel grid for O(1) neighbor lookup instead of a KD-tree;
   this is the step that makes the GPU win big.

3. **Build the linear system** — point-to-plane residuals → a 6×6 normal-equation
   system (`A`, `b`) summed over all correspondences.
   *CUDA:* large parallel reduction of per-correspondence 6×6 + 6×1 contributions.

4. **Solve** — solve the 6×6 for the incremental twist (small enough to do on host
   or with a tiny GPU solver), compose onto the running transform.

5. **Iterate** until convergence (delta below threshold or max iterations).

## The benchmark harness (start here)

The whole project hangs off one offline harness, using rosbags already on disk:

1. Decode `/barracuda/zed_node/point_cloud/cloud_registered` out of
   `zed_20260621_225845` → dump consecutive clouds to a simple binary (`x,y,z` floats).
2. **CPU baseline** — PCL (or a plain C++ loop) for each stage; time it.
3. **GPU version** — the CUDA kernel; validate output matches CPU within tolerance,
   then time it and record the speedup.

Every stage grows under this harness, so the CUDA work is always the only new thing
and is always measured on real data.

## Milestones

- **M1 (MVP):** GPU voxel-grid downsample + the CPU-vs-GPU timing harness on one bag.
  Deliverable: "CPU X ms → GPU Y ms, output matches within ε."
- **M2:** GPU correspondence via the voxel grid + point-to-plane residual build.
- **M3:** full ICP loop (build → solve → iterate); align consecutive scans; compare
  the recovered trajectory against the ZED VIO track from the bag.
- **M4:** batched / multi-resolution ICP, tuning (occupancy, shared memory).
- **Stretch:** ROS 2 node wrapper; run live on the Jetson against the ZED cloud.

## Technology

- **CUDA C++**, single CLI binary, `make`/`CMake` build.
- Dev on a desktop GPU or the **Jetson Orin** (where it would eventually run).
- Minimal deps: a point-cloud loader (custom binary), optional PCL for the CPU
  baseline only. No ROS at runtime.

## Testing

- **Correctness:** every GPU stage must match its CPU baseline within a numerical
  tolerance on the same input cloud (golden-output tests).
- **Performance:** record CPU vs GPU timing per stage, per bag; the headline metric
  is end-to-end ICP time per keyframe on real data.
- **Trajectory sanity (M3+):** the relative poses chained together should track the
  ZED VIO trajectory from the bag, not drift arbitrarily.

## Success criteria

FlashICP succeeds when it aligns two real ZED point clouds on the GPU in a fraction
of the CPU time, with output that matches the CPU baseline — demonstrated end-to-end
on a Barracuda rosbag, with the speedup numbers to back it up.
