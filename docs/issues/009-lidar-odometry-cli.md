# 009 — Build sequential LiDAR odometry

**Status:** Implemented (CPU; optional CUDA point-to-plane fallback is explicit)
**Priority:** P0
**Labels:** `area:odometry`, `area:core`, `priority:p0`
**Depends on:** 007, 008

## Goal

Register consecutive scans and accumulate relative transforms into a trajectory.
This issue is odometry only; it does not add loop closure, mapping, or graph
optimization.

## Scope

Add an `odometry` subcommand with the requested shape:

```bash
flashicp odometry \
  --dataset kitti \
  --sequence /data/kitti/sequences/00 \
  --method point-to-plane
```

For each pair, register scan `i` against scan `i-1`, use the previous relative
motion as an optional initial guess when appropriate, and compose transforms in a
single documented convention. Preserve the initial pose and write a trajectory
plus per-frame registration records. Make voxel size, radius, max iterations,
convergence tolerance, minimum correspondences, output path, and CPU/CUDA mode
configurable.

A failed frame must be recorded and handled according to an explicit policy
(stop, hold pose, or skip with a gap); it must not silently generate an invalid
pose. Avoid retaining every full-resolution scan unless the chosen algorithm
requires it.

## Acceptance criteria

- The command processes a multi-frame sequence in order and writes a trajectory.
- Each frame records status, relative transform, accumulated pose, error,
  correspondences, iterations, and latency.
- CPU-only execution works without ROS, CUDA, or Python.
- The output states the transform direction, coordinate frame, and failure policy.
- A short deterministic multi-frame fixture verifies transform composition.
