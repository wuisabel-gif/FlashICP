# 001 — Audit the existing implementation and establish a baseline

**Status:** Complete  
**Priority:** P0  
**Labels:** `area:core`, `priority:p0`

## Goal

Record what the repository actually implements before architectural changes are
made. Preserve the AUV/ZED path and the useful negative benchmark result.

## Findings

Implemented: CPU/CUDA voxel downsampling, CPU brute-force correspondence, CUDA
fixed-radius hash-grid correspondence, the custom `int32 + float32 xyz` format,
the Barracuda/ZED SQLite bag extractor, the `bench`/`corr` CLI, and an Orin voxel
run log.

Not implemented: generic registration API, transforms, normals, iterative ICP,
6x6 solve, KITTI, odometry, trajectory metrics, runtime ROS 2, Rerun, or GTSAM.
CUDA correspondence is checked in but has no recorded hardware execution in this
checkout. CUDA architecture `87` is hard-coded.

## Baseline

```bash
cmake -B /tmp/flashicp-baseline-build -DUSE_CUDA=OFF
cmake --build /tmp/flashicp-baseline-build --parallel
c++ -std=c++17 -Wall -Wextra -Isrc tools/test_corr.cpp \
  -o /tmp/flashicp-test-corr
/tmp/flashicp-test-corr
```

The CPU build and self-check pass. The existing Jetson log records 0.7x for the
sort voxel path and 3.0x for the atomic-hash voxel path on one real ZED cloud;
these are voxel measurements, not end-to-end ICP measurements.

## Acceptance criteria

- The implemented/partial/planned inventory is committed in `docs/roadmap.md`.
- No dataset, build output, or benchmark result is added to Git.
- Existing CPU-only build and self-check remain green.
- AUV/ZED extraction and the custom binary format remain documented.
