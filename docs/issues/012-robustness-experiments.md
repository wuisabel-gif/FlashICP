# 012 — Add reproducible robustness experiments

**Status:** Planned  
**Priority:** P1  
**Labels:** `area:core`, `area:benchmark`, `priority:p1`  
**Depends on:** 005, 007, 009

## Goal

Measure where ICP succeeds and fails so parameter limits are evidence-based.

## Scope

Add a deterministic synthetic experiment driver that varies initial translation
and rotation error, overlap, Gaussian noise, outlier percentage, voxel size,
correspondence radius, and point count. Record every trial, including failures,
with the random seed, options, method, CPU/CUDA mode, recovered pose, residual,
iterations, correspondences, status, and latency.

Keep this separate from KITTI evaluation. Synthetic trials expose registration
basins and degeneracy; KITTI measures behavior on real driving data. Do not
selectively omit failed or slow runs.

## Acceptance criteria

- The same seed and configuration produce byte-stable or schema-stable CSV/JSON
  output within documented floating-point limits.
- Every requested sweep value appears in the output, including failed trials.
- The report distinguishes convergence failure, insufficient overlap,
  degeneracy, invalid input, and runtime failure.
- A short default sweep runs on a CPU-only machine for CI or local regression.
- Results include enough metadata to reproduce a trial without a dataset commit.
