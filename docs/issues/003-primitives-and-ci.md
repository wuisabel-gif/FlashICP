# 003 — Harden existing primitives and CI

**Status:** Partial  
**Priority:** P0  
**Labels:** `area:core`, `area:cuda`, `priority:p0`  
**Depends on:** 002

## Goal

Make the existing voxel and correspondence stages safe inputs to an iterative
registration loop and improve portability without discarding working kernels.

## Scope

- Reject or report empty input, nonfinite points, invalid leaf sizes, invalid
  correspondence radii, oversized counts, truncated binary files, and integer
  overflow risks.
- Give CPU and CUDA correspondence the same documented radius semantics. The
  CUDA path currently assumes a positive radius while the CPU path treats a
  nonpositive radius as unbounded.
- Test spatial-hash behavior under negative coordinates, hash collisions, sparse
  cells, duplicate points, and boundary points. Store cell identity if a
  correctness test shows bucket collisions can affect behavior.
- Define scratch-buffer lifetime and stream behavior before calls become
  concurrent or iterative. Avoid process-global state where it can make results
  race.
- Replace the fixed CUDA architecture with a configurable CMake cache option,
  retaining an easy Orin (`87`) configuration and a CPU-only path.
- Add CTest or an equivalent test target; keep CUDA tests conditional on a CUDA
  compiler/device.

## Acceptance criteria

- Invalid inputs return a documented error/status instead of producing invalid
  poses or undefined CUDA behavior.
- CPU tests run on a machine without CUDA.
- CUDA compile CI tests the configurable architecture, and hardware tests run
  voxel/correspondence correctness when a GPU is available.
- The existing Orin voxel benchmark remains reproducible and its historical
  sort-versus-hash results are not overwritten.
