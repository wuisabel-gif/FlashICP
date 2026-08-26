# 002 — Introduce the generic core API

**Status:** Implemented  
**Priority:** P0  
**Labels:** `area:core`, `priority:p0`  
**Depends on:** 001

## Goal

Make registration independent of ZED, Velodyne, Ouster, ROS, or KITTI. Move the
public types into a stable include boundary while adapting existing voxel and
correspondence code behind it.

## Scope

Define a minimal API equivalent to:

```cpp
RegistrationResult align(const PointCloud& source,
                         const PointCloud& target,
                         const Transform& initial_guess,
                         const ICPOptions& options);
```

The public model should include `PointXYZ`, an owning or view-based `PointCloud`,
a numerically well-defined SE(3) `Transform`, `ICPMethod`, `ICPOptions`, and
`RegistrationResult`. The result needs transform, status/failure reason,
converged flag, iteration count, final error, correspondence count, and timing
fields. Keep CUDA headers and device pointers out of the public API.

Define frame and transform conventions in the API docs, including whether the
result maps source coordinates into target coordinates. Preserve adapters for
`load_cloud`, the existing binary files, and AUV callers.

## Acceptance criteria

- CPU-only consumers can include the public header without CUDA or ROS.
- Options validate finite voxel size, radius, tolerance, iteration, and minimum
  correspondence values before work begins.
- Existing `bench` and `corr` behavior remains available or has a documented
  migration path.
- A small compile-only example demonstrates the API with two in-memory clouds.
- No algorithmic replacement is made solely for directory aesthetics.
