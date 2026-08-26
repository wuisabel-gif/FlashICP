#pragma once

#include "flashicp.hpp"
#include "flashicp/normals.hpp"

namespace flashicp {

std::vector<Corr> correspond_gpu(const std::vector<Point>& source,
                                 const std::vector<Point>& target,
                                 float radius);

namespace internal {

enum class SolveStatus { Ok, Degenerate, NumericalFailure };

SolveStatus solve_rigid(const std::vector<PointXYZ>& source,
                        const std::vector<PointXYZ>& target,
                        const std::vector<Corr>& correspondences,
                        Transform& transform, float& rms);

SolveStatus solve_point_to_plane(const std::vector<PointXYZ>& source,
                                 const std::vector<PointXYZ>& target,
                                 const std::vector<Corr>& correspondences,
                                 const NormalCloud& normals, Transform& delta,
                                 float& rms, std::size_t& valid_correspondences);

float transform_step(const Transform& transform);

#ifdef FLASHICP_CUDA
RegistrationResult align_cuda(const PointCloud& source,
                              const PointCloud& target,
                              const Transform& initial_guess,
                              const ICPOptions& options);
#endif

}  // namespace internal
}  // namespace flashicp
