#ifndef ULA_GPU_SOLVER_AUDIT_HH
#define ULA_GPU_SOLVER_AUDIT_HH

namespace ula::port {

// Replay-only checks. Both libraries receive the same float matrix; double
// precision is used to measure residuals, not to alter the controller.
struct SolverAudit {
  int tick=0, lu_count=0, svd_count=0, crossings=0, rank_changes=0;
  double lu_residual=0, eigen_lu_residual=0, svd_residual=0, sigma_difference=0;
  double min_cutoff_distance=1;

  void lu(const float* matrix, const float* inverse, int n);
  void svd(const float* matrix, const float* u, const float* v, const float* sigma,
           int m, int n, bool weighted, int task, double cutoff);
  void report() const;
};

}
#endif
