#include "port/solver_audit.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace ula::port {

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using View = Eigen::Map<const MatrixXf>;

void SolverAudit::lu(const float* matrix, const float* inverse, int n) {
  const View a(matrix,n,n), inv(inverse,n,n);
  if (!a.allFinite() || !inv.allFinite()) throw std::runtime_error("non-finite matrix in LU audit");
  const MatrixXd identity=MatrixXd::Identity(n,n);
  const MatrixXf ref=a.partialPivLu().inverse();
  lu_residual=std::max(lu_residual,(a.cast<double>()*inv.cast<double>()-identity).cwiseAbs().maxCoeff());
  eigen_lu_residual=std::max(eigen_lu_residual,(a.cast<double>()*ref.cast<double>()-identity).cwiseAbs().maxCoeff());
  ++lu_count;
}

void SolverAudit::svd(const float* matrix,const float* u,const float* v,const float* sigma,
                      int m,int n,bool weighted,int task,double cutoff) {
  const View a(matrix,m,n), u_view(u,m,m), v_view(v,n,n);
  const Eigen::Map<const Eigen::VectorXf> s(sigma,n);
  if (!a.allFinite() || !u_view.allFinite() || !v_view.allFinite() || !s.allFinite())
    throw std::runtime_error("non-finite result in SVD audit");
  // The device factors J^T for the unweighted case. Give Eigen J itself,
  // so its QR preconditioning is the same as the reference path.
  const MatrixXf original=weighted ? MatrixXf(a) : MatrixXf(a.transpose());
  const Eigen::JacobiSVD<MatrixXf> ref(original);
  int ref_rank=0, gpu_rank=0;
  for (int i=0;i<n;++i) {
    const double r=ref.singularValues()[i], g=s[i];
    const bool rk=r>cutoff, gk=g>cutoff;
    ref_rank+=rk;
    gpu_rank+=gk;
    sigma_difference=std::max(sigma_difference,std::abs(r-g));
    min_cutoff_distance=std::min({min_cutoff_distance,std::abs(r-cutoff),std::abs(g-cutoff)});
    if (rk!=gk) {
      if (crossings<8)
        std::fprintf(stderr,"CUTOFF REVIEW tick=%d %s task=%d %dx%d sigma[%d]: Eigen %.9g, CUDA %.9g, cutoff %.9g\n",
                     tick,weighted?"weighted":"kinematic",task,int(original.rows()),int(original.cols()),i,r,g,cutoff);
      ++crossings;
    }
  }
  rank_changes+=ref_rank!=gpu_rank;
  const MatrixXd reconstructed=u_view.leftCols(n).cast<double>()*s.cast<double>().asDiagonal()*v_view.cast<double>().transpose();
  const MatrixXd target=a.cast<double>();
  const double norm=target.norm();
  svd_residual=std::max(svd_residual,(reconstructed-target).norm()/(norm!=0 ? norm : 1));
  ++svd_count;
}

void SolverAudit::report() const {
  std::fprintf(stderr,"SOLVER AUDIT same-input samples: %d LU matrices, max |M inverse - I| %.3g (Eigen %.3g)\n",
               lu_count,lu_residual,eigen_lu_residual);
  std::fprintf(stderr,
               "SOLVER AUDIT %d SVD matrices, max relative reconstruction error %.3g, max singular difference %.3g, "
               "nearest cutoff %.3g, %d cutoff crossings in %d rank-changing solves\n",
               svd_count,svd_residual,sigma_difference,min_cutoff_distance,crossings,rank_changes);
  if (crossings) std::fprintf(stderr,"REVIEW: sampled SVD cutoff decisions differ; discuss before accepting\n");
}

}
