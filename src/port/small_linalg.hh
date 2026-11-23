#ifndef ULA_GPU_SMALL_LINALG_HH
#define ULA_GPU_SMALL_LINALG_HH

#include <Eigen/Householder>
#include <Eigen/Jacobi>

#include "port/rigid_body.hh"
#include "qp/gi_qp.hh"

namespace ula::small {

// Eigen views use the active extents; the backing storage stays fixed and
// can be shared between host and device without Eigen objects in the wire ABI.
constexpr int kMax = rigid::kDof;
template<class T> using Mat = rigid::Mat<T, kMax, kMax>;
template<class T> using BoundedMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                                   Eigen::RowMajor, kMax, kMax>;
using OuterStride = Eigen::OuterStride<kMax>;

template<class T> ULA_HD auto view(Mat<T>& a, int n, int m) {
  return Eigen::Map<BoundedMat<T>, Eigen::Unaligned, OuterStride>(&a.data[0][0], n, m);
}

template<class T> ULA_HD auto view(const Mat<T>& a, int n, int m) {
  return Eigen::Map<const BoundedMat<T>, Eigen::Unaligned, OuterStride>(&a.data[0][0], n, m);
}

template<class T> ULA_HD auto vec(const T* a, int n) {
  return Eigen::Map<const Eigen::Matrix<T,Eigen::Dynamic,1,Eigen::ColMajor,kMax,1>>(a,n);
}

template<class T> ULA_HD T abs(T x) { return qp::qfabs(x); }
template<class T> ULA_HD void swap(T& a, T& b) { T t = a; a = b; b = t; }

template<class T> ULA_HD void identity(Mat<T>& a, int n, int m) {
  view(a,n,m).setIdentity();
}

template<class T> ULA_HD void multiply(const Mat<T>& a, const Mat<T>& b, Mat<T>& c,
                                       int n, int k, int m, bool bt=false) {
  auto av = view(a,n,k);
  auto out = view(c,n,m);
  if (bt) out.noalias() = av.lazyProduct(view(b,m,k).transpose());
  else out.noalias() = av.lazyProduct(view(b,k,m));
}

// Draft fallback: the full PartialPivLU API is host-only in Eigen 3.4.
// Partial-pivot LU, with the same swaps applied to the identity RHS.
template<class T> ULA_HD bool invert(Mat<T>& lu, Mat<T>& inv, int n) {
  auto a=view(lu,n,n);
  auto rhs=view(inv,n,n);
  rhs.setIdentity();
  for (int k=0; k<n; ++k) {
    Eigen::Index offset;
    const T pivot=a.col(k).tail(n-k).cwiseAbs().maxCoeff(&offset);
    if (!std::isfinite(pivot) || pivot == T(0)) return false;
    const int p=k+static_cast<int>(offset);
    if (p!=k) {
      a.row(k).swap(a.row(p));
      rhs.row(k).swap(rhs.row(p));
    }
    for (int i=k+1; i<n; ++i) {
      const T factor = a(i,k) /= a(k,k);
      a.row(i).tail(n-k-1) -= factor * a.row(k).tail(n-k-1);
      rhs.row(i) -= factor * rhs.row(k);
    }
  }
  // Eigen's triangular solve dispatch also calls host-only code here.
  for (int i=n-1; i>=0; --i) {
    for (int k=i+1; k<n; ++k) rhs.row(i) -= a(i,k) * rhs.row(k);
    rhs.row(i) /= a(i,i);
  }
  return true;
}

template<class T> struct Svd {
  Mat<T> work, u, v, qr;
  T tau[kMax];
  int column[kMax];
};

// Draft fallback: the full JacobiSVD and ColPivHouseholderQR APIs are
// host-only in Eigen 3.4, but the individual reflectors/rotations work.
// Thin Jacobi SVD for n <= m. For a wide matrix, A^T P = Q R gives
// A = P R^T Q^T. Diagonalize the square core with two-sided rotations.
template<class T> ULA_HD bool pinverse(const Mat<T>& a, int n, int m, double cutoff,
                                       Mat<T>& out, Svd<T>& w) {
  T scale = 0;
  for (int i=0; i<n; ++i) for (int j=0; j<m; ++j) {
    const T v = abs(a.data[i][j]);
    if (!std::isfinite(v)) return false;
    if (v > scale) scale = v;
  }
  if (scale == T(0)) {
    view(out,m,n).setZero();
    return true;
  }
  identity(w.v,m,n);
  if (n==m) {
    identity(w.u,n,n);
    view(w.work,n,n) = view(a,n,n)/scale;
  } else {
    auto qr = view(w.qr,m,n);
    auto v = view(w.v,m,n);
    qr = view(a,n,m).transpose()/scale;
    for (int j=0; j<n; ++j) w.column[j]=j;
    T scratch[kMax];
    for (int k=0; k<n; ++k) {
      int p=k;
      T best=-1;
      for (int j=k; j<n; ++j) {
        const T norm=qr.col(j).tail(m-k).squaredNorm();
        if (norm>best) { best=norm; p=j; }
      }
      if (p!=k) {
        swap(w.column[k],w.column[p]);
        qr.col(k).swap(qr.col(p));
      }
      qr.col(k).tail(m-k).makeHouseholderInPlace(w.tau[k],qr(k,k));
      if (k+1<n)
        qr.bottomRightCorner(m-k,n-k-1).applyHouseholderOnTheLeft(qr.col(k).tail(m-k-1),w.tau[k],scratch);
    }
    for (int k=n-1; k>=0; --k)
      v.bottomRows(m-k).applyHouseholderOnTheLeft(qr.col(k).tail(m-k-1),w.tau[k],scratch);
    for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) {
      w.work.data[i][j]=i>=j ? w.qr.data[j][i] : T(0);
      w.u.data[i][j]=T(i==w.column[j]);
    }
  }
  auto s=view(w.work,n,n);
  auto u=view(w.u,n,n);
  auto v=view(w.v,m,n);
  T largest=s.diagonal().cwiseAbs().maxCoeff();
  const T tiny=qp::Lim<T>::tiny();
  bool done=false;
  for (int sweep=0; sweep<64 && !done; ++sweep) {
    done=true;
    for (int p=1; p<n; ++p) for (int q=0; q<p; ++q) {
      const T tol=2*qp::Lim<T>::eps()*largest;
      const T threshold=tol>tiny ? tol : tiny;
      if (abs(s(p,q))<=threshold && abs(s(q,p))<=threshold) continue;
      done=false;
      const T skew=s(q,p)-s(p,q);
      Eigen::JacobiRotation<T> symmetrize(T(1),T(0));
      if (abs(skew)>=tiny) {
        const T t=(s(p,p)+s(q,q))/skew;
        const T norm=qp::qsqrt(1+t*t);
        symmetrize=Eigen::JacobiRotation<T>(t/norm,1/norm);
      }
      Eigen::Matrix<T,2,2> block;
      block << s(p,p),s(p,q),s(q,p),s(q,q);
      block.applyOnTheLeft(0,1,symmetrize);
      Eigen::JacobiRotation<T> right;
      right.makeJacobi(block,0,1);
      const auto left=symmetrize*right.transpose();
      s.applyOnTheLeft(p,q,left);
      u.applyOnTheRight(p,q,left.transpose());
      s.applyOnTheRight(p,q,right);
      v.applyOnTheRight(p,q,right);
      if (abs(s(p,p))>largest) largest=abs(s(p,p));
      if (abs(s(q,q))>largest) largest=abs(s(q,q));
    }
  }
  if (!done) return false;
  // Sorting is not needed mathematically, but summing largest sigma first
  // makes accumulation predictable and matches the reference convention.
  for (int k=0; k<n; ++k) {
    int p=k;
    for (int i=k+1; i<n; ++i) if (abs(s(i,i))>abs(s(p,p))) p=i;
    if (p!=k) {
      swap(s(k,k),s(p,p));
      u.col(k).swap(u.col(p));
      v.col(k).swap(v.col(p));
    }
    const T sigma=abs(s(k,k))*scale;
    w.tau[k]=sigma>cutoff ? T((s(k,k)<0 ? -1.0 : 1.0)/sigma) : T(0);
  }
  for (int k=0; k<n; ++k) v.col(k) *= w.tau[k];
  view(out,m,n).noalias() = v.lazyProduct(u.transpose());
  return true;
}

}  // namespace ula::small
#endif
