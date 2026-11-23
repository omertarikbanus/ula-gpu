#ifndef ULA_GPU_WBIC_PROBLEM_HH
#define ULA_GPU_WBIC_PROBLEM_HH

#include "qp/gi_qp.hh"

namespace ula::qp {

// Packed row-major active extents, not padded row strides. Safe to copy
// between host and device without Eigen or QuadProg++ objects.
template <class T>
struct WbicProblem {
  enum { N = 18, P = 6, M = 24 };
  int n{}, p{}, m{};
  T G[N * N]{}, g0[N]{}, CE[N * P]{}, ce0[P]{}, CI[N * M]{}, ci0[M]{};
};

template <class T>
struct WbicSolution {
  T x[WbicProblem<T>::N]{};
  Result<T> result{};
};

template <class T>
ULA_HD inline WbicSolution<T> solve_wbic(const WbicProblem<T>& in) {
  using B = WbicProblem<T>;
  Workspace<T, B::N, B::P, B::M> workspace;
  WbicSolution<T> out{};
  out.result = solve(workspace, in.n, in.p, in.m, in.G, in.g0,
                     in.CE, in.ce0, in.CI, in.ci0, out.x);
  return out;
}

// Dynamics and task acceleration from the host. The device packs the QP,
// solves it and computes joint feedforward torques in one call.
template <class T>
struct WbicTorqueProblem {
  enum { Dof = 18, Base = 6, Joints = Dof - Base, MaxContact = 12, MaxIneq = 24 };
  int contact_rows{}, inequality_rows{};
  T acceleration_pre[Dof]{};
  T desired_force[MaxContact]{};
  T cost[Dof]{};
  T mass[Dof][Dof]{};
  T contact_transpose[Dof][MaxContact]{};
  T coriolis[Dof]{}, gravity[Dof]{};
  T force_constraint[MaxIneq][MaxContact]{}, force_lower_bound[MaxIneq]{};
};

template <class T>
struct WbicTorqueSolution {
  using B = WbicTorqueProblem<T>;
  WbicSolution<T> qp{};
  T acceleration[B::Dof]{}, force[B::MaxContact]{}, torque[B::Joints]{};
};

template <class T>
ULA_HD inline bool assemble_wbic_qp(const WbicTorqueProblem<T>& in, WbicProblem<T>& q) {
  using B = WbicTorqueProblem<T>;
  const int contact = in.contact_rows;
  const int inequalities = in.inequality_rows;
  if (contact < 0 || contact > B::MaxContact || inequalities < 0 || inequalities > B::MaxIneq ||
      (contact == 0 && inequalities != 0)) return false;
  q = WbicProblem<T>{};
  q.n = B::Base + contact;
  q.p = B::Base;
  // Match WBIC's existing single zero inequality when there are no contacts.
  q.m = contact > 0 ? inequalities : 1;
  for (int i = 0; i < q.n; ++i) q.G[i * q.n + i] = in.cost[i];
  for (int row = 0; row < B::Base; ++row) {
    T inertial = 0, support = 0;
    for (int i = 0; i < B::Dof; ++i) inertial += in.mass[row][i] * in.acceleration_pre[i];
    for (int i = 0; i < contact; ++i) support += in.contact_transpose[row][i] * in.desired_force[i];
    q.ce0[row] = ((inertial + in.coriolis[row]) + in.gravity[row]) - support;
    for (int i = 0; i < B::Base; ++i) q.CE[i * q.p + row] = in.mass[row][i];
    for (int i = 0; i < contact; ++i) q.CE[(B::Base + i) * q.p + row] = -in.contact_transpose[row][i];
  }
  for (int row = 0; row < inequalities; ++row) {
    T desired = 0;
    for (int i = 0; i < contact; ++i) {
      const T f = in.force_constraint[row][i];
      q.CI[(B::Base + i) * q.m + row] = f;
      desired += f * in.desired_force[i];
    }
    q.ci0[row] = desired - in.force_lower_bound[row];
  }
  return true;
}

template <class T>
struct WbicWorkspace {
  WbicProblem<T> problem;
  Workspace<T, WbicProblem<T>::N, WbicProblem<T>::P, WbicProblem<T>::M> solver;
};

template <class T>
ULA_HD inline void solve_wbic_torque(const WbicTorqueProblem<T>& in, WbicTorqueSolution<T>& out,
                                    WbicWorkspace<T>& work) {
  using B = WbicTorqueProblem<T>;
  out = WbicTorqueSolution<T>{};
  auto& q = work.problem;
  if (!assemble_wbic_qp(in, q)) {
    out.qp.result.status = kBadSize;
    return;
  }
  out.qp.result = solve(work.solver, q.n, q.p, q.m, q.G, q.g0, q.CE, q.ce0, q.CI, q.ci0, out.qp.x);
  if (out.qp.result.status != kOk) return;
  const int contact = in.contact_rows;
  for (int i = 0; i < B::Dof; ++i) {
    out.acceleration[i] = in.acceleration_pre[i];
    if (i < B::Base) out.acceleration[i] += out.qp.x[i];
  }
  for (int i = 0; i < contact; ++i) out.force[i] = in.desired_force[i] + out.qp.x[B::Base + i];
  for (int j = 0; j < B::Joints; ++j) {
    const int row = B::Base + j;
    T inertial = 0, support = 0;
    for (int i = 0; i < B::Dof; ++i) inertial += in.mass[row][i] * out.acceleration[i];
    for (int i = 0; i < contact; ++i) support += in.contact_transpose[row][i] * out.force[i];
    out.torque[j] = (inertial + in.coriolis[row]) + in.gravity[row] - support;
  }
}

template <class T>
ULA_HD inline void solve_wbic_torque(const WbicTorqueProblem<T>& in, WbicTorqueSolution<T>& out) {
  WbicWorkspace<T> work;
  solve_wbic_torque(in, out, work);
}

// Optional per-controller host dispatch. Device code only sees the POD above.
template <class T>
struct WbicDispatch {
  using Run = void (*)(void*, const WbicTorqueProblem<T>&, WbicTorqueSolution<T>&);
  void* context{};
  Run run{};
};

}  // namespace ula::qp

#endif
