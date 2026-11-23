// gi_qp.hh: fixed-storage, bounded-iteration port of the Goldfarb-Idnani dual
// active-set QP solver, mirroring third-party/Goldfarb_Optimizer/QuadProg++.cc
// step by step (same tolerances, same control flow, same sign conventions).
//
//   minimize   1/2 x'Gx + g0'x
//   subject to CE'x + ce0  = 0      (p equality constraints)
//              CI'x + ci0 >= 0      (m inequality constraints)
//
// Inputs are flat ROW-MAJOR arrays, laid out exactly like QuadProg++'s
// matrices (so CE and CI are n x p and n x m, one constraint per COLUMN):
//   G[i*n + j], CE[i*p + k], CI[i*m + k], g0[n], ce0[p], ci0[m].
// G is not modified (the workspace holds its Cholesky factor).
//
// Properties: header-only; templated on the scalar type and on the capacities
// (max n, p, m); all storage in `Workspace` (no heap, no std containers, no
// exceptions, no I/O, no virtuals); bounded iterations; every function is
// ULA_HD so the same code compiles for host and CUDA device; math goes
// through the tiny shim below.
//
// Differences from QuadProg++ (all deliberate):
//  * errors are status codes instead of exceptions / exit();
//  * "infinity" is the largest finite number of the scalar type (the only
//    uses are `t = min(t1, t2)` and the `t >= inf` test, which behave the same);
//  * for float, the step-1 'no infeasibility left' threshold uses 1e-9 instead of
//    FLT_EPSILON (pass psi_eps = FLT_EPSILON to get the literal rule);
//  * an iteration cap (`max_steps`, counted over every pass of step 2a);
//  * on a dependent/degenerate equality set the status is LinearlyDependent.
#ifndef ULA_QP_GI_QP_HH
#define ULA_QP_GI_QP_HH

#include <float.h>
#include <math.h>

#ifdef __CUDACC__
#define ULA_HD __host__ __device__
#else
#define ULA_HD
#endif

namespace ula {
namespace qp {

// ---- math shim: the only math used by the solver ---------------------------
ULA_HD inline float qsqrt(float x) { return ::sqrtf(x); }
ULA_HD inline double qsqrt(double x) { return ::sqrt(x); }
ULA_HD inline float qfabs(float x) { return x < 0.f ? -x : x; }
ULA_HD inline double qfabs(double x) { return x < 0. ? -x : x; }
template <class S> struct Lim;
template <> struct Lim<float> {
  ULA_HD static float eps() { return FLT_EPSILON; }
  ULA_HD static float tiny() { return FLT_MIN; }
  ULA_HD static float inf() { return FLT_MAX; }
  // eps of the step-1 'no infeasibility left' test. FLT_EPSILON (the literal
  // QuadProg++ rule) is far too loose for float on this QP (accepts ~0.1 N
  // violations); 1e-9 is accurate and stable, see docs/QP_SOLVER.md.
  ULA_HD static float psi_eps() { return 1e-9f; }
};
template <> struct Lim<double> {
  ULA_HD static double eps() { return DBL_EPSILON; }
  ULA_HD static double tiny() { return DBL_MIN; }
  ULA_HD static double inf() { return DBL_MAX; }
  ULA_HD static double psi_eps() { return DBL_EPSILON; }  // exactly QuadProg++
};

enum Status {
  kOk = 0,
  kInfeasible = 1,           // no step in primal or dual space (QuadProg++ returns +inf)
  kNotPositiveDefinite = 2,  // Cholesky of G failed
  kIterationLimit = 3,
  kLinearlyDependent = 4,    // equality constraints dependent (QuadProg++ throws)
  kBadSize = 5               // n, p or m outside the compile-time capacities
};

template <class S>
struct Result {
  S f;         // objective value at the solution
  int status;  // Status
  int iter;    // passes through step 1 (what QuadProg++ calls iter)
  int steps;   // passes through step 2a (total work; this is what max_steps caps)
  int iq;      // size of the final active set (equalities first)
};

// Capacities: n <= N, p <= P, m <= M. WBIC needs N=18, P=6, M=24.
template <class S, int N, int P, int M>
struct Workspace {
  enum { K = (M + P > N + 1) ? (M + P) : (N + 1) };
  S L[N][N];  // Cholesky factor of G (both triangles filled, as in QuadProg++)
  S R[N][N];
  S J[N][N];
  S s[K], z[N], r[K], d[N], np[N], u[K], x_old[N], u_old[K];
  int A[K], A_old[K], iai[K];
  bool iaexcl[K];
  int iq;  // active set size after solve; A[0..p) = -(i+1) equalities,
           // A[p..iq) = inequality indices
};

namespace detail {

template <class S>
ULA_HD inline S qmin(S a, S b) { return b < a ? b : a; }

template <class S>
ULA_HD inline S distance(S a, S b) {
  S a1 = qfabs(a), b1 = qfabs(b), t;
  if (a1 > b1) {
    t = b1 / a1;
    return a1 * qsqrt(S(1) + t * t);
  } else if (b1 > a1) {
    t = a1 / b1;
    return b1 * qsqrt(S(1) + t * t);
  }
  return a1 * qsqrt(S(2));
}

template <class S>
ULA_HD inline S dot(const S* x, const S* y, int n) {
  S sum = 0;
  for (int i = 0; i < n; i++) sum += x[i] * y[i];
  return sum;
}

// d = J' np
template <class S, int N>
ULA_HD inline void compute_d(S* d, const S (*J)[N], const S* np, int n) {
  for (int i = 0; i < n; i++) {
    S sum = 0;
    for (int j = 0; j < n; j++) sum += J[j][i] * np[j];
    d[i] = sum;
  }
}

// z = J(:, iq:n) d(iq:n)
template <class S, int N>
ULA_HD inline void update_z(S* z, const S (*J)[N], const S* d, int n, int iq) {
  for (int i = 0; i < n; i++) {
    z[i] = 0;
    for (int j = iq; j < n; j++) z[i] += J[i][j] * d[j];
  }
}

// r = R^-1 d (back substitution on the iq x iq upper-triangular R)
template <class S, int N>
ULA_HD inline void update_r(const S (*R)[N], S* r, const S* d, int iq) {
  for (int i = iq - 1; i >= 0; i--) {
    S sum = 0;
    for (int j = i + 1; j < iq; j++) sum += R[i][j] * r[j];
    r[i] = (d[i] - sum) / R[i][i];
  }
}

// Givens rotations that zero d[iq+1..n) and rotate J accordingly, then append
// d as a new column of R. Returns false if the new diagonal is ~0 (dependent).
template <class S, int N>
ULA_HD inline bool add_constraint(S (*R)[N], S (*J)[N], S* d, int n, int& iq, S& R_norm) {
  const S eps = Lim<S>::eps();
  for (int j = n - 1; j >= iq + 1; j--) {
    S cc = d[j - 1], ss = d[j];
    S h = distance(cc, ss);
    if (qfabs(h) < eps) continue;  // h == 0
    d[j] = 0;
    ss = ss / h;
    cc = cc / h;
    if (cc < 0) {
      cc = -cc;
      ss = -ss;
      d[j - 1] = -h;
    } else {
      d[j - 1] = h;
    }
    S xny = ss / (S(1) + cc);
    for (int k = 0; k < n; k++) {
      S t1 = J[k][j - 1], t2 = J[k][j];
      J[k][j - 1] = t1 * cc + t2 * ss;
      J[k][j] = xny * (t1 + J[k][j - 1]) - t2;
    }
  }
  iq++;
  for (int i = 0; i < iq; i++) R[i][iq - 1] = d[i];
  if (qfabs(d[iq - 1]) <= eps * R_norm) return false;  // degenerate
  R_norm = qfabs(d[iq - 1]) > R_norm ? qfabs(d[iq - 1]) : R_norm;
  return true;
}

// Remove active constraint l (index into the constraint list), restoring R to
// upper-triangular form with Givens rotations (also applied to J).
template <class S, int N>
ULA_HD inline void delete_constraint(S (*R)[N], S (*J)[N], int* A, S* u, int n, int p,
                                     int& iq, int l) {
  const S eps = Lim<S>::eps();
  int qq = -1;
  for (int i = p; i < iq; i++)
    if (A[i] == l) {
      qq = i;
      break;
    }
  if (qq < 0) return;  // not active: cannot happen in exact flow; keep memory safe

  for (int i = qq; i < iq - 1; i++) {
    A[i] = A[i + 1];
    u[i] = u[i + 1];
    for (int j = 0; j < n; j++) R[j][i] = R[j][i + 1];
  }
  A[iq - 1] = A[iq];
  u[iq - 1] = u[iq];
  A[iq] = 0;
  u[iq] = 0;
  for (int j = 0; j < iq; j++) R[j][iq - 1] = 0;
  iq--;
  if (iq == 0) return;

  for (int j = qq; j < iq; j++) {
    S cc = R[j][j], ss = R[j + 1][j];
    S h = distance(cc, ss);
    if (qfabs(h) < eps) continue;  // h == 0
    cc = cc / h;
    ss = ss / h;
    R[j + 1][j] = 0;
    if (cc < 0) {
      R[j][j] = -h;
      cc = -cc;
      ss = -ss;
    } else {
      R[j][j] = h;
    }
    S xny = ss / (S(1) + cc);
    for (int k = j + 1; k < iq; k++) {
      S t1 = R[j][k], t2 = R[j + 1][k];
      R[j][k] = t1 * cc + t2 * ss;
      R[j + 1][k] = xny * (t1 + R[j][k]) - t2;
    }
    for (int k = 0; k < n; k++) {
      S t1 = J[k][j], t2 = J[k][j + 1];
      J[k][j] = t1 * cc + t2 * ss;
      J[k][j + 1] = xny * (J[k][j] + t1) - t2;
    }
  }
}

}  // namespace detail

// Solve one QP. Returns the status etc.; x[0..n) receives the solution.
// max_steps caps the passes through step 2a (default generous for n,m <= 24).
// psi_eps (0 = Lim<S>::psi_eps(): DBL_EPSILON for double = QuadProg++, 1e-9 for float) scales the step-1
// threshold |sum of infeasibilities| <= m*psi_eps*c1*c2*100; see docs/QP_SOLVER.md
// for why float wants a smaller value than FLT_EPSILON.
template <class S, int N, int P, int M>
ULA_HD inline Result<S> solve(Workspace<S, N, P, M>& w, int n, int p, int m, const S* G,
                              const S* g0, const S* CE, const S* ce0, const S* CI,
                              const S* ci0, S* x, int max_steps = 400, S psi_eps = 0) {
  using namespace detail;
  Result<S> res;
  res.f = 0;
  res.status = kOk;
  res.iter = 0;
  res.steps = 0;
  res.iq = 0;
  w.iq = 0;
  if (n < 1 || n > N || p < 0 || p > P || m < 0 || m > M) {
    res.status = kBadSize;
    return res;
  }

  const S eps = Lim<S>::eps();
  const S inf = Lim<S>::inf();
  // eps used in the 'no infeasibility left' test of step 1 (QuadProg++: machine eps)
  if (!(psi_eps > 0)) psi_eps = Lim<S>::psi_eps();
  int i, j, k, l;
  int ip;  // index of the constraint to be added to the active set
  S f_value, psi, c1, c2, sum, ss, R_norm;
  S t, t1, t2;  // step length: min of partial step t1 and full step t2
  int q, iq, iter = 0, steps = 0;

  S(*L)[N] = w.L;
  S(*R)[N] = w.R;
  S(*J)[N] = w.J;
  S *s = w.s, *z = w.z, *r = w.r, *d = w.d, *np = w.np, *u = w.u, *x_old = w.x_old,
    *u_old = w.u_old;
  int *A = w.A, *A_old = w.A_old, *iai = w.iai;
  bool* iaexcl = w.iaexcl;

  q = 0;  // size of the active set A

  // ---- Preprocessing ---------------------------------------------------------
  // trace of the original G
  c1 = 0;
  for (i = 0; i < n; i++) c1 += G[i * n + i];

  // decompose G = L L^T (cholesky_decomposition; copy first, G stays intact)
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) L[i][j] = G[i * n + j];
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      sum = L[i][j];
      for (k = i - 1; k >= 0; k--) sum -= L[i][k] * L[j][k];
      if (i == j) {
        if (!(sum > 0)) {  // QuadProg++ throws on sum <= 0 (also catches NaN here)
          res.status = kNotPositiveDefinite;
          return res;
        }
        L[i][i] = qsqrt(sum);
      } else {
        L[j][i] = sum / L[i][i];
      }
    }
    for (k = i + 1; k < n; k++) L[i][k] = L[k][i];
  }

  // initialize R
  for (i = 0; i < n; i++) {
    d[i] = 0;
    for (j = 0; j < n; j++) R[i][j] = 0;
  }
  R_norm = 1;  // norm of R

  // J = L^-T (columns of the identity pushed through forward elimination);
  // this is the initial H = G^-1 factor. c2 accumulates trace(L^-1).
  c2 = 0;
  for (i = 0; i < n; i++) {
    d[i] = 1;
    // forward_elimination(L, z, d)
    z[0] = d[0] / L[0][0];
    for (int a = 1; a < n; a++) {
      z[a] = d[a];
      for (int b = 0; b < a; b++) z[a] -= L[a][b] * z[b];
      z[a] = z[a] / L[a][a];
    }
    for (j = 0; j < n; j++) J[i][j] = z[j];
    c2 += z[i];
    d[i] = 0;
  }
  // c1 * c2 estimates cond(G)

  // unconstrained minimizer x = -G^-1 g0 (cholesky_solve: L y = g0, L' x = y)
  {
    S* y = np;  // np is free here; used as the temporary y
    y[0] = g0[0] / L[0][0];
    for (int a = 1; a < n; a++) {
      y[a] = g0[a];
      for (int b = 0; b < a; b++) y[a] -= L[a][b] * y[b];
      y[a] = y[a] / L[a][a];
    }
    x[n - 1] = y[n - 1] / L[n - 1][n - 1];
    for (int a = n - 2; a >= 0; a--) {
      x[a] = y[a];
      for (int b = a + 1; b < n; b++) x[a] -= L[a][b] * x[b];
      x[a] = x[a] / L[a][a];
    }
  }
  for (i = 0; i < n; i++) x[i] = -x[i];
  f_value = S(0.5) * dot(g0, x, n);

  // ---- Add the equality constraints to the working set A ---------------------
  iq = 0;
  for (i = 0; i < p; i++) {
    for (j = 0; j < n; j++) np[j] = CE[j * p + i];
    compute_d<S, N>(d, J, np, n);
    update_z<S, N>(z, J, d, n, iq);
    update_r<S, N>(R, r, d, iq);

    // full step length t2: minimum primal step making the constraint feasible
    t2 = 0;
    if (qfabs(dot(z, z, n)) > eps)  // z != 0
      t2 = (-dot(np, x, n) - ce0[i]) / dot(z, np, n);

    for (k = 0; k < n; k++) x[k] += t2 * z[k];  // x = x + t2 z

    u[iq] = t2;  // u = u+
    for (k = 0; k < iq; k++) u[k] -= t2 * r[k];

    f_value += S(0.5) * (t2 * t2) * dot(z, np, n);
    A[i] = -i - 1;

    if (!add_constraint<S, N>(R, J, d, n, iq, R_norm)) {
      res.status = kLinearlyDependent;  // QuadProg++: throws "Constraints are linearly dependent"
      res.f = f_value;
      res.iq = w.iq = iq;
      return res;
    }
  }

  // iai = K \ A
  for (i = 0; i < m; i++) iai[i] = i;

l1:
  iter++;
  // ---- Step 1: choose a violated constraint ----------------------------------
  for (i = p; i < iq; i++) {
    ip = A[i];
    iai[ip] = -1;
  }

  // s[x] = ci' x + ci0 for all of K \ A
  ss = 0;
  psi = 0;  // sum of all infeasibilities
  ip = 0;   // index of the chosen violated constraint
  for (i = 0; i < m; i++) {
    iaexcl[i] = true;
    sum = 0;
    for (j = 0; j < n; j++) sum += CI[j * m + i] * x[j];
    sum += ci0[i];
    s[i] = sum;
    psi += qmin<S>(0, sum);
  }

  if (qfabs(psi) <= m * psi_eps * c1 * c2 * S(100)) {
    // numerically no infeasibilities left
    q = iq;
    goto done;
  }

  // save old values for u, A and x
  for (i = 0; i < iq; i++) {
    u_old[i] = u[i];
    A_old[i] = A[i];
  }
  for (i = 0; i < n; i++) x_old[i] = x[i];

l2:  // ---- Step 2: check feasibility and determine a new S-pair -----------------
  for (i = 0; i < m; i++) {
    if (s[i] < ss && iai[i] != -1 && iaexcl[i]) {
      ss = s[i];
      ip = i;
    }
  }
  if (ss >= 0) {
    q = iq;
    goto done;
  }

  // np = n[ip]
  for (i = 0; i < n; i++) np[i] = CI[i * m + ip];
  u[iq] = 0;   // u = [u 0]'
  A[iq] = ip;  // add ip to the active set A

l2a:  // ---- Step 2a: determine the step direction -------------------------------
  if (++steps > max_steps) {
    res.status = kIterationLimit;
    goto done;
  }
  // z = H np (primal step direction, through J)
  compute_d<S, N>(d, J, np, n);
  update_z<S, N>(z, J, d, n, iq);
  // r = N* np (negative of the dual step direction), if q > 0
  update_r<S, N>(R, r, d, iq);

  // ---- Step 2b: compute the step length --------------------------------------
  l = 0;
  // t1: partial step (max dual step keeping dual feasibility)
  t1 = inf;
  for (k = p; k < iq; k++) {
    if (r[k] > 0) {
      if (u[k] / r[k] < t1) {
        t1 = u[k] / r[k];
        l = A[k];
      }
    }
  }
  // t2: full step (min primal step making constraint ip feasible)
  if (qfabs(dot(z, z, n)) > eps) {  // z != 0
    t2 = -s[ip] / dot(z, np, n);
    if (t2 < 0) t2 = inf;  // patch for numerical inconsistencies (Takano Akio)
  } else {
    t2 = inf;
  }
  t = qmin<S>(t1, t2);

  // ---- Step 2c: determine the new S-pair and take the step -------------------
  // case (i): no step in primal or dual space -> infeasible
  if (t >= inf) {
    q = iq;
    res.status = kInfeasible;
    goto done;
  }
  // case (ii): step in dual space
  if (t2 >= inf) {
    // u = u + t [-r 1]; drop constraint l from A
    for (k = 0; k < iq; k++) u[k] -= t * r[k];
    u[iq] += t;
    iai[l] = l;
    delete_constraint<S, N>(R, J, A, u, n, p, iq, l);
    goto l2a;
  }

  // case (iii): step in primal and dual space
  for (k = 0; k < n; k++) x[k] += t * z[k];
  f_value += t * dot(z, np, n) * (S(0.5) * t + u[iq]);
  for (k = 0; k < iq; k++) u[k] -= t * r[k];
  u[iq] += t;

  if (qfabs(t - t2) < eps) {
    // full step taken: add constraint ip to the active set
    if (!add_constraint<S, N>(R, J, d, n, iq, R_norm)) {
      // degenerate: exclude ip, drop it, restore the old state, retry step 2
      iaexcl[ip] = false;
      delete_constraint<S, N>(R, J, A, u, n, p, iq, ip);
      for (i = 0; i < m; i++) iai[i] = i;
      for (i = p; i < iq; i++) {
        A[i] = A_old[i];
        u[i] = u_old[i];
        iai[A[i]] = -1;
      }
      for (i = 0; i < n; i++) x[i] = x_old[i];
      goto l2;
    } else {
      iai[ip] = -1;
    }
    goto l1;
  }

  // partial step taken: drop constraint l
  iai[l] = l;
  delete_constraint<S, N>(R, J, A, u, n, p, iq, l);

  // update s[ip] = ci' x + ci0
  sum = 0;
  for (k = 0; k < n; k++) sum += CI[k * m + ip] * x[k];
  s[ip] = sum + ci0[ip];
  goto l2a;

done:
  res.f = f_value;
  res.iter = iter;
  res.steps = steps;
  res.iq = w.iq = iq;
  (void)q;
  return res;
}

}  // namespace qp
}  // namespace ula

#endif  // ULA_QP_GI_QP_HH
