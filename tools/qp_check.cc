// qp_check: solve every QP recorded by qp_dump with the fixed-storage solver
// (src/qp/gi_qp.hh) in double and float and compare with QuadProg++.
//
// usage: qp_check <in.qp> [timing_repeats=200]
//
//  (a) double port vs QuadProg++ (must agree to ~1e-10)
//  (b) float solver vs QuadProg++: accuracy, constraint violation (evaluated in
//      double on the recorded data), active-set mismatches, iterations, status
//  (c) CPU time per solve: QuadProg++ (incl. copying the inputs into its
//      matrices, since it factorises G in place), port double, port float
//      (float inputs pre-converted).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <vector>

#include "Goldfarb_Optimizer/QuadProg++.hh"
#include "qp/gi_qp.hh"
#include "qp_file.hh"

static const int CN = 18, CP = 6, CM = 24;
using WsD = ula::qp::Workspace<double, CN, CP, CM>;
using WsF = ula::qp::Workspace<float, CN, CP, CM>;

template <class S>
struct FQp {  // inputs converted to scalar S
  std::vector<S> G, g0, CE, ce0, CI, ci0;
  explicit FQp(const RecordedQp& q) {
    auto cv = [](const std::vector<double>& a) { return std::vector<S>(a.begin(), a.end()); };
    G = cv(q.G); g0 = cv(q.g0); CE = cv(q.CE); ce0 = cv(q.ce0); CI = cv(q.CI); ci0 = cv(q.ci0);
  }
};

static void violations(const RecordedQp& q, const double* x, double& eq, double& ineq) {
  eq = 0; ineq = 0;
  for (uint32_t k = 0; k < q.p; ++k) {
    double s = q.ce0[k];
    for (uint32_t i = 0; i < q.n; ++i) s += q.CE[i * q.p + k] * x[i];
    eq = std::max(eq, std::fabs(s));
  }
  for (uint32_t k = 0; k < q.m; ++k) {
    double s = q.ci0[k];
    for (uint32_t i = 0; i < q.n; ++i) s += q.CI[i * q.m + k] * x[i];
    ineq = std::max(ineq, -s);
  }
}

static std::vector<int> activeSet(const ula::qp::Result<double>& r, const WsD& w, int p) {
  std::vector<int> a(w.A + p, w.A + r.iq);
  std::sort(a.begin(), a.end());
  return a;
}
static std::vector<int> activeSet(const ula::qp::Result<float>& r, const WsF& w, int p) {
  std::vector<int> a(w.A + p, w.A + r.iq);
  std::sort(a.begin(), a.end());
  return a;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s <in.qp> [repeats=200]\n", argv[0]); return 2; }
  const int repeats = argc > 2 ? std::atoi(argv[2]) : 200;
  std::vector<RecordedQp> qps;
  if (!loadQps(argv[1], qps)) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  const size_t N = qps.size();
  int maxn = 0, maxp = 0, maxm = 0;
  for (auto& q : qps) { maxn = std::max<int>(maxn, q.n); maxp = std::max<int>(maxp, q.p); maxm = std::max<int>(maxm, q.m); }
  std::printf("qp_check: %zu QPs, max n=%d p=%d m=%d (capacities %d %d %d)\n", N, maxn, maxp, maxm, CN, CP, CM);

  WsD wd; WsF wf;
  // (a) double port vs QuadProg++
  double a_dx = 0, a_df = 0, a_df_rel = 0;
  int d_bad = 0, sdmax = 0, idmax = 0;
  double sdsum = 0, idsum = 0;
  std::vector<std::vector<int>> actD(N);
  for (size_t k = 0; k < N; ++k) {
    const RecordedQp& q = qps[k];
    std::vector<double> xd(q.n);
    ula::qp::Result<double> rd = ula::qp::solve(wd, q.n, q.p, q.m, q.G.data(), q.g0.data(), q.CE.data(),
                                                q.ce0.data(), q.CI.data(), q.ci0.data(), xd.data());
    if (rd.status != ula::qp::kOk) ++d_bad;
    actD[k] = activeSet(rd, wd, q.p);
    double dx = 0;
    for (uint32_t i = 0; i < q.n; ++i) dx = std::max(dx, std::fabs(xd[i] - q.x[i]));
    a_dx = std::max(a_dx, dx);
    a_df = std::max(a_df, std::fabs(rd.f - q.f));
    a_df_rel = std::max(a_df_rel, std::fabs(rd.f - q.f) / std::max(std::fabs(q.f), 1e-300));
    sdmax = std::max(sdmax, rd.steps); sdsum += rd.steps;
    idmax = std::max(idmax, rd.iter); idsum += rd.iter;
  }
  std::printf("\n(a) double port vs QuadProg++ (%zu QPs)\n", N);
  std::printf("    max |dx| = %.3e   max |df| = %.3e   max rel |df| = %.3e   non-ok status: %d\n",
              a_dx, a_df, a_df_rel, d_bad);
  std::printf("    iterations (steps of 2a): max %d mean %.2f ; step-1 passes: max %d mean %.2f\n",
              sdmax, sdsum / N, idmax, idsum / N);

  // (b) float solver, once with the shipped tolerance and once with the literal
  // QuadProg++ rule (eps = FLT_EPSILON in the step-1 test)
  const float psiList[2] = {0.f, FLT_EPSILON};
  const char* psiName[2] = {"default psi_eps (1e-9)", "literal rule, psi_eps = FLT_EPSILON"};
  for (int v = 0; v < 2; ++v) {
    double b_dx = 0, b_dxrel = 0, b_dfrel = 0, b_eq = 0, b_ineq = 0, qp_eq = 0, qp_ineq = 0, rel_sum = 0;
    int b_actmis = 0, b_bad = 0, sfmax = 0, ifmax = 0, statusHist[8] = {0};
    double sfsum = 0, ifsum = 0;
    size_t worst = 0;
    for (size_t k = 0; k < N; ++k) {
      const RecordedQp& q = qps[k];
      FQp<float> fq(q);
      std::vector<float> xf(q.n);
      ula::qp::Result<float> rf = ula::qp::solve(wf, q.n, q.p, q.m, fq.G.data(), fq.g0.data(), fq.CE.data(),
                                                 fq.ce0.data(), fq.CI.data(), fq.ci0.data(), xf.data(), 400,
                                                 psiList[v]);
      statusHist[rf.status]++;
      if (rf.status != ula::qp::kOk) ++b_bad;
      sfmax = std::max(sfmax, rf.steps); sfsum += rf.steps;
      ifmax = std::max(ifmax, rf.iter); ifsum += rf.iter;
      std::vector<double> xfd(xf.begin(), xf.end());
      double fdx = 0, xn = 0;
      for (uint32_t i = 0; i < q.n; ++i) { fdx = std::max(fdx, std::fabs(xfd[i] - q.x[i])); xn = std::max(xn, std::fabs(q.x[i])); }
      const double rel = fdx / std::max(xn, 1e-12);
      if (fdx > b_dx) worst = k;
      b_dx = std::max(b_dx, fdx);
      b_dxrel = std::max(b_dxrel, rel);
      rel_sum += rel;
      b_dfrel = std::max(b_dfrel, std::fabs((double)rf.f - q.f) / std::max(std::fabs(q.f), 1e-12));
      double eq, ineq;
      violations(q, xfd.data(), eq, ineq);
      b_eq = std::max(b_eq, eq); b_ineq = std::max(b_ineq, ineq);
      violations(q, q.x.data(), eq, ineq);
      qp_eq = std::max(qp_eq, eq); qp_ineq = std::max(qp_ineq, ineq);
      if (activeSet(rf, wf, q.p) != actD[k]) ++b_actmis;
    }
    std::printf("\n(b%d) float solver vs QuadProg++, %s\n", v + 1, psiName[v]);
    std::printf("    max |dx| abs = %.3e   max |dx|/|x|inf = %.3e (mean %.3e)   max rel |df| = %.3e\n",
                b_dx, b_dxrel, rel_sum / N, b_dfrel);
    std::printf("    max violation (double check on recorded data): float eq %.3e ineq %.3e | QuadProg++ eq %.3e ineq %.3e\n",
                b_eq, b_ineq, qp_eq, qp_ineq);
    std::printf("    active-set mismatches vs double port: %d / %zu\n", b_actmis, N);
    std::printf("    iterations (steps of 2a): max %d mean %.2f ; step-1 passes: max %d mean %.2f\n",
                sfmax, sfsum / N, ifmax, ifsum / N);
    std::printf("    status counts: ok %d infeasible %d notPD %d iterLimit %d dependent %d badSize %d (non-ok total %d)\n",
                statusHist[0], statusHist[1], statusHist[2], statusHist[3], statusHist[4], statusHist[5], b_bad);
    std::printf("    worst abs |dx| at QP #%zu (tick %u, n=%u m=%u)\n", worst, qps[worst].tick, qps[worst].n, qps[worst].m);
  }

  // (c) timing
  using Clock = std::chrono::steady_clock;
  std::vector<FQp<float>> fqs;
  for (auto& q : qps) fqs.emplace_back(q);
  double sink = 0;
  double tQ = 0, tD = 0, tF = 0;
  {  // QuadProg++: GMatr copies are part of the call (G is destroyed by the solve)
    auto t0 = Clock::now();
    for (int rep = 0; rep < repeats; ++rep)
      for (auto& q : qps) {
        GolDIdnani::GMatr<double> G(q.n, q.n), CE(q.n, q.p), CI(q.n, q.m);
        GolDIdnani::GVect<double> g0(q.n), ce0(q.p), ci0(q.m), x(q.n);
        for (uint32_t i = 0; i < q.n; ++i) {
          for (uint32_t j = 0; j < q.n; ++j) G[i][j] = q.G[i * q.n + j];
          for (uint32_t j = 0; j < q.p; ++j) CE[i][j] = q.CE[i * q.p + j];
          for (uint32_t j = 0; j < q.m; ++j) CI[i][j] = q.CI[i * q.m + j];
          g0[i] = q.g0[i];
        }
        for (uint32_t j = 0; j < q.p; ++j) ce0[j] = q.ce0[j];
        for (uint32_t j = 0; j < q.m; ++j) ci0[j] = q.ci0[j];
        sink += solve_quadprog(G, g0, CE, ce0, CI, ci0, x);
      }
    tQ = std::chrono::duration<double>(Clock::now() - t0).count();
  }
  {
    std::vector<double> x(CN);
    auto t0 = Clock::now();
    for (int rep = 0; rep < repeats; ++rep)
      for (auto& q : qps)
        sink += ula::qp::solve(wd, q.n, q.p, q.m, q.G.data(), q.g0.data(), q.CE.data(), q.ce0.data(),
                               q.CI.data(), q.ci0.data(), x.data()).f;
    tD = std::chrono::duration<double>(Clock::now() - t0).count();
  }
  {
    std::vector<float> x(CN);
    auto t0 = Clock::now();
    for (int rep = 0; rep < repeats; ++rep)
      for (auto& fq : fqs) {
        const RecordedQp& q = qps[&fq - fqs.data()];
        sink += ula::qp::solve(wf, q.n, q.p, q.m, fq.G.data(), fq.g0.data(), fq.CE.data(), fq.ce0.data(),
                               fq.CI.data(), fq.ci0.data(), x.data()).f;
      }
    tF = std::chrono::duration<double>(Clock::now() - t0).count();
  }
  const double cnt = (double)repeats * N;
  std::printf("\n(c) CPU time per solve (%d repeats x %zu QPs, this machine, -O%s)\n", repeats, N,
#ifdef NDEBUG
              "2/3"
#else
              "0"
#endif
  );
  std::printf("    QuadProg++ : %8.2f us\n    port double: %8.2f us  (%.2fx)\n    port float : %8.2f us  (%.2fx)\n",
              tQ / cnt * 1e6, tD / cnt * 1e6, tQ / tD, tF / cnt * 1e6, tQ / tF);
  if (sink == 12345.678) std::printf("\n");
  return 0;
}
