// Optional observer of the WBIC quadratic program (used by tools/qp_dump.cc).
// When no recorder is installed (the default) WBIC behaves exactly as before.
// Layout of the arrays is QuadProg++'s: row-major, G n x n, CE n x p,
// CI n x m (one constraint per column), exactly as they are handed to the
// solver. problem() is called just before the solve (G is still unfactored),
// solution() right after it with the solver's objective value and solution.
#ifndef ULA_WBIC_QP_RECORDER_HPP
#define ULA_WBIC_QP_RECORDER_HPP

struct WbicQpRecorder {
  virtual ~WbicQpRecorder() {}
  virtual void problem(int n, int p, int m, const double* G, const double* g0,
                       const double* CE, const double* ce0, const double* CI,
                       const double* ci0) = 0;
  virtual void solution(double f, const double* x, int n) = 0;
};

inline WbicQpRecorder*& wbicQpRecorder() {
  static WbicQpRecorder* rec = nullptr;
  return rec;
}

#endif
