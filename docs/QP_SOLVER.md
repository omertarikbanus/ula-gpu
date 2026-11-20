# QP solver: fixed-storage float Goldfarb-Idnani

`src/qp/gi_qp.hh` is a header-only port of the dual active-set algorithm in
`third-party/Goldfarb_Optimizer/QuadProg++.cc` (Goldfarb and Idnani 1983), made
for a batched GPU kernel (one robot per thread). It is validated against
QuadProg++ on the 384 QPs the real WBIC produces in the parity sweep.

## The QP in WBIC

Convention (same as QuadProg++): minimize `1/2 x'Gx + g0'x` subject to
`CE'x + ce0 = 0` and `CI'x + ci0 >= 0`.

`x = [6 floating-base terms, 3 force terms per contact]`. Each contact is a
`SingleContact` with 6 friction-cone / unilateral rows (`Uf` is 6 x 3). So
`n = 6 + 3c`, `p = 6`, `m = 6c`, with `c` = number of contacts (1 to 4), `n <= 18`,
`m <= 24`. With no contact WBIC builds a degenerate `m = 1` problem with a zero
column and `ci0 = 0` (never violated). `G` is diagonal (weights 10 for the base,
1 for the forces in the training config). WBIC<float> builds the data from
float quantities and hands QuadProg++ doubles.

Recorded sweep (`tools/qp_dump.cc`, 384 ticks, no failure, 384 QPs): n=18/m=24
(four contacts) 192 QPs, n=15/m=18 (three contacts) 132, n=12/m=12 (two contacts)
60; p = 6 always. The sweep never has fewer than two contacts and never the
degenerate m = 1 case. File layout is documented at the top of `qp_dump.cc`.
The mean number of dual steps is only 0.14 per QP (max 4): in most ticks the
unconstrained solution of the equality-constrained problem already satisfies the
friction cones, so the solve time is dominated by the setup (Cholesky, `J = L^-T`,
six equality-constraint additions), not by the active-set loop.

## Design and conventions

* `ula::qp::solve<S, N, P, M>(Workspace&, n, p, m, G, g0, CE, ce0, CI, ci0, x,
  max_steps = 400, psi_eps = 0)`.
* Inputs are flat **row-major** arrays in QuadProg++'s layout: `G[i*n+j]`,
  `CE[i*p+k]` (n x p, one constraint per column), `CI[i*m+k]` (n x m), `g0[n]`,
  `ce0[p]`, `ci0[m]`. `G` is not modified.
* All storage lives in `Workspace<S,N,P,M>` (about 5 KB for float at 18/6/24:
  `L`, `R`, `J` n x n plus vectors). No heap, std containers, exceptions, I/O,
  virtuals. Math is only `qsqrt`/`qfabs` (shim) and `<float.h>` constants.
  Every function is `ULA_HD` (`__host__ __device__` under `__CUDACC__`, else
  empty).
* The control flow is a line-by-line mirror, `goto` labels `l1`, `l2`, `l2a`
  included, each phase commented with its algorithm step: preprocessing
  (Cholesky, `J`, unconstrained minimum), equality constraints, step 1 (most
  violated constraint), step 2 / 2a / 2b / 2c (direction, partial and full step
  length, dual-space step, full step with Givens `add_constraint`, partial step
  with `delete_constraint`), including the degenerate-add recovery path.
* `Result{f, status, iter, steps, iq}`. `iter` counts passes through step 1 (what
  QuadProg++ calls `iter`); `steps` counts passes through step 2a and is what
  `max_steps` caps. After the call `ws.A[p..iq)` are the active inequality
  indices (`ws.A[0..p)` are `-(i+1)` for the equalities).

Status codes: `kOk`, `kInfeasible` (no step in primal or dual space; QuadProg++
returns +inf), `kNotPositiveDefinite` (Cholesky of `G`; QuadProg++ throws),
`kIterationLimit`, plus two extras: `kLinearlyDependent` (equalities dependent;
QuadProg++ throws) and `kBadSize` (n, p or m above the capacities).

Capacities: `N = 18, P = 6, M = 24` (the WBIC maximum). `K = max(M+P, N+1)`
entries for the per-constraint vectors (`u[iq]` is indexed up to `n`).

Deliberate differences from QuadProg++: status codes instead of exceptions;
"infinity" is the largest finite number of the scalar type; the iteration cap;
and the float step-1 tolerance below.

### Tolerance finding (important for float)

QuadProg++ stops when `|psi| <= m * eps * c1 * c2 * 100`, where `psi` is the sum of
the inequality violations, `c1 = trace(G)`, `c2 = trace(L^-1)` and `eps` is
machine epsilon. With `eps = FLT_EPSILON` (the literal port) that threshold is
about 0.1 on this QP, so float stops with real violations of 0.1 N (see table
b2). `Lim<float>::psi_eps()` is therefore 1e-9 (a bound that is equivalent in
spirit to the double rule: relative to the data scale, well above float noise
in the residuals, which are ~1e-6). Results are identical for any value from
1e-8 down to 1e-16 on the sweep, and the solver neither cycles nor hits the
iteration cap there. For double it stays `DBL_EPSILON` (exact QuadProg++).
All other tolerances (`eps` in the `z != 0`, Givens `h == 0`, full-step
`|t - t2| < eps`, degenerate-constraint tests) are `eps` of the scalar type, as
in the original, and were not a problem.

## Results (`./build/qp_check /tmp/wbic.qp`, this laptop, GCC 13, -O3)

(a) double port vs QuadProg++, 384 QPs: **max |dx| = 0, max |df| = 0** (bit
identical), no non-ok status. Iterations: dual steps max 4 mean 0.14, step-1
passes max 5 mean 1.14.

(b) float vs QuadProg++:

| | default `psi_eps` 1e-9 | literal rule `FLT_EPSILON` |
|---|---|---|
| max abs `|dx|` | 2.4e-05 | 9.2e-02 |
| max `|dx|/|x|inf` (mean) | 1.3e-06 (3.6e-07) | 6.6e-03 (2.7e-05) |
| max relative `|df|` | 5.7e-07 | 1.2e-05 |
| max equality violation (recorded data, double check) | 4.5e-05 | 4.5e-05 |
| max inequality violation | 1.5e-06 | 1.1e-01 |
| active-set mismatches vs double port | 0 / 384 | 2 / 384 |
| dual steps max / mean | 4 / 0.14 | 4 / 0.13 |
| step-1 passes max / mean | 5 / 1.14 | 5 / 1.13 |
| non-ok status | 0 | 0 |

(QuadProg++ itself: equality violation 8.6e-14, inequality 8.9e-15.) The
equality residual 4.5e-5 is float rounding on dynamics terms of order 1e2.

(c) CPU time per solve (200 x 384 QPs, single thread; QuadProg++ includes
copying the inputs into its matrices, because it factorises `G` in place):

| solver | us / solve | speed-up |
|---|---|---|
| QuadProg++ (double) | 8.9 | 1 |
| port, double | 6.9 | 1.29x |
| port, float | 6.6 | 1.34x |

Float is barely faster on the CPU (setup is scalar, no SIMD); the value of the
port is fixed storage and device-readiness, not CPU speed.

## In the controller: `ULA_QP_SOLVER`

```bash
cmake -S . -B build-gi -DULA_QP_SOLVER=gi        # default: quadprog (unchanged)
cmake --build build-gi -j6 --target parity_dump
./build-gi/parity_dump models/go2/go2.urdf /tmp/parity_gi.bin
./build/parity_compare /tmp/parity_default.bin /tmp/parity_gi.bin 1e-3
```

`WBIC<T>::_SolveQP()` (WBIC.cpp) either calls `solve_quadprog` (default) or
converts the matrices to `T` and calls `ula::qp::solve` in the controller's scalar
type (float for the float controller, so the QP is solved in single precision;
default build solves it in double). The recorder hook `wbicQpRecorder()`
(`QpRecorder.hpp`) is null by default, and the default build still prints
fnv1a64 `58a2b3ac27ab48cd`.

Default vs `gi` build, 384 records, tol 1e-3 (`parity_compare`; relative errors
are floored at |v| = 1e-3):

| group | max abs | mean abs | max rel | records > 1e-3 |
|---|---|---|---|---|
| qDes | 0 | 0 | 0 | 0 |
| qdDes | 0 | 0 | 0 | 0 |
| tauFF | 6.2e-06 | 3.8e-07 | 1.1e-03 | 0 |
| Fr | 2.4e-05 | 1.2e-06 | 2.2e-03 | 0 |
| qddot (base 6) | 9.2e-06 | 3.5e-07 | 2.9e-04 | 0 |
| qddot (joints 12) | 0 | 0 | 0 | 0 |

(With tol 1e-5 one record of `Fr` exceeds it.) The `gi` build's hash is
`4ddda3e1e08939da`, so it is not bit-identical, as expected; the outputs agree
to float rounding of the QP. Joint-space outputs that do not depend on the
QP solution (qDes, qdDes, joint qddot) are unchanged.

## GPU check on Colab

`tools/qp_cuda_check.cu` solves the recorded QPs (replicated, default 256 copies)
with one thread per QP in float, compares with the host float solve of the same
header and reports max |dx|, status/iteration/active-set mismatches and
throughput. It could not be compiled here (no nvcc); the host logic was
exercised with a mock CUDA runtime, and the header itself is compiled and tested
on the host by `qp_check`.

It runs as the last cells of the Colab benchmark notebook (`bench/colab_bench.ipynb`):

1. On this machine: `cmake -S . -B build && cmake --build build -j6 --target qp_dump`,
   then `bench/make_colab_tarball.sh` (writes `/tmp/opencode/ula-gpu-bench.tar.gz`
   including a fresh `data/wbic.qp`).
2. Colab: pick a GPU runtime, open `bench/colab_bench.ipynb`, upload the
   package and run all cells; the QP check only needs Colab's `nvcc`
   (`nvcc -O2 -std=c++17 -arch=native -I src -I tools tools/qp_cuda_check.cu`).
3. Expect `RESULT: PASS`: device-vs-host differences around 1e-5 relative (FMA
   contraction; the last cell rebuilds with `--fmad=false`), no non-ok status.
   Small status/iteration mismatches are possible only for marginal QPs; report
   them.

## Concerns about float for this QP

* The step-1 tolerance must not be the literal `FLT_EPSILON` rule (above). The
  chosen 1e-9 works on all 384 recorded QPs but was tuned on this sweep only.
* Equality constraints are satisfied only to ~5e-5 (relative 1e-7 of the
  dynamics terms); WBIC then recomputes torques from `z` so this appears as
  tiny torque noise (max 6e-6 Nm in the sweep).
* The sweep is benign: 0.14 dual steps per QP, never infeasible, G is
  well-conditioned diagonal. Edge cases the sweep does not exercise (very small
  normal force, wide swing transitions, large disturbances where many cones
  activate, near-dependent constraints) are where float cycling or degenerate
  adds would show; the iteration cap and status code make that visible rather
  than hanging. A broader state sweep (including saturated forces) should be
  recorded before relying on it.
* `qp_check` compares against QuadProg++ on QPs built from float data; QuadProg++
  was run in double on them. The real controller's float-vs-double effect (QP
  solved in float) is what the parity table above measures.
