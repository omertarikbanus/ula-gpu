// qp_cuda_check: solve all recorded WBIC QPs on the GPU, one thread per QP,
// with the float solver of src/qp/gi_qp.hh, and compare with the same solver
// run on the host (same header, same float inputs).
//
// build (Colab / any machine with nvcc):
//   nvcc -O2 -std=c++17 -I src -I tools tools/qp_cuda_check.cu -o qp_cuda_check
// run:
//   ./qp_cuda_check wbic.qp [replicate=256]
//
// `replicate` tiles the 384 QPs that many times for the throughput figure
// (the comparison itself uses the first copy of each QP and checks all copies
// give the same result as the first).
//
// Differences between host and device are expected at the 1e-5 level (the
// device contracts a*b+c into FMA, the host build usually does not; compile
// with `--fmad=false` to remove that source). Status codes and iteration
// counts should match except possibly in marginal cases; they are reported.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "qp/gi_qp.hh"
#include "qp_file.hh"

#define CK(call)                                                                      \
  do {                                                                                \
    cudaError_t e_ = (call);                                                          \
    if (e_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
                   __LINE__);                                                         \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

enum { CN = 18, CP = 6, CM = 24 };
using Ws = ula::qp::Workspace<float, CN, CP, CM>;

// One QP in a fixed-size block, inputs packed with the QP's own n/p/m strides.
struct QpBlock {
  int n, p, m;
  float G[CN * CN], g0[CN], CE[CN * CP], ce0[CP], CI[CN * CM], ci0[CM];
};
struct QpOut {
  float x[CN], f;
  int status, iter, steps, iq;
  int active[CN];  // A[p..iq), sorted
};

ULA_HD inline void solveOne(const QpBlock& b, QpOut& o) {
  Ws ws;
  ula::qp::Result<float> r = ula::qp::solve(ws, b.n, b.p, b.m, b.G, b.g0, b.CE, b.ce0, b.CI, b.ci0, o.x);
  o.f = r.f; o.status = r.status; o.iter = r.iter; o.steps = r.steps; o.iq = r.iq;
  int k = 0;
  for (int i = b.p; i < r.iq && k < CN; ++i) o.active[k++] = ws.A[i];
  for (int i = 1; i < k; ++i) {  // insertion sort
    int v = o.active[i], j = i - 1;
    while (j >= 0 && o.active[j] > v) { o.active[j + 1] = o.active[j]; --j; }
    o.active[j + 1] = v;
  }
  for (; k < CN; ++k) o.active[k] = -1;
}

__global__ void solveKernel(const QpBlock* in, QpOut* out, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) solveOne(in[i], out[i]);
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s wbic.qp [replicate=256]\n", argv[0]); return 2; }
  const int rep = argc > 2 ? std::atoi(argv[2]) : 256;
  std::vector<RecordedQp> qps;
  if (!loadQps(argv[1], qps)) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  const int N = (int)qps.size();
  std::vector<QpBlock> blocks(N);
  for (int k = 0; k < N; ++k) {
    const RecordedQp& q = qps[k];
    QpBlock& b = blocks[k];
    b = QpBlock();
    b.n = q.n; b.p = q.p; b.m = q.m;
    for (size_t i = 0; i < q.G.size(); ++i) b.G[i] = (float)q.G[i];
    for (size_t i = 0; i < q.g0.size(); ++i) b.g0[i] = (float)q.g0[i];
    for (size_t i = 0; i < q.CE.size(); ++i) b.CE[i] = (float)q.CE[i];
    for (size_t i = 0; i < q.ce0.size(); ++i) b.ce0[i] = (float)q.ce0[i];
    for (size_t i = 0; i < q.CI.size(); ++i) b.CI[i] = (float)q.CI[i];
    for (size_t i = 0; i < q.ci0.size(); ++i) b.ci0[i] = (float)q.ci0[i];
  }
  // host reference (float, same header)
  std::vector<QpOut> host(N);
  for (int k = 0; k < N; ++k) solveOne(blocks[k], host[k]);

  // device: first the plain set, then the tiled set for throughput
  const int total = N * rep;
  std::vector<QpBlock> tiled(total);
  for (int i = 0; i < total; ++i) tiled[i] = blocks[i % N];
  QpBlock* dIn; QpOut* dOut;
  CK(cudaMalloc(&dIn, sizeof(QpBlock) * total));
  CK(cudaMalloc(&dOut, sizeof(QpOut) * total));
  CK(cudaMemcpy(dIn, tiled.data(), sizeof(QpBlock) * total, cudaMemcpyHostToDevice));
  const int bs = 64;
  solveKernel<<<(total + bs - 1) / bs, bs>>>(dIn, dOut, total);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<QpOut> dev(total);
  CK(cudaMemcpy(dev.data(), dOut, sizeof(QpOut) * total, cudaMemcpyDeviceToHost));

  double maxdx = 0, maxrel = 0, maxdf = 0;
  int statusDiff = 0, iterDiff = 0, actDiff = 0, devNotOk = 0, copyDiff = 0;
  for (int i = 0; i < total; ++i) {
    const QpOut& d = dev[i];
    const QpOut& h = host[i % N];
    if (d.status != 0) ++devNotOk;
    if (i >= N) {  // copies must equal the first device copy bit for bit
      if (std::memcmp(&d, &dev[i % N], sizeof d) != 0) ++copyDiff;
      continue;
    }
    if (d.status != h.status) ++statusDiff;
    if (d.iter != h.iter) ++iterDiff;
    if (std::memcmp(d.active, h.active, sizeof d.active) != 0) ++actDiff;
    double dx = 0, xn = 1e-12;
    for (int j = 0; j < blocks[i].n; ++j) {
      dx = std::max(dx, (double)std::fabs(d.x[j] - h.x[j]));
      xn = std::max(xn, (double)std::fabs(h.x[j]));
    }
    maxdx = std::max(maxdx, dx);
    maxrel = std::max(maxrel, dx / xn);
    maxdf = std::max(maxdf, (double)std::fabs(d.f - h.f) / std::max((double)std::fabs(h.f), 1e-12));
  }
  std::printf("qp_cuda_check: %d QPs on the device (%d recorded x %d copies)\n", total, N, rep);
  std::printf("  device vs host float: max |dx| = %.3e  max |dx|/|x|inf = %.3e  max rel |df| = %.3e\n",
              maxdx, maxrel, maxdf);
  std::printf("  status mismatches %d, iteration mismatches %d, active-set mismatches %d (of %d)\n",
              statusDiff, iterDiff, actDiff, N);
  std::printf("  non-ok status on device: %d, copies differing from the first copy: %d\n", devNotOk,
              copyDiff);

  // throughput
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const int iters = 10;
  CK(cudaEventRecord(e0));
  for (int it = 0; it < iters; ++it) solveKernel<<<(total + bs - 1) / bs, bs>>>(dIn, dOut, total);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  std::printf("  kernel: %.3f ms per launch of %d QPs = %.2f M solves/s\n", ms / iters, total,
              total / (ms / iters * 1e-3) / 1e6);

  const bool pass = maxrel < 1e-3 && statusDiff == 0 && devNotOk == 0 && copyDiff == 0;
  std::printf("RESULT: %s\n", pass ? "PASS" : "CHECK (see numbers above)");
  CK(cudaFree(dIn)); CK(cudaFree(dOut));
  return pass ? 0 : 1;
}
