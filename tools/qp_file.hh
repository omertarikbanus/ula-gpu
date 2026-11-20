// qp_file.hh: reader for the file written by qp_dump (layout in qp_dump.cc).
// Host-only, plain C++ (also used by the CUDA check).
#ifndef ULA_GPU_TOOLS_QP_FILE_HH
#define ULA_GPU_TOOLS_QP_FILE_HH
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct RecordedQp {
  uint32_t tick, n, p, m;
  std::vector<double> G, g0, CE, ce0, CI, ci0, x;
  double f;
};

inline bool loadQps(const std::string& path, std::vector<RecordedQp>& out) {
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) return false;
  uint32_t hdr[8];
  if (std::fread(hdr, sizeof hdr, 1, fp) != 1 || hdr[0] != 0x53445051u) { std::fclose(fp); return false; }
  out.resize(hdr[2]);
  bool ok = true;
  auto rd = [&](std::vector<double>& v, size_t len) {
    v.resize(len);
    if (len && std::fread(v.data(), sizeof(double), len, fp) != len) ok = false;
  };
  for (auto& q : out) {
    uint32_t head[4];
    if (std::fread(head, sizeof head, 1, fp) != 1) { ok = false; break; }
    q.tick = head[0]; q.n = head[1]; q.p = head[2]; q.m = head[3];
    rd(q.G, (size_t)q.n * q.n); rd(q.g0, q.n); rd(q.CE, (size_t)q.n * q.p); rd(q.ce0, q.p);
    rd(q.CI, (size_t)q.n * q.m); rd(q.ci0, q.m); rd(q.x, q.n);
    double f;
    if (std::fread(&f, sizeof f, 1, fp) != 1) ok = false;
    q.f = f;
    if (!ok) break;
  }
  std::fclose(fp);
  return ok;
}
#endif
