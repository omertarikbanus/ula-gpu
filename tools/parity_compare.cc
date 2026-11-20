// parity_compare: compare two parity files (parity_dump layout) field by field.
//
// usage: parity_compare <a.bin> <b.bin> [tol=1e-3]
//
// For every output group (qDes, qdDes, tauFF, Fr, qddot) prints the maximum
// absolute difference, the maximum relative difference |a-b|/max(|a|,|b|,1e-3)
// (floored so near-zero values do not dominate) and the number of records in
// which any value of the group differs by more than tol (absolute, or relative
// when |value| > 1: |a-b| > tol * max(1, |a|, |b|)). Inputs are checked to be
// identical.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool load(const char* path, uint32_t hdr[8], std::vector<float>& d) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  bool ok = std::fread(hdr, 4, 8, f) == 8 && hdr[0] == 0x59545250u;
  if (ok) {
    d.resize((size_t)hdr[2] * (hdr[3] + hdr[4]));
    ok = std::fread(d.data(), 4, d.size(), f) == d.size();
  }
  std::fclose(f);
  return ok;
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s a.bin b.bin [tol=1e-3]\n", argv[0]); return 2; }
  const double tol = argc > 3 ? std::atof(argv[3]) : 1e-3;
  uint32_t ha[8], hb[8];
  std::vector<float> a, b;
  if (!load(argv[1], ha, a) || !load(argv[2], hb, b)) { std::fprintf(stderr, "cannot read inputs\n"); return 1; }
  if (ha[2] != hb[2] || ha[3] != hb[3] || ha[4] != hb[4]) { std::fprintf(stderr, "shape mismatch\n"); return 1; }
  const uint32_t R = ha[2], NI = ha[3], NO = ha[4];
  const size_t stride = NI + NO;
  size_t inDiff = 0;
  for (uint32_t r = 0; r < R; ++r)
    for (uint32_t i = 0; i < NI; ++i)
      if (a[r * stride + i] != b[r * stride + i]) ++inDiff;
  std::printf("parity_compare: %u records, tol %.1e, input values that differ: %zu\n", R, tol, inDiff);

  struct Grp { const char* name; int off, len; } grp[] = {
      {"qDes", 0, 12}, {"qdDes", 12, 12}, {"tauFF", 24, 12}, {"Fr", 36, 12}, {"qddot base", 48, 6},
      {"qddot joints", 54, 12}};
  std::printf("%-13s %12s %12s %12s %10s   %s\n", "group", "max abs", "mean abs", "max rel", "records>tol", "worst record");
  for (auto& g : grp) {
    double mabs = 0, sabs = 0, mrel = 0;
    uint32_t over = 0, worst = 0;
    size_t cnt = 0;
    for (uint32_t r = 0; r < R; ++r) {
      bool o = false;
      for (int i = 0; i < g.len; ++i) {
        const double x = a[r * stride + NI + g.off + i], y = b[r * stride + NI + g.off + i];
        const double d = std::fabs(x - y);
        const double sc = std::max(std::max(std::fabs(x), std::fabs(y)), 1e-3);
        if (d > mabs) { mabs = d; worst = r; }
        sabs += d; ++cnt;
        mrel = std::max(mrel, d / sc);
        if (d > tol * std::max(1.0, std::max(std::fabs(x), std::fabs(y)))) o = true;
      }
      over += o;
    }
    std::printf("%-13s %12.3e %12.3e %12.3e %10u   %u\n", g.name, mabs, sabs / cnt, mrel, over, worst);
  }
  return 0;
}
