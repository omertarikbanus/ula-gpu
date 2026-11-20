// parity_dump: deterministic sweep of WBC ticks -> float32 binary file.
// Ground truth for checking a GPU / refactored build of the controller core.
//
// usage: parity_dump <urdf> <out.bin> [n_states=128] [seed=12345]
//
// File layout (little endian):
//   header, 8 x uint32: magic 'PRTY' (0x59545250), version 1, n_records,
//                       n_in (77), n_out (66), n_states, n_cmds (3), seed
//   n_records x { float32 in[77]; float32 out[66]; }, records ordered
//   state-major: record = state * 3 + cmd (cmd 0 stand, 1 forward 0.5 m/s,
//   2 yaw 0.2 rad/s).
// Leg order everywhere: FR, FL, HR, HL; joints (abad, hip, knee); index 3*leg+j.
//
// in[77]:
//    0..11  q               joint angles [rad]
//   12..23  qd              joint velocities [rad/s]
//   24..26  pos             body position, world [m]
//   27..30  quat            body orientation (w,x,y,z), world->body convention
//                           of ori::quaternionToRotationMatrix
//   31..33  vBody           linear velocity, body frame [m/s]
//   34..36  omega           angular velocity, body frame [rad/s]
//   37..40  contact         1 = stance, 0 = swing
//   41..43  pBody_des       world [m]
//   44..46  vBody_des       world [m/s]
//   47..49  rpy_des         [rad]
//   50..52  omega_des       [rad/s] (WBC vBody_Ori_des)
//   53..64  Fr_des          per leg (x,y,z) [N], 0 for swing legs
//   65..76  pFoot_des       per leg, world [m]; used for swing legs only
// out[66]:
//    0..11  qDes            desired joint positions (leg command, after the
//                           WBC knee non-flip clamp)
//   12..23  qdDes           desired joint velocities
//   24..35  tauFF           feed-forward joint torques [Nm]
//   36..47  Fr              WBIC contact reaction forces per leg (0 if swing)
//   48..65  qddot           WBIC solution: base (6) + joint (12) accelerations
// A fresh controller is built for every record, so records are independent.
// WBC settings: ula's MIT_UserParameters defaults, floating_base_weight = 10
// (training config).
#include <tinyxml2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "sweep.hh"

using T = float;
static const int N_IN = 77, N_OUT = 66;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <urdf> <out.bin> [n_states=128] [seed=12345]\n", argv[0]);
    return 2;
  }
  const std::string urdf = argv[1], outpath = argv[2];
  const uint32_t n_states = argc > 3 ? std::atoi(argv[3]) : 128;
  const uint32_t seed = argc > 4 ? std::atoi(argv[4]) : 12345;
  const uint32_t n_cmds = 3;

  std::vector<float> data;
  data.reserve((size_t)n_states * n_cmds * (N_IN + N_OUT));
  uint32_t nonfinite = 0, nrec = 0;

  auto onTick = [&](const wbch::TickInput& in, uint32_t) {
      wbch::Harness h;
      wbch::TickOutput out;
      std::memset(&out, 0, sizeof out);
      bool ok = h.init(urdf);
      bool bad = !ok;
      if (ok) {
        try { h.tick(in, out); } catch (...) { bad = true; }
      }
      float rec_in[N_IN], rec_out[N_OUT];
      int k = 0;
      auto put = [&](const float* v, int n) { for (int i = 0; i < n; ++i) rec_in[k++] = v[i]; };
      put(in.q, 12); put(in.qd, 12); put(in.pos, 3); put(in.quat, 4); put(in.vBody, 3);
      put(in.omega, 3); put(in.contact, 4); put(in.pBody_des, 3); put(in.vBody_des, 3);
      put(in.rpy_des, 3); put(in.omega_des, 3); put(&in.Fr_des[0][0], 12);
      put(&in.pFoot_des[0][0], 12);
      k = 0;
      auto puto = [&](const float* v, int n) { for (int i = 0; i < n; ++i) rec_out[k++] = v[i]; };
      puto(out.qDes, 12); puto(out.qdDes, 12); puto(out.tauFF, 12); puto(out.Fr, 12);
      puto(out.qddot, 18);
      for (int i = 0; i < N_OUT; ++i) if (!std::isfinite(rec_out[i])) bad = true;
      if (bad) ++nonfinite;
      data.insert(data.end(), rec_in, rec_in + N_IN);
      data.insert(data.end(), rec_out, rec_out + N_OUT);
      ++nrec;
  };
  if (!sweep::forEachTick(urdf, n_states, seed, onTick)) return 1;

  FILE* f = std::fopen(outpath.c_str(), "wb");
  if (!f) { std::perror("open out"); return 1; }
  const uint32_t hdr[8] = {0x59545250u, 1u, nrec, N_IN, N_OUT, n_states, n_cmds, seed};
  std::fwrite(hdr, sizeof hdr, 1, f);
  std::fwrite(data.data(), sizeof(float), data.size(), f);
  std::fclose(f);

  // FNV-1a over all float bytes + double sum of outputs
  uint64_t hsh = 1469598103934665603ull;
  const unsigned char* b = reinterpret_cast<const unsigned char*>(data.data());
  for (size_t i = 0; i < data.size() * sizeof(float); ++i) { hsh ^= b[i]; hsh *= 1099511628211ull; }
  double osum = 0;
  for (uint32_t r = 0; r < nrec; ++r)
    for (int i = 0; i < N_OUT; ++i) {
      const float v = data[(size_t)r * (N_IN + N_OUT) + N_IN + i];
      if (std::isfinite(v)) osum += v;
    }
  std::printf("parity_dump: %u records (%u states x %u cmds), %u non-finite, seed %u\n",
              nrec, n_states, n_cmds, nonfinite, seed);
  std::printf("parity_dump: file %s, %zu bytes, fnv1a64 %016llx, output sum %.6f\n",
              outpath.c_str(), sizeof hdr + data.size() * sizeof(float),
              (unsigned long long)hsh, osum);
  return nonfinite ? 1 : 0;
}
