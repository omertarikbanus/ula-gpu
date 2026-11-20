// sweep.hh: the deterministic state/command sweep shared by parity_dump and
// qp_dump (128 states x 3 commands, fixed seed, one TickInput per record).
// See parity_dump.cc for the meaning of the inputs.
#ifndef ULA_GPU_TOOLS_SWEEP_HH
#define ULA_GPU_TOOLS_SWEEP_HH
#include <tinyxml2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <cstdio>
#include <string>

#include "core/WbcHarness.hh"

namespace sweep {
using T = float;

static bool readLimits(const std::string& urdf, float lo[12], float hi[12]) {
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(urdf.c_str()) != tinyxml2::XML_SUCCESS) return false;
  auto* robot = doc.FirstChildElement("robot");
  if (!robot) return false;
  const std::string rn = robot->Attribute("name");
  const char* jn[3] = {"abad", "hip", "knee"};
  int found = 0;
  for (auto* j = robot->FirstChildElement("joint"); j; j = j->NextSiblingElement("joint")) {
    for (int leg = 0; leg < 4; ++leg)
      for (int k = 0; k < 3; ++k) {
        const std::string name = rn + "/" + jn[k] + "_" + std::to_string(leg + 1) + "_joint";
        if (name == j->Attribute("name")) {
          auto* lim = j->FirstChildElement("limit");
          if (!lim) return false;
          lo[3 * leg + k] = lim->FloatAttribute("lower");
          hi[3 * leg + k] = lim->FloatAttribute("upper");
          ++found;
        }
      }
  }
  return found == 12;
}


// Calls fn(const wbch::TickInput&, uint32_t record) for every record in order
// (record = state * 3 + cmd). Returns false if the URDF cannot be read.
template <class Fn>
bool forEachTick(const std::string& urdf, uint32_t n_states, uint32_t seed, Fn fn) {
  const uint32_t n_cmds = 3;
  float lo[12], hi[12];
  if (!readLimits(urdf, lo, hi)) { std::fprintf(stderr, "cannot read joint limits\n"); return false; }

  wbch::Harness probe;
  if (!probe.init(urdf)) { std::fprintf(stderr, "urdf parse failed\n"); return false; }
  wbch::T stance[3];
  probe.stanceJoints(0.32f, stance);
  const T weight = probe.total_mass * 9.81f;
  Vec3<T> hipLoc[4];
  for (int l = 0; l < 4; ++l) hipLoc[l] = probe.quad.getHipLocation(l);

  std::mt19937 rng(seed);
  auto U = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };

  const float dt = 0.002f;

  for (uint32_t s = 0; s < n_states; ++s) {
    // ---- feasible state ----------------------------------------------------
    wbch::TickInput base{};
    for (int l = 0; l < 4; ++l)
      for (int j = 0; j < 3; ++j) {
        const float spread = j == 0 ? 0.15f : 0.3f;
        const float q = stance[j] + U(-spread, spread);
        base.q[3 * l + j] = std::min(std::max(q, lo[3 * l + j] + 0.02f), hi[3 * l + j] - 0.02f);
        base.qd[3 * l + j] = U(-1.5f, 1.5f);
      }
    base.pos[0] = U(-1, 1); base.pos[1] = U(-1, 1); base.pos[2] = U(0.26f, 0.36f);
    const Vec3<T> rpy(U(-0.15f, 0.15f), U(-0.15f, 0.15f), U(-3.0f, 3.0f));
    const Vec4<T> qt = ori::rpyToQuat(rpy);
    for (int i = 0; i < 4; ++i) base.quat[i] = qt[i];
    for (int i = 0; i < 3; ++i) { base.vBody[i] = U(-0.4f, 0.4f); base.omega[i] = U(-0.4f, 0.4f); }
    // contact pattern: all stance (50%), else one swing leg or a diagonal pair
    const int pat = (s % 2 == 0) ? 0 : 1 + (s / 2) % 6;
    for (int l = 0; l < 4; ++l) base.contact[l] = 1.f;
    if (pat >= 1 && pat <= 4) base.contact[pat - 1] = 0.f;
    if (pat == 5) { base.contact[0] = 0.f; base.contact[3] = 0.f; }
    if (pat == 6) { base.contact[1] = 0.f; base.contact[2] = 0.f; }
    int nc = 0;
    for (int l = 0; l < 4; ++l) nc += base.contact[l] > 0.f;

    const Mat3<T> R = ori::quaternionToRotationMatrix(qt);
    const Vec3<T> p(base.pos[0], base.pos[1], base.pos[2]);

    for (uint32_t c = 0; c < n_cmds; ++c) {
      wbch::TickInput in = base;
      const float vfwd = c == 1 ? 0.5f : 0.f, yaw = c == 2 ? 0.2f : 0.f;
      const Vec3<T> vdes_world = R.transpose() * Vec3<T>(vfwd, 0, 0);
      in.pBody_des[0] = p[0]; in.pBody_des[1] = p[1]; in.pBody_des[2] = 0.32f;
      for (int i = 0; i < 3; ++i) in.vBody_des[i] = vdes_world[i];
      in.vBody_des[2] = 0.f;
      in.rpy_des[2] = rpy[2] + yaw * dt;
      in.omega_des[2] = yaw;
      for (int l = 0; l < 4; ++l) {
        if (in.contact[l] > 0.f) {
          in.Fr_des[l][2] = weight / nc;
        } else {
          const Vec3<T> pf = p + R.transpose() * (hipLoc[l] + Vec3<T>(0, 0, -0.2f));
          for (int i = 0; i < 3; ++i) in.pFoot_des[l][i] = pf[i];
        }
      }

      fn(in, s * n_cmds + c);
    }
  }
  return true;
}
}  // namespace sweep
#endif
