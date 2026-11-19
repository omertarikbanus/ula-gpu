// Standalone check of the controller core: build the Go2 floating-base model
// from the URDF and run one WBC tick from a standing pose.
//
// The full input/output sweep a GPU build is checked against is
// tools/parity_dump.cc; this demo is the readable single-tick version.
#include <cstdio>
#include <cmath>

#include "dynamics/Quadruped.h"
#include "dynamics/ParseURDFtoQuadruped.h"
#include "core/WbcHarness.hh"

using T = float;

int main(int argc, char** argv) {
  const std::string urdf = argc > 1 ? argv[1] : "models/go2/go2.urdf";

  // --- model ---------------------------------------------------------------
  bool urdf_ok = false;
  Quadruped<T> quad(ParseURDFtoQuadruped<T>(urdf, RobotType::MINI_CHEETAH, &urdf_ok));
  if (!urdf_ok) {
    std::printf("urdf parse failed: %s\n", urdf.c_str());
    return 1;
  }
  FloatingBaseModel<T> model;
  quad.buildModel(model);
  std::printf("model: nDof=%zu nq=%zu mass=%.3f hip_links=%.3f/%.3f/%.3f\n",
              static_cast<size_t>(model._nDof), static_cast<size_t>(cheetah::num_q),
              static_cast<double>(quad._bodyMass),
              static_cast<double>(quad._abadLinkLength),
              static_cast<double>(quad._hipLinkLength),
              static_cast<double>(quad._kneeLinkLength));

  // --- one WBC tick: standing crouch at rest -------------------------------
  {
    wbch::Harness h;
    if (!h.init(urdf)) { std::printf("urdf parse failed (harness)\n"); return 1; }
    wbch::T leg[3];
    h.stanceJoints(0.32f, leg);
    wbch::TickInput in{};
    wbch::TickOutput out{};
    for (int l = 0; l < 4; ++l)
      for (int j = 0; j < 3; ++j) in.q[3 * l + j] = leg[j];
    in.pos[2] = 0.32f;
    in.quat[0] = 1.f;
    for (int l = 0; l < 4; ++l) {
      in.contact[l] = 1.f;
      in.Fr_des[l][2] = h.total_mass * 9.81f / 4.f;  // as SupervisorStates BalanceStand
    }
    in.pBody_des[2] = 0.32f;
    h.tick(in, out);

    std::printf("wbc: stance leg q = (%.4f, %.4f, %.4f), total mass %.3f kg, Fz_des/leg %.3f N\n",
                leg[0], leg[1], leg[2], h.total_mass, in.Fr_des[0][2]);
    const char* nm[4] = {"FR", "FL", "HR", "HL"};
    bool finite = true;
    double checksum = 0;
    for (int l = 0; l < 4; ++l) {
      std::printf("wbc %s: qDes %8.4f %8.4f %8.4f | qdDes %8.4f %8.4f %8.4f | tauFF %8.4f %8.4f %8.4f | Fr %8.3f %8.3f %8.3f\n",
                  nm[l], out.qDes[3*l], out.qDes[3*l+1], out.qDes[3*l+2],
                  out.qdDes[3*l], out.qdDes[3*l+1], out.qdDes[3*l+2],
                  out.tauFF[3*l], out.tauFF[3*l+1], out.tauFF[3*l+2],
                  out.Fr[3*l], out.Fr[3*l+1], out.Fr[3*l+2]);
    }
    const wbch::T* arrs[5] = {out.qDes, out.qdDes, out.tauFF, out.Fr, out.qddot};
    const int lens[5] = {12, 12, 12, 12, 18};
    for (int a = 0; a < 5; ++a)
      for (int i = 0; i < lens[a]; ++i) {
        if (!std::isfinite(arrs[a][i])) finite = false;
        checksum += static_cast<double>(arrs[a][i]) * (1 + 0.001 * (a * 18 + i));
      }
    double fz = 0;
    for (int l = 0; l < 4; ++l) fz += out.Fr[3 * l + 2];
    std::printf("wbc: sum Fz = %.3f N (weight %.3f N), all finite = %s, checksum = %.6f\n",
                fz, h.total_mass * 9.81, finite ? "yes" : "NO", checksum);
  }

  FBModelState<T> state;
  state.bodyOrientation = Quat<T>(1, 0, 0, 0);
  state.bodyPosition = Vec3<T>(0, 0, 0.32);
  state.bodyVelocity = SVec<T>::Zero();
  state.q = DVec<T>::Zero(12);
  state.qd = DVec<T>::Zero(12);
  // FBModelState::q holds the 12 leg joints only (the base is separate).
  const int q0 = 0;
  const T stand[3] = {static_cast<T>(0), static_cast<T>(-0.8), static_cast<T>(1.6)};
  for (int leg = 0; leg < 4; ++leg)
    for (int j = 0; j < 3; ++j) state.q[q0 + 3 * leg + j] = stand[j];
  model.setState(state);

  model.forwardKinematics();
  model.massMatrix();  // getMassMatrix() only returns the cache massMatrix() fills
  double hmax = 0.0;
  for (int i = 0; i < model.getMassMatrix().rows(); ++i)
    hmax = std::max(hmax, std::fabs(static_cast<double>(model.getMassMatrix()(i, i))));
  std::printf("model: max |H_ii| after massMatrix() = %.6f\n", hmax);

  return 0;
}
