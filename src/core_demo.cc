// Standalone check of the controller core: build the Go2 floating-base model
// from the URDF and exercise forward/inverse kinematics.
//
// This is the parity harness for the standalone core: run it here and in ula
// with the same controller sources and compare the numbers. A WBC tick from a
// fixed stance command is the next step (see docs/CONTEXT.md).
#include <cstdio>
#include <cmath>
#include <random>

#include "dynamics/Quadruped.h"
#include "dynamics/ParseURDFtoQuadruped.h"
#include "dynamics/QuadrupedKinematics.hh"
#include "parameters/QuadrupedParams.hh"

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
  std::printf("model: nDof=%zu	nq=%d	mass=%.3f hip_links=%.3f/%.3f/%.3f\n",
              static_cast<size_t>(model._nDof), cheetah::num_q,
              static_cast<double>(quad._bodyMass),
              static_cast<double>(quad._abadLinkLength),
              static_cast<double>(quad._hipLinkLength),
              static_cast<double>(quad._kneeLinkLength));

  FBModelState<T> state;
  state.bodyOrientation = Quat<T>(1, 0, 0, 0);
  state.bodyPosition = Vec3<T>(0, 0, 0.32);
  state.bodyVelocity = SVec<T>::Zero();
  state.q = DVec<T>::Zero(cheetah::num_q);
  state.qd = DVec<T>::Zero(cheetah::num_q);
  // The URDF build has 12 leg joints after the 6 floating-base coordinates
  // (nDof = 18), so leg joints start at q[6].
  const int q0 = cheetah::num_q - 12;
  const T stand[3] = {static_cast<T>(0), static_cast<T>(-0.8), static_cast<T>(1.6)};
  for (int leg = 0; leg < 4; ++leg)
    for (int j = 0; j < 3; ++j) state.q[q0 + 3 * leg + j] = stand[j];
  model.setState(state);

  model.forwardKinematics();
  model.forwardKinematics();
  double hmax = 0.0;
  for (int i = 0; i < model.getMassMatrix().rows(); ++i)
    hmax = std::max(hmax, std::fabs(static_cast<double>(model.getMassMatrix()(i, i))));
  std::printf("model: max |H_ii| after one FK pass = %.6f\n", hmax);

  // --- forward / inverse kinematics round trip -----------------------------
  // forwardKinematics returns the foot position relative to the hip (the robot
  // runtime compares against hip-relative points); inverseKinematics expects a
  // body-frame target and subtracts the hip itself.
  const auto kin_params = quadruped::createGo2Config();
  QuadrupedKinematics kin(kin_params);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  double max_err = 0.0;
  for (int leg = 0; leg < QuadrupedKinematics::NUM_LEGS; ++leg) {
    for (int i = 0; i < 25; ++i) {
      Eigen::Vector3d q(dist(rng), dist(rng) - 0.5236, dist(rng) - 1.5);
      Eigen::Vector3d p_rel, q2, p2_rel;
      const Eigen::Vector3d hip = kin_params.hip_positions.row(leg).transpose();
      if (!kin.forwardKinematics(leg, q, p_rel)) continue;
      if (!kin.inverseKinematics(leg, p_rel + hip, q2)) continue;
      kin.forwardKinematics(leg, q2, p2_rel);
      max_err = std::max(max_err, (p_rel - p2_rel).norm());
    }
  }
  std::printf("kinematics: max round-trip error (hip frame) %.3e\n", max_err);
  return 0;
}
