#include "actuation/sim_motor.hh"

#include <cmath>
#include <cstdio>
#include <limits>

int main() {
  using namespace ula::actuation;
  LegMeasurement m;
  LegTarget t;
  m.joint_velocity[0] = 1.0f;
  t.joint_position[0] = 2.0f;
  t.joint_kp[0][0] = 2.0f;
  t.joint_kd[0][0] = 1.0f;
  t.foot_position[1] = 1.0f;
  t.foot_kp[1][1] = 3.0f;
  t.force_ff[1] = 1.0f;
  m.foot_jacobian[1][0] = 2.0f;
  m.foot_jacobian[1][1] = -2.0f;
  t.joint_kp[2][0] = 3.0f;  // off-diagonal term contributes before the clamp

  const float limits[3] = {7.0f, 6.0f, 0.0f};
  const auto out = prepare_leg_command(m, t, limits);
  const auto check = [](const char* label, double actual, double expected) {
    if (std::abs(actual - expected) < 1e-9) return true;
    std::fprintf(stderr, "%s: got %.10g, expected %.10g\n", label, actual, expected);
    return false;
  };
  bool ok = true;
  ok &= check("positive limit", out.limited_torque[0], 7.0);
  ok &= check("negative limit", out.limited_torque[1], -6.0);
  ok &= check("off-diagonal joint gain", out.limited_torque[2], 6.0);
  ok &= check("compensated feedforward", out.command[0].feedforward, 4.0);
  ok &= check("callback at measured position", sim_motor_torque(out.command[0], 0.0, 1.0, true), 7.0);
  ok &= check("callback uses new actuator sample", sim_motor_torque(out.command[0], 0.5, 2.0, true), 5.0);
  ok &= check("callback default damping", sim_motor_torque(out.command[1], 0.0, 1.0, true), -11.0);
  ok &= check("disabled actuator", sim_motor_torque(out.command[0], 0.0, 1.0, false), 0.0);
  ok &= check("non-finite torque", sim_motor_torque(out.command[0],
              std::numeric_limits<double>::quiet_NaN(), 1.0, true), 0.0);
  return ok ? 0 : 1;
}
