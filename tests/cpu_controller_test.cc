#include <cmath>
#include <iostream>
#include <limits>

#include "port/cpu_controller.hh"

int main(int argc, char** argv) {
  using namespace ula::port;
  CpuController controller;
  CpuConfig cfg;
  if (argc != 2 || !controller.init(argv[1], cfg)) return 1;
  CpuInput in;
  in.planner.plant.position_w[2] = 0.32f;
  for (int l = 0; l < 4; ++l) {
    const float q[3]{0, -0.7752f, 1.5504f};
    compute_leg_foot_position(cfg.planner.footstep.kinematics, l, q, in.planner.plant.foot_position_leg[l]);
    for (int j = 0; j < 3; ++j) {
      const int i = 3 * l + j;
      in.joint_position[i] = in.motor_joint_position[i] = q[j];
      in.actuator_position[i] = q[j];
    }
  }
  in.planner.base_command[0] = 0.45f;
  in.planner.base_command[2] = 0.3f;
  in.planner.action[0] = std::numeric_limits<float>::quiet_NaN();
  in.planner.action[23] = 0.4f;
  in.planner.action_count = 0;
  const auto initial = controller.step(in);
  in.planner.action_count = 25;
  const auto full = controller.step(in);
  if (!initial.planner.gait.planned_contact[0] || initial.planner.gait.planned_contact[1] ||
      initial.planner.decoded.foot_forces[0][2] != 0 || full.planner.decoded.foot_forces[0][2] != 125 ||
      full.planner.decoded.desired_velocity[0] != in.planner.base_command[0] ||
      std::abs(full.wbic_command.pBody_des[2] - 0.34f) > 1e-6f ||
      std::abs(full.wbic_command.vBody_des[0] - 0.4f) > 1e-6f) {
    std::cerr << "unexpected reset/action/body command\n";
    return 1;
  }
  const auto height = full.wbic_command.pFoot_des[1][2];
  if (initial.wbic_ran || !full.wbic_ran || full.wbic_command.vFoot_des[1].norm() != 0 ||
      height == full.planner.footstep.foot_position_target[1][2]) {
    std::cerr << "liftoff did not align with the measured foot\n";
    return 1;
  }
  int reversals = 0;
  bool contact = true;
  for (int t = 0; t < 220; ++t) {
    const auto out = controller.step(in);
    if (contact != out.planner.gait.planned_contact[0]) {
      ++reversals;
      contact = !contact;
    }
    for (int j = 0; j < 12; ++j)
      if (!std::isfinite(out.feedforward_torque[j]) || !std::isfinite(out.motor_torque[j])) return 1;
  }
  if (reversals != 2) return 1;
  controller.reset();
  in.planner.action_count = 0;
  const auto reset = controller.step(in);
  if (reset.planner.gait.gait_phase != initial.planner.gait.gait_phase) return 1;
  for (int j = 0; j < 12; ++j)
    if (reset.motor_torque[j] != initial.motor_torque[j]) return 1;
  return 0;
}
