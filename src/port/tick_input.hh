#ifndef ULA_GPU_TICK_INPUT_HH
#define ULA_GPU_TICK_INPUT_HH
#include "port/controller.hh"

namespace ula::port {
struct CpuInput {
  ControllerInput planner{};
  float orientation_wxyz[4]{1, 0, 0, 0};
  float linear_velocity_b[3]{}, angular_velocity_b[3]{};
  float joint_position[12]{}, joint_velocity[12]{};  // cached sample for WBIC
  float motor_joint_position[12]{}, motor_joint_velocity[12]{};  // fresh leg sample
  double actuator_position[12]{}, actuator_velocity[12]{};
  bool actuator_enabled[12]{true, true, true, true, true, true,
                            true, true, true, true, true, true};
};
}
#endif
