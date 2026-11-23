#ifndef ULA_GPU_ACTUATION_SIM_MOTOR_HH
#define ULA_GPU_ACTUATION_SIM_MOTOR_HH

#include <cmath>

#ifdef __CUDACC__
#define ULA_MOTOR_HD __host__ __device__
#else
#define ULA_MOTOR_HD
#endif

namespace ula::actuation {

template <class T = float>
struct LegMeasurementT {
  T joint_position[3]{};
  T joint_velocity[3]{};
  T foot_position[3]{};
  T foot_velocity[3]{};
  T foot_jacobian[3][3]{};  // foot coordinate, then joint
};

template <class T = float>
struct LegTargetT {
  T joint_position[3]{};
  T joint_velocity[3]{};
  T foot_position[3]{};
  T foot_velocity[3]{};
  T torque_ff[3]{};
  T force_ff[3]{};
  T joint_kp[3][3]{};
  T joint_kd[3][3]{};
  T foot_kp[3][3]{};
  T foot_kd[3][3]{};
};

template <class T = float>
struct MotorCommandT {
  T position{};
  T velocity{};
  T feedforward{};
  T kp{};
  T kd{};
};

template <class T = float>
struct LegOutputT {
  MotorCommandT<T> command[3]{};
  T limited_torque[3]{};
};

using LegMeasurement = LegMeasurementT<float>;
using LegTarget = LegTargetT<float>;
using MotorCommand = MotorCommandT<float>;
using LegOutput = LegOutputT<float>;

template <class T>
ULA_MOTOR_HD inline T dot3(const T* a, const T* b) {
  return a[0] * b[0] + (a[1] * b[1] + a[2] * b[2]);
}

// Match ULA's MdlLegController::applyCommands. Its limit applies to the
// combined command; subtract the diagonal PD term before handing it off.
template <class T>
ULA_MOTOR_HD inline LegOutputT<T> prepare_leg_command(const LegMeasurementT<T>& measured,
                                                     const LegTargetT<T>& target,
                                                     const T (&max_torque)[3]) {
  T foot_error[3], foot_rate_error[3], joint_error[3], joint_rate_error[3];
  for (int i = 0; i < 3; ++i) {
    foot_error[i] = target.foot_position[i] - measured.foot_position[i];
    foot_rate_error[i] = target.foot_velocity[i] - measured.foot_velocity[i];
    joint_error[i] = target.joint_position[i] - measured.joint_position[i];
    joint_rate_error[i] = target.joint_velocity[i] - measured.joint_velocity[i];
  }

  T force[3];
  for (int i = 0; i < 3; ++i) {
    force[i] = target.force_ff[i] + dot3(target.foot_kp[i], foot_error);
    force[i] += dot3(target.foot_kd[i], foot_rate_error);
  }

  LegOutputT<T> out;
  for (int i = 0; i < 3; ++i) {
    const T jacobian_column[3] = {measured.foot_jacobian[0][i],
                                  measured.foot_jacobian[1][i],
                                  measured.foot_jacobian[2][i]};
    T torque = target.torque_ff[i] + dot3(jacobian_column, force);
    torque += dot3(target.joint_kp[i], joint_error);
    torque += dot3(target.joint_kd[i], joint_rate_error);
    const T limit = max_torque[i];
    if (limit > T(0)) {
      if (torque < -limit) torque = -limit;
      if (torque > limit) torque = limit;
    }
    out.limited_torque[i] = torque;
    auto& cmd = out.command[i];
    cmd.position = target.joint_position[i];
    cmd.velocity = target.joint_velocity[i];
    cmd.kp = target.joint_kp[i][i];
    cmd.kd = target.joint_kd[i][i];
    cmd.feedforward = torque - cmd.kp * joint_error[i] - cmd.kd * joint_rate_error[i];
  }
  return out;
}

// Match the CPU MuJoCo callback at the ctrl[] write, BEFORE the simulator's
// own actuator force clamp. The callback uses actuator length and velocity,
// not necessarily the earlier joint measurement used by prepare_leg_command.
template <class T>
ULA_MOTOR_HD inline double sim_motor_torque(const MotorCommandT<T>& cmd, double actuator_length,
                                            double actuator_velocity, bool enabled) {
  if (!enabled) return 0.0;
  const double kp = cmd.kp <= T(0) ? 0.0 : static_cast<double>(cmd.kp);
  const double kd = cmd.kd <= T(0) ? 5.0 : static_cast<double>(cmd.kd);
  const double torque = -kp * (actuator_length - static_cast<double>(cmd.position))
                        -kd * (actuator_velocity - static_cast<double>(cmd.velocity)) +
                        static_cast<double>(cmd.feedforward);
  return std::isfinite(torque) ? torque : 0.0;
}

}  // namespace ula::actuation

#endif  // ULA_GPU_ACTUATION_SIM_MOTOR_HH
