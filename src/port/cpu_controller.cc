#include "port/cpu_controller.hh"

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "dynamics/ParseURDFtoQuadruped.h"
#include "port/rigid_body_adapter.hh"

namespace ula::port {

template <class T>
bool CpuControllerT<T>::init(const std::string& urdf, const CpuConfig& config, ReplayStages* stages) {
  if constexpr (!std::is_same_v<T, float>) {
    if (stages) return false;  // the staged dispatch is float-only
  }
  bool ok = false;
  quad_ = ParseURDFtoQuadruped<T>(urdf, RobotType::MINI_CHEETAH, &ok);
  if (!ok) return false;
  config_ = config;
  config_.planner.body_height = static_cast<float>(config_.wbic.body_height);
  stages_ = stages;
  auto& geometry = config_.planner.footstep.kinematics;
  for (int i = 0; i < 3; ++i) geometry.abad_location[i] = static_cast<float>(quad_._abadLocation[i]);
  reset();
  return true;
}

template <class T>
void CpuControllerT<T>::reset() {
  if (stages_) stages_->reset(config_.planner);
  else ula::port::reset(planner_, config_.planner);
  wbic_ = std::make_unique<LocomotionCtrl<T>>(quad_.buildModel());
  wbic_->setFloatingBaseWeight(static_cast<T>(config_.floating_base_weight));
  if constexpr (std::is_same_v<T, float>) {
    if (stages_) {
      stages_->set_model(rigid::pack_model(wbic_->getModel()));
      wbic_->setDynamicsDispatch({stages_, [](void* context, const rigid::State<float>& in,
                                            rigid::Result<float>& out) {
        static_cast<ReplayStages*>(context)->dynamics(in, out);
      }});
      wbic_->setQpDispatch({stages_, [](void* context, const ReplayQp& in, ReplayQpSolution& sol) {
        static_cast<ReplayStages*>(context)->solve(in, sol);
        if (sol.qp.result.status != qp::kOk || !std::isfinite(sol.qp.result.f)) {
          throw std::runtime_error("WBIC QP failed, status=" + std::to_string(sol.qp.result.status));
        }
        for (int i = 0; i < ReplayQp::Base + in.contact_rows; ++i)
          if (!std::isfinite(sol.qp.x[i])) throw std::runtime_error("WBIC QP returned a non-finite solution");
        for (const float tau : sol.torque)
          if (!std::isfinite(tau)) throw std::runtime_error("WBIC recovered a non-finite torque");
      }});
    }
  }
  leg_ = std::make_unique<LegController<T>>(quad_);
}

template <class T>
CpuOutputT<T> CpuControllerT<T>::step(const CpuInput& in) {
  CpuOutputT<T> out{};
  out.planner = stages_ ? stages_->plan(in.planner) : ula::port::step(planner_, config_.planner, in.planner);
  const auto& plant = in.planner.plant;
  const auto& targets = out.planner.wbic;
  const auto& gait = out.planner.gait;
  out.wbic_ran = targets.run_wbic;

  using Map3 = Eigen::Map<const Vec3<float>>;
  using Map4 = Eigen::Map<const Vec4<float>>;
  quadruped::StateEstimate<T> state{};
  state.position = Map3(plant.position_w).cast<T>();
  state.orientation = Map4(in.orientation_wxyz).cast<T>();
  state.vBody = Map3(in.linear_velocity_b).cast<T>();
  state.omegaBody = Map3(in.angular_velocity_b).cast<T>();
  using RowMajor3 = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
  const Mat3<float> Rwb = Eigen::Map<const RowMajor3>(&plant.rotation_wb[0][0]);
  state.rBody = Rwb.transpose().cast<T>();

  auto& cmd = out.wbic_command;
  cmd.setZero();
  cmd.pBody_des = Map3(targets.position_w).cast<T>();
  cmd.vBody_des = Map3(targets.velocity_w).cast<T>();
  cmd.vBody_Ori_des = Map3(targets.angular_velocity).cast<T>();
  cmd.pBody_RPY_des = Map3(targets.orientation_rpy).cast<T>();

  for (int leg = 0; leg < 4; ++leg) {
    auto& data = leg_->datas[leg];
    data.q = Map3(in.joint_position + 3 * leg).cast<T>();
    data.qd = Map3(in.joint_velocity + 3 * leg).cast<T>();
    cmd.contact_state[leg] = gait.planned_contact[leg] ? T(1) : T(0);
    cmd.Fr_des[leg] = Map3(targets.force_w[leg]).cast<T>();
    cmd.pFoot_des[leg] = Map3(targets.foot.foot_position_target[leg]).cast<T>();
    cmd.vFoot_des[leg] = Map3(targets.foot.foot_velocity_target[leg]).cast<T>();
    cmd.aFoot_des[leg] = Map3(targets.foot.foot_acceleration_target[leg]).cast<T>();
  }

  // In ULA, WBIC discovers the later-created leg module on the first tick
  // and constructs its model on the next one. Leg commands stay zero until then.
  if (out.wbic_ran) wbic_->run(&cmd, state, leg_.get(), &config_.wbic);
  MotorTickInputT<T> motor{};
  for (int j = 0; j < 3; ++j) motor.limit[j] = static_cast<T>(config_.max_torque[j]);
  for (int leg = 0; leg < 4; ++leg) {
    const auto& leg_cmd = leg_->commands[leg];
    auto& measured = motor.measured[leg];
    auto& target = motor.target[leg];
    // WBIC emits joint-space commands only; the Cartesian commands stay zero.
    for (int j = 0; j < 3; ++j) {
      const int i = 3 * leg + j;
      measured.joint_position[j] = static_cast<T>(in.motor_joint_position[i]);
      measured.joint_velocity[j] = static_cast<T>(in.motor_joint_velocity[i]);
      motor.length[i] = in.actuator_position[i];
      motor.velocity[i] = in.actuator_velocity[i];
      motor.enabled[i] = in.actuator_enabled[i];
      out.joint_position_target[i] = target.joint_position[j] = leg_cmd.qDes[j];
      out.joint_velocity_target[i] = target.joint_velocity[j] = leg_cmd.qdDes[j];
      out.feedforward_torque[i] = target.torque_ff[j] = leg_cmd.tauFeedForward[j];
      target.joint_kp[j][j] = leg_cmd.kpJoint(j, j);
      target.joint_kd[j][j] = leg_cmd.kdJoint(j, j);
    }
  }
  MotorTickOutputT<T> motor_out;
  if constexpr (std::is_same_v<T, float>) motor_out = stages_ ? stages_->act(motor) : motor_tick(motor);
  else motor_out = motor_tick(motor);
  for (int i = 0; i < 12; ++i) {
    out.motor_feedforward[i] = motor_out.feedforward[i];
    out.motor_torque[i] = motor_out.torque[i];
  }
  return out;
}

template class CpuControllerT<float>;
template class CpuControllerT<double>;

}  // namespace ula::port
