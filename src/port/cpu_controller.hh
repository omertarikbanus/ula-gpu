#ifndef ULA_GPU_PORT_CPU_CONTROLLER_HH
#define ULA_GPU_PORT_CPU_CONTROLLER_HH

#include <memory>
#include <string>

#include "control/LocomotionCtrlData.hh"
#include "controllers/LegController.h"
#include "controllers/wbc/WBC_Ctrl/LocomotionCtrl/LocomotionCtrl.hpp"
#include "port/replay_stages.hh"
#include "port/tick_input.hh"

namespace ula::port {

struct CpuConfig {
  ControllerConfig planner{};
  quadruped::MIT_UserParameters wbic{};
  float floating_base_weight = 50.0f;
  float max_torque[3]{23.7f, 23.7f, 45.43f};
};

template <class T>
struct CpuOutputT {
  ControllerOutput planner{};
  LocomotionCtrlData<T> wbic_command{};  // after liftoff alignment if WBIC ran
  bool wbic_ran = false;
  T joint_position_target[12]{}, joint_velocity_target[12]{};
  T feedforward_torque[12]{};  // WBIC output before limit/PD compensation
  T motor_feedforward[12]{};
  double motor_torque[12]{};  // ctrl[] before the simulator's force range
};

// CPU bridge around ULA's original controller classes. The planner always runs
// in float, as in ULA. T selects the precision of everything after it (model,
// tasks, WBIC, QP, leg and motor commands): float is ULA as deployed, double is
// the reference that measures float rounding noise. Replay stages apply to
// float only.
template <class T>
class CpuControllerT {
 public:
  // stages, if provided, belongs to the caller and must outlive this object.
  bool init(const std::string& urdf, const CpuConfig& config, ReplayStages* stages = nullptr);
  void reset();  // after init(); constructs a fresh WBIC, as ULA does
  CpuOutputT<T> step(const CpuInput& in);
  const FloatingBaseModel<T>& model() const { return wbic_->getModel(); }
  const CpuConfig& config() const { return config_; }

 private:
  Quadruped<T> quad_;
  CpuConfig config_{};
  ControllerState planner_{};
  ReplayStages* stages_ = nullptr;
  std::unique_ptr<LocomotionCtrl<T>> wbic_;
  std::unique_ptr<LegController<T>> leg_;
};

using CpuOutput = CpuOutputT<float>;
using CpuController = CpuControllerT<float>;
using ReferenceController = CpuControllerT<double>;

extern template class CpuControllerT<float>;
extern template class CpuControllerT<double>;

}  // namespace ula::port

#endif
