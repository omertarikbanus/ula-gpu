#ifndef ULA_GPU_PORT_REPLAY_STAGES_HH
#define ULA_GPU_PORT_REPLAY_STAGES_HH

#include <memory>

#include "actuation/sim_motor.hh"
#include "port/controller.hh"
#include "port/rigid_body.hh"
#include "qp/wbic_problem.hh"

namespace ula::port {

// Only plain, fixed-size data crosses the CUDA boundary. The host still
// assembles and projects the tasks to obtain the joint and acceleration targets.
template <class T = float>
struct MotorTickInputT {
  actuation::LegMeasurementT<T> measured[4]{};
  actuation::LegTargetT<T> target[4]{};
  T limit[3]{};
  double length[12]{}, velocity[12]{};
  bool enabled[12]{};
};

template <class T = float>
struct MotorTickOutputT {
  T feedforward[12]{};
  double torque[12]{};
};

using MotorTickInput = MotorTickInputT<float>;
using MotorTickOutput = MotorTickOutputT<float>;

template <class T>
ULA_MOTOR_HD inline MotorTickOutputT<T> motor_tick(const MotorTickInputT<T>& in) {
  MotorTickOutputT<T> out{};
  for (int leg = 0; leg < 4; ++leg) {
    const auto motor = actuation::prepare_leg_command(in.measured[leg], in.target[leg], in.limit);
    for (int joint = 0; joint < 3; ++joint) {
      const int i = 3 * leg + joint;
      const auto& cmd = motor.command[joint];
      out.feedforward[i] = cmd.feedforward;
      out.torque[i] = actuation::sim_motor_torque(cmd, in.length[i], in.velocity[i], in.enabled[i]);
    }
  }
  return out;
}

using ReplayQp = qp::WbicTorqueProblem<float>;
using ReplayQpSolution = qp::WbicTorqueSolution<float>;

// Host-side dispatch only. No vtables, allocation, or Eigen objects in kernels.
class ReplayStages {
 public:
  virtual ~ReplayStages() = default;
  virtual const char* name() const = 0;
  virtual void reset(const ControllerConfig& config) = 0;
  virtual void set_model(const rigid::Model<float>& model) = 0;
  virtual ControllerOutput plan(const ControllerInput& in) = 0;
  virtual void dynamics(const rigid::State<float>& in, rigid::Result<float>& out) = 0;
  virtual void solve(const ReplayQp& in, ReplayQpSolution& out) = 0;
  virtual MotorTickOutput act(const MotorTickInput& in) = 0;
};

// Exercises the same dispatch on a machine without a CUDA device.
class HostReplayStages final : public ReplayStages {
 public:
  const char* name() const override { return "host-stages"; }
  void reset(const ControllerConfig& config) override {
    config_ = config;
    ula::port::reset(planner_, config_);
  }
  ControllerOutput plan(const ControllerInput& in) override {
    return ula::port::step(planner_, config_, in);
  }
  void set_model(const rigid::Model<float>& model) override { model_ = model; }
  void dynamics(const rigid::State<float>& in, rigid::Result<float>& out) override {
    rigid::calculate(model_, in, out);
  }
  void solve(const ReplayQp& in, ReplayQpSolution& out) override { qp::solve_wbic_torque(in, out); }
  MotorTickOutput act(const MotorTickInput& in) override { return motor_tick(in); }

 private:
  ControllerConfig config_{};
  ControllerState planner_{};
  rigid::Model<float> model_{};
};

// Throws if a CUDA device is unavailable. There is no automatic CPU fallback.
std::unique_ptr<ReplayStages> make_cuda_replay_stages();

}  // namespace ula::port

#endif
