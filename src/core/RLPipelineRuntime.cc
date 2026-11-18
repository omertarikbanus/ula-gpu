#include "supervisor/RLPipelineRuntime.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <mutex>

#include "control/ControlMessages.hh"
#include "core/ControllerPeriod.hh"
#include "estimators/StateEstimatorTypes.hh"
#include "hardware/IMUHW.hh"
#include "hardware/MotorHW.hh"
#include "hardware/SimHW.hh"
#include "modules/MdlControlParams.hh"
#include "modules/MdlCtGaitScheduler.hh"
#include "modules/MdlFootstepPlanner.hh"
#include "modules/MdlLegController.hh"
#include "modules/MdlRLCommandSource.hh"
#include "modules/MdlRLLocomotionState.hh"
#include "modules/MdlRLJointPD.hh"
#include "modules/MdlStateEstimator.hh"
#include "modules/MdlWBIC.hh"
#include "rtcore/ClockHW.hh"
#include "rtcore/ConfigInput.hh"
#include "rtcore/ConfigTable.hh"
#include "rtcore/LogServer.hh"
#include "rtcore/ModuleManager.hh"
#include "rtclient/LogClient.hh"
#include "rtclient/WriteCSV.hh"

namespace quadruped {
namespace {
constexpr const char* kLocalLogPath = "/app/data/current/logs.csv";
constexpr int kLocalLogPeriod = 1;
constexpr int kOrderControlParams = 100;
constexpr int kOrderRlCommandSource = 150;
constexpr int kOrderGaitScheduler = 200;
constexpr int kOrderFootstepPlanner = 210;
constexpr int kOrderRlState = 220;
constexpr int kOrderWBIC = 300;
constexpr int kOrderJointPD = 300;
constexpr int kOrderLegController = 400;
constexpr int kOrderStateEstimator = 32000;
constexpr bool kEnableLocalLoggingWithModuleRebuild = true;
std::mutex g_runtime_exec_mutex;
}  // namespace

RLPipelineRuntime::HardwareThreadGuard::HardwareThreadGuard(
    MotorHW* motor, IMUHW* imu, rtcore::ClockHW* clock)
    : prev_motor_(MotorHW::threadInstance()),
      prev_imu_(IMUHW::threadInstance()),
      prev_clock_(rtcore::ClockHW::threadInstance()) {
  MotorHW::setThreadInstance(motor);
  IMUHW::setThreadInstance(imu);
  rtcore::ClockHW::setThreadInstance(clock);
}

RLPipelineRuntime::HardwareThreadGuard::~HardwareThreadGuard() {
  MotorHW::setThreadInstance(prev_motor_);
  IMUHW::setThreadInstance(prev_imu_);
  rtcore::ClockHW::setThreadInstance(prev_clock_);
}

RLPipelineRuntime::RLPipelineRuntime() : options_(Options{}) {}

RLPipelineRuntime::RLPipelineRuntime(const Options& opts) : options_(opts) {}

RLPipelineRuntime::~RLPipelineRuntime() { shutdown(); }

bool RLPipelineRuntime::initialize() {
  std::lock_guard<std::mutex> lock(g_runtime_exec_mutex);
  if (initialized_) {
    return !disabled_;
  }
  initializeQuadcontrol();
  if (!disabled_) {
    resetSimulation();
    cacheModuleState();
  }
  initialized_ = true;
  return !disabled_;
}

void RLPipelineRuntime::shutdown() {
  std::lock_guard<std::mutex> lock(g_runtime_exec_mutex);
  if (!initialized_) {
    return;
  }

  // Stop the log job before deleting any log variable.
  //
  // rtrobot now synchronizes LogEntry ownership itself (branch
  // fix/logentry-refcount-race), so this ordering is a second line of defence,
  // not the only one. Before that fix LogEntry::_count/_deleted were not
  // synchronized with the log server thread: LogJob::kill() calls release()
  // (and deletes an entry whose count reaches zero) while deleteVar() reads
  // inUse() under _dict_lock only. Disconnecting first left the job running
  // concurrently with the deleteVar() calls below and was an intermittent
  // double free at teardown.
  // A RESET makes the listener thread delete the job and acknowledge it before
  // returning (LogClient::reset() waits for the ACK), so every later
  // deleteVar() stays sequential with it. If the reset fails the job may still
  // be running, so the variables stay registered and are leaked: leaking at
  // shutdown is safe, racing the log server is not.
  bool log_job_stopped = true;
  if (logclient_) {
    log_job_stopped = logclient_->reset();
    if (!log_job_stopped) {
      std::cerr << "[RLPipelineRuntime] LogClient reset failed; leaving env "
                   "debug log variables registered (leaked at shutdown)."
                << std::endl;
    }
  }
  if (log_job_stopped) {
    unregisterEnvDebugLogs_();
  }
  shutdownLocalLogging();
  destroyPipelineModules_();
  if (!disabled_) {
    HardwareThreadGuard hw_guard(motor_hw_, imu_hw_, clock_hw_);
    module_manager_.shutdown();
  } else {
    module_manager_.shutdown();
  }

  cleanupHardware(&module_manager_);

  if (owns_logserver_) {
    delete logserver_;
  }

  control_params_ = nullptr;
  state_estimator_ = nullptr;
  leg_controller_module_ = nullptr;
  rl_command_source_ = nullptr;
  ct_gait_scheduler_ = nullptr;
  footstep_planner_ = nullptr;
  rl_state_module_ = nullptr;
  rl_joint_pd_ = nullptr;
  wbic_ = nullptr;
  sim_driver_ = nullptr;
  motor_hw_ = nullptr;
  imu_hw_ = nullptr;
  clock_hw_ = nullptr;
  logserver_ = nullptr;
  owns_logserver_ = false;
  local_logging_enabled_ = false;
  local_logging_initialized_ = false;
  disabled_ = false;
  initialized_ = false;
}

void RLPipelineRuntime::resetEpisode() {
  std::lock_guard<std::mutex> lock(g_runtime_exec_mutex);
  if (disabled_) {
    return;
  }

  // Full module reconstruction re-registers module log variables.
  // Stop local streaming first to avoid clients reading stale var layouts.
  shutdownLocalLogging();
  resetSimulation();
  ++command_episode_;
  resetModules();
  // Prime-frame logs belong to the new episode.  Clear the persistent debug
  // buffer before those frames so a terminal health/done flag from the prior
  // episode cannot leak into an otherwise fully reset trajectory.
  setEnvDebugFrame(EnvDebugFrame{});
  // Ensure first post-reset control cycles do not consume stale terminal action.
  setPolicyAction(nullptr, 0);
  if (control_params_) {
    control_params_->setControlMode(quadruped::ControlMode::kRlLocomotion);
  }

  HardwareThreadGuard hw_guard(motor_hw_, imu_hw_, clock_hw_);
  constexpr int kResetPrimeFrames = 5;
  for (int i = 0; i < kResetPrimeFrames; ++i) {
    module_manager_.stepOnce();
  }
  module_manager_.stepOnce();
  cacheModuleState();
  maybeInitLocalLogging();
  drainLocalLogging();
}

void RLPipelineRuntime::stepFrame() {
  if (disabled_) {
    return;
  }
  HardwareThreadGuard hw_guard(motor_hw_, imu_hw_, clock_hw_);
  module_manager_.stepOnce();
  cacheModuleState();
  maybeInitLocalLogging();
  drainLocalLogging();
}

void RLPipelineRuntime::stepFrames(int count) {
  if (count <= 0) {
    return;
  }
  for (int idx = 0; idx < count; ++idx) {
    stepFrame();
  }
}

void RLPipelineRuntime::setDesiredVelocity(
    const quadruped::RlDesiredVelocity& desired) {
  if (rl_command_source_) {
    rl_command_source_->setPolicyDesiredVelocity(desired);
  }
}

void RLPipelineRuntime::setPolicyBaseVelocity(
    const quadruped::RlDesiredVelocity& base) {
  if (rl_command_source_) {
    rl_command_source_->setPolicyBaseVelocity(base);
  }
}

quadruped::RlDesiredVelocity RLPipelineRuntime::policyBaseVelocity() const {
  return rl_command_source_ ? rl_command_source_->policyBaseVelocity()
                            : quadruped::RlDesiredVelocity{};
}

void RLPipelineRuntime::setFootForces(
    const quadruped::FootForceTargets& forces) {
  if (rl_command_source_) {
    rl_command_source_->setPolicyFootForces(forces);
  }
}

void RLPipelineRuntime::setFootstepResiduals(
    const quadruped::FootstepResiduals& residuals) {
  if (rl_command_source_) {
    rl_command_source_->setPolicyFootstepResiduals(residuals);
  }
}

void RLPipelineRuntime::setPhaseDelta(float delta) {
  if (rl_command_source_) {
    rl_command_source_->setPolicyPhaseDelta(delta);
  }
  if (control_params_) {
    control_params_->setControlMode(quadruped::ControlMode::kRlLocomotion);
  }
}

void RLPipelineRuntime::setPolicyAction(const mjtNum* action,
                                        std::size_t action_count) {
  if (options_.pipeline == Pipeline::kJointPD) {
    if (!rl_joint_pd_) return;
    std::array<float, MdlRLJointPD::kActionDim> action_buffer{};
    const std::size_t count = action
                                  ? std::min(action_count,
                                             MdlRLJointPD::kActionDim)
                                  : 0;
    for (std::size_t idx = 0; idx < count; ++idx) {
      action_buffer[idx] = std::isfinite(static_cast<double>(action[idx]))
                               ? static_cast<float>(action[idx])
                               : 0.0f;
    }
    rl_joint_pd_->setPolicyAction(action_buffer, count);
    return;
  }
  if (!rl_command_source_) {
    return;
  }
  std::array<float, MdlRLCommandSource::kPolicyActionDim> action_buffer{};
  action_buffer.fill(0.0f);
  const std::size_t count = action
                                ? std::min(action_count,
                                           MdlRLCommandSource::kPolicyActionDim)
                                : 0;
  for (std::size_t idx = 0; idx < count; ++idx) {
    const mjtNum raw = action[idx];
    action_buffer[idx] = std::isfinite(static_cast<double>(raw))
                             ? static_cast<float>(raw)
                             : 0.0f;
  }
  rl_command_source_->setPolicyActionNormalized(action_buffer, count);
  if (control_params_) {
    control_params_->setControlMode(quadruped::ControlMode::kRlLocomotion);
  }
}

void RLPipelineRuntime::setEnvDebugFrame(const EnvDebugFrame& frame) {
  if (options_.env_id != 0) {
    return;
  }
  if (!env_dbg_logs_registered_) {
    registerEnvDebugLogs_();
  }
  env_dbg_env_id_ = frame.env_id;
  env_dbg_step_ = frame.step;
  env_dbg_done_ = frame.done;
  env_dbg_done_reason_ = frame.done_reason;
  env_dbg_is_healthy_ = frame.is_healthy;
  env_dbg_reward_total_ = frame.reward_total;
  env_dbg_ctrl_cost_ = frame.ctrl_cost;
  env_dbg_contact_cost_ = frame.contact_cost;
  env_dbg_healthy_reward_ = frame.healthy_reward;
  env_dbg_action_ = frame.action;
  env_dbg_obs_ = frame.observation;
  env_dbg_reward_terms_ = frame.reward_terms;
  env_dbg_reward_penalties_ = frame.reward_penalties;
  env_dbg_base_pos_ = frame.base_pos;
  env_dbg_base_vel_world_ = frame.base_vel_world;
}

bool RLPipelineRuntime::usesPolicyCommands() const {
  if (!rl_command_source_) {
    return true;
  }
  return rl_command_source_->usesPolicyInputs();
}

const quadruped::RlDesiredVelocity& RLPipelineRuntime::desiredVelocity() const {
  if (rl_command_source_) {
    return rl_command_source_->desiredVelocity();
  }
  static const quadruped::RlDesiredVelocity kZeroDesired = [] {
    quadruped::RlDesiredVelocity d;
    d.body_velocity.setZero();
    return d;
  }();
  return kZeroDesired;
}

double RLPipelineRuntime::computeControlCost(double weight) const {
  if (!motor_hw_) {
    return 0.0;
  }
  double ctrl_cost = 0.0;
  MotorHW::cmd_t cmd{};
  auto* motor_hw = const_cast<MotorHW*>(motor_hw_);
  for (int axis = 0; axis < 12; ++axis) {
    motor_hw->getCommand(axis, cmd);
    const double tau = static_cast<double>(cmd.tau);
    if (std::isfinite(static_cast<double>(tau))) {
      ctrl_cost += weight * tau * tau;
    }
  }
  return ctrl_cost;
}

float RLPipelineRuntime::nominalBodyHeight() const {
  return control_params_
             ? static_cast<float>(control_params_->userParams()->body_height)
             : 0.32f;
}

LegController<float>* RLPipelineRuntime::legController() const {
  return leg_controller_module_ ? leg_controller_module_->legController() : nullptr;
}

std::array<float, 12> RLPipelineRuntime::jointDefaultPosition() const {
  return rl_joint_pd_ ? rl_joint_pd_->defaultJointPosition()
                      : std::array<float, 12>{};
}

void RLPipelineRuntime::initializeQuadcontrol() {
  if (disabled_) {
    return;
  }

  module_manager_.clearConfig();
  if (options_.sim_config_path.empty()) {
    if (options_.env_id == 0) {
      std::cerr << "[RLPipelineRuntime] sim_config_path is empty." << std::endl;
    }
    disabled_ = true;
    return;
  }
  const std::filesystem::path sim_cfg_path(options_.sim_config_path);
  if (sim_cfg_path.has_parent_path()) {
    rtcore::ConfigInput::addDirectoryFirst(
        sim_cfg_path.parent_path().string().c_str());
  }
  if (!module_manager_.appendConfigFile(options_.sim_config_path.c_str())) {
    if (options_.env_id == 0) {
      std::cerr << "[RLPipelineRuntime] Failed to load sim config: "
                << options_.sim_config_path << std::endl;
    }
    disabled_ = true;
    return;
  }

  // Layered last so it overrides the file, matching how the temporary
  // %include stub it replaces behaved.
  if (!options_.sim_config_overlay.empty() &&
      !module_manager_.appendConfigString(options_.sim_config_overlay.c_str())) {
    if (options_.env_id == 0) {
      std::cerr << "[RLPipelineRuntime] Failed to apply sim config overlay."
                << std::endl;
    }
    disabled_ = true;
    return;
  }

  if (!module_manager_.finalizeConfig()) {
    if (options_.env_id == 0) {
      std::cerr << "[RLPipelineRuntime] Failed to finalize config." << std::endl;
    }
    disabled_ = true;
    return;
  }

  local_logging_enabled_ =
      (options_.enable_local_log_for_env0 && options_.env_id == 0);
  rtcore::ConfigTable logserver_config;
  if (module_manager_.getConfigTable("logserver", logserver_config)) {
    local_logging_enabled_ =
        local_logging_enabled_ && logserver_config.getBool("enabled", false);
  } else {
    local_logging_enabled_ = false;
  }
  local_logging_enabled_ =
      local_logging_enabled_ && kEnableLocalLoggingWithModuleRebuild;

  if (local_logging_enabled_) {
    static constexpr int kOrderLogServer = 0;
    logserver_ = dynamic_cast<rtcore::LogServer*>(
        module_manager_.findModule(LOGSERVER_NAME, 0));
    if (!logserver_) {
      logserver_ = new rtcore::LogServer();
      module_manager_.addModule(logserver_, 1, 0, kOrderLogServer);
      owns_logserver_ = true;
    }
    module_manager_.activateModule(logserver_);
  }

  initHardware(&module_manager_, false, options_.env_id == 0,
               options_.env_id);
  sim_driver_ = simHWGetDriver(&module_manager_);
  motor_hw_ = simHWGetMotorHW(&module_manager_);
  imu_hw_ = simHWGetIMUHW(&module_manager_);
  clock_hw_ = simHWGetClockHW(&module_manager_);
  if (!sim_driver_) {
    std::cerr << "[RLPipelineRuntime] SimHW context not available." << std::endl;
    cleanupHardware(&module_manager_);
    disabled_ = true;
    return;
  }
  if (!motor_hw_ || !imu_hw_ || !clock_hw_) {
    std::cerr << "[RLPipelineRuntime] SimHW hardware instances missing."
              << std::endl;
    cleanupHardware(&module_manager_);
    disabled_ = true;
    return;
  }

  HardwareThreadGuard hw_guard(motor_hw_, imu_hw_, clock_hw_);

  // Period comes from [control_parameters] controller_dt, matching
  // supervisor/main.cc: control rate must not be coupled to the MuJoCo
  // timestep, or physics cannot be refined without changing the control rate.
  // The model timestep remains the fallback when the key is absent, exactly
  // as before. (The old simHWSetPeriod() call wrote a driver field that
  // nothing reads; integration uses getStepPeriod() every update.)
  const rtcore::CLOCK step_period = quadruped::applyControllerPeriod(
      &module_manager_, simHWGetStepPeriod(sim_driver_));
  step_period_s_ = static_cast<double>(step_period) * 1e-6;
  if (options_.env_id == 0 && MdlControlParams::parityDumpEnabled()) {
    std::cout << "PARITY step_period_us=" << module_manager_.getStepPeriod()
              << std::endl;
  }

  createPipelineModules_();
  registerEnvDebugLogs_();
}

void RLPipelineRuntime::createPipelineModules_() {
  if (control_params_ || rl_command_source_ || ct_gait_scheduler_ ||
      footstep_planner_ || rl_state_module_ || rl_joint_pd_ || wbic_ ||
      leg_controller_module_ || state_estimator_) {
    return;
  }

  control_params_ = new quadruped::MdlControlParams();
  module_manager_.addModule(control_params_, 1, 0, kOrderControlParams);
  module_manager_.activateModule(control_params_);

  rl_command_source_ = new quadruped::MdlRLCommandSource();
  const std::uint64_t command_stream =
      (command_episode_ << 32U) |
      static_cast<std::uint32_t>(options_.env_id);
  rl_command_source_->setRandomStream(command_stream);
  module_manager_.addModule(rl_command_source_, 1, 0, kOrderRlCommandSource);
  module_manager_.activateModule(rl_command_source_);

  if (options_.pipeline == Pipeline::kWbic) {
    ct_gait_scheduler_ = new quadruped::MdlCtGaitScheduler();
    module_manager_.addModule(ct_gait_scheduler_, 1, 0, kOrderGaitScheduler);
    module_manager_.activateModule(ct_gait_scheduler_);

    footstep_planner_ = new quadruped::MdlFootstepPlanner();
    module_manager_.addModule(footstep_planner_, 1, 0, kOrderFootstepPlanner);
    module_manager_.activateModule(footstep_planner_);

    rl_state_module_ = new quadruped::MdlRLLocomotionState();
    module_manager_.addModule(rl_state_module_, 1, 0, kOrderRlState);
    module_manager_.activateModule(rl_state_module_);

    wbic_ = new quadruped::MdlWBIC();
    module_manager_.addModule(wbic_, 1, 0, kOrderWBIC);
    module_manager_.activateModule(wbic_);
  }

  leg_controller_module_ = new quadruped::MdlLegController();
  module_manager_.addModule(leg_controller_module_, 1, 0, kOrderLegController);
  module_manager_.activateModule(leg_controller_module_);

  if (options_.pipeline == Pipeline::kJointPD) {
    rl_joint_pd_ = new quadruped::MdlRLJointPD();
    module_manager_.addModule(rl_joint_pd_, 1, 0, kOrderJointPD);
    module_manager_.activateModule(rl_joint_pd_);
  }

  state_estimator_ = new quadruped::MdlStateEstimator();
  module_manager_.addModule(state_estimator_, 1, 0, kOrderStateEstimator);
  module_manager_.activateModule(state_estimator_);

  if (control_params_) {
    control_params_->setControlMode(quadruped::ControlMode::kRlLocomotion);
  }

  if (control_params_ && options_.env_id == 0 &&
      MdlControlParams::parityDumpEnabled()) {
    control_params_->dumpEffectiveParameters(std::cout);
  }
}

void RLPipelineRuntime::destroyPipelineModules_() {
  auto destroy_module = [this](rtcore::Module*& module) {
    if (!module) {
      return;
    }
    module_manager_.deactivateModule(module);
    module_manager_.removeModule(module);
    delete module;
    module = nullptr;
  };

  rtcore::Module* state_estimator = state_estimator_;
  destroy_module(state_estimator);
  state_estimator_ = nullptr;

  rtcore::Module* leg_controller = leg_controller_module_;
  destroy_module(leg_controller);
  leg_controller_module_ = nullptr;

  rtcore::Module* wbic = wbic_;
  destroy_module(wbic);
  wbic_ = nullptr;

  rtcore::Module* joint_pd = rl_joint_pd_;
  destroy_module(joint_pd);
  rl_joint_pd_ = nullptr;

  rtcore::Module* rl_state = rl_state_module_;
  destroy_module(rl_state);
  rl_state_module_ = nullptr;

  rtcore::Module* footstep = footstep_planner_;
  destroy_module(footstep);
  footstep_planner_ = nullptr;

  rtcore::Module* gait = ct_gait_scheduler_;
  destroy_module(gait);
  ct_gait_scheduler_ = nullptr;

  rtcore::Module* rl_cmd = rl_command_source_;
  destroy_module(rl_cmd);
  rl_command_source_ = nullptr;

  rtcore::Module* control = control_params_;
  destroy_module(control);
  control_params_ = nullptr;
}

void RLPipelineRuntime::resetModules() {
  if (disabled_) {
    return;
  }

  HardwareThreadGuard hw_guard(motor_hw_, imu_hw_, clock_hw_);
  destroyPipelineModules_();
  // The sim driver is hardware: initHardware() built it and it survives this
  // rebuild, still holding pointers to the modules just deleted. Clear them
  // before the replacements are registered, or the driver keeps reading freed
  // memory -- which stalled the gait phase it samples for the force onset, so
  // most episodes after the first reset were never pushed at all.
  simHWInvalidateModuleCache(sim_driver_);
  createPipelineModules_();
}

SimForceState RLPipelineRuntime::forceState() const {
  return simHWGetForceState(sim_driver_);
}

std::array<float, 4> RLPipelineRuntime::footContactState(
    float min_normal_force) const {
  return simHWGetFootContactState(sim_driver_, min_normal_force);
}

int RLPipelineRuntime::penalizedCollisionCount() const {
  return simHWGetPenalizedCollisionCount(sim_driver_);
}

SimRobotState RLPipelineRuntime::robotState() const {
  return simHWGetRobotState(sim_driver_);
}

void RLPipelineRuntime::setRobotState(const SimRobotState& state) {
  simHWSetRobotState(sim_driver_, state);
}

void RLPipelineRuntime::addBaseLinearVelocityXY(double vx, double vy) {
  simHWAddBaseLinearVelocityXY(sim_driver_, vx, vy);
}

void RLPipelineRuntime::setFriction(double mu) {
  simHWSetFriction(sim_driver_, mu);
}

double RLPipelineRuntime::groundFriction() const {
  return simHWGetGroundFriction(sim_driver_);
}

double RLPipelineRuntime::baseContactForceNorm() const {
  return simHWGetBaseContactForceNorm(sim_driver_);
}

double RLPipelineRuntime::stepPeriodSeconds() const {
  return step_period_s_;
}

void RLPipelineRuntime::resetSimulation() {
  if (!sim_driver_) {
    return;
  }
  simHWResetSimulation(sim_driver_);
}

void RLPipelineRuntime::cacheModuleState() {
  if (ct_gait_scheduler_ && ct_gait_scheduler_->hasSchedule()) {
    last_gait_schedule_ = ct_gait_scheduler_->gaitSchedule();
  } else {
    // The joint-PD pipeline has no gait scheduler. Never retain a schedule
    // copied from a module that was destroyed during an episode reset.
    last_gait_schedule_ = quadruped::GaitSchedule{};
  }
  if (state_estimator_) {
    last_state_est_ = state_estimator_->getCheetahStateEstimate();
  }
}

void RLPipelineRuntime::registerEnvDebugLogs_() {
  if (env_dbg_logs_registered_ || options_.env_id != 0) {
    return;
  }
  if (!logserver_) {
    logserver_ = dynamic_cast<rtcore::LogServer*>(
        module_manager_.findModule(LOGSERVER_NAME, 0));
  }
  if (!logserver_) {
    return;
  }

  std::fill(env_dbg_log_ids_.begin(), env_dbg_log_ids_.end(), 0);
  const char* module = "RLPipelineRuntime";
  std::size_t i = 0;
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_INT, 1, module, "rl_dbg_env_id",
      reinterpret_cast<unsigned char*>(&env_dbg_env_id_));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_INT, 1, module, "rl_dbg_step",
      reinterpret_cast<unsigned char*>(&env_dbg_step_));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_INT, 1, module, "rl_dbg_done",
      reinterpret_cast<unsigned char*>(&env_dbg_done_));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_INT, 1, module, "rl_dbg_done_reason",
      reinterpret_cast<unsigned char*>(&env_dbg_done_reason_));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_INT, 1, module, "rl_dbg_is_healthy",
      reinterpret_cast<unsigned char*>(&env_dbg_is_healthy_));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_FLOAT, 1, module, "rl_dbg_reward_total",
      reinterpret_cast<unsigned char*>(&env_dbg_reward_total_));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_FLOAT, 1, module, "rl_dbg_ctrl_cost",
      reinterpret_cast<unsigned char*>(&env_dbg_ctrl_cost_));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_FLOAT, 1, module, "rl_dbg_contact_cost",
      reinterpret_cast<unsigned char*>(&env_dbg_contact_cost_));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_FLOAT, 1, module, "rl_dbg_healthy_reward",
      reinterpret_cast<unsigned char*>(&env_dbg_healthy_reward_));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_FLOAT, static_cast<int>(env_dbg_action_.size()), module,
      "rl_dbg_action",
      reinterpret_cast<unsigned char*>(env_dbg_action_.data()));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_FLOAT, static_cast<int>(env_dbg_obs_.size()), module,
      "rl_dbg_obs", reinterpret_cast<unsigned char*>(env_dbg_obs_.data()));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_FLOAT, static_cast<int>(env_dbg_reward_terms_.size()), module,
      "rl_dbg_reward_terms",
      reinterpret_cast<unsigned char*>(env_dbg_reward_terms_.data()));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_FLOAT, static_cast<int>(env_dbg_reward_penalties_.size()),
      module, "rl_dbg_reward_penalties",
      reinterpret_cast<unsigned char*>(env_dbg_reward_penalties_.data()));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_FLOAT, static_cast<int>(env_dbg_base_pos_.size()), module,
      "rl_dbg_base_pos",
      reinterpret_cast<unsigned char*>(env_dbg_base_pos_.data()));
  env_dbg_log_ids_[i++] = logserver_->registerVar(
      rtcore::LOG_FLOAT, static_cast<int>(env_dbg_base_vel_world_.size()),
      module, "rl_dbg_base_vel_world",
      reinterpret_cast<unsigned char*>(env_dbg_base_vel_world_.data()));

  env_dbg_logs_registered_ = true;
}

void RLPipelineRuntime::unregisterEnvDebugLogs_() {
  if (!env_dbg_logs_registered_) {
    return;
  }
  if (!logserver_) {
    env_dbg_log_ids_.fill(0);
    env_dbg_logs_registered_ = false;
    return;
  }
  for (auto& id : env_dbg_log_ids_) {
    if (id != 0) {
      logserver_->deleteVar(id);
      id = 0;
    }
  }
  env_dbg_logs_registered_ = false;
}

void RLPipelineRuntime::maybeInitLocalLogging() {
  if (!local_logging_enabled_ || local_logging_initialized_) {
    return;
  }
  if (!logserver_) {
    logserver_ = dynamic_cast<rtcore::LogServer*>(
        module_manager_.findModule(LOGSERVER_NAME, 0));
    if (!logserver_) {
      return;
    }
  }

  static std::once_flag s_enet_once;
  std::call_once(s_enet_once, []() { rtclient::enet_initialize(); });

  if (!logclient_) {
    logclient_ = new rtclient::LogClient("localhost", logserver_->getPort(),
                                         logserver_->getChannel());
  }
  if (!logclient_) {
    return;
  }
  if (!logclient_->query()) {
    return;
  }
  const unsigned int var_count = logclient_->listVars();
  if (var_count == 0) {
    return;
  }

  if (logtask_) {
    logclient_->destroyLog(logtask_);
    logtask_ = nullptr;
  }
  logtask_ = logclient_->newLog();
  if (!logtask_) {
    return;
  }

  bool all_added = true;
  for (rtcore::log_name_t* var = logclient_->getNextVar(); var != nullptr;
       var = logclient_->getNextVar()) {
    if (!logtask_->addVar(var->name)) {
      all_added = false;
      break;
    }
  }
  if (!all_added) {
    logclient_->destroyLog(logtask_);
    logtask_ = nullptr;
    return;
  }

  if (kLocalLogPath[0] != '\0') {
    std::filesystem::path log_path(kLocalLogPath);
    std::error_code ec;
    if (!log_path.parent_path().empty()) {
      std::filesystem::create_directories(log_path.parent_path(), ec);
    }
  }

  logwriter_ = std::make_unique<rtclient::WriteCSV>(
      kLocalLogPath, logtask_->varList(), "EnvPool local data log");
  logtask_->startLog(kLocalLogPeriod, 0);
  local_logging_initialized_ = true;
}

void RLPipelineRuntime::drainLocalLogging() {
  if (!local_logging_initialized_ || !logtask_ || !logwriter_) {
    return;
  }
  rtcore::log_line_t* line = nullptr;
  while ((line = logtask_->getData(0)) != nullptr) {
    logwriter_->appendLine(line);
  }
}

void RLPipelineRuntime::shutdownLocalLogging() {
  if (logtask_) {
    logtask_->abortLog();
    if (logclient_) {
      logclient_->destroyLog(logtask_);
    }
    logtask_ = nullptr;
  }
  logwriter_.reset();
  if (logclient_) {
    delete logclient_;
    logclient_ = nullptr;
  }
  local_logging_initialized_ = false;
}

}  // namespace quadruped
