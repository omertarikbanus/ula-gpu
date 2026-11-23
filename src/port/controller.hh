#ifndef ULA_GPU_PORT_CONTROLLER_HH
#define ULA_GPU_PORT_CONTROLLER_HH

#include <cmath>
#include <cstddef>
#include <Eigen/Geometry>

#ifdef __CUDACC__
#define ULA_PORT_HD __host__ __device__
#else
#define ULA_PORT_HD
#endif

namespace ula::port {

inline constexpr int kPolicyActionDim = 25;
inline constexpr int kNumLegs = 4;       // Leg order: FR, FL, HR, HL
inline constexpr int kJointsPerLeg = 3;  // Joint order: abad, hip, knee
inline constexpr float kDefaultDtSeconds = 0.002f;
inline constexpr float kDefaultPhaseDelta = 1.0f / 200.0f;
inline constexpr float kMaxPhaseDelta = 2.0f * kDefaultPhaseDelta;

/// Action command mode matching quadruped::MdlRLCommandSource::Mode.
enum class CommandMode : int {
  kPolicy = 0,
  kTomlFixed = 1,
};

/// Configuration for policy action decoding (mirrors MdlRLCommandSource).
struct ActionDecoderConfig {
  CommandMode mode = CommandMode::kPolicy;
  float policy_linear_limit = 0.5f;
  float policy_yaw_limit = 0.3f;
  float force_xy_limit = 50.0f;
  float force_z_max = 250.0f;
  float residual_xy_scale = 0.5f;
  float height_residual_limit = 0.05f;
  float phase_delta_min = 0.0f;
  float phase_delta_max = kMaxPhaseDelta;

  // When true, velocity clamp caps use the configured random sampling range
  // bounds (matching MdlRLCommandSource::applyCommand_). Base command sampling
  // itself is performed at the episode boundary and passed in via base_command.
  bool policy_random_velocity_enabled = false;
  float policy_random_vx_min = -0.5f;
  float policy_random_vx_max = 0.5f;
  float policy_random_vy_min = -0.3f;
  float policy_random_vy_max = 0.3f;
  float policy_random_yaw_min = -0.3f;
  float policy_random_yaw_max = 0.3f;

  // Values used when mode == CommandMode::kTomlFixed.
  float fixed_desired_velocity[3] = {0.0f, 0.0f, 0.0f};
  float fixed_foot_forces[kNumLegs][3] = {};
  float fixed_footstep_residuals[kNumLegs][3] = {};
  float fixed_phase_delta = kDefaultPhaseDelta;

  /// Defaults matching ULA's envpool.toml [rl_command_source].
  static constexpr ActionDecoderConfig envpool_defaults() {
    ActionDecoderConfig cfg{};
    cfg.policy_random_velocity_enabled = true;
    cfg.policy_random_vx_min = -0.5f;
    cfg.policy_random_vx_max = 2.7f;
    cfg.policy_random_vy_min = -0.1f;
    cfg.policy_random_vy_max = 0.1f;
    cfg.policy_random_yaw_min = 0.0f;
    cfg.policy_random_yaw_max = 0.0f;
    cfg.fixed_foot_forces[0][2] = 150.0f;
    cfg.fixed_foot_forces[1][2] = 150.0f;
    cfg.fixed_foot_forces[2][2] = 150.0f;
    cfg.fixed_foot_forces[3][2] = 150.0f;
    return cfg;
  }

  /// Defaults matching ULA's envpool_train_Gamma5.toml / v1 / ramp123.
  static constexpr ActionDecoderConfig gamma5_defaults() {
    ActionDecoderConfig cfg = envpool_defaults();
    cfg.policy_random_vx_min = 0.0f;
    cfg.policy_random_vx_max = 1.0f;
    cfg.policy_random_vy_min = -0.2f;
    cfg.policy_random_vy_max = 0.2f;
    cfg.policy_random_yaw_min = -0.2f;
    cfg.policy_random_yaw_max = 0.2f;
    cfg.fixed_desired_velocity[0] = 0.8f;
    return cfg;
  }
};

/// Configuration for continuous-time gait scheduling (mirrors MdlCtGaitScheduler).
struct GaitSchedulerConfig {
  float dt_s = kDefaultDtSeconds;
  float phase_offsets[kNumLegs] = {0.0f, 0.5f, 0.5f, 0.0f};
  float contact_durations[kNumLegs] = {0.5f, 0.5f, 0.5f, 0.5f};
};

/// Leg geometry for forward kinematics and hip locations (Go2 URDF defaults).
struct LegKinematicsConfig {
  float abad_location[3] = {0.1934f, 0.0465f, 0.0f};
  float abad_x_offset = 0.0f;
  float abad_link_length = 0.0955f;
  float hip_link_length = 0.213f;
  float knee_link_length = 0.235f;  // 0.213 m calf + 0.022 m foot sphere radius
  float knee_link_y_offset = 0.0f;
};

/// Configuration for analytic + residual footstep planning (mirrors MdlFootstepPlanner).
struct FootstepPlannerConfig {
  float swing_height = 0.10f;
  float gravity = 9.81f;
  float p_rel_max = 0.30f;
  float foot_target_z = -0.003f;
  LegKinematicsConfig kinematics{};
};

/// Combined configuration for the ported controller slice.
struct ControllerConfig {
  ActionDecoderConfig decoder{};
  GaitSchedulerConfig gait{};
  FootstepPlannerConfig footstep{};
  float body_height = 0.32f;
};

/// Decoded policy commands (mirrors MdlRLCommandSource outputs).
struct DecodedCommand {
  float desired_velocity[3] = {0.0f, 0.0f, 0.0f};  // [vx, vy, yaw_rate]
  float height_residual = 0.0f;                    // [m]
  float foot_forces[kNumLegs][3] = {};             // [N], leg-major
  float footstep_residuals[kNumLegs][3] = {};      // [m], z always 0
  float phase_delta = kDefaultPhaseDelta;          // [cycles / tick]
};

/// Persistent state of the gait scheduler across ticks.
struct GaitSchedulerState {
  float cycle_time_seconds = 0.08f;
  float time_into_cycle = 0.0f;
  float theta = 0.0f;
  bool has_schedule = false;
  float phase_offsets[kNumLegs] = {0.0f, 0.5f, 0.5f, 0.0f};
  float contact_durations[kNumLegs] = {0.5f, 0.5f, 0.5f, 0.5f};
  float swing_durations[kNumLegs] = {0.5f, 0.5f, 0.5f, 0.5f};
  float contact_state[kNumLegs] = {1.0f, 1.0f, 1.0f, 1.0f};
  float contact_phase[kNumLegs] = {0.0f, 0.0f, 0.0f, 0.0f};
  float swing_phase[kNumLegs] = {0.0f, 0.0f, 0.0f, 0.0f};
};

/// Gait schedule emitted on each tick (mirrors quadruped::GaitSchedule).
struct GaitSchedule {
  float contact_state[kNumLegs] = {1.0f, 1.0f, 1.0f, 1.0f};
  bool planned_contact[kNumLegs] = {true, true, true, true};
  float contact_phase[kNumLegs] = {0.0f, 0.0f, 0.0f, 0.0f};
  float swing_phase[kNumLegs] = {0.0f, 0.0f, 0.0f, 0.0f};
  float stance_duration[kNumLegs] = {0.0f, 0.0f, 0.0f, 0.0f};
  float swing_duration[kNumLegs] = {0.0f, 0.0f, 0.0f, 0.0f};
  float gait_phase = 0.0f;
  float gait_cycle_time = 0.0f;
  float delta_theta = 0.0f;
};

/// Per-leg Bezier swing trajectory state (mirrors FootSwingTrajectory<float>).
struct FootSwingState {
  float p0[3] = {0.0f, 0.0f, 0.0f};
  float pf[3] = {0.0f, 0.0f, 0.0f};
  float height = 0.10f;
};

/// Persistent state of the footstep planner across ticks.
struct FootstepPlannerState {
  float prev_contact_state[kNumLegs] = {1.0f, 1.0f, 1.0f, 1.0f};
  FootSwingState swing_traj[kNumLegs]{};
  bool has_targets = false;
};

/// World-frame foot targets emitted by the footstep planner (mirrors FootstepTargets).
struct FootstepTargets {
  float foot_position_target[kNumLegs][3] = {};      // pFoot_des
  float foot_velocity_target[kNumLegs][3] = {};      // vFoot_des
  float foot_acceleration_target[kNumLegs][3] = {};  // aFoot_des
};

// Independently rounded estimator values and cached leg kinematics, as
// consumed by ULA. Do not reconstruct these from a float quaternion/velocity
// or from the newer joint measurements used for the motor command.
struct PlannerMeasurements {
  float position_w[3]{};
  float rotation_wb[3][3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};  // body -> world
  float linear_velocity_w[3]{};
  float foot_position_leg[kNumLegs][3]{};  // relative to each hip, body axes
  float yaw{};  // estimator's yaw, not recomputed from the rounded quaternion
};

struct BodyCommandState {
  float position_w[2]{};
  float velocity_b[2]{};
  bool first_tick = true;
  bool previous_contact[kNumLegs]{true, true, true, true};
};

// Targets consumed by WBIC, distinct from the planner's unmodified foot targets.
struct WholeBodyTargets {
  float position_w[3]{}, velocity_w[3]{};
  float orientation_rpy[3]{}, angular_velocity[3]{};
  FootstepTargets foot{};
  float force_w[kNumLegs][3]{};
  bool run_wbic = false;
};

/// Complete persistent state of the ported controller slice.
struct ControllerState {
  GaitSchedulerState gait{};
  FootstepPlannerState footstep{};
  BodyCommandState body{};
};

/// Explicit per-tick input to the ported controller slice.
struct ControllerInput {
  float action[kPolicyActionDim] = {};
  int action_count = kPolicyActionDim;  // < 25 triggers baseline pre-action fallback
  float base_command[3] = {0.0f, 0.0f, 0.0f};  // [vx, vy, yaw_rate]
  PlannerMeasurements plant{};
};

/// Explicit per-tick output from the ported controller slice.
struct ControllerOutput {
  DecodedCommand decoded{};
  GaitSchedule gait{};
  FootstepTargets footstep{};
  WholeBodyTargets wbic{};
};

namespace detail {

ULA_PORT_HD inline float clamp_f(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

ULA_PORT_HD inline float max_f(float a, float b) { return a > b ? a : b; }

ULA_PORT_HD inline float abs_f(float x) { return x < 0.0f ? -x : x; }

ULA_PORT_HD inline float finite_abs(float x) {
  return std::isfinite(x) ? abs_f(x) : 0.0f;
}

ULA_PORT_HD inline float non_negative_finite(float x) {
  return (std::isfinite(x) && x >= 0.0f) ? x : 0.0f;
}

ULA_PORT_HD inline float map_range(float x, float in_min, float in_max,
                                   float out_min, float out_max) {
  if (in_max == in_min) {
    return out_min;
  }
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

ULA_PORT_HD inline float wrap_unit_interval(float value) {
  float wrapped = std::fmod(value, 1.0f);
  if (wrapped < 0.0f) {
    wrapped += 1.0f;
  }
  return wrapped;
}

ULA_PORT_HD inline float cubic_bezier(float y0, float yf, float x) {
  const float y_diff = yf - y0;
  const float bezier = x * x * x + 3.0f * (x * x * (1.0f - x));
  return y0 + bezier * y_diff;
}

ULA_PORT_HD inline float cubic_bezier_d1(float y0, float yf, float x) {
  const float y_diff = yf - y0;
  const float bezier = 6.0f * x * (1.0f - x);
  return bezier * y_diff;
}

ULA_PORT_HD inline float cubic_bezier_d2(float y0, float yf, float x) {
  const float y_diff = yf - y0;
  const float bezier = 6.0f - 12.0f * x;
  return bezier * y_diff;
}

ULA_PORT_HD inline void update_gait_leg_states(GaitSchedulerState& st) {
  for (int leg = 0; leg < kNumLegs; ++leg) {
    const float duration = st.contact_durations[leg];
    float contact_progress = 0.0f;
    if (duration > 1e-5f) {
      const float shifted = wrap_unit_interval(st.theta - st.phase_offsets[leg]);
      if (shifted < duration) {
        contact_progress = shifted / duration;
      }
      st.contact_state[leg] = (shifted < duration) ? 1.0f : 0.0f;
    } else {
      st.contact_state[leg] = 0.0f;
    }
    st.contact_phase[leg] = contact_progress;

    const float swing_duration = st.swing_durations[leg];
    const float swing_offset =
        wrap_unit_interval(st.phase_offsets[leg] + st.contact_durations[leg]);
    float swing_progress = 0.0f;
    if (swing_duration > 1e-5f) {
      const float shifted = wrap_unit_interval(st.theta - swing_offset);
      if (shifted < swing_duration) {
        swing_progress = shifted / swing_duration;
      }
    }
    st.swing_phase[leg] = swing_progress;
  }
}

/// Converts a body->world quaternion (w,x,y,z) to Rwb (body->world 3x3).
ULA_PORT_HD inline void quat_wxyz_to_Rwb(const float (&q_wxyz)[4],
                                         float (&Rwb)[3][3]) {
  float w = q_wxyz[0];
  float x = q_wxyz[1];
  float y = q_wxyz[2];
  float z = q_wxyz[3];
  const float norm_sq = w * w + x * x + y * y + z * z;
  if (norm_sq > 0.0f && std::isfinite(norm_sq)) {
    const float inv_norm = 1.0f / std::sqrt(norm_sq);
    w *= inv_norm;
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
  } else {
    w = 1.0f;
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
  }

  const float tx = 2.0f * x;
  const float ty = 2.0f * y;
  const float tz = 2.0f * z;
  const float twx = tx * w;
  const float twy = ty * w;
  const float twz = tz * w;
  const float txx = tx * x;
  const float txy = ty * x;
  const float txz = tz * x;
  const float tyy = ty * y;
  const float tyz = tz * y;
  const float tzz = tz * z;

  Rwb[0][0] = 1.0f - (tyy + tzz);
  Rwb[0][1] = txy - twz;
  Rwb[0][2] = txz + twy;
  Rwb[1][0] = txy + twz;
  Rwb[1][1] = 1.0f - (txx + tzz);
  Rwb[1][2] = tyz - twx;
  Rwb[2][0] = txz - twy;
  Rwb[2][1] = tyz + twx;
  Rwb[2][2] = 1.0f - (txx + tyy);
}

ULA_PORT_HD inline void mat3_mul_vec3(const float (&m)[3][3],
                                      const float (&v)[3],
                                      float (&out)[3]) {
  using RowMajor3 = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
  Eigen::Map<Eigen::Vector3f> result(out);
  result.noalias() = Eigen::Map<const RowMajor3>(&m[0][0]) * Eigen::Map<const Eigen::Vector3f>(v);
}

ULA_PORT_HD inline void mat3_transpose_mul_vec3(const float (&m)[3][3],
                                                const float (&v)[3],
                                                float (&out)[3]) {
  using RowMajor3 = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
  Eigen::Map<Eigen::Vector3f> result(out);
  result.noalias() = Eigen::Map<const RowMajor3>(&m[0][0]).transpose() * Eigen::Map<const Eigen::Vector3f>(v);
}

}  // namespace detail

/// Sanitize and zero-fill normalized policy actions (mirrors RLPipelineRuntime::setPolicyAction).
ULA_PORT_HD inline int sanitize_policy_action(const float* raw_action,
                                              int action_count,
                                              float (&out_action)[kPolicyActionDim]) {
  for (int i = 0; i < kPolicyActionDim; ++i) {
    out_action[i] = 0.0f;
  }
  if (!raw_action || action_count <= 0) {
    return 0;
  }
  const int count = action_count < kPolicyActionDim ? action_count : kPolicyActionDim;
  for (int i = 0; i < count; ++i) {
    const float v = raw_action[i];
    out_action[i] = std::isfinite(v) ? v : 0.0f;
  }
  return count;
}

/// Pure fixed-size action decoding (mirrors MdlRLCommandSource::applyCommand_).
ULA_PORT_HD inline DecodedCommand decode_action(const ActionDecoderConfig& cfg,
                                                const float (&base_command)[3],
                                                const float* raw_action,
                                                int action_count) {
  DecodedCommand out{};
  const bool use_fixed = (cfg.mode == CommandMode::kTomlFixed);

  if (use_fixed) {
    for (int i = 0; i < 3; ++i) {
      out.desired_velocity[i] = cfg.fixed_desired_velocity[i];
    }
    for (int leg = 0; leg < kNumLegs; ++leg) {
      for (int axis = 0; axis < 3; ++axis) {
        out.foot_forces[leg][axis] = cfg.fixed_foot_forces[leg][axis];
        out.footstep_residuals[leg][axis] = cfg.fixed_footstep_residuals[leg][axis];
      }
    }
    out.height_residual = 0.0f;
    out.phase_delta = cfg.fixed_phase_delta;
    return out;
  }

  float sanitized[kPolicyActionDim]{};
  const int count = sanitize_policy_action(raw_action, action_count, sanitized);

  if (count >= kPolicyActionDim) {
    const float residual_vx = detail::map_range(
        sanitized[0], -1.0f, 1.0f, -cfg.policy_linear_limit, cfg.policy_linear_limit);
    const float residual_vy = detail::map_range(
        sanitized[1], -1.0f, 1.0f, -cfg.policy_linear_limit, cfg.policy_linear_limit);
    const float residual_yaw = detail::map_range(
        sanitized[2], -1.0f, 1.0f, -cfg.policy_yaw_limit, cfg.policy_yaw_limit);

    out.desired_velocity[0] = base_command[0] + residual_vx;
    out.desired_velocity[1] = base_command[1] + residual_vy;
    out.desired_velocity[2] = base_command[2] + residual_yaw;

    for (int leg = 0; leg < kNumLegs; ++leg) {
      const int force_base = 3 + leg * 3;
      const int residual_base = 15 + leg * 2;
      out.foot_forces[leg][0] = detail::map_range(
          sanitized[force_base + 0], -1.0f, 1.0f, -cfg.force_xy_limit, cfg.force_xy_limit);
      out.foot_forces[leg][1] = detail::map_range(
          sanitized[force_base + 1], -1.0f, 1.0f, -cfg.force_xy_limit, cfg.force_xy_limit);
      out.foot_forces[leg][2] = detail::map_range(
          sanitized[force_base + 2], -1.0f, 1.0f, 0.0f, cfg.force_z_max);
      out.footstep_residuals[leg][0] = sanitized[residual_base + 0] * cfg.residual_xy_scale;
      out.footstep_residuals[leg][1] = sanitized[residual_base + 1] * cfg.residual_xy_scale;
      out.footstep_residuals[leg][2] = 0.0f;
    }

    out.height_residual = detail::map_range(
        sanitized[23], -1.0f, 1.0f, -cfg.height_residual_limit, cfg.height_residual_limit);

    float pd_min = cfg.phase_delta_min;
    float pd_max = cfg.phase_delta_max;
    if (pd_max < pd_min) {
      const float tmp = pd_min;
      pd_min = pd_max;
      pd_max = tmp;
    }
    out.phase_delta = detail::map_range(sanitized[kPolicyActionDim - 1], -1.0f, 1.0f,
                                        pd_min, pd_max);
  } else {
    // Baseline fallback before 25-dim policy action is supplied (e.g. reset priming ticks).
    out.desired_velocity[0] = base_command[0];
    out.desired_velocity[1] = base_command[1];
    out.desired_velocity[2] = base_command[2];
    out.height_residual = 0.0f;
    out.phase_delta = kDefaultPhaseDelta;
  }

  float base_vx_cap = 0.0f;
  float base_vy_cap = 0.0f;
  float base_yaw_cap = 0.0f;
  if (cfg.policy_random_velocity_enabled) {
    base_vx_cap = detail::max_f(detail::finite_abs(cfg.policy_random_vx_min),
                                detail::finite_abs(cfg.policy_random_vx_max));
    base_vy_cap = detail::max_f(detail::finite_abs(cfg.policy_random_vy_min),
                                detail::finite_abs(cfg.policy_random_vy_max));
    base_yaw_cap = detail::max_f(detail::finite_abs(cfg.policy_random_yaw_min),
                                 detail::finite_abs(cfg.policy_random_yaw_max));
  } else {
    base_vx_cap = detail::finite_abs(base_command[0]);
    base_vy_cap = detail::finite_abs(base_command[1]);
    base_yaw_cap = detail::finite_abs(base_command[2]);
  }

  const float vx_abs_cap = base_vx_cap + detail::non_negative_finite(cfg.policy_linear_limit);
  const float vy_abs_cap = base_vy_cap + detail::non_negative_finite(cfg.policy_linear_limit);
  const float yaw_abs_cap = base_yaw_cap + detail::non_negative_finite(cfg.policy_yaw_limit);

  out.desired_velocity[0] = detail::clamp_f(out.desired_velocity[0], -vx_abs_cap, vx_abs_cap);
  out.desired_velocity[1] = detail::clamp_f(out.desired_velocity[1], -vy_abs_cap, vy_abs_cap);
  out.desired_velocity[2] = detail::clamp_f(out.desired_velocity[2], -yaw_abs_cap, yaw_abs_cap);

  return out;
}

/// Reset gait scheduler state (mirrors MdlCtGaitScheduler::reset_).
ULA_PORT_HD inline void reset_gait_scheduler(GaitSchedulerState& st,
                                             const GaitSchedulerConfig& cfg) {
  const float dt = cfg.dt_s > 0.0f ? cfg.dt_s : kDefaultDtSeconds;
  st.cycle_time_seconds = detail::max_f(dt * 40.0f, 1e-4f);
  st.time_into_cycle = 0.0f;
  st.theta = 0.0f;
  st.has_schedule = false;
  for (int leg = 0; leg < kNumLegs; ++leg) {
    st.phase_offsets[leg] = detail::wrap_unit_interval(cfg.phase_offsets[leg]);
    st.contact_durations[leg] = detail::clamp_f(cfg.contact_durations[leg], 0.0f, 1.0f);
    st.swing_durations[leg] = 1.0f - st.contact_durations[leg];
  }
  detail::update_gait_leg_states(st);
}

/// Advance gait scheduler by one tick (mirrors MdlCtGaitScheduler::update).
ULA_PORT_HD inline GaitSchedule step_gait_scheduler(GaitSchedulerState& st,
                                                    float phase_delta_input,
                                                    bool has_phase_delta,
                                                    float dt_s) {
  float delta_theta = has_phase_delta ? phase_delta_input : kDefaultPhaseDelta;
  delta_theta = detail::clamp_f(delta_theta, 0.0f, kMaxPhaseDelta);

  const float dt = detail::max_f(dt_s > 0.0f ? dt_s : kDefaultDtSeconds, 1e-6f);
  const float safe_delta = detail::max_f(delta_theta, 1e-5f);

  const float inst_cycle = detail::clamp_f(dt / safe_delta, 0.02f, 2.0f);
  constexpr float kAlpha = 0.1f;
  st.cycle_time_seconds = (1.0f - kAlpha) * st.cycle_time_seconds + kAlpha * inst_cycle;

  const float prev_theta = st.theta;
  st.theta = detail::wrap_unit_interval(st.theta + delta_theta);

  st.time_into_cycle += dt;
  if (st.theta < prev_theta) {
    const float measured = detail::clamp_f(st.time_into_cycle, 0.02f, 2.0f);
    constexpr float kBeta = 0.5f;
    st.cycle_time_seconds = (1.0f - kBeta) * st.cycle_time_seconds + kBeta * measured;
    st.time_into_cycle = 0.0f;
  }

  detail::update_gait_leg_states(st);
  st.has_schedule = true;

  GaitSchedule schedule{};
  for (int leg = 0; leg < kNumLegs; ++leg) {
    schedule.contact_state[leg] = st.contact_state[leg];
    schedule.planned_contact[leg] = st.contact_state[leg] > 0.0f;
    schedule.contact_phase[leg] = st.contact_phase[leg];
    schedule.swing_phase[leg] = st.swing_phase[leg];
    schedule.stance_duration[leg] = st.contact_durations[leg] * st.cycle_time_seconds;
    schedule.swing_duration[leg] = st.swing_durations[leg] * st.cycle_time_seconds;
  }
  schedule.gait_phase = st.theta;
  schedule.gait_cycle_time = st.cycle_time_seconds;
  schedule.delta_theta = delta_theta;
  return schedule;
}

/// Hip location in body frame (mirrors Quadruped<float>::getHipLocation).
ULA_PORT_HD inline void get_hip_location(const LegKinematicsConfig& cfg,
                                         int leg,
                                         float (&hip)[3]) {
  hip[0] = (leg == 0 || leg == 1) ? cfg.abad_location[0] : -cfg.abad_location[0];
  hip[1] = (leg == 1 || leg == 3) ? cfg.abad_location[1] : -cfg.abad_location[1];
  hip[2] = cfg.abad_location[2];
}

/// Foot position in hip/leg frame (mirrors computeLegJacobianAndPosition<float>).
ULA_PORT_HD inline void compute_leg_foot_position(const LegKinematicsConfig& cfg,
                                                  int leg,
                                                  const float (&q)[3],
                                                  float (&p_local)[3]) {
  const float side_sign = (leg == 1 || leg == 3) ? 1.0f : -1.0f;
  const float front_sign = (leg < 2) ? 1.0f : -1.0f;

  const float s1 = std::sin(q[0]);
  const float s2 = std::sin(q[1]);
  const float s3 = std::sin(q[2]);
  const float c1 = std::cos(q[0]);
  const float c2 = std::cos(q[1]);
  const float c3 = std::cos(q[2]);

  const float c23 = c2 * c3 - s2 * s3;
  const float s23 = s2 * c3 + c2 * s3;
  const float l14 = cfg.abad_link_length + cfg.knee_link_y_offset;

  p_local[0] = cfg.knee_link_length * s23 + cfg.hip_link_length * s2 +
               cfg.abad_x_offset * front_sign;
  p_local[1] = l14 * side_sign * c1 + cfg.knee_link_length * (s1 * c23) +
               cfg.hip_link_length * c2 * s1;
  p_local[2] = l14 * side_sign * s1 - cfg.knee_link_length * (c1 * c23) -
               cfg.hip_link_length * c1 * c2;
}

/// Bezier swing trajectory evaluation (mirrors FootSwingTrajectory<float>::computeSwingTrajectoryBezier).
ULA_PORT_HD inline void compute_swing_trajectory_bezier(const FootSwingState& traj,
                                                        float phase,
                                                        float swing_time,
                                                        float (&pos)[3],
                                                        float (&vel)[3],
                                                        float (&acc)[3]) {
  const float clamped_phase = detail::clamp_f(phase, 0.0f, 1.0f);
  const float inv_t = 1.0f / swing_time;
  const float inv_t2 = inv_t * inv_t;

  for (int axis = 0; axis < 3; ++axis) {
    pos[axis] = detail::cubic_bezier(traj.p0[axis], traj.pf[axis], clamped_phase);
    vel[axis] = detail::cubic_bezier_d1(traj.p0[axis], traj.pf[axis], clamped_phase) * inv_t;
    acc[axis] = detail::cubic_bezier_d2(traj.p0[axis], traj.pf[axis], clamped_phase) * inv_t2;
  }

  if (clamped_phase < 0.5f) {
    const float t = clamped_phase * 2.0f;
    pos[2] = detail::cubic_bezier(traj.p0[2], traj.p0[2] + traj.height, t);
    vel[2] = detail::cubic_bezier_d1(traj.p0[2], traj.p0[2] + traj.height, t) * 2.0f * inv_t;
    acc[2] = detail::cubic_bezier_d2(traj.p0[2], traj.p0[2] + traj.height, t) * 4.0f * inv_t2;
  } else {
    const float t = detail::clamp_f(clamped_phase * 2.0f - 1.0f, 0.0f, 1.0f);
    pos[2] = detail::cubic_bezier(traj.p0[2] + traj.height, traj.pf[2], t);
    vel[2] = detail::cubic_bezier_d1(traj.p0[2] + traj.height, traj.pf[2], t) * 2.0f * inv_t;
    acc[2] = detail::cubic_bezier_d2(traj.p0[2] + traj.height, traj.pf[2], t) * 4.0f * inv_t2;
  }
}

/// Reset footstep planner state (mirrors MdlFootstepPlanner::reset_).
ULA_PORT_HD inline void reset_footstep_planner(FootstepPlannerState& st,
                                               const FootstepPlannerConfig& cfg) {
  for (int leg = 0; leg < kNumLegs; ++leg) {
    st.prev_contact_state[leg] = 1.0f;
    for (int axis = 0; axis < 3; ++axis) {
      st.swing_traj[leg].p0[axis] = 0.0f;
      st.swing_traj[leg].pf[axis] = 0.0f;
    }
    st.swing_traj[leg].height = cfg.swing_height;
  }
  st.has_targets = false;
}

/// Advance footstep planner by one tick (mirrors MdlFootstepPlanner::update).
ULA_PORT_HD inline FootstepTargets step_footstep_planner(
    FootstepPlannerState& st,
    const FootstepPlannerConfig& cfg,
    const GaitSchedule& gait,
    const float (&desired_velocity)[3],
    const float (&footstep_residuals)[kNumLegs][3],
    const PlannerMeasurements& plant) {
  const auto& Rwb = plant.rotation_wb;
  const auto& v_world = plant.linear_velocity_w;
  const auto& position_w = plant.position_w;

  const float v_body_des[3] = {desired_velocity[0], desired_velocity[1], 0.0f};
  float v_des_world[3]{};
  detail::mat3_mul_vec3(Rwb, v_body_des, v_des_world);
  const float yaw_rate_des = desired_velocity[2];

  FootstepTargets targets{};

  for (int leg = 0; leg < kNumLegs; ++leg) {
    const auto& foot_local = plant.foot_position_leg[leg];

    float hip[3]{};
    get_hip_location(cfg.kinematics, leg, hip);

    const float foot_body[3] = {hip[0] + foot_local[0],
                                hip[1] + foot_local[1],
                                hip[2] + foot_local[2]};
    float foot_world_rel[3]{};
    detail::mat3_mul_vec3(Rwb, foot_body, foot_world_rel);
    const float foot_world[3] = {position_w[0] + foot_world_rel[0],
                                 position_w[1] + foot_world_rel[1],
                                 position_w[2] + foot_world_rel[2]};

    const bool was_in_contact = st.prev_contact_state[leg] > 0.0f;
    const bool leg_in_contact = gait.contact_state[leg] > 0.0f;
    if (was_in_contact && !leg_in_contact) {
      for (int axis = 0; axis < 3; ++axis) {
        st.swing_traj[leg].p0[axis] = foot_world[axis];
      }
      st.swing_traj[leg].height = cfg.swing_height;
    }

    if (!leg_in_contact) {
      const float stance_duration = gait.stance_duration[leg];
      const float swing_duration = detail::max_f(gait.swing_duration[leg], 1e-3f);
      const float clamped_phase = detail::clamp_f(gait.swing_phase[leg], 0.0f, 1.0f);
      const float swing_time_remaining =
          detail::max_f((1.0f - clamped_phase) * swing_duration, 0.0f);

      const float yaw_angle = -yaw_rate_des * stance_duration * 0.5f;
      const Eigen::AngleAxisf yaw_correction(yaw_angle, Eigen::Vector3f::UnitZ());
      const Eigen::Vector3f p_yaw_corrected = yaw_correction * Eigen::Map<const Eigen::Vector3f>(hip);

      float commanded_velocity_body[3]{};
      detail::mat3_transpose_mul_vec3(Rwb, v_des_world, commanded_velocity_body);
      commanded_velocity_body[2] = 0.0f;

      const float nominal_body[3] = {
          p_yaw_corrected[0] + commanded_velocity_body[0] * swing_time_remaining,
          p_yaw_corrected[1] + commanded_velocity_body[1] * swing_time_remaining,
          p_yaw_corrected[2] + commanded_velocity_body[2] * swing_time_remaining,
      };
      float nominal_world_rel[3]{};
      detail::mat3_mul_vec3(Rwb, nominal_body, nominal_world_rel);

      float foot_target_world[3] = {
          position_w[0] + nominal_world_rel[0],
          position_w[1] + nominal_world_rel[1],
          position_w[2] + nominal_world_rel[2],
      };

      float pfx_rel = v_world[0] * 0.5f * stance_duration +
                      0.03f * (v_world[0] - v_des_world[0]) +
                      (0.5f * position_w[2] / cfg.gravity) *
                          (v_world[1] * yaw_rate_des);
      float pfy_rel = v_world[1] * 0.5f * stance_duration +
                      0.03f * (v_world[1] - v_des_world[1]) +
                      (0.5f * position_w[2] / cfg.gravity) *
                          (-v_world[0] * yaw_rate_des);

      pfx_rel = detail::clamp_f(pfx_rel, -cfg.p_rel_max, cfg.p_rel_max) +
                footstep_residuals[leg][0];
      pfy_rel = detail::clamp_f(pfy_rel, -cfg.p_rel_max, cfg.p_rel_max) +
                footstep_residuals[leg][1];

      foot_target_world[0] += pfx_rel;
      foot_target_world[1] += pfy_rel;
      foot_target_world[2] = cfg.foot_target_z;

      for (int axis = 0; axis < 3; ++axis) {
        st.swing_traj[leg].pf[axis] = foot_target_world[axis];
      }
      st.swing_traj[leg].height = cfg.swing_height;

      compute_swing_trajectory_bezier(st.swing_traj[leg], clamped_phase,
                                      swing_duration,
                                      targets.foot_position_target[leg],
                                      targets.foot_velocity_target[leg],
                                      targets.foot_acceleration_target[leg]);
    } else {
      for (int axis = 0; axis < 3; ++axis) {
        targets.foot_position_target[leg][axis] = foot_world[axis];
        targets.foot_velocity_target[leg][axis] = 0.0f;
        targets.foot_acceleration_target[leg][axis] = 0.0f;
      }
    }
  }

  for (int leg = 0; leg < kNumLegs; ++leg) {
    st.prev_contact_state[leg] = gait.contact_state[leg];
  }
  st.has_targets = true;
  return targets;
}

// ULA integrates the rate-limited body motion in world axes, caps position
// error, and aligns a new swing with the current foot before running WBIC.
ULA_PORT_HD inline WholeBodyTargets step_body_command(
    BodyCommandState& state, const ControllerConfig& cfg,
    const PlannerMeasurements& plant, const ControllerOutput& plan) {
  WholeBodyTargets out{};
  out.run_wbic = !state.first_tick;  // WBIC is not ready on the first tick after reset
  if (state.first_tick) {
    state.position_w[0] = plant.position_w[0];
    state.position_w[1] = plant.position_w[1];
    state.first_tick = false;
  }
  const float dt = cfg.gait.dt_s;
  for (int i = 0; i < 2; ++i) {
    state.velocity_b[i] += detail::clamp_f(plan.decoded.desired_velocity[i] - state.velocity_b[i], -0.2f, 0.2f);
  }
  for (int i = 0; i < 2; ++i) {
    out.velocity_w[i] = plant.rotation_wb[i][0] * state.velocity_b[0]
                      + plant.rotation_wb[i][1] * state.velocity_b[1];
    state.position_w[i] = detail::clamp_f(state.position_w[i] + dt * out.velocity_w[i],
                                         plant.position_w[i] - 0.1f, plant.position_w[i] + 0.1f);
    out.position_w[i] = state.position_w[i];
  }
  out.position_w[2] = cfg.body_height + plan.decoded.height_residual;
  out.angular_velocity[2] = plan.decoded.desired_velocity[2];
  out.orientation_rpy[2] = plant.yaw + out.angular_velocity[2] * dt;

  out.foot = plan.footstep;
  for (int leg = 0; leg < kNumLegs; ++leg) {
    const bool contact = plan.gait.planned_contact[leg];
    if (contact) {
      for (int i = 0; i < 3; ++i) out.force_w[leg][i] = plan.decoded.foot_forces[leg][i];
    }
    if (!out.run_wbic) continue;
    if (state.previous_contact[leg] && !contact) {
      float foot_b[3];
      get_hip_location(cfg.footstep.kinematics, leg, foot_b);
      for (int i = 0; i < 3; ++i) foot_b[i] += plant.foot_position_leg[leg][i];
      float foot_w[3];
      detail::mat3_mul_vec3(plant.rotation_wb, foot_b, foot_w);
      for (int i = 0; i < 3; ++i) {
        out.foot.foot_position_target[leg][i] = plant.position_w[i] + foot_w[i];
        out.foot.foot_velocity_target[leg][i] = 0.0f;
        out.foot.foot_acceleration_target[leg][i] = 0.0f;
      }
    }
    state.previous_contact[leg] = contact;
  }
  return out;
}

/// Reset all persistent state for a fresh episode.
ULA_PORT_HD inline void reset(ControllerState& state,
                              const ControllerConfig& cfg) {
  reset_gait_scheduler(state.gait, cfg.gait);
  reset_footstep_planner(state.footstep, cfg.footstep);
  state.body = BodyCommandState{};
}

/// Execute one 500 Hz planner tick through WBIC target generation.
ULA_PORT_HD inline ControllerOutput step(ControllerState& state,
                                         const ControllerConfig& cfg,
                                         const ControllerInput& in) {
  ControllerOutput out{};
  out.decoded = decode_action(cfg.decoder, in.base_command, in.action, in.action_count);
  out.gait = step_gait_scheduler(state.gait, out.decoded.phase_delta, true, cfg.gait.dt_s);
  out.footstep = step_footstep_planner(
      state.footstep, cfg.footstep, out.gait, out.decoded.desired_velocity,
      out.decoded.footstep_residuals, in.plant);
  out.wbic = step_body_command(state.body, cfg, in.plant, out);
  return out;
}

}  // namespace ula::port

#endif  // ULA_GPU_PORT_CONTROLLER_HH
