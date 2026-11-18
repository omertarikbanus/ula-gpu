/*
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 *
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium is
 * strictly prohibited.
 */

#ifndef QUADRUPED_CONTROL_MESSAGES_HH
#define QUADRUPED_CONTROL_MESSAGES_HH

#include <array>
#include <cstdint>
#include <cstring>

#include "controllers/LegController.h"
#include "input/GamepadCommand.h"
#include "input/rt_rc_interface.h"
#include "types/cppTypes.h"

namespace quadruped {

enum class ControlMode : int {
  kPassive = 0,
  kStandUp = 1,
  kBalanceStand = 3,
  kLocomotion = 4,
  kRlLocomotion = 5,
  kInvalid = 100,
};

inline ControlMode toControlMode(double mode_value) {
  return static_cast<ControlMode>(static_cast<int>(mode_value));
}

struct DesiredCommand {
  GamepadCommand gamepad{};
  rc_control_settings rc{};
  Vec2<float> leftAnalog = Vec2<float>::Zero();
  Vec2<float> rightAnalog = Vec2<float>::Zero();
  bool trigger_pressed = false;

  void reset() {
    gamepad.zero();
    std::memset(&rc, 0, sizeof(rc));
    leftAnalog.setZero();
    rightAnalog.setZero();
    trigger_pressed = false;
  }
};

struct RlDesiredVelocity {
  Vec3<float> body_velocity = Vec3<float>::Zero();  // [vx, vy, yaw_rate]
  // Policy offset on the commanded stance height (m). Zero for every non-RL
  // path and for toml_fixed mode, so the baseline keeps a constant height.
  float height_residual = 0.0f;
};

struct FootForceTargets {
  Vec3<float> forces[4];

  void reset() {
    for (auto& force : forces) {
      force.setZero();
    }
  }
};

struct FootstepResiduals {
  Vec3<float> offsets[4];

  void reset() {
    for (auto& offset : offsets) {
      offset.setZero();
    }
  }
};

struct GaitSchedule {
  std::array<float, 4> contact_state{{1.f, 1.f, 1.f, 1.f}};
  std::array<float, 4> contact_phase{{0.f, 0.f, 0.f, 0.f}};
  std::array<float, 4> swing_phase{{0.f, 0.f, 0.f, 0.f}};
  std::array<float, 4> stance_duration{{0.f, 0.f, 0.f, 0.f}};
  std::array<float, 4> swing_duration{{0.f, 0.f, 0.f, 0.f}};
  float gait_phase = 0.0f;
  float gait_cycle_time = 0.0f;
  float delta_theta = 0.0f;
};

struct FootstepTargets {
  Vec3<float> pFoot_des[4];
  Vec3<float> vFoot_des[4];
  Vec3<float> aFoot_des[4];

  void reset() {
    for (int i = 0; i < 4; ++i) {
      pFoot_des[i].setZero();
      vFoot_des[i].setZero();
      aFoot_des[i].setZero();
    }
  }
};

struct LegCommand {
  LegControllerCommand<float> commands[4];
  ControlMode source_mode = ControlMode::kInvalid;
  double timestamp = 0.0;

  void reset() {
    for (auto& cmd : commands) {
      cmd.zero();
    }
    source_mode = ControlMode::kInvalid;
    timestamp = 0.0;
  }
};

struct WbcConfig {
  float floating_base_weight = 0.1f;
};

struct ModeCommand {
  ControlMode mode = ControlMode::kInvalid;
};

struct HealthStatus {
  bool ok = false;
  double timestamp = 0.0;
};

}  // namespace quadruped

#endif  // QUADRUPED_CONTROL_MESSAGES_HH
