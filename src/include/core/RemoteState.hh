/*
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 *
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium, is
 * strictly prohibited.
 */

#ifndef REMOTESTATE_HH
#define REMOTESTATE_HH

#include <cstdint>

namespace quadruped {

/** \brief Decoded state of the handheld remote controller.
 *
 * Filled by hardware backends that receive a remote (e.g. the Go2 driver
 * decodes LowState.wireless_remote). Units and axes follow the Unitree
 * joystick layout; sticks are in [-1, 1].
 */
struct RemoteState {
  bool fresh = false;  // data received within the provider's timeout
  uint16_t buttons = 0;
  float lx = 0.0f;  // left stick x
  float ly = 0.0f;  // left stick y
  float rx = 0.0f;  // right stick x
  float ry = 0.0f;  // right stick y
  float l2 = 0.0f;  // L2 analog trigger
  double t = -1.0;  // last frame time (s, manager timebase); <0 = never
};

// Unitree wireless controller button bits (unitree_joystick.hpp layout).
constexpr uint16_t kRemoteBtnR1 = 1u << 0;
constexpr uint16_t kRemoteBtnL1 = 1u << 1;
constexpr uint16_t kRemoteBtnStart = 1u << 2;
constexpr uint16_t kRemoteBtnSelect = 1u << 3;
constexpr uint16_t kRemoteBtnR2 = 1u << 4;
constexpr uint16_t kRemoteBtnL2 = 1u << 5;
constexpr uint16_t kRemoteBtnF1 = 1u << 6;
constexpr uint16_t kRemoteBtnF2 = 1u << 7;
constexpr uint16_t kRemoteBtnA = 1u << 8;
constexpr uint16_t kRemoteBtnB = 1u << 9;
constexpr uint16_t kRemoteBtnX = 1u << 10;
constexpr uint16_t kRemoteBtnY = 1u << 11;
constexpr uint16_t kRemoteBtnUp = 1u << 12;
constexpr uint16_t kRemoteBtnRight = 1u << 13;
constexpr uint16_t kRemoteBtnDown = 1u << 14;
constexpr uint16_t kRemoteBtnLeft = 1u << 15;

/** \brief Interface through which modules read the remote state.
 *
 * Implemented by hardware driver modules (e.g. MdlUnitreeDriver) and located
 * via ModuleManager::findModule + dynamic_cast, so the module stack stays
 * independent of any particular hardware backend.
 */
class RemoteStateProvider {
 public:
  virtual ~RemoteStateProvider() = default;

  /** \brief Copies the latest remote state into `out`.
   * \return false if no remote data has ever been received.
   */
  virtual bool getRemoteState( RemoteState& out ) = 0;
};

}  // namespace quadruped

#endif  // REMOTESTATE_HH
