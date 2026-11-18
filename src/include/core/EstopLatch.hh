/* 
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 * 
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium, is
 * strictly prohibited.
*/

#ifndef QUADCONTROL_CORE_ESTOPLATCH_HH
#define QUADCONTROL_CORE_ESTOPLATCH_HH

#include <cstdint>

namespace quadruped {

// E-stop reason bits. These are the driver's estop_flags log contract, so
// the values are part of the observable output -- do not renumber.
namespace estop {
constexpr uint32_t kStateStale = 1u << 0;
constexpr uint32_t kRemote = 1u << 1;     // operator stop / deadman
constexpr uint32_t kOverrun = 1u << 2;    // loop stall or overrun streak
constexpr uint32_t kFall = 1u << 3;       // fall detection
}  // namespace estop

/**
 * Backend-neutral emergency-stop latch, in the same spirit as
 * RemoteStateProvider: safety logic calls this without knowing the hardware
 * backend. The unitreehw driver implements it; while any bit is latched the
 * backend publishes damping commands only.
 */
class EstopLatch {
 public:
  virtual ~EstopLatch() = default;

  /** \brief Latch the given estop reason bits (OR-ed into the current set). */
  virtual void latchEstop( uint32_t flags ) = 0;
};

}  // namespace quadruped

#endif  // QUADCONTROL_CORE_ESTOPLATCH_HH
