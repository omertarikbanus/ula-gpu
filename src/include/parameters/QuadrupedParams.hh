/* 
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 *
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium is
 * strictly prohibited.
 */

#ifndef QUADRUPEDPARAMS_HH
#define QUADRUPEDPARAMS_HH

#include "dynamics/QuadrupedKinematics.hh"

namespace quadruped {

/**
 * @brief Helper factory for the default Unitree Go2 kinematic configuration.
 */
QuadrupedKinematics::params_t createGo2Config();

}  // namespace quadruped

#endif  // QUADRUPEDPARAMS_HH

