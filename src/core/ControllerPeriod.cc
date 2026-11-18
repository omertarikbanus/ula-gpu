/*
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 *
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium, is
 * strictly prohibited.
 */

#include "core/ControllerPeriod.hh"

#include "rtcore/ConfigTable.hh"
#include "rtcore/ModuleManager.hh"

namespace quadruped {

rtcore::CLOCK applyControllerPeriod( rtcore::ModuleManager* mm,
                                     rtcore::CLOCK fallback_us ) {
  if ( mm == nullptr ) return 0;

  rtcore::ConfigTable control_config;
  double controller_dt = 0.0;
  if ( mm->getConfigTable( "control_parameters", control_config ) ) {
    controller_dt = control_config.getDouble( "controller_dt", 0.0 );
  }

  rtcore::CLOCK period = 0;
  if ( controller_dt > 0.0 ) {
    period = static_cast<rtcore::CLOCK>( controller_dt * 1e6 );
  } else if ( fallback_us > 0 ) {
    period = fallback_us;
  }

  if ( period > 0 ) mm->setStepPeriod( period );
  return mm->getStepPeriod();
}

}  // namespace quadruped
