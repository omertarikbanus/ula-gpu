/*
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 *
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium, is
 * strictly prohibited.
 */

#ifndef CONTROLLERPERIOD_HH
#define CONTROLLERPERIOD_HH

#include "rtcore/ClockHW.hh"

namespace rtcore {
class ModuleManager;
}

namespace quadruped {

/** \brief Fixes the ModuleManager step period from config, for all backends.
 *
 * Sets the module update period from [control_parameters] controller_dt.
 * The control period is a property of the controller, fixed in config and
 * identical for every hardware backend; backends may validate it (e.g. the
 * Go2 low-level stream requires 500 Hz) but never negotiate it. Both
 * supervisor/main.cc and RLPipelineRuntime call this so the two pipelines
 * cannot diverge in control rate, and so hardware backends adopt the
 * configured period too (previously the adoption was sim-only).
 *
 * \param mm ModuleManager whose step period is set.
 * \param fallback_us Period applied when controller_dt is absent from
 *        config (>0). The MuJoCo backend passes its model timestep here to
 *        preserve the historical fallback; the period is a property of the
 *        controller, not of the integrator, so the config key always wins.
 * \return The effective step period in microseconds.
 */
rtcore::CLOCK applyControllerPeriod( rtcore::ModuleManager* mm,
                                     rtcore::CLOCK fallback_us = 0 );

}  // namespace quadruped

#endif  // CONTROLLERPERIOD_HH
