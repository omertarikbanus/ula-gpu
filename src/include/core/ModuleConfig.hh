/* 
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 * 
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium is
 * strictly prohibited.
*/

#ifndef MODULECONFIG_HH
#define MODULECONFIG_HH

#include "rtcore/ClockHW.hh"
#include "rtcore/ModuleManager.hh"

#include <map>
#include <string>

typedef struct {
  std::string name;
  int index;        // Use -1 to cover all indices
} ModuleId_t;

struct ModuleCompare_t  {
  bool operator()(const ModuleId_t& m1, const ModuleId_t& m2)const {
    return ((m1.name < m2.name)
            || (m1.name == m2.name
                && m1.index != -1
                && m2.index != -1
                && (m1.index < m2.index)));
  }
};

typedef struct {
  unsigned int period;
  unsigned int offset;
  unsigned int order;
} ModuleConfig_t;

const std::map<ModuleId_t, ModuleConfig_t, ModuleCompare_t> moduleConfig = {
  // Sim HW related modules
  { {"MdlSimDriver", -1},  { 1, 0, ACTUATOR_MODULES } },

  // Robot HW related modules
  { {"MdlMotorMaster", -1},  { 500, 0, ACTUATOR_MODULES } },
  { {"MdlUnitreeDriver", -1}, { 1, 0, ACTUATOR_MODULES } },

  // Bench command generator (go2bench low-level testing tool)
  { {"MdlBench", -1}, { 1, 0, USER_CONTROLLERS } },

  // Core modules
  { {"LogServer", -1},        { 1, 0, LOGGING_MODULES } },
  { {"MdlControlParams", -1}, { 1, 0, SENSING_MODULES } },
  { {"MdlLegController", -1}, { 1, 0, SENSING_MODULES + 1 } },
  { {"MdlStateEstimator", -1}, { 1, 0, SENSING_MODULES + 2 } },
  { {"MdlDesiredState", -1},  { 1, 0, USER_CONTROLLERS } },
  { {"MdlConvexMPC", -1},     { 1, 0, USER_CONTROLLERS + 1 } },
  { {"MdlCtGaitScheduler", -1}, { 1, 0, USER_CONTROLLERS + 2 } },
  { {"MdlFootstepPlanner", -1}, { 1, 0, USER_CONTROLLERS + 3 } },
  { {"MdlRLLocomotionState", -1}, { 1, 0, USER_CONTROLLERS + 4 } },
  // RL deployment (see docs/GO2_HARDWARE_MODE_PLAN.md §7): the policy runs
  // after state estimation (12) and before its consumers -- MdlRLJointPD
  // (joint_pd line) or the MdlRLCommandSource -> gait (10002) chain
  // (residual line), hence the 94xx-96xx slots below USER_CONTROLLERS.
  { {"MdlRLPolicy", -1},        { 1, 0, 9400 } },
  { {"MdlRLCommandSource", -1}, { 1, 0, 9500 } },
  { {"MdlRLJointPD", -1},       { 1, 0, 9600 } },
  // Safety gate (audit B3, plan §6.2 layer 2): fall detection + deadman.
  // Runs after sensing/the RL chain and before the command consumers, so
  // its estop latch reaches the driver (order 30000) within the same tick.
  { {"MdlGo2Safety", -1},       { 1, 0, 9900 } },
  { {"MdlStandUp", -1},       { 1, 0, BEHAVIORAL_CONTROLLERS } },
  { {"MdlWBIC", -1},          { 1, 0, LOW_LEVEL_CONTROLLERS } },
};


#endif
