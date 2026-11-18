/* 
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 * 
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium is
 * strictly prohibited.
*/

#include "rtcore/ModuleManager.hh"
#include "rtcore/LogServer.hh"

#include "core/CoreModules.hh"
#include "modules/MdlStateEstimator.hh"
#include "core/ModuleDefs.hh"
#include "modules/MdlControlParams.hh"
#include "modules/MdlConvexMPC.hh"
#include "modules/MdlDesiredState.hh"
#include "modules/MdlLegController.hh"
#include "modules/MdlCtGaitScheduler.hh"
#include "modules/MdlFootstepPlanner.hh"
#include "modules/MdlRLJointPD.hh"
#include "modules/MdlGo2Safety.hh"
#include "modules/MdlRLPolicy.hh"
#include "modules/MdlStandUp.hh"
#include "modules/MdlWBIC.hh"

using namespace rtcore;
using quadruped::MdlStateEstimator;

static LogServer* _logserver = nullptr;
static quadruped::MdlControlParams* _controlParams = nullptr;
static quadruped::MdlLegController* _legController = nullptr;
static MdlStateEstimator* _stateEstimator = nullptr;
static quadruped::MdlDesiredState* _desiredState = nullptr;
static quadruped::MdlConvexMPC* _convexMpc = nullptr;
static quadruped::MdlCtGaitScheduler* _ctGaitScheduler = nullptr;
static quadruped::MdlFootstepPlanner* _footstepPlanner = nullptr;
static quadruped::MdlWBIC* _wbic = nullptr;
static quadruped::MdlStandUp* _standUp = nullptr;
static quadruped::MdlRLPolicy* _rlPolicy = nullptr;
static quadruped::MdlRLJointPD* _rlJointPD = nullptr;
static quadruped::MdlGo2Safety* _go2Safety = nullptr;

void AddCoreModules( ModuleManager *mgr ) {
  CREATE_MODULE(mgr, LogServer, _logserver );
  CREATE_MODULE(mgr, quadruped::MdlControlParams, _controlParams);
  CREATE_MODULE(mgr, quadruped::MdlLegController, _legController);
  CREATE_MODULE(mgr, MdlStateEstimator, _stateEstimator );
  CREATE_MODULE(mgr, quadruped::MdlDesiredState, _desiredState);
  CREATE_MODULE(mgr, quadruped::MdlConvexMPC, _convexMpc);
  CREATE_MODULE(mgr, quadruped::MdlCtGaitScheduler, _ctGaitScheduler);
  CREATE_MODULE(mgr, quadruped::MdlFootstepPlanner, _footstepPlanner);
  CREATE_MODULE(mgr, quadruped::MdlWBIC, _wbic);
  CREATE_MODULE(mgr, quadruped::MdlStandUp, _standUp);
  // RL deployment modules: inert unless [rl_policy] is enabled and the
  // Supervisor grabs them in RL_LOCOMOTION (docs/GO2_HARDWARE_MODE_PLAN.md
  // §7). Created unconditionally like the other behavioral modules so the
  // supervisor binary and RLPipelineRuntime share one creation path.
  CREATE_MODULE(mgr, quadruped::MdlRLPolicy, _rlPolicy);
  CREATE_MODULE(mgr, quadruped::MdlRLJointPD, _rlJointPD);
  // Safety gate (fall detection + deadman): always active, not a behavior
  // the supervisor engages -- see docs/GO2_HARDWARE_READINESS_AUDIT.md B3.
  CREATE_MODULE(mgr, quadruped::MdlGo2Safety, _go2Safety);
}

void ActivateCoreModules(ModuleManager *mgr ) {
  ACTIVATE_MODULE(mgr, _logserver);
  ACTIVATE_MODULE(mgr, _controlParams);
  ACTIVATE_MODULE(mgr, _legController);
  ACTIVATE_MODULE(mgr, _stateEstimator);
  ACTIVATE_MODULE(mgr, _desiredState);
  ACTIVATE_MODULE(mgr, _go2Safety);
  // Behavioral controllers are activated by Supervisor.
}

void DeactivateCoreModules(ModuleManager *mgr ) {
  DEACTIVATE_MODULE(mgr, _go2Safety);
  DEACTIVATE_MODULE(mgr, _rlJointPD);
  DEACTIVATE_MODULE(mgr, _rlPolicy);
  DEACTIVATE_MODULE(mgr, _standUp);
  DEACTIVATE_MODULE(mgr, _wbic);
  DEACTIVATE_MODULE(mgr, _footstepPlanner);
  DEACTIVATE_MODULE(mgr, _ctGaitScheduler);
  DEACTIVATE_MODULE(mgr, _convexMpc);
  DEACTIVATE_MODULE(mgr, _desiredState);
  DEACTIVATE_MODULE(mgr, _stateEstimator);
  DEACTIVATE_MODULE(mgr, _legController);
  DEACTIVATE_MODULE(mgr, _controlParams);
  DEACTIVATE_MODULE(mgr, _logserver);
}

void RemoveCoreModules(ModuleManager *mgr ) {
  // Log jobs hold references to module-owned variables. Stop and join the
  // LogServer listener before any module unregisters or destroys those variables.
  DESTROY_MODULE(mgr, _logserver);
  DESTROY_MODULE(mgr, _go2Safety);
  DESTROY_MODULE(mgr, _standUp);
  DESTROY_MODULE(mgr, _rlJointPD);
  DESTROY_MODULE(mgr, _rlPolicy);
  DESTROY_MODULE(mgr, _wbic);
  DESTROY_MODULE(mgr, _footstepPlanner);
  DESTROY_MODULE(mgr, _ctGaitScheduler);
  DESTROY_MODULE(mgr, _convexMpc);
  DESTROY_MODULE(mgr, _desiredState);
  DESTROY_MODULE(mgr, _stateEstimator);
  DESTROY_MODULE(mgr, _legController);
  DESTROY_MODULE(mgr, _controlParams);
}
