/*
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 *
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium is
 * strictly prohibited.
 */

#ifndef QUADRUPED_CONTROLPARAMS_HH
#define QUADRUPED_CONTROLPARAMS_HH

#include "estimators/StateEstimatorTypes.hh"

namespace quadruped {

/**
 * @brief MIT controller parameters ported from Cheetah defaults.
 */
struct MIT_UserParameters {
  double cmpc_gait = 9.0;
  double cmpc_x_drag = 3.0;
  double cmpc_use_sparse = 0.0;
  // 1 = run the dense ConvexMPC QP solve on a worker thread so a slow solve
  // can never stall the 500 Hz control loop (the published solution is
  // applied at most one control tick later, against a dtMPC-long hold).
  // 0 = legacy synchronous solve inside the control tick. The sparse solver
  // path (cmpc_use_sparse = 1) is always synchronous.
  double cmpc_async_solve = 1.0;
  double use_wbc = 1.0;
  double cmpc_bonus_swing = 0.0;

  // SHARED. Coulomb friction coefficient assumed by the WBIC contact cone and
  // the convex-MPC force constraints. Held at 75% of the simulator's 0.6 so
  // the controllers keep a margin against an uncertain surface.
  double contact_mu = 0.45;

  // SHARED. Commanded trunk height in metres. Both the convex MPC and the RL
  // locomotion state issue this as pBody_des[2], so the two controllers hold
  // the same stance and the comparison is not confounded by posture.
  double body_height = 0.32;

  // SEPARABLE (see the parity spec): this is a legitimate tuning knob and may
  // differ between training and inference. It must nonetheless be EQUAL across
  // controllers in any run whose results are compared, and the value must be
  // reported alongside those results.
  //
  // Weight on relaxing the floating-base acceleration in the WBIC QP, against
  // W_rf = 1 on deviating from the commanded foot forces. High values make the
  // QP discard Fr_des to hold base acceleration -- at 1000 the RL policy's
  // vertical force command reached the robot with a gain of 0.02.
  double floating_base_weight = 50.0;

  Vec3<double> Kp_body = Vec3<double>(100.0, 100.0, 100.0);
  Vec3<double> Kd_body = Vec3<double>(10.0, 10.0, 10.0);

  Vec3<double> Kp_ori = Vec3<double>(100.0, 100.0, 100.0);
  Vec3<double> Kd_ori = Vec3<double>(10.0, 10.0, 10.0);

  Vec3<double> Kp_foot = Vec3<double>(500.0, 500.0, 500.0);
  Vec3<double> Kd_foot = Vec3<double>(10.0, 10.0, 10.0);

  Vec3<double> Kp_joint = Vec3<double>(3.0, 3.0, 3.0);
  Vec3<double> Kd_joint = Vec3<double>(1.0, 0.2, 0.2);

  Vec3<double> Q_pos = Vec3<double>(20.0, 20.0, 20000.0);
  Vec3<double> Q_vel = Vec3<double>(0.0, 0.0, 0.0);
  Vec3<double> Q_ori = Vec3<double>(1500.0, 1900.0, 900.0);
  Vec3<double> Q_ang = Vec3<double>(30.0, 40.0, 10.0);
  double R_control = 1.1;
  double R_prev = 2.0;
  Vec3<double> two_leg_orient = Vec3<double>(0.225, 9.1, 0.0);
  double stance_legs = 4.0;

  double use_jcqp = 0.0;
  double jcqp_max_iter = 10000.0;
  double jcqp_rho = 1e-7;
  double jcqp_sigma = 1e-8;
  double jcqp_alpha = 1.5;
  double jcqp_terminate = 0.1;

  Vec3<double> Swing_Kp_cartesian = Vec3<double>(350.0, 350.0, 75.0);
  Vec3<double> Swing_Kd_cartesian = Vec3<double>(5.5, 5.5, 5.5);
  Vec3<double> Swing_Kp_joint = Vec3<double>(0.0, 0.0, 0.0);
  Vec3<double> Swing_Kd_joint = Vec3<double>(0.2, 0.2, 0.2);
  Vec3<double> Swing_step_offset = Vec3<double>(0.0, 0.05, -0.003);
  double Swing_traj_height = 0.07;
  double Swing_use_tau_ff = 0.0;

  Vec3<double> RPC_Q_p = Vec3<double>(10000.0, 10000.0, 100000.0);
  Vec3<double> RPC_Q_theta = Vec3<double>(5000.0, 10000.0, 2500.0);
  Vec3<double> RPC_Q_dp = Vec3<double>(2500.0, 2500.0, 500.0);
  Vec3<double> RPC_Q_dtheta = Vec3<double>(2.5, 2.5, 50.0);
  Vec3<double> RPC_R_r = Vec3<double>(350000.0, 350000.0, 350000.0);
  Vec3<double> RPC_R_f = Vec3<double>(0.001, 0.001, 0.001);
  Vec3<double> RPC_H_r_trans = Vec3<double>(1.0, 2.0, 1.0);
  Vec3<double> RPC_H_r_rot = Vec3<double>(1.0, 1.0, 0.0);
  Vec3<double> RPC_H_theta0 = Vec3<double>(1.25, 0.0, 0.0);
  Vec3<double> RPC_H_phi0 = Vec3<double>(-0.3, 0.02, 0.0);
  double RPC_mass = 9.0;
  Vec3<double> RPC_inertia = Vec3<double>(0.07, 0.26, 0.242);
  Vec3<double> RPC_gravity = Vec3<double>(0.0, 0.0, -9.81);
  double RPC_mu = 0.5;
  Vec3<double> RPC_filter = Vec3<double>(0.5, 0.1, 0.0);
  double RPC_use_pred_comp = 1.0;
  double RPC_use_async_filt = 1.0;
  double RPC_visualize_pred = 1.0;
  double RPC_use_separate = 1.0;

  Vec3<double> des_p = Vec3<double>(0.0, 0.0, 0.26);
  Vec3<double> des_theta = Vec3<double>(0.0, 0.0, 0.0);
  Vec3<double> des_dp = Vec3<double>(0.0, 0.0, 0.0);
  Vec3<double> des_dtheta = Vec3<double>(0.0, 0.0, 0.0);
  Vec3<double> des_theta_max = Vec3<double>(0.0, 0.4, 0.0);
  Vec3<double> des_dp_max = Vec3<double>(1.0, 0.5, 0.0);
  Vec3<double> des_dtheta_max = Vec3<double>(0.0, 0.0, 3.0);

  double gait_type = 4.0;
  double gait_period_time = 0.5;
  double gait_switching_phase = 0.5;
  double gait_override = 4.0;
  double gait_max_leg_angle = 15.0;
  double gait_max_stance_time = 0.25;
  double gait_min_stance_time = 0.1;
};

}  // namespace quadruped

#endif  // QUADRUPED_CONTROLPARAMS_HH
