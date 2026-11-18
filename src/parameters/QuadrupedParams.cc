/*
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 *
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium is
 * strictly prohibited.
 */

#include "parameters/QuadrupedParams.hh"

namespace quadruped {

QuadrupedKinematics::params_t createGo2Config() {
  QuadrupedKinematics::params_t params;

  // Robot identification
  params.robot_name = "Demir-1";

  // Hip positions in body frame [x, y, z] for each leg (meters)
  // From demir_1/robot.xml - abad body positions
  // Leg order: FR, FL, HR, HL (Cheetah convention)
  params.hip_positions <<  0.223895, -0.0655252, 0.0,  // Front Right (FR) - abad_1
                           0.223895,  0.0655252, 0.0,  // Front Left (FL)  - abad_2
                          -0.223895, -0.0655252, 0.0,  // Hind Right (HR)  - abad_3
                          -0.223895,  0.0655252, 0.0;  // Hind Left (HL)   - abad_4

  // Link lengths [thigh_length, calf_length] for each leg (meters)
  // From robot.xml: knee_1 pos="0 0.07885 -0.27" → thigh = 0.27
  // From robot.xml: ROBOT_FOOT_1 pos="0 0 -0.288398" → calf = 0.288398
  params.link_lengths << 0.27, 0.288398,  // Front Right
                          0.27, 0.288398,  // Front Left
                          0.27, 0.288398,  // Hind Right
                          0.27, 0.288398;  // Hind Left

  // Hip flexion axis offset for each leg (meters)
  // From robot.xml: hip_1 pos="0.0644992 -0.025 0" - offset along Y from abad to hip
  // FR/HR: hip at negative Y from abad → hip_2,4 have offset at y=0.025, then knee at y=-0.07885 from hip
  // Looking at the geometry: the offset from abad to the hip flexion axis
  // abad_1 is at y=-0.0655, hip_1 at relative y=-0.025, knee_1 at relative y=+0.07885 from hip
  params.hip_flexion_offset << -0.07885,  // Front Right (negative for right legs)
                                0.07885,  // Front Left  (positive for left legs)
                               -0.07885,  // Hind Right
                                0.07885;  // Hind Left

  // Hip abduction limits (radians)
  params.hip_abduction_limits << -1.0472, 1.0472,  // Front Right
                                 -1.0472, 1.0472,  // Front Left
                                 -1.0472, 1.0472,  // Hind Right
                                 -1.0472, 1.0472;  // Hind Left

  // Hip flexion limits (radians)
  params.hip_flexion_limits << -1.5708, 3.4907,  // Front Right
                               -1.5708, 3.4907,  // Front Left
                               -0.5236, 4.5379,  // Hind Right
                               -0.5236, 4.5379;  // Hind Left

  // Knee limits (radians)
  params.knee_limits << -2.7227, -0.83776,  // Front Right
                        -2.7227, -0.83776,  // Front Left
                        -2.7227, -0.83776,  // Hind Right
                        -2.7227, -0.83776;  // Hind Left

  // Joint directions for sign conventions
  // Note: Joint polarity is already handled in MdlSimDriver::_readJointStates()
  // so all directions here should be 1.0 (no additional sign flip)
  params.joint_directions << 1.0, 1.0, 1.0,  // Front Right
                              1.0, 1.0, 1.0,  // Front Left
                              1.0, 1.0, 1.0,  // Hind Right
                              1.0, 1.0, 1.0;  // Hind Left

  return params;
}

}  // namespace quadruped

