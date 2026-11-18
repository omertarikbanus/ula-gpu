/*
 * Copyright (C) 2005-2025 Uluc Saranli. All Rights Reserved
 *
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium is
 * strictly prohibited.
 */

#ifndef QUADRUPED_LOCOMOTION_CTRL_DATA_HH
#define QUADRUPED_LOCOMOTION_CTRL_DATA_HH

#include "types/cppTypes.h"

template<typename T>
class LocomotionCtrlData {
 public:
  Vec3<T> pBody_des;
  Vec3<T> vBody_des;
  Vec3<T> aBody_des;
  Vec3<T> pBody_RPY_des;
  Vec3<T> vBody_Ori_des;

  Vec3<T> pFoot_des[4];
  Vec3<T> vFoot_des[4];
  Vec3<T> aFoot_des[4];
  Vec3<T> Fr_des[4];
  Vec3<T> Fr_se[4];

  Vec4<T> contact_state;

  void setZero() {
    pBody_des.setZero();
    vBody_des.setZero();
    aBody_des.setZero();
    pBody_RPY_des.setZero();
    vBody_Ori_des.setZero();
    contact_state.setZero();
    for (int leg = 0; leg < 4; ++leg) {
      pFoot_des[leg].setZero();
      vFoot_des[leg].setZero();
      aFoot_des[leg].setZero();
      Fr_des[leg].setZero();
      Fr_se[leg].setZero();
    }
  }
};

#endif  // QUADRUPED_LOCOMOTION_CTRL_DATA_HH
