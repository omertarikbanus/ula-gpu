// Shared harness for driving one WBC tick outside the robot runtime. Mirrors
// how ula builds the inputs: MdlRLLocomotionState / SupervisorStates build the
// LocomotionCtrlData command, MdlStateEstimator provides StateEstimate,
// MdlLegController owns the LegController data, MdlWBIC calls
// LocomotionCtrl<float>::run with MIT_UserParameters. No algorithm changes.
#ifndef ULA_GPU_WBC_HARNESS_HH
#define ULA_GPU_WBC_HARNESS_HH

#include <cmath>
#include <memory>
#include <string>

#include "control/LocomotionCtrlData.hh"
#include "controllers/LegController.h"
#include "controllers/wbc/WBC_Ctrl/LocomotionCtrl/LocomotionCtrl.hpp"
#include "dynamics/ParseURDFtoQuadruped.h"
#include "dynamics/Quadruped.h"
#include "estimators/StateEstimatorTypes.hh"
#include "math/orientation_tools.h"
#include "parameters/MIT_UserParameters.h"

namespace wbch {

using T = float;

// Joint order everywhere: leg-major in the WBC order FR, FL, HR, HL, each
// (abad, hip, knee) -> index 3*leg + joint.
struct TickInput {
  T q[12];
  T qd[12];
  T pos[3];     // body position, world
  T quat[4];    // body orientation (w, x, y, z)
  T vBody[3];   // linear velocity, body frame
  T omega[3];   // angular velocity, body frame
  T contact[4];
  // command (LocomotionCtrlData)
  T pBody_des[3], vBody_des[3], rpy_des[3], omega_des[3];
  T Fr_des[4][3];
  T pFoot_des[4][3];
};

struct TickOutput {
  T qDes[12];
  T qdDes[12];
  T tauFF[12];
  T Fr[12];     // WBIC reaction forces per leg (0 for legs not in contact)
  T qddot[18];  // WBIC solution (floating base 6 + 12 joints)
};

struct Harness {
  Quadruped<T> quad;
  std::unique_ptr<LocomotionCtrl<T>> wbc;
  std::unique_ptr<LegController<T>> leg;
  quadruped::MIT_UserParameters user;
  T total_mass = 0;

  bool init(const std::string& urdf, T floating_base_weight = 10.0f) {
    bool ok = false;
    quad = ParseURDFtoQuadruped<T>(urdf, RobotType::MINI_CHEETAH, &ok);
    if (!ok) return false;
    wbc = std::make_unique<LocomotionCtrl<T>>(quad.buildModel());
    wbc->setFloatingBaseWeight(floating_base_weight);  // training config: 10
    leg = std::make_unique<LegController<T>>(quad);
    total_mass = quad._bodyMass +
                 4.0f * (quad._abadInertia.getMass() + quad._hipInertia.getMass() +
                         quad._kneeInertia.getMass());
    return true;
  }

  // One WBC tick (what MdlWBIC::update does when ready).
  void tick(const TickInput& in, TickOutput& out) {
    quadruped::StateEstimate<T> st;
    st.orientation = Vec4<T>(in.quat[0], in.quat[1], in.quat[2], in.quat[3]);
    st.position = Vec3<T>(in.pos[0], in.pos[1], in.pos[2]);
    st.vBody = Vec3<T>(in.vBody[0], in.vBody[1], in.vBody[2]);
    st.omegaBody = Vec3<T>(in.omega[0], in.omega[1], in.omega[2]);
    st.rBody = ori::quaternionToRotationMatrix(st.orientation);
    st.rpy = ori::quatToRPY(st.orientation);
    st.vWorld = st.rBody.transpose() * st.vBody;
    st.omegaWorld = st.rBody.transpose() * st.omegaBody;
    for (int l = 0; l < 4; ++l) st.contactEstimate[l] = in.contact[l];

    for (int l = 0; l < 4; ++l)
      for (int j = 0; j < 3; ++j) {
        leg->datas[l].q[j] = in.q[3 * l + j];
        leg->datas[l].qd[j] = in.qd[3 * l + j];
      }

    LocomotionCtrlData<T> cmd{};
    cmd.setZero();
    for (int i = 0; i < 3; ++i) {
      cmd.pBody_des[i] = in.pBody_des[i];
      cmd.vBody_des[i] = in.vBody_des[i];
      cmd.pBody_RPY_des[i] = in.rpy_des[i];
      cmd.vBody_Ori_des[i] = in.omega_des[i];
    }
    for (int l = 0; l < 4; ++l) {
      cmd.contact_state[l] = in.contact[l];
      for (int i = 0; i < 3; ++i) {
        cmd.Fr_des[l][i] = in.Fr_des[l][i];
        cmd.pFoot_des[l][i] = in.pFoot_des[l][i];
      }
    }

    wbc->run(&cmd, st, leg.get(), &user);

    const auto* wd = wbc->getWBICData();
    for (int l = 0; l < 4; ++l)
      for (int j = 0; j < 3; ++j) {
        const auto& c = leg->commands[l];
        out.qDes[3 * l + j] = c.qDes[j];
        out.qdDes[3 * l + j] = c.qdDes[j];
        out.tauFF[3 * l + j] = c.tauFeedForward[j];
        out.Fr[3 * l + j] = 0;
      }
    int src = 0;
    for (int l = 0; l < 4; ++l) {
      if (in.contact[l] > 0.f) {
        for (int j = 0; j < 3; ++j)
          if (src + j < wd->_Fr.size()) out.Fr[3 * l + j] = (T)wd->_Fr[src + j];
        src += 3;
      }
    }
    for (int i = 0; i < 18; ++i)
      out.qddot[i] = i < wd->_qddot.size() ? (T)wd->_qddot[i] : T(0);
  }

  // Foot contact point heights for the stance (all legs symmetric): finds the
  // hip angle (knee = -2*hip, abad = 0) that puts the feet on z=0 with the
  // body at height h, using the same model the WBC uses.
  void stanceJoints(T h, T out_leg[3]) {
    FloatingBaseModel<T> m = quad.buildModel();
    auto foot_z = [&](T hip) {
      FBModelState<T> s;
      s.bodyOrientation = Quat<T>(1, 0, 0, 0);
      s.bodyPosition = Vec3<T>(0, 0, h);
      s.bodyVelocity = SVec<T>::Zero();
      s.q = DVec<T>::Zero(12);
      s.qd = DVec<T>::Zero(12);
      for (int l = 0; l < 4; ++l) {
        s.q[3 * l + 1] = hip;
        s.q[3 * l + 2] = -2 * hip;
      }
      m.setState(s);
      m.forwardKinematics();
      m.contactJacobians();
      return m._pGC[linkID::FR][2];
    };
    // hip in (-1.4, -0.05): the foot rises monotonically as |hip| grows.
    T lo = -1.4f, hi = -0.05f;
    for (int it = 0; it < 60; ++it) {
      T mid = 0.5f * (lo + hi);
      if (foot_z(mid) > 0) lo = mid; else hi = mid;  // too high: crouched too much
    }
    out_leg[0] = 0;
    out_leg[1] = 0.5f * (lo + hi);
    out_leg[2] = -2 * out_leg[1];
  }
};

}  // namespace wbch

#endif
