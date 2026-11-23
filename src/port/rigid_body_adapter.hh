#ifndef ULA_GPU_PORT_RIGID_BODY_ADAPTER_HH
#define ULA_GPU_PORT_RIGID_BODY_ADAPTER_HH

#include <stdexcept>

#include "dynamics/FloatingBaseModel.h"
#include "dynamics/Quadruped.h"
#include "port/rigid_body.hh"

namespace ula::rigid {

inline constexpr size_t foot_index[kFeet]{linkID::FR, linkID::FL, linkID::HR, linkID::HL};

// Build once from the existing model setup (including rotor inertias and
// mirrored legs). No geometry is re-derived or inferred from a trace.
template <class T> Model<T> pack_model(const FloatingBaseModel<T>& src) {
  if (src._nDof != kDof) throw std::runtime_error("fixed dynamics expects an 18-DOF Go2 model");
  Model<T> dst;
  for (int i = 0; i < 3; ++i) dst.gravity.data[i] = src._gravity[i];
  for (int j = kBase; j < kDof; ++j) {
    const auto body = src._Ibody[j].getMatrix();
    const auto rotor = src._Irot[j].getMatrix();
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) {
      dst.body_inertia[j].data[r][c] = body(r, c);
      dst.rotor_inertia[j].data[r][c] = rotor(r, c);
      dst.tree[j].data[r][c] = src._Xtree[j](r, c);
      dst.rotor_tree[j].data[r][c] = src._Xrot[j](r, c);
    }
    if (j == kBase) continue;
    const int axis = static_cast<int>(src._jointAxes[j]);
    const int parent = src._parents[j];
    if (axis < 0 || axis > 2 || parent < kBase || parent >= j ||
        src._jointTypes[j] != spatial::JointType::Revolute) {
      throw std::runtime_error("fixed dynamics expects a parent-ordered revolute tree");
    }
    dst.axis[j] = axis;
    dst.parent[j] = parent;
    dst.gear[j] = src._gearRatios[j];
  }
  for (int leg = 0; leg < kFeet; ++leg) {
    const size_t foot = foot_index[leg];
    if (foot >= src._nGroundContact || !src._compute_contact_info[foot] ||
        src._gcParent[foot] < kBase || src._gcParent[foot] >= kDof) {
      throw std::runtime_error("fixed dynamics requires the four enabled ULA foot points");
    }
    dst.foot_parent[leg] = src._gcParent[foot];
    for (int i = 0; i < 3; ++i) dst.foot_local[leg].data[i] = src._gcLocation[foot][i];
  }
  return dst;
}

template <class T> State<T> pack_state(const FBModelState<T>& src) {
  State<T> dst;
  for (int i = 0; i < 4; ++i) dst.quaternion.data[i] = src.bodyOrientation[i];
  for (int i = 0; i < 3; ++i) dst.position.data[i] = src.bodyPosition[i];
  for (int i = 0; i < 6; ++i) dst.velocity.data[i] = src.bodyVelocity[i];
  for (int i = 0; i < 12; ++i) {
    dst.q[i] = src.q[i];
    dst.qd[i] = src.qd[i];
  }
  return dst;
}

}  // namespace ula::rigid

#endif
