/* 
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 *
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium is
 * strictly prohibited.
 */

#include "estimators/StateEstimators.hh"

#include <algorithm>
#include <iostream>
namespace quadruped {

template <typename T>
void ImuOrientationEstimator<T>::run() {
  auto* result = this->_stateEstimatorData.result;
  auto* imu = this->_stateEstimatorData.imuMeasurement;
  if (!result || !imu) return;

  Vec4<T> orientation = ori::quaternionToVector(imu->orientation.template cast<T>());

  result->orientation = orientation;
  result->rBody = ori::quaternionToRotationMatrix(orientation);
  result->rpy = ori::quatToRPY(orientation);
  result->omegaBody = imu->gyro.template cast<T>();
  result->omegaWorld = result->rBody.transpose() * result->omegaBody;
  result->aBody = imu->accelerometer.template cast<T>();
  result->aWorld = result->rBody.transpose() * result->aBody;
}

template class ImuOrientationEstimator<float>;
template class ImuOrientationEstimator<double>;

template <typename T>
LinearKFPositionVelocityEstimator<T>::LinearKFPositionVelocityEstimator() {}

template <typename T>
void LinearKFPositionVelocityEstimator<T>::setup() {
  auto* params = this->_stateEstimatorData.parameters;
  if (!params) return;

  T dt = static_cast<T>(params->controller_dt);
  _xhat.setZero();
  _ps.setZero();
  _vs.setZero();

  _A.setZero();
  _A.block(0, 0, 3, 3) = Mat3<T>::Identity();
  _A.block(0, 3, 3, 3) = dt * Mat3<T>::Identity();
  _A.block(3, 3, 3, 3) = Mat3<T>::Identity();
  _A.block(6, 6, 12, 12) = Eigen::Matrix<T, 12, 12>::Identity();

  _B.setZero();
  _B.block(3, 0, 3, 3) = dt * Mat3<T>::Identity();

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> C1(3, 6);
  C1 << Mat3<T>::Identity(), Mat3<T>::Zero();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> C2(3, 6);
  C2 << Mat3<T>::Zero(), Mat3<T>::Identity();

  _C.setZero();
  _C.block(0, 0, 3, 6) = C1;
  _C.block(3, 0, 3, 6) = C1;
  _C.block(6, 0, 3, 6) = C1;
  _C.block(9, 0, 3, 6) = C1;
  _C.block(0, 6, 12, 12) = static_cast<T>(-1) * Eigen::Matrix<T, 12, 12>::Identity();
  _C.block(12, 0, 3, 6) = C2;
  _C.block(15, 0, 3, 6) = C2;
  _C.block(18, 0, 3, 6) = C2;
  _C.block(21, 0, 3, 6) = C2;
  _C(27, 17) = static_cast<T>(1);
  _C(26, 14) = static_cast<T>(1);
  _C(25, 11) = static_cast<T>(1);
  _C(24, 8) = static_cast<T>(1);

  _P = Mat18<T>::Identity() * static_cast<T>(100);

  _Q0.setIdentity();
  _Q0.block(0, 0, 3, 3) = (dt / static_cast<T>(20.0)) * Mat3<T>::Identity();
  _Q0.block(3, 3, 3, 3) =
      (dt * static_cast<T>(9.8) / static_cast<T>(20.0)) * Mat3<T>::Identity();
  _Q0.block(6, 6, 12, 12) = dt * Eigen::Matrix<T, 12, 12>::Identity();

  _R0.setIdentity();
}

template <typename T>
void LinearKFPositionVelocityEstimator<T>::initialize(
    const Vec3<T>& position, const Vec3<T>& velocity) {
  auto* result = this->_stateEstimatorData.result;
  auto* legData = this->_stateEstimatorData.legData;
  auto* model = this->_stateEstimatorData.model;
  if (!result || !legData || !model) return;

  _xhat.setZero();
  _xhat.template segment<3>(0) = position;
  _xhat.template segment<3>(3) = velocity;

  // Foot states in the world frame, mirroring the measurement model in
  // run(): p_foot_world = p_base + R_body_world * (hip + p_leg_body).
  const Mat3<T> Rbod = result->rBody.transpose();
  for (int i = 0; i < kNumLegs; i++) {
    Vec3<T> p_rel = model->template getHipLocation<T>(i) + legData[i].p;
    _xhat.template segment<3>(6 + 3 * i) = position + Rbod * p_rel;
  }
}

template <typename T>
void LinearKFPositionVelocityEstimator<T>::run() {
  auto* params = this->_stateEstimatorData.parameters;
  auto* result = this->_stateEstimatorData.result;
  auto* legData = this->_stateEstimatorData.legData;
  auto* contactPhase = this->_stateEstimatorData.contactPhase;
  auto* model = this->_stateEstimatorData.model;
  if (!params || !result || !legData || !contactPhase || !model) return;

  T process_noise_pimu = static_cast<T>(params->imu_process_noise_position);
  T process_noise_vimu = static_cast<T>(params->imu_process_noise_velocity);
  T process_noise_pfoot = static_cast<T>(params->foot_process_noise_position);
  T sensor_noise_pimu_rel_foot = static_cast<T>(params->foot_sensor_noise_position);
  T sensor_noise_vimu_rel_foot = static_cast<T>(params->foot_sensor_noise_velocity);
  T sensor_noise_zfoot = static_cast<T>(params->foot_height_sensor_noise);

  Mat18<T> Q = Mat18<T>::Identity();
  Q.block(0, 0, 3, 3) = _Q0.block(0, 0, 3, 3) * process_noise_pimu;
  Q.block(3, 3, 3, 3) = _Q0.block(3, 3, 3, 3) * process_noise_vimu;
  Q.block(6, 6, 12, 12) = _Q0.block(6, 6, 12, 12) * process_noise_pfoot;

  Mat28<T> R = Mat28<T>::Identity();
  R.block(0, 0, 12, 12) = _R0.block(0, 0, 12, 12) * sensor_noise_pimu_rel_foot;
  R.block(12, 12, 12, 12) = _R0.block(12, 12, 12, 12) * sensor_noise_vimu_rel_foot;
  R.block(24, 24, 4, 4) = _R0.block(24, 24, 4, 4) * sensor_noise_zfoot;

  int qindex = 0;
  int rindex1 = 0;
  int rindex2 = 0;
  int rindex3 = 0;

  Vec3<T> g(static_cast<T>(0), static_cast<T>(0), static_cast<T>(-9.81));
  Mat3<T> Rbod = result->rBody.transpose();
  Vec3<T> a = result->aWorld + g;
  Vec4<T> pzs = Vec4<T>::Zero();
  Vec4<T> trusts = Vec4<T>::Zero();
  Vec3<T> p0 = _xhat.template segment<3>(0);
  Vec3<T> v0 = _xhat.template segment<3>(3);

  for (int i = 0; i < kNumLegs; i++) {
    int i1 = 3 * i;
    Vec3<T> ph = model->template getHipLocation<T>(i);
    Vec3<T> p_rel = ph + legData[i].p;
    Vec3<T> dp_rel = legData[i].v;

    Vec3<T> p_f = Rbod * p_rel;
    Vec3<T> dp_f = Rbod * (result->omegaBody.cross(p_rel) + dp_rel);

    qindex = 6 + i1;
    rindex1 = i1;
    rindex2 = 12 + i1;
    rindex3 = 24 + i;

    T trust = static_cast<T>(1);
    T phase = std::min((*contactPhase)(i), static_cast<T>(1));
    T trust_window = static_cast<T>(0.2);

    if (phase < trust_window) {
      trust = phase / trust_window;
    } else if (phase > (static_cast<T>(1) - trust_window)) {
      trust = (static_cast<T>(1) - phase) / trust_window;
    }
    T high_suspect_number = static_cast<T>(100);

    Q.block(qindex, qindex, 3, 3) =
        (static_cast<T>(1) + (static_cast<T>(1) - trust) * high_suspect_number) *
        Q.block(qindex, qindex, 3, 3);
    R.block(rindex1, rindex1, 3, 3) = R.block(rindex1, rindex1, 3, 3);
    R.block(rindex2, rindex2, 3, 3) =
        (static_cast<T>(1) + (static_cast<T>(1) - trust) * high_suspect_number) *
        R.block(rindex2, rindex2, 3, 3);
    R(rindex3, rindex3) =
        (static_cast<T>(1) + (static_cast<T>(1) - trust) * high_suspect_number) *
        R(rindex3, rindex3);

    trusts(i) = trust;

    _ps.segment(i1, 3) = -p_f;
    _vs.segment(i1, 3) = (static_cast<T>(1) - trust) * v0 + trust * (-dp_f);
    pzs(i) = (static_cast<T>(1) - trust) * (p0(2) + p_f(2));
  }

  Vec28<T> y;
  y << _ps, _vs, pzs;
  _xhat = _A * _xhat + _B * a;
  Mat18<T> At = _A.transpose();
  Mat18<T> Pm = _A * _P * At + Q;
  Eigen::Matrix<T, 18, 28> Ct = _C.transpose();
  Vec28<T> yModel = _C * _xhat;
  Vec28<T> ey = y - yModel;
  Mat28<T> S = _C * Pm * Ct + R;

  Vec28<T> S_ey = S.lu().solve(ey);
  _xhat += Pm * Ct * S_ey;

  Eigen::Matrix<T, 28, 18> S_C = S.lu().solve(_C);
  _P = (Mat18<T>::Identity() - Pm * Ct * S_C) * Pm;

  Mat18<T> Pt = _P.transpose();
  _P = (_P + Pt) / static_cast<T>(2);

  if (_P.block(0, 0, 2, 2).determinant() > static_cast<T>(0.000001)) {
    _P.block(0, 2, 2, 16).setZero();
    _P.block(2, 0, 16, 2).setZero();
    _P.block(0, 0, 2, 2) /= static_cast<T>(10);
  }

  result->position = _xhat.block(0, 0, 3, 1);
  result->vWorld = _xhat.block(3, 0, 3, 1);
  result->vBody = result->rBody * result->vWorld;
}

template class LinearKFPositionVelocityEstimator<float>;
template class LinearKFPositionVelocityEstimator<double>;

}  // namespace quadruped
