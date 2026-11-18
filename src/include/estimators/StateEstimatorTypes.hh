/* 
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 *
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium is
 * strictly prohibited.
 */

#ifndef QUADRUPED_STATEESTIMATORTYPES_HH
#define QUADRUPED_STATEESTIMATORTYPES_HH

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <vector>

namespace quadruped {

constexpr int kNumLegs = 4;

template <typename T>
using Vec3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using Vec4 = Eigen::Matrix<T, 4, 1>;

template <typename T>
using Vec12 = Eigen::Matrix<T, 12, 1>;

template <typename T>
using Vec18 = Eigen::Matrix<T, 18, 1>;

template <typename T>
using Vec28 = Eigen::Matrix<T, 28, 1>;

template <typename T>
using Mat3 = Eigen::Matrix<T, 3, 3>;

template <typename T>
using Mat18 = Eigen::Matrix<T, 18, 18>;

template <typename T>
using Mat28 = Eigen::Matrix<T, 28, 28>;

template <typename T>
using Mat86 = Eigen::Matrix<T, 18, 3>;

template <typename T>
using Mat283 = Eigen::Matrix<T, 28, 18>;

/**
 * @brief Lightweight kinematic model providing hip locations for the estimator.
 */
struct RobotModel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RobotModel() = default;

  template <typename T>
  Vec3<T> getHipLocation(int leg) const {
    return hipLocations[leg].template cast<T>();
  }

  Eigen::Vector3d hipLocations[kNumLegs] = {
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero()};
};

/**
 * @brief IMU measurement aligned with the estimator conventions.
 */
struct ImuMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d accelerometer = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

/**
 * @brief Leg data (position/velocity in body frame) used by the estimator.
 */
template <typename T>
struct LegEstimate {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3<T> p = Vec3<T>::Zero();
  Vec3<T> v = Vec3<T>::Zero();
};

/**
 * @brief Full state estimate produced by the estimator stack.
 */
template <typename T>
struct StateEstimate {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec4<T> contactEstimate = Vec4<T>::Zero();
  Vec3<T> position = Vec3<T>::Zero();
  Vec3<T> vBody = Vec3<T>::Zero();
  Vec4<T> orientation = Vec4<T>::Zero();  // (w, x, y, z)
  Vec3<T> omegaBody = Vec3<T>::Zero();
  Mat3<T> rBody = Mat3<T>::Identity();
  Vec3<T> rpy = Vec3<T>::Zero();
  Vec3<T> omegaWorld = Vec3<T>::Zero();
  Vec3<T> vWorld = Vec3<T>::Zero();
  Vec3<T> aBody = Vec3<T>::Zero();
  Vec3<T> aWorld = Vec3<T>::Zero();
};

/**
 * @brief Control parameters used by the estimators (ported from Cheetah).
 */
struct RobotControlParameters {
  double controller_dt = 0.002;
  double control_mode = 0.0;
  double myValue = 0.0;
  double testValue = 0.0;
  Vec3<double> stand_kp_cartesian = Vec3<double>(50.0, 50.0, 50.0);
  Vec3<double> stand_kd_cartesian = Vec3<double>(2.5, 2.5, 2.5);
  Vec3<double> kpCOM = Vec3<double>(50.0, 50.0, 50.0);
  Vec3<double> kdCOM = Vec3<double>(10.0, 10.0, 10.0);
  Vec3<double> kpBase = Vec3<double>(300.0, 200.0, 100.0);
  Vec3<double> kdBase = Vec3<double>(20.0, 10.0, 10.0);
  Vec3<double> max_torque = Vec3<double>(23.7, 23.7, 45.43);
  double imu_process_noise_position = 0.02;
  double imu_process_noise_velocity = 0.02;
  double foot_process_noise_position = 0.002;
  double foot_sensor_noise_position = 0.001;
  double foot_sensor_noise_velocity = 0.1;
  double foot_height_sensor_noise = 0.001;
  // Where the controller's base state comes from. Three explicit modes, so a
  // run always declares which one it used instead of inferring it from a bool:
  //   kEstimator  - the quadcontrol KF estimate from the (possibly corrupted)
  //                 sensors. The honest setting, and the hardest.
  //   kNoisyTruth - simulator truth perturbed by the configured half-widths.
  //                 Isolates estimator quality from sensor quality: the
  //                 controller sees a known, bounded error with no filter
  //                 dynamics behind it.
  //   kTruth      - simulator truth, unperturbed. The old `cheater_mode = true`.
  enum class StateSource { kEstimator, kNoisyTruth, kTruth };
  StateSource state_source = StateSource::kEstimator;
  // kNoisyTruth half-widths (uniform, resampled per controller tick).
  double truth_noise_position = 0.0;
  double truth_noise_orientation = 0.0;
  double truth_noise_velocity = 0.0;
  double truth_noise_angular_velocity = 0.0;
  unsigned int truth_noise_seed = 1001;
  bool use_rc = false;
};

/**
 * @brief Shared data passed to all estimator blocks.
 */
template <typename T>
struct StateEstimatorData {
  StateEstimate<T>* result = nullptr;
  ImuMeasurement* imuMeasurement = nullptr;
  LegEstimate<T>* legData = nullptr;
  Vec4<T>* contactPhase = nullptr;
  RobotControlParameters* parameters = nullptr;
  const RobotModel* model = nullptr;
};

/**
 * @brief Base class for estimator blocks (matches Cheetah's interface).
 */
template <typename T>
class GenericEstimator {
 public:
  virtual ~GenericEstimator() = default;
  virtual void run() = 0;
  virtual void setup() = 0;

  void setData(StateEstimatorData<T> data) { _stateEstimatorData = data; }

 protected:
  StateEstimatorData<T> _stateEstimatorData;
};

/**
 * @brief Quaternion helper utilities (Eigen-based reimplementation).
 */
namespace ori {

template <typename T>
Eigen::Quaternion<T> toEigenQuat(const Vec4<T>& q) {
  return Eigen::Quaternion<T>(q(0), q(1), q(2), q(3));
}

// Rotation matrix of a body->world quaternion q (v_world = R v_body),
// returned in the MIT Cheetah convention as the WORLD->BODY transform
// (v_body = R v_world). This mirrors ori::quaternionToRotationMatrix in
// math/orientation_tools.h, which also transposes: the MIT estimator and
// controller math is written against rBody = world->body
// (vBody = rBody * vWorld, footWorld = position + rBody^T * p_body), so the
// transpose is part of the contract. Returning the untransposed matrix here
// silently flips every one of those products.
template <typename T>
Mat3<T> quaternionToRotationMatrix(const Vec4<T>& qvec) {
  Eigen::Quaternion<T> q(qvec(0), qvec(1), qvec(2), qvec(3));
  q.normalize();
  return q.toRotationMatrix().transpose();
}

template <typename T>
Vec4<T> quaternionToVector(const Eigen::Quaternion<T>& q) {
  Eigen::Quaternion<T> qn = q.normalized();
  return Vec4<T>(qn.w(), qn.x(), qn.y(), qn.z());
}

template <typename T>
Vec3<T> quatToRPY(const Vec4<T>& qvec) {
  Eigen::Quaternion<T> q(qvec(0), qvec(1), qvec(2), qvec(3));
  q.normalize();
  Eigen::Matrix<T, 3, 1> rpy;
  rpy(0) = std::atan2(2.0 * (q.w() * q.x() + q.y() * q.z()),
                      1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y()));
  T sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
  if (std::abs(sinp) >= 1.0)
    rpy(1) = std::copysign(static_cast<T>(M_PI / 2.0), sinp);
  else
    rpy(1) = std::asin(sinp);
  rpy(2) = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                      1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
  return rpy;
}

template <typename T>
Vec4<T> rpyToQuat(const Vec3<T>& rpy) {
  Eigen::AngleAxis<T> roll(rpy(0), Vec3<T>::UnitX());
  Eigen::AngleAxis<T> pitch(rpy(1), Vec3<T>::UnitY());
  Eigen::AngleAxis<T> yaw(rpy(2), Vec3<T>::UnitZ());
  Eigen::Quaternion<T> q = yaw * pitch * roll;
  return quaternionToVector(q);
}

template <typename T>
Vec4<T> quatProduct(const Vec4<T>& lhs, const Vec4<T>& rhs) {
  Eigen::Quaternion<T> ql(lhs(0), lhs(1), lhs(2), lhs(3));
  Eigen::Quaternion<T> qr(rhs(0), rhs(1), rhs(2), rhs(3));
  return quaternionToVector(ql * qr);
}

}  // namespace ori

/**
 * @brief Container orchestrating all estimator stages.
 */
template <typename T>
class StateEstimatorContainer {
 public:
  explicit StateEstimatorContainer(StateEstimatorData<T> data) : _data(data) {}

  template <typename EstimatorToAdd>
  EstimatorToAdd* addEstimator() {
    auto* estimator = new EstimatorToAdd();
    estimator->setData(_data);
    estimator->setup();
    _estimators.push_back(estimator);
    return estimator;
  }

  void run() {
    for (auto* estimator : _estimators) {
      estimator->run();
    }
  }

  void setContactPhase(const Vec4<T>& phase) {
    if (_data.contactPhase) {
      *_data.contactPhase = phase;
    }
  }

  StateEstimate<T>* getResultHandle() { return _data.result; }

  ~StateEstimatorContainer() {
    for (auto* estimator : _estimators) {
      delete estimator;
    }
    _estimators.clear();
  }

 private:
  StateEstimatorData<T> _data;
  std::vector<GenericEstimator<T>*> _estimators;
};

}  // namespace quadruped

#endif  // QUADRUPED_STATEESTIMATORTYPES_HH
