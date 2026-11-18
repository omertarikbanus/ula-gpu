/* 
 * Copyright (C) 2005-2025 Uluç Saranlı. All Rights Reserved
 *
 * This file is part of the RoboMETU robot control software library
 * collection. Unauthorized copying of this file, via any medium is
 * strictly prohibited.
 */

#ifndef QUADRUPED_STATEESTIMATORS_HH
#define QUADRUPED_STATEESTIMATORS_HH

#include "estimators/StateEstimatorTypes.hh"

namespace quadruped {

/**
 * @brief Contact estimator pass-through (mirrors Cheetah implementation).
 */
template <typename T>
class ContactEstimator : public GenericEstimator<T> {
 public:
  void run() override {
    if (this->_stateEstimatorData.result && this->_stateEstimatorData.contactPhase) {
      this->_stateEstimatorData.result->contactEstimate =
          *this->_stateEstimatorData.contactPhase;
    }
  }
  void setup() override {}
};

/**
 * @brief Orientation estimator that consumes IMU measurements.
 *
 * The IMU reports an absolute (world-referenced) orientation, which is passed
 * through unchanged so the estimator's world frame is the IMU's world frame.
 * The previous behavior latched the first sample and removed its yaw, which
 * silently rotated the whole world frame whenever the robot did not start at
 * yaw zero (for example after a random reset).
 */
template <typename T>
class ImuOrientationEstimator : public GenericEstimator<T> {
 public:
  void run() override;
  void setup() override {}
};

/**
 * @brief Linear Kalman filter estimator for body position and velocity.
 */
template <typename T>
class LinearKFPositionVelocityEstimator : public GenericEstimator<T> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LinearKFPositionVelocityEstimator();

  void run() override;
  void setup() override;

  /**
   * @brief Seeds the filter state from the robot's current state.
   *
   * Call once at initialization or reset, before the first run(), when the
   * robot's current base pose is known: the filter starts at the actual
   * position/velocity instead of assuming rest at the origin, and the
   * per-foot world positions are seeded from the current leg kinematics so
   * the first measurement update is consistent rather than pulling the state
   * toward an assumed origin. Requires result->rBody to already reflect the
   * current orientation.
   */
  void initialize(const Vec3<T>& position, const Vec3<T>& velocity);

 private:
  Vec18<T> _xhat = Vec18<T>::Zero();
  Vec12<T> _ps = Vec12<T>::Zero();
  Vec12<T> _vs = Vec12<T>::Zero();
  Mat18<T> _A = Mat18<T>::Zero();
  Mat18<T> _Q0 = Mat18<T>::Zero();
  Mat18<T> _P = Mat18<T>::Identity();
  Mat28<T> _R0 = Mat28<T>::Identity();
  Eigen::Matrix<T, 18, 3> _B = Eigen::Matrix<T, 18, 3>::Zero();
  Eigen::Matrix<T, 28, 18> _C = Eigen::Matrix<T, 28, 18>::Zero();
};

}  // namespace quadruped

#endif  // QUADRUPED_STATEESTIMATORS_HH

