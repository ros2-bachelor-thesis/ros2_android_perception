#pragma once

#include <Eigen/Dense>

namespace perception {

/**
 * Kalman Filter for 8D constant velocity tracking model
 *
 * State vector (8D): [x, y, a, h, vx, vy, va, vh]
 * where:
 *   (x, y) = bounding box center position
 *   a = aspect ratio (width / height)
 *   h = height
 *   v* = velocities
 *
 * Observation vector (4D): [x, y, a, h]
 * (only position is observed, velocities are estimated)
 *
 * Model: Constant velocity with linear observation
 * Reference: Deep SORT paper (Wojke et al. 2017)
 */
class KalmanFilter {
 public:
  KalmanFilter();

  /**
   * Initialize new track from measurement
   *
   * Creates initial state estimate with zero velocity and covariance
   * matrix proportional to measurement uncertainty.
   *
   * @param measurement 4D measurement [x, y, a, h]
   * @param mean Output 8D mean vector (position + zero velocity)
   * @param covariance Output 8x8 covariance matrix
   */
  void Initiate(const float measurement[4],
                Eigen::VectorXf& mean,
                Eigen::MatrixXf& covariance);

  /**
   * Predict next state distribution (Kalman prediction step)
   *
   * Propagates state forward in time using constant velocity model:
   *   x_pred = F * x
   *   P_pred = F * P * F' + Q
   *
   * @param mean 8D mean vector (modified in-place)
   * @param covariance 8x8 covariance matrix (modified in-place)
   */
  void Predict(Eigen::VectorXf& mean, Eigen::MatrixXf& covariance);

  /**
   * Update state with measurement (Kalman correction step)
   *
   * Corrects predicted state using measurement:
   *   y = z - H*x (innovation)
   *   S = H*P*H' + R (innovation covariance)
   *   K = P*H' * S^-1 (Kalman gain)
   *   x = x + K*y (updated state)
   *   P = P - K*H*P (updated covariance)
   *
   * @param mean 8D mean vector (modified in-place)
   * @param covariance 8x8 covariance matrix (modified in-place)
   * @param measurement 4D measurement [x, y, a, h]
   */
  void Update(Eigen::VectorXf& mean,
              Eigen::MatrixXf& covariance,
              const float measurement[4]);

  /**
   * Project state to measurement space
   *
   * Projects 8D state to 4D measurement space:
   *   z_proj = H * x
   *   S = H * P * H' + R
   *
   * Used for gating and data association.
   *
   * @param mean 8D mean vector
   * @param covariance 8x8 covariance matrix
   * @param proj_mean Output 4D projected mean
   * @param proj_cov Output 4x4 projected covariance
   */
  void Project(const Eigen::VectorXf& mean,
               const Eigen::MatrixXf& covariance,
               Eigen::VectorXf& proj_mean,
               Eigen::MatrixXf& proj_cov) const;

  /**
   * Compute Mahalanobis distance for gating
   *
   * Calculates squared Mahalanobis distance between state distribution
   * and measurements:
   *   d^2 = (z - H*x)' * S^-1 * (z - H*x)
   *
   * Used to reject outlier associations (chi-square test).
   * Threshold: 9.4877 (4 DOF, 95% confidence)
   *
   * @param mean 8D mean vector
   * @param covariance 8x8 covariance matrix
   * @param measurements Nx4 matrix of N measurements
   * @param only_position If true, use only (x,y) for distance (2 DOF)
   * @return Vector of N squared Mahalanobis distances
   */
  Eigen::VectorXf GatingDistance(const Eigen::VectorXf& mean,
                                  const Eigen::MatrixXf& covariance,
                                  const Eigen::MatrixXf& measurements,
                                  bool only_position = false) const;

  /**
   * Chi-square thresholds for 95% confidence gating
   */
  static constexpr float CHI2_4DOF = 9.4877f;  ///< 4 DOF (full measurement)
  static constexpr float CHI2_2DOF = 5.9915f;  ///< 2 DOF (position only)

 private:
  Eigen::MatrixXf motion_mat_;     ///< 8x8 state transition matrix F
  Eigen::MatrixXf update_mat_;     ///< 4x8 observation matrix H

  // Standard deviation weights for uncertainty modeling
  float std_weight_position_;      ///< Position uncertainty: 1/20
  float std_weight_velocity_;      ///< Velocity uncertainty: 1/160
};

}  // namespace perception
