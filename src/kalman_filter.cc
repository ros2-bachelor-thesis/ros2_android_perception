#include "perception/kalman_filter.h"
#include <Eigen/Cholesky>

namespace perception {

KalmanFilter::KalmanFilter()
    : std_weight_position_(1.0f / 20.0f),
      std_weight_velocity_(1.0f / 160.0f) {

  // Initialize 8x8 motion matrix F (constant velocity model, dt=1)
  // State: [x, y, a, h, vx, vy, va, vh]
  // Transition: position += velocity
  motion_mat_ = Eigen::MatrixXf::Identity(8, 8);
  for (int i = 0; i < 4; i++) {
    motion_mat_(i, i + 4) = 1.0f;  // position[i] += velocity[i] * dt
  }

  // Initialize 4x8 update matrix H (observation model)
  // Observation: [x, y, a, h] (only position, not velocity)
  update_mat_ = Eigen::MatrixXf::Zero(4, 8);
  for (int i = 0; i < 4; i++) {
    update_mat_(i, i) = 1.0f;  // observe position directly
  }
}

void KalmanFilter::Initiate(const float measurement[4],
                             Eigen::VectorXf& mean,
                             Eigen::MatrixXf& covariance) {
  // Initialize mean: [x, y, a, h, 0, 0, 0, 0]
  mean = Eigen::VectorXf::Zero(8);
  mean(0) = measurement[0];  // x
  mean(1) = measurement[1];  // y
  mean(2) = measurement[2];  // aspect ratio
  mean(3) = measurement[3];  // height
  // velocities initialized to zero (indices 4-7)

  // Initialize covariance: diagonal matrix with uncertainty proportional to height
  std::vector<float> std(8);
  std[0] = 2.0f * std_weight_position_ * measurement[3];  // std_x
  std[1] = 2.0f * std_weight_position_ * measurement[3];  // std_y
  std[2] = 1e-2f;                                          // std_a (small, aspect ratio stable)
  std[3] = 2.0f * std_weight_position_ * measurement[3];  // std_h
  std[4] = 10.0f * std_weight_velocity_ * measurement[3]; // std_vx
  std[5] = 10.0f * std_weight_velocity_ * measurement[3]; // std_vy
  std[6] = 1e-5f;                                          // std_va (very small)
  std[7] = 10.0f * std_weight_velocity_ * measurement[3]; // std_vh

  covariance = Eigen::MatrixXf::Zero(8, 8);
  for (int i = 0; i < 8; i++) {
    covariance(i, i) = std[i] * std[i];  // variance = std^2
  }
}

void KalmanFilter::Predict(Eigen::VectorXf& mean,
                            Eigen::MatrixXf& covariance) {
  // Predict state: x = F * x
  mean = motion_mat_ * mean;

  // Process noise covariance Q (uncertainty grows with prediction)
  std::vector<float> std_pos(4);
  std_pos[0] = std_weight_position_ * mean(3);  // std_x
  std_pos[1] = std_weight_position_ * mean(3);  // std_y
  std_pos[2] = 1e-2f;                           // std_a
  std_pos[3] = std_weight_position_ * mean(3);  // std_h

  std::vector<float> std_vel(4);
  std_vel[0] = std_weight_velocity_ * mean(3);  // std_vx
  std_vel[1] = std_weight_velocity_ * mean(3);  // std_vy
  std_vel[2] = 1e-5f;                           // std_va
  std_vel[3] = std_weight_velocity_ * mean(3);  // std_vh

  Eigen::MatrixXf motion_cov = Eigen::MatrixXf::Zero(8, 8);
  for (int i = 0; i < 4; i++) {
    motion_cov(i, i) = std_pos[i] * std_pos[i];
    motion_cov(i + 4, i + 4) = std_vel[i] * std_vel[i];
  }

  // Predict covariance: P = F * P * F' + Q
  covariance = motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;
}

void KalmanFilter::Project(const Eigen::VectorXf& mean,
                            const Eigen::MatrixXf& covariance,
                            Eigen::VectorXf& proj_mean,
                            Eigen::MatrixXf& proj_cov) {
  // Measurement noise covariance R
  std::vector<float> std(4);
  std[0] = std_weight_position_ * mean(3);  // std_x
  std[1] = std_weight_position_ * mean(3);  // std_y
  std[2] = 1e-1f;                           // std_a
  std[3] = std_weight_position_ * mean(3);  // std_h

  Eigen::MatrixXf innovation_cov = Eigen::MatrixXf::Zero(4, 4);
  for (int i = 0; i < 4; i++) {
    innovation_cov(i, i) = std[i] * std[i];
  }

  // Project mean: z = H * x
  proj_mean = update_mat_ * mean;

  // Project covariance: S = H * P * H' + R
  proj_cov = update_mat_ * covariance * update_mat_.transpose() + innovation_cov;
}

void KalmanFilter::Update(Eigen::VectorXf& mean,
                           Eigen::MatrixXf& covariance,
                           const float measurement[4]) {
  // Project state to measurement space
  Eigen::VectorXf proj_mean;
  Eigen::MatrixXf proj_cov;
  Project(mean, covariance, proj_mean, proj_cov);

  // Compute Kalman gain: K = P * H' * S^-1
  // Use Cholesky decomposition for numerical stability
  Eigen::LLT<Eigen::MatrixXf> llt(proj_cov);
  Eigen::MatrixXf cov_update_t = covariance * update_mat_.transpose();
  Eigen::MatrixXf kalman_gain = llt.solve(cov_update_t.transpose()).transpose();

  // Innovation: y = z - H*x
  Eigen::VectorXf innovation(4);
  for (int i = 0; i < 4; i++) {
    innovation(i) = measurement[i] - proj_mean(i);
  }

  // Update mean: x = x + K * y
  mean = mean + kalman_gain * innovation;

  // Update covariance: P = P - K * H * P
  covariance = covariance - kalman_gain * update_mat_ * covariance;
}

Eigen::VectorXf KalmanFilter::GatingDistance(
    const Eigen::VectorXf& mean,
    const Eigen::MatrixXf& covariance,
    const Eigen::MatrixXf& measurements,
    bool only_position) {

  // Project state to measurement space
  Eigen::VectorXf proj_mean;
  Eigen::MatrixXf proj_cov;
  Project(mean, covariance, proj_mean, proj_cov);

  // If only position, reduce to 2D (x, y)
  if (only_position) {
    proj_mean = proj_mean.head(2);
    proj_cov = proj_cov.topLeftCorner(2, 2);
  }

  int dim = proj_mean.size();
  int num_measurements = measurements.rows();

  // Compute Cholesky decomposition of projected covariance
  Eigen::LLT<Eigen::MatrixXf> llt(proj_cov);
  Eigen::MatrixXf cholesky = llt.matrixL();

  // Compute squared Mahalanobis distance for each measurement
  Eigen::VectorXf distances(num_measurements);

  for (int i = 0; i < num_measurements; i++) {
    // Innovation: d = z_i - proj_mean
    Eigen::VectorXf d;
    if (only_position) {
      d = measurements.row(i).head(2).transpose() - proj_mean;
    } else {
      d = measurements.row(i).transpose() - proj_mean;
    }

    // Solve: cholesky * z = d (forward substitution)
    Eigen::VectorXf z = llt.matrixL().solve(d);

    // Squared Mahalanobis distance: d' * S^-1 * d = z' * z
    distances(i) = z.squaredNorm();
  }

  return distances;
}

}  // namespace perception
