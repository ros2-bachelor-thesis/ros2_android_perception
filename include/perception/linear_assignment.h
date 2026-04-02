#pragma once

#include <Eigen/Dense>
#include <vector>
#include <utility>

#include "perception/detection.h"
#include "perception/track.h"
#include "perception/kalman_filter.h"

namespace perception {

/**
 * Result of linear assignment matching
 */
struct MatchResult {
  std::vector<std::pair<int, int>> matches;  ///< (track_idx, detection_idx) pairs
  std::vector<int> unmatched_tracks;         ///< Track indices without matches
  std::vector<int> unmatched_detections;     ///< Detection indices without matches
};

/**
 * Linear assignment for track-detection association
 *
 * Implements Hungarian algorithm for optimal assignment and
 * cascade matching strategy from Deep SORT paper.
 */
class LinearAssignment {
 public:
  /**
   * Solve minimum cost matching with Hungarian algorithm
   *
   * Finds optimal assignment minimizing total cost. Assignments
   * with cost > max_distance are rejected.
   *
   * @param cost_matrix NxM cost matrix (tracks x detections)
   * @param max_distance Maximum allowed cost (assignments above this are invalid)
   * @param track_indices Indices of tracks (maps rows to track list)
   * @param detection_indices Indices of detections (maps cols to detection list)
   * @return Match result with assignments and unmatched indices
   */
  static MatchResult MinCostMatching(
      const Eigen::MatrixXf& cost_matrix,
      float max_distance,
      const std::vector<int>& track_indices,
      const std::vector<int>& detection_indices);

  /**
   * Cascade matching strategy from Deep SORT
   *
   * Prioritizes recently seen tracks by running matching at different
   * cascade levels based on time_since_update. Removes matched detections
   * from pool after each level.
   *
   * Level 0: tracks with time_since_update == 0 (just updated)
   * Level 1: tracks with time_since_update == 1
   * ...
   * Level N: tracks with time_since_update == N (up to max_age)
   *
   * @param cost_matrix NxM cost matrix (tracks x detections)
   * @param max_distance Maximum allowed cost
   * @param cascade_depth Maximum cascade level (usually max_age = 30)
   * @param tracks List of all tracks
   * @param track_indices Indices of tracks to match
   * @param detection_indices Indices of detections available
   * @return Match result after cascade matching
   */
  static MatchResult MatchingCascade(
      const Eigen::MatrixXf& cost_matrix,
      float max_distance,
      int cascade_depth,
      const std::vector<Track>& tracks,
      const std::vector<int>& track_indices,
      const std::vector<int>& detection_indices);

  /**
   * Gate cost matrix using Mahalanobis distance
   *
   * Sets cost to infinity for track-detection pairs that are too far
   * apart (unlikely associations). Uses Kalman filter's gating distance
   * with chi-square threshold.
   *
   * @param cost_matrix NxM cost matrix (modified in-place)
   * @param kf Kalman filter for gating distance computation
   * @param tracks List of tracks
   * @param detections List of detections
   * @param track_indices Track indices (maps cost matrix rows)
   * @param detection_indices Detection indices (maps cost matrix cols)
   * @param gating_threshold Chi-square threshold (default: 9.4877 for 4 DOF)
   */
  static void GateCostMatrix(
      Eigen::MatrixXf& cost_matrix,
      const KalmanFilter& kf,
      const std::vector<Track>& tracks,
      const std::vector<Detection>& detections,
      const std::vector<int>& track_indices,
      const std::vector<int>& detection_indices,
      float gating_threshold = KalmanFilter::CHI2_4DOF);

  /**
   * Compute IoU-based cost matrix
   *
   * Cost = 1 - IoU (lower cost = better match)
   * Used as fallback when appearance features unavailable.
   *
   * @param tracks List of tracks
   * @param detections List of detections
   * @param track_indices Track indices to include
   * @param detection_indices Detection indices to include
   * @return Cost matrix (NxM, tracks x detections)
   */
  static Eigen::MatrixXf IoUCostMatrix(
      const std::vector<Track>& tracks,
      const std::vector<Detection>& detections,
      const std::vector<int>& track_indices,
      const std::vector<int>& detection_indices);

 private:
  /**
   * Hungarian algorithm implementation
   *
   * Solves assignment problem in O(n^3) time.
   *
   * @param cost_matrix NxM cost matrix
   * @return Vector of (row, col) assignment pairs
   */
  static std::vector<std::pair<int, int>> Hungarian(
      const Eigen::MatrixXf& cost_matrix);

  /**
   * Calculate IoU between two bounding boxes
   */
  static float CalculateIoU(const float bbox1[4], const float bbox2[4]);

  static constexpr float INFTY_COST = 1e5f;  ///< Infinity cost for invalid assignments
};

}  // namespace perception
