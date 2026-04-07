#include "perception/linear_assignment.h"
#include <algorithm>
#include <limits>
#include <set>

namespace perception {

// Optimal O(n^3) Hungarian (Kuhn-Munkres) algorithm
// Matches scipy.optimize.linear_sum_assignment used in Python Deep SORT
std::vector<std::pair<int, int>> LinearAssignment::Hungarian(
    const Eigen::MatrixXf& cost_matrix) {
  int rows = cost_matrix.rows();
  int cols = cost_matrix.cols();

  if (rows == 0 || cols == 0) {
    return {};
  }

  // Pad to square matrix (algorithm requires square)
  // Pad with INFTY_COST so dummy assignments are never preferred over real ones
  int n = std::max(rows, cols);
  Eigen::MatrixXf C = Eigen::MatrixXf::Constant(n, n, INFTY_COST);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      C(i, j) = cost_matrix(i, j);
    }
  }

  // u[i], v[j] = potentials for row i and column j
  std::vector<float> u(n + 1, 0.0f), v(n + 1, 0.0f);
  // p[j] = row assigned to column j, way[j] = predecessor column
  std::vector<int> p(n + 1, 0), way(n + 1, 0);

  for (int i = 1; i <= n; i++) {
    p[0] = i;
    int j0 = 0;
    std::vector<float> minv(n + 1, std::numeric_limits<float>::max());
    std::vector<bool> used(n + 1, false);

    do {
      used[j0] = true;
      int i0 = p[j0], j1 = 0;
      float delta = std::numeric_limits<float>::max();

      for (int j = 1; j <= n; j++) {
        if (!used[j]) {
          float cur = C(i0 - 1, j - 1) - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      }

      for (int j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }

      j0 = j1;
    } while (p[j0] != 0);

    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  // Extract assignments (only for original rows/cols)
  std::vector<std::pair<int, int>> assignments;
  for (int j = 1; j <= n; j++) {
    if (p[j] > 0 && p[j] <= rows && j <= cols) {
      assignments.push_back({p[j] - 1, j - 1});
    }
  }

  return assignments;
}

MatchResult LinearAssignment::MinCostMatching(
    const Eigen::MatrixXf& cost_matrix,
    float max_distance,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices) {

  MatchResult result;

  // Handle empty cases
  if (track_indices.empty() || detection_indices.empty()) {
    result.unmatched_tracks = track_indices;
    result.unmatched_detections = detection_indices;
    return result;
  }

  // Create cost matrix subset
  int num_tracks = track_indices.size();
  int num_detections = detection_indices.size();
  Eigen::MatrixXf subset_cost(num_tracks, num_detections);

  for (int i = 0; i < num_tracks; i++) {
    for (int j = 0; j < num_detections; j++) {
      float cost = cost_matrix(track_indices[i], detection_indices[j]);
      // Mask costs exceeding threshold
      subset_cost(i, j) = (cost > max_distance) ? INFTY_COST : cost;
    }
  }

  // Run Hungarian algorithm
  auto assignments = Hungarian(subset_cost);

  // Track which indices are matched
  std::set<int> matched_track_idx;
  std::set<int> matched_detection_idx;

  // Process assignments
  for (const auto& [row, col] : assignments) {
    float cost = subset_cost(row, col);
    if (cost > max_distance) {
      // Invalid assignment (cost too high)
      continue;
    }

    int track_idx = track_indices[row];
    int detection_idx = detection_indices[col];

    result.matches.push_back({track_idx, detection_idx});
    matched_track_idx.insert(track_idx);
    matched_detection_idx.insert(detection_idx);
  }

  // Collect unmatched tracks
  for (int idx : track_indices) {
    if (!matched_track_idx.count(idx)) {
      result.unmatched_tracks.push_back(idx);
    }
  }

  // Collect unmatched detections
  for (int idx : detection_indices) {
    if (!matched_detection_idx.count(idx)) {
      result.unmatched_detections.push_back(idx);
    }
  }

  return result;
}

MatchResult LinearAssignment::MinCostMatchingLocal(
    const Eigen::MatrixXf& cost_matrix,
    float max_distance,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices) {

  MatchResult result;

  if (track_indices.empty() || detection_indices.empty()) {
    result.unmatched_tracks = track_indices;
    result.unmatched_detections = detection_indices;
    return result;
  }

  int num_tracks = track_indices.size();
  int num_detections = detection_indices.size();

  // Cost matrix is already locally indexed (rows=0..N-1, cols=0..M-1)
  // Mask costs exceeding threshold
  Eigen::MatrixXf masked_cost(num_tracks, num_detections);
  for (int i = 0; i < num_tracks; i++) {
    for (int j = 0; j < num_detections; j++) {
      float cost = cost_matrix(i, j);
      masked_cost(i, j) = (cost > max_distance) ? INFTY_COST : cost;
    }
  }

  auto assignments = Hungarian(masked_cost);

  std::set<int> matched_track_idx;
  std::set<int> matched_detection_idx;

  for (const auto& [row, col] : assignments) {
    if (masked_cost(row, col) > max_distance) {
      continue;
    }

    int track_idx = track_indices[row];
    int detection_idx = detection_indices[col];

    result.matches.push_back({track_idx, detection_idx});
    matched_track_idx.insert(track_idx);
    matched_detection_idx.insert(detection_idx);
  }

  for (int idx : track_indices) {
    if (!matched_track_idx.count(idx)) {
      result.unmatched_tracks.push_back(idx);
    }
  }

  for (int idx : detection_indices) {
    if (!matched_detection_idx.count(idx)) {
      result.unmatched_detections.push_back(idx);
    }
  }

  return result;
}

MatchResult LinearAssignment::MatchingCascade(
    const Eigen::MatrixXf& cost_matrix,
    float max_distance,
    int cascade_depth,
    const std::vector<Track>& tracks,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices) {

  MatchResult result;
  std::vector<int> unmatched_detections = detection_indices;

  // Cascade matching: prioritize tracks by time_since_update
  // Python parity (linear_assignment.py:124-139)
  for (int level = 0; level < cascade_depth; level++) {
    if (unmatched_detections.empty()) {
      break;
    }

    // Find tracks at this cascade level
    // Python: tracks[k].time_since_update == 1 + level
    std::vector<int> level_track_indices;
    for (int idx : track_indices) {
      if (tracks[idx].time_since_update == 1 + level) {
        level_track_indices.push_back(idx);
      }
    }

    if (level_track_indices.empty()) {
      continue;
    }

    // Match tracks at this level with remaining detections
    MatchResult level_result = MinCostMatching(
        cost_matrix, max_distance,
        level_track_indices, unmatched_detections);

    // Accumulate matches
    result.matches.insert(result.matches.end(),
                          level_result.matches.begin(),
                          level_result.matches.end());

    // Remove matched detections from pool for next level
    unmatched_detections = level_result.unmatched_detections;
  }

  // Python parity (linear_assignment.py:140):
  // unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
  std::set<int> matched_track_set;
  for (const auto& [t, d] : result.matches) {
    matched_track_set.insert(t);
  }
  for (int idx : track_indices) {
    if (!matched_track_set.count(idx)) {
      result.unmatched_tracks.push_back(idx);
    }
  }

  // Remaining unmatched detections
  result.unmatched_detections = unmatched_detections;

  return result;
}

void LinearAssignment::GateCostMatrix(
    Eigen::MatrixXf& cost_matrix,
    const KalmanFilter& kf,
    const std::vector<Track>& tracks,
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices,
    float gating_threshold) {

  // Build measurements matrix from detections
  int num_detections = detection_indices.size();
  Eigen::MatrixXf measurements(num_detections, 4);

  for (int i = 0; i < num_detections; i++) {
    const Detection& det = detections[detection_indices[i]];
    float cx = (det.bbox[0] + det.bbox[2]) / 2.0f;
    float cy = (det.bbox[1] + det.bbox[3]) / 2.0f;
    float w = det.bbox[2] - det.bbox[0];
    float h = det.bbox[3] - det.bbox[1];
    float a = (h > 0.0f) ? (w / h) : 1.0f;

    measurements(i, 0) = cx;
    measurements(i, 1) = cy;
    measurements(i, 2) = a;
    measurements(i, 3) = h;
  }

  // Gate each track-detection pair
  for (size_t i = 0; i < track_indices.size(); i++) {
    const Track& track = tracks[track_indices[i]];

    // Compute gating distances for all detections using track's covariance
    Eigen::VectorXf distances = kf.GatingDistance(
        track.state, track.covariance, measurements, false);

    // Set cost to infinity for gates that exceed threshold
    for (int j = 0; j < num_detections; j++) {
      if (distances(j) > gating_threshold) {
        cost_matrix(track_indices[i], detection_indices[j]) = INFTY_COST;
      }
    }
  }
}

float LinearAssignment::CalculateIoU(const float bbox1[4], const float bbox2[4]) {
  // Calculate intersection
  float x1 = std::max(bbox1[0], bbox2[0]);
  float y1 = std::max(bbox1[1], bbox2[1]);
  float x2 = std::min(bbox1[2], bbox2[2]);
  float y2 = std::min(bbox1[3], bbox2[3]);

  float inter_w = std::max(0.0f, x2 - x1);
  float inter_h = std::max(0.0f, y2 - y1);
  float inter_area = inter_w * inter_h;

  // Calculate union
  float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
  float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
  float union_area = area1 + area2 - inter_area;

  if (union_area <= 0.0f) {
    return 0.0f;
  }

  return inter_area / union_area;
}

Eigen::MatrixXf LinearAssignment::IoUCostMatrix(
    const std::vector<Track>& tracks,
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices) {

  int num_tracks = track_indices.size();
  int num_detections = detection_indices.size();

  Eigen::MatrixXf cost_matrix(num_tracks, num_detections);

  for (int i = 0; i < num_tracks; i++) {
    for (int j = 0; j < num_detections; j++) {
      const Track& track = tracks[track_indices[i]];
      const Detection& det = detections[detection_indices[j]];

      float iou = CalculateIoU(track.bbox, det.bbox);
      cost_matrix(i, j) = 1.0f - iou;  // Cost = 1 - IoU
    }
  }

  return cost_matrix;
}

}  // namespace perception
