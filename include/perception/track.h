#pragma once

#include <Eigen/Dense>
#include <vector>

namespace perception {

/**
 * Track state for Deep SORT multi-object tracking
 *
 * Represents a tracked object across multiple frames with
 * Kalman filter state, appearance features, and lifecycle management.
 */
struct Track {
  /** Unique track identifier (assigned sequentially) */
  int track_id;

  /** Current bounding box in absolute pixel coordinates [x1, y1, x2, y2] */
  float bbox[4];

  /**
   * Kalman filter state vector (8D):
   * [x, y, a, h, vx, vy, va, vh]
   * where (x,y) is center, a is aspect ratio, h is height, v* are velocities
   */
  Eigen::VectorXf state;

  /**
   * Kalman filter covariance matrix (8x8)
   */
  Eigen::MatrixXf covariance;

  /**
   * Smoothed appearance feature (512-dim from OSNet-AIN)
   * Updated as exponential moving average of detection features
   */
  std::vector<float> feature;

  /** Number of frames since last successful detection match */
  int time_since_update;

  /** Total number of successful detection matches */
  int hits;

  /**
   * Track confirmation status
   * Confirmed after N consecutive hits (default N=3)
   */
  bool is_confirmed;

  /** Class ID inherited from detection: 0=cpb_beetle, 1=cpb_larva, 2=cpb_eggs */
  int class_id;

  Track()
      : track_id(-1),
        state(Eigen::VectorXf::Zero(8)),
        covariance(Eigen::MatrixXf::Identity(8, 8)),
        time_since_update(0),
        hits(0),
        is_confirmed(false),
        class_id(-1) {
    bbox[0] = bbox[1] = bbox[2] = bbox[3] = 0.0f;
  }

  Track(int id, const float* detection_bbox, int cls)
      : track_id(id),
        state(Eigen::VectorXf::Zero(8)),
        covariance(Eigen::MatrixXf::Identity(8, 8)),
        time_since_update(0),
        hits(1),
        is_confirmed(false),
        class_id(cls) {
    bbox[0] = detection_bbox[0];
    bbox[1] = detection_bbox[1];
    bbox[2] = detection_bbox[2];
    bbox[3] = detection_bbox[3];

    // Initialize state from detection bbox
    float cx = (detection_bbox[0] + detection_bbox[2]) / 2.0f;
    float cy = (detection_bbox[1] + detection_bbox[3]) / 2.0f;
    float w = detection_bbox[2] - detection_bbox[0];
    float h = detection_bbox[3] - detection_bbox[1];

    state(0) = cx;  // x
    state(1) = cy;  // y
    state(2) = w;   // width
    state(3) = h;   // height
    // velocities initialized to zero (indices 4-7)
  }

  /** Check if track should be deleted (exceeds max age) */
  bool ShouldDelete(int max_age = 30) const {
    return time_since_update > max_age;
  }

  /** Check if track should be confirmed (minimum consecutive hits) */
  bool ShouldConfirm(int min_hits = 3) const {
    return hits >= min_hits && time_since_update == 0;
  }

  /** Update bbox from Kalman state */
  void UpdateBboxFromState() {
    float cx = state(0);
    float cy = state(1);
    float w = state(2);
    float h = state(3);

    bbox[0] = cx - w / 2.0f;  // x1
    bbox[1] = cy - h / 2.0f;  // y1
    bbox[2] = cx + w / 2.0f;  // x2
    bbox[3] = cy + h / 2.0f;  // y2
  }
};

}  // namespace perception
