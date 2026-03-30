#pragma once

#include "perception/detection.h"
#include <vector>

namespace perception {

/**
 * Non-Maximum Suppression utilities
 *
 * Implements IoU-based NMS for filtering overlapping detections.
 */
class NMS {
 public:
  /**
   * Calculate Intersection over Union (IoU) between two bounding boxes
   *
   * @param bbox1 First bounding box [x1, y1, x2, y2]
   * @param bbox2 Second bounding box [x1, y1, x2, y2]
   * @return IoU score [0.0, 1.0]
   */
  static float CalculateIoU(const float bbox1[4], const float bbox2[4]);

  /**
   * Calculate IoU between two detections
   *
   * @param det1 First detection
   * @param det2 Second detection
   * @return IoU score [0.0, 1.0]
   */
  static float CalculateIoU(const Detection& det1, const Detection& det2);

  /**
   * Apply Non-Maximum Suppression (greedy algorithm)
   *
   * Filters detections by removing overlapping boxes with lower confidence.
   * Detections are sorted by confidence in descending order, then iteratively
   * filtered by IoU threshold.
   *
   * @param detections Input detections (will be modified - sorted by confidence)
   * @param iou_threshold IoU threshold for suppression (default: 0.45)
   * @return Filtered detections after NMS
   */
  static std::vector<Detection> Apply(std::vector<Detection>& detections,
                                       float iou_threshold = 0.45f);

  /**
   * Apply class-aware NMS
   *
   * Performs NMS separately for each class to avoid suppressing
   * overlapping detections of different classes.
   *
   * @param detections Input detections
   * @param iou_threshold IoU threshold for suppression (default: 0.45)
   * @return Filtered detections after class-aware NMS
   */
  static std::vector<Detection> ApplyPerClass(std::vector<Detection> detections,
                                                float iou_threshold = 0.45f);

 private:
  /**
   * Calculate area of bounding box
   */
  static float CalculateArea(const float bbox[4]);

  /**
   * Calculate intersection area between two bounding boxes
   */
  static float CalculateIntersection(const float bbox1[4], const float bbox2[4]);
};

}  // namespace perception
