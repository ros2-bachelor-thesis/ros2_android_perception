#include "perception/nms.h"
#include <algorithm>
#include <cmath>

namespace perception {

float NMS::CalculateArea(const float bbox[4]) {
  float width = bbox[2] - bbox[0];
  float height = bbox[3] - bbox[1];
  return std::max(0.0f, width) * std::max(0.0f, height);
}

float NMS::CalculateIntersection(const float bbox1[4], const float bbox2[4]) {
  // Calculate intersection rectangle
  float x1 = std::max(bbox1[0], bbox2[0]);
  float y1 = std::max(bbox1[1], bbox2[1]);
  float x2 = std::min(bbox1[2], bbox2[2]);
  float y2 = std::min(bbox1[3], bbox2[3]);

  // Calculate intersection area
  float width = std::max(0.0f, x2 - x1);
  float height = std::max(0.0f, y2 - y1);

  return width * height;
}

float NMS::CalculateIoU(const float bbox1[4], const float bbox2[4]) {
  // Calculate areas
  float area1 = CalculateArea(bbox1);
  float area2 = CalculateArea(bbox2);

  // Calculate intersection
  float intersection = CalculateIntersection(bbox1, bbox2);

  // Calculate union
  float union_area = area1 + area2 - intersection;

  // Avoid division by zero
  if (union_area <= 0.0f) {
    return 0.0f;
  }

  // Return IoU
  return intersection / union_area;
}

float NMS::CalculateIoU(const Detection& det1, const Detection& det2) {
  return CalculateIoU(det1.bbox, det2.bbox);
}

std::vector<Detection> NMS::Apply(std::vector<Detection>& detections,
                                   float iou_threshold) {
  // Return early if empty
  if (detections.empty()) {
    return {};
  }

  // Sort detections by confidence in descending order
  std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) {
              return a.confidence > b.confidence;
            });

  std::vector<Detection> result;
  std::vector<bool> suppressed(detections.size(), false);

  // Greedy NMS algorithm
  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }

    // Keep this detection
    result.push_back(detections[i]);

    // Suppress overlapping detections with lower confidence
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }

      // Calculate IoU with current detection
      float iou = CalculateIoU(detections[i], detections[j]);

      // Suppress if IoU exceeds threshold
      if (iou > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

std::vector<Detection> NMS::ApplyPerClass(std::vector<Detection> detections,
                                           float iou_threshold) {
  // Return early if empty
  if (detections.empty()) {
    return {};
  }

  // Find all unique class IDs
  std::vector<int> class_ids;
  for (const auto& det : detections) {
    if (std::find(class_ids.begin(), class_ids.end(), det.class_id) == class_ids.end()) {
      class_ids.push_back(det.class_id);
    }
  }

  // Apply NMS per class
  std::vector<Detection> result;
  for (int class_id : class_ids) {
    // Filter detections by class
    std::vector<Detection> class_detections;
    for (auto& det : detections) {
      if (det.class_id == class_id) {
        class_detections.push_back(det);
      }
    }

    // Apply NMS to this class
    std::vector<Detection> class_result = Apply(class_detections, iou_threshold);

    // Add to final result
    result.insert(result.end(), class_result.begin(), class_result.end());
  }

  return result;
}

}  // namespace perception
