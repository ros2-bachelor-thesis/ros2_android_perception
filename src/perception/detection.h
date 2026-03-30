#pragma once

#include <vector>

namespace perception {

/**
 * Detection result from YOLOv9 object detector
 *
 * Represents a single detected object with bounding box,
 * confidence score, class ID, and optional appearance feature
 * for tracking.
 */
struct Detection {
  /** Bounding box in absolute pixel coordinates [x1, y1, x2, y2] */
  float bbox[4];

  /** Detection confidence score [0.0, 1.0] */
  float confidence;

  /** Class ID: 0=cpb_beetle, 1=cpb_larva, 2=cpb_eggs */
  int class_id;

  /** 128-dimensional appearance feature from ReID network (mars-small128) */
  std::vector<float> feature;

  Detection() : confidence(0.0f), class_id(-1) {
    bbox[0] = bbox[1] = bbox[2] = bbox[3] = 0.0f;
  }

  Detection(float x1, float y1, float x2, float y2, float conf, int cls)
      : confidence(conf), class_id(cls) {
    bbox[0] = x1;
    bbox[1] = y1;
    bbox[2] = x2;
    bbox[3] = y2;
  }

  /** Get bounding box center X coordinate */
  float CenterX() const { return (bbox[0] + bbox[2]) / 2.0f; }

  /** Get bounding box center Y coordinate */
  float CenterY() const { return (bbox[1] + bbox[3]) / 2.0f; }

  /** Get bounding box width */
  float Width() const { return bbox[2] - bbox[0]; }

  /** Get bounding box height */
  float Height() const { return bbox[3] - bbox[1]; }

  /** Get bounding box area */
  float Area() const { return Width() * Height(); }
};

}  // namespace perception
