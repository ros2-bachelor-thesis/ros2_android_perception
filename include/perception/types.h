#pragma once

#include <vector>
#include <cstdint>

#include "perception/detection.h"
#include "perception/track.h"

namespace perception {

/**
 * Complete perception processing result
 *
 * Contains raw YOLO detections, Deep SORT tracks, and annotated
 * visualization frames for debugging.
 *
 * Annotated frames are returned as raw BGR buffers (not cv::Mat)
 * to avoid exposing OpenCV in the public API.
 */
struct PerceptionResult {
  /**
   * Raw YOLO detections (before Deep SORT tracking)
   * These are published to ROS topics when depth+cloud available
   */
  std::vector<Detection> detections;

  /**
   * Deep SORT tracks (for visualization only)
   * Contains track IDs and smoothed bounding boxes
   */
  std::vector<Track> tracks;

  /**
   * RGB image with YOLO bounding boxes + Deep SORT track IDs
   * Format: Raw BGR bytes (interleaved, 8-bit per channel)
   * Size: rgb_width * rgb_height * 3 bytes
   */
  std::vector<uint8_t> annotated_rgb_bgr;
  int rgb_width = 0;
  int rgb_height = 0;

  PerceptionResult() = default;
};

}  // namespace perception
