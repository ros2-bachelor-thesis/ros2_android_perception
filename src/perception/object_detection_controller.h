#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "perception/detection.h"
#include "perception/track.h"

namespace perception {

// Forward declarations of internal components
class NcnnDetector;
class DeepSortTracker;

/**
 * Complete YOLOv9 + Deep SORT object detection and tracking pipeline
 *
 * This class provides a single, simple API for the entire ML pipeline.
 * No ROS 2 dependencies - pure computer vision library.
 *
 * Usage:
 *   ObjectDetectionController detector(yolo_param, yolo_bin, reid_param, reid_bin);
 *   auto tracks = detector.ProcessFrame(image);
 *   for (const auto& track : tracks) {
 *     // Use track.track_id, track.bbox, track.class_id, track.confidence
 *   }
 */
class ObjectDetectionController {
 public:
  /**
   * Constructor - loads NCNN models
   *
   * @param yolo_param Path to yolov9_s_pobed.ncnn.param
   * @param yolo_bin Path to yolov9_s_pobed.ncnn.bin
   * @param reid_param Path to mars-small128.ncnn.param
   * @param reid_bin Path to mars-small128.ncnn.bin
   * @param use_vulkan Use GPU acceleration (default: false, CPU NEON faster on ARM)
   */
  ObjectDetectionController(
      const std::string& yolo_param,
      const std::string& yolo_bin,
      const std::string& reid_param,
      const std::string& reid_bin,
      bool use_vulkan = false);

  ~ObjectDetectionController();

  /**
   * Process single frame - complete detection + tracking pipeline
   *
   * Pipeline:
   * 1. Convert RGB to BGR
   * 2. Preprocess image (letterbox resize to 1280x736, normalize)
   * 3. YOLOv9 detection
   * 4. NMS filtering
   * 5. Extract ReID features
   * 6. Deep SORT tracking (Kalman + Hungarian matching)
   * 7. Return confirmed tracks only (hits >= n_init)
   *
   * @param rgb_data Raw RGB buffer (interleaved RGB, 8-bit per channel)
   * @param width Image width in pixels
   * @param height Image height in pixels
   * @param conf_threshold Confidence threshold (default: 0.25)
   * @param iou_threshold NMS IoU threshold (default: 0.45)
   * @return Vector of confirmed tracks (sorted by track_id)
   */
  std::vector<Track> ProcessFrame(
      const uint8_t* rgb_data,
      int width,
      int height,
      float conf_threshold = 0.25f,
      float iou_threshold = 0.45f);

  /**
   * Check if models loaded successfully
   */
  bool IsReady() const { return ready_; }

  /**
   * Get total number of active tracks (confirmed + tentative)
   */
  size_t GetTrackCount() const;

  /**
   * Get only confirmed tracks from last ProcessFrame call
   */
  std::vector<Track> GetConfirmedTracks() const;

  /**
   * Reset tracker state (clears all tracks)
   */
  void Reset();

 private:
  std::unique_ptr<NcnnDetector> detector_;
  std::unique_ptr<DeepSortTracker> tracker_;
  bool ready_;

  // Cache last results
  std::vector<Track> last_confirmed_tracks_;
};

}  // namespace perception
