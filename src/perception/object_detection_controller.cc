#include "perception/object_detection_controller.h"

namespace perception {

ObjectDetectionController::ObjectDetectionController(
    const std::string& yolo_param,
    const std::string& yolo_bin,
    const std::string& reid_param,
    const std::string& reid_bin,
    bool use_vulkan)
    : ready_(false) {

  // Load YOLOv9 detector
  detector_ = std::make_unique<NcnnDetector>(
      yolo_param, yolo_bin, use_vulkan);

  if (!detector_->IsLoaded()) {
    // Model load failed
    return;
  }

  // Load Deep SORT tracker (includes ReID model)
  DeepSortConfig config;
  // Use default config:
  // - max_cosine_distance: 0.4
  // - nn_budget: 100
  // - max_age: 30 frames
  // - n_init: 3 frames to confirm
  // - max_iou_distance: 0.7

  tracker_ = std::make_unique<DeepSortTracker>(
      reid_param, reid_bin, config);

  if (!tracker_->IsReady()) {
    // ReID model load failed
    return;
  }

  ready_ = true;
}

ObjectDetectionController::~ObjectDetectionController() {
  // Unique_ptr handles cleanup
}

std::vector<Track> ObjectDetectionController::ProcessFrame(
    const cv::Mat& image,
    float conf_threshold,
    float iou_threshold) {

  if (!ready_ || image.empty()) {
    return {};
  }

  // Step 1: Run YOLOv9 detector
  // - Preprocesses image (letterbox resize to 1280x736, normalize)
  // - NCNN inference
  // - Decodes output (bbox, confidence, class)
  // - Applies NMS
  auto detections = detector_->Detect(image, conf_threshold, iou_threshold);

  // Step 2: Update Deep SORT tracker
  // - Extracts ReID features for each detection
  // - Kalman filter prediction
  // - Cascade matching (appearance + motion)
  // - IoU matching for unmatched
  // - Updates confirmed tracks, creates new tentative tracks
  auto all_tracks = tracker_->Update(image, detections);

  // Step 3: Get only confirmed tracks (hits >= 3)
  last_confirmed_tracks_ = tracker_->GetConfirmedTracks();

  return last_confirmed_tracks_;
}

size_t ObjectDetectionController::GetTrackCount() const {
  return tracker_ ? tracker_->GetTrackCount() : 0;
}

std::vector<Track> ObjectDetectionController::GetConfirmedTracks() const {
  return last_confirmed_tracks_;
}

void ObjectDetectionController::Reset() {
  // Clear cached tracks
  last_confirmed_tracks_.clear();

  // Note: To fully reset tracker state, would need to recreate tracker
  // but that requires storing model paths. For now, just clear cache.
  // Full reset can be implemented if needed by storing constructor params.
}

}  // namespace perception
