#include "perception/object_detection_controller.h"

#include <chrono>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <ncnn/gpu.h>

#include "perception/types.h"
#include "perception/ncnn_detector.h"
#include "perception/deep_sort_tracker.h"
#include "perception/visualization/annotator.h"
#include "perception/log.h"

namespace perception
{

  namespace
  {
    /**
     * Detect if Vulkan GPU acceleration is available
     * @return true if Vulkan is supported, false otherwise
     */
    bool HasVulkanSupport()
    {
      ncnn::create_gpu_instance();
      int gpu_count = ncnn::get_gpu_count();
      ncnn::destroy_gpu_instance();

      if (gpu_count > 0)
      {
        LOGI("Vulkan GPU detected: %d device(s) available", gpu_count);
        return true;
      }
      else
      {
        LOGI("No Vulkan GPU detected, will use CPU");
        return false;
      }
    }
  } // anonymous namespace

  ObjectDetectionController::ObjectDetectionController(
      const std::string &yolo_param,
      const std::string &yolo_bin,
      const std::string &reid_param,
      const std::string &reid_bin,
      bool use_vulkan)
      : ready_(false)
  {

    // Auto-detect Vulkan support if requested
    bool vulkan_enabled = use_vulkan && HasVulkanSupport();

    if (use_vulkan && !vulkan_enabled)
    {
      LOGW("Vulkan requested but not available, falling back to CPU");
    }
    else if (vulkan_enabled)
    {
      LOGD("Vulkan GPU acceleration enabled for perception");
    }
    else
    {
      LOGD("Using CPU (NEON) for perception");
    }

    // Load YOLOv9 detector
    detector_ = std::make_unique<NcnnDetector>(
        yolo_param, yolo_bin, vulkan_enabled);

    if (!detector_->IsLoaded())
    {
      // Model load failed
      return;
    }

    // Load Deep SORT tracker (includes ReID model)
    DeepSortConfig config;
    // Tuned for lower false positives:
    config.max_cosine_distance = 0.3f; // Stricter appearance matching (was 0.4)
    config.nn_budget = 100;            // Feature gallery size (Android memory constraint)
    config.max_age = 15;               // Delete lost tracks faster (was 30)
    config.n_init = 5;                 // Require more hits to confirm (was 3)
    config.max_iou_distance = 0.7f;    // IoU fallback matching

    tracker_ = std::make_unique<DeepSortTracker>(
        reid_param, reid_bin, config);

    if (!tracker_->IsReady())
    {
      // ReID model load failed
      return;
    }

    ready_ = true;
  }

  ObjectDetectionController::~ObjectDetectionController()
  {
    // Unique_ptr handles cleanup
  }

  std::vector<Track> ObjectDetectionController::ProcessFrame(
      const uint8_t *rgb_data,
      int width,
      int height,
      float conf_threshold,
      float iou_threshold)
  {

    if (!ready_ || !rgb_data || width <= 0 || height <= 0)
    {
      return {};
    }

    // Input is already BGR from TurboJPEG decompression (matches Python cv_bridge)
    cv::Mat bgr_image(height, width, CV_8UC3, const_cast<uint8_t *>(rgb_data));

    // Step 1: Run YOLOv9 detector
    // - Preprocesses image (simple resize to 1280x736, normalize)
    // - NCNN inference
    // - Decodes output (bbox, confidence, class)
    // - Applies NMS
    auto detections = detector_->Detect(bgr_image, conf_threshold, iou_threshold);

    // Log detection count (helps debug if YOLO is detecting anything)
    if (detections.size() > 0)
    {
      // TODO: Add actual logging when available
    }

    // Step 2: Update Deep SORT tracker
    // - Extracts ReID features for each detection
    // - Kalman filter prediction
    // - Cascade matching (appearance + motion)
    // - IoU matching for unmatched
    // - Updates confirmed tracks, creates new tentative tracks
    auto all_tracks = tracker_->Update(bgr_image, detections);

    // Step 3: Get only confirmed tracks (hits >= n_init=5)
    last_confirmed_tracks_ = tracker_->GetConfirmedTracks();

    return last_confirmed_tracks_;
  }

  size_t ObjectDetectionController::GetTrackCount() const
  {
    return tracker_ ? tracker_->GetTrackCount() : 0;
  }

  std::vector<Track> ObjectDetectionController::GetConfirmedTracks() const
  {
    return last_confirmed_tracks_;
  }

  void ObjectDetectionController::Reset()
  {
    // Clear cached tracks
    last_confirmed_tracks_.clear();

    // Note: To fully reset tracker state, would need to recreate tracker
    // but that requires storing model paths. For now, just clear cache.
    // Full reset can be implemented if needed by storing constructor params.
  }

  PerceptionResult ObjectDetectionController::ProcessFrame(
      const uint8_t *bgr_data,
      int width,
      int height,
      const float *depth_data,
      int depth_width,
      int depth_height,
      float conf_threshold,
      float iou_threshold)
  {

    PerceptionResult result;

    if (!ready_ || !bgr_data || width <= 0 || height <= 0)
    {
      LOGE("ProcessFrame: Not ready or invalid input");
      return result;
    }

    // Wrap raw BGR buffer in cv::Mat (zero-copy, internal use only)
    cv::Mat rgb_bgr(height, width, CV_8UC3, const_cast<uint8_t *>(bgr_data));

    // Wrap depth buffer in cv::Mat if provided (zero-copy, internal use only)
    cv::Mat depth;
    if (depth_data && depth_width > 0 && depth_height > 0)
    {
      depth = cv::Mat(depth_height, depth_width, CV_32FC1,
                      const_cast<float *>(depth_data));
    }

    // Step 1: Run YOLOv9 detector
    auto detect_start = std::chrono::high_resolution_clock::now();
    auto all_detections = detector_->Detect(rgb_bgr, conf_threshold, iou_threshold);
    auto detect_end = std::chrono::high_resolution_clock::now();
    double detect_ms = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();

    LOGD("YOLO detection: %zu detections in %.1f ms", all_detections.size(), detect_ms);

    // Step 2: Filter detections by confidence threshold (matches Python line 245)
    std::vector<Detection> filtered_detections;
    for (const auto &det : all_detections)
    {
      if (det.confidence > conf_threshold)
      {
        filtered_detections.push_back(det);
      }
    }

    LOGD("Filtered detections: %zu (conf > %.2f)", filtered_detections.size(), conf_threshold);

    // Step 3: Update Deep SORT tracker
    auto track_start = std::chrono::high_resolution_clock::now();
    auto all_tracks = tracker_->Update(rgb_bgr, filtered_detections);
    auto track_end = std::chrono::high_resolution_clock::now();
    double track_ms = std::chrono::duration<double, std::milli>(track_end - track_start).count();

    LOGD("Deep SORT tracking: %zu tracks in %.1f ms (ReID + matching + Kalman)",
         all_tracks.size(), track_ms);

    // Step 4: Store detections and tracks
    result.detections = filtered_detections;
    result.tracks = all_tracks;

    // Step 5: Generate annotated RGB visualization
    cv::Mat annotated_rgb = rgb_bgr.clone();

    visualization::Annotator annotator(2); // line_width = 2 (matches Python)

    // Fixed colors for each class (matches Python visualization)
    cv::Scalar colors[3] = {
        cv::Scalar(15, 180, 15), // Green for cpb_beetle
        cv::Scalar(20, 20, 215), // Red for cpb_larva
        cv::Scalar(215, 20, 20)  // Blue for cpb_eggs
    };

    // Draw YOLO detections on RGB (matches Python lines 248-249)
    for (const auto &det : result.detections)
    {
      std::string label = std::string(GetClassName(det.class_id)) +
                          " " + std::to_string(det.confidence).substr(0, 4);
      annotator.DrawBoundingBox(annotated_rgb, det.bbox, label,
                                colors[det.class_id]);
    }

    // Draw Deep SORT track IDs on RGB (matches Python lines 374-381)
    for (const auto &track : result.tracks)
    {
      if (!track.is_confirmed)
        continue; // Only draw confirmed tracks

      // Generate color based on track_id (matches Python line 377-378)
      int color_idx = track.track_id % 100;
      cv::Scalar track_color(
          (color_idx * 71) % 256, // Simple pseudo-random color generation
          (color_idx * 151) % 256,
          (color_idx * 211) % 256);

      annotator.DrawTrackId(annotated_rgb, track.track_id, track.bbox, track_color);
    }

    // Copy annotated RGB to result (cv::Mat → std::vector)
    result.rgb_width = annotated_rgb.cols;
    result.rgb_height = annotated_rgb.rows;
    size_t rgb_size = result.rgb_width * result.rgb_height * 3;
    result.annotated_rgb_bgr.resize(rgb_size);
    std::memcpy(result.annotated_rgb_bgr.data(), annotated_rgb.data, rgb_size);

    // Step 6: Generate depth colormap visualization if depth available
    if (!depth.empty())
    {
      result.has_depth_visualization = true;
      cv::Mat annotated_depth = GenerateDepthColormap(depth);

      // Draw YOLO detections on depth colormap (matches Python lines 268-269)
      // NOTE: Does NOT draw track IDs, only YOLO boxes
      for (const auto &det : result.detections)
      {
        std::string label = std::string(GetClassName(det.class_id)) +
                            " " + std::to_string(det.confidence).substr(0, 4);
        annotator.DrawBoundingBox(annotated_depth, det.bbox, label,
                                  colors[det.class_id]);
      }

      // Copy annotated depth to result (cv::Mat → std::vector)
      result.depth_width = annotated_depth.cols;
      result.depth_height = annotated_depth.rows;
      size_t depth_size = result.depth_width * result.depth_height * 3;
      result.annotated_depth_bgr.resize(depth_size);
      std::memcpy(result.annotated_depth_bgr.data(), annotated_depth.data, depth_size);
    }

    return result;
  }

  cv::Mat ObjectDetectionController::GenerateDepthColormap(const cv::Mat &depth)
  {
    if (depth.empty())
    {
      return cv::Mat();
    }

    // Python code (line 150):
    // depth_color_map = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=100), cv2.COLORMAP_JET)

    // Convert depth (32FC1, meters) to 8-bit for colormap
    cv::Mat depth_8u;
    cv::convertScaleAbs(depth, depth_8u, 100.0); // alpha=100 (scale factor)

    // Apply JET colormap
    cv::Mat colormap;
    cv::applyColorMap(depth_8u, colormap, cv::COLORMAP_JET);

    return colormap;
  }

  const char *ObjectDetectionController::GetClassName(int class_id)
  {
    if (class_id < 0 || class_id >= 3)
    {
      return "unknown";
    }
    return CLASS_NAMES[class_id];
  }

} // namespace perception
