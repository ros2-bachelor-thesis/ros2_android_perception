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

    // Disable Vulkan for 352x640 model - non-standard dimensions cause
    // Vulkan shader crashes (BinaryOp_vulkan SIGSEGV). CPU+NEON is sufficient.
    bool vulkan_enabled = false;

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
    // Match Python reference exactly (object_detection.py + tracker.py defaults)
    config.max_cosine_distance = 0.4f; // Python: max_cosine_distance=0.4
    config.nn_budget = 100;            // Cap gallery size for Android memory + O(100) search bound
    config.max_age = 30;               // Python: max_age=30
    config.n_init = 3;                 // Python: n_init=3
    config.max_iou_distance = 0.7f;    // Python: max_iou_distance=0.7

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
    last_confirmed_tracks_.clear();
    if (tracker_)
    {
      tracker_->Reset();
    }
  }

  PerceptionResult ObjectDetectionController::ProcessFrame(
      const uint8_t *bgr_data,
      int width,
      int height,
      const float *depth_data,
      int depth_width,
      int depth_height,
      float conf_threshold,
      float iou_threshold,
      bool enable_tracking)
  {

    PerceptionResult result;

    if (!ready_ || !bgr_data || width <= 0 || height <= 0)
    {
      LOGE("ProcessFrame: Not ready or invalid input");
      return result;
    }

    // Wrap raw BGR buffer in cv::Mat (zero-copy, internal use only)
    cv::Mat rgb_bgr(height, width, CV_8UC3, const_cast<uint8_t *>(bgr_data));

    // Match Python preprocessing (object_detection.py yolov9 branch lines 224-226):
    //   img = cv2.resize(self.rgb_image, (640, 360))  # color_depth_size
    //   img = img[4:-4,:]                              # crop to 640x352
    //   im0 = img.copy()                               # detection space = 640x352
    // NCNN model expects 1280x736, so we pass the 640x352 image and NCNN
    // resizes internally. Detections are then scaled back to 640x352 (im0 space).
    cv::Mat resized;
    cv::resize(rgb_bgr, resized, cv::Size(640, 360));
    cv::Mat model_input = resized(cv::Rect(0, 4, 640, 352)).clone();

    // Wrap depth buffer in cv::Mat if provided (zero-copy, internal use only)
    cv::Mat depth;
    if (depth_data && depth_width > 0 && depth_height > 0)
    {
      depth = cv::Mat(depth_height, depth_width, CV_32FC1,
                      const_cast<float *>(depth_data));
    }

    // Step 1: Run YOLOv9 detector
    // NCNN internally resizes 640x352 -> 1280x736, then scale_boxes maps
    // detections back to model_input size (640x352) via gain/pad math.
    auto detect_start = std::chrono::high_resolution_clock::now();
    auto all_detections = detector_->Detect(model_input, conf_threshold, iou_threshold);
    auto detect_end = std::chrono::high_resolution_clock::now();
    double detect_ms = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();

    LOGD("YOLO detection: %zu detections in %.1f ms", all_detections.size(), detect_ms);

    // Step 2: Conditionally update Deep SORT tracker
    std::vector<Track> all_tracks;
    if (enable_tracking)
    {
      auto track_start = std::chrono::high_resolution_clock::now();
      all_tracks = tracker_->Update(model_input, all_detections);
      auto track_end = std::chrono::high_resolution_clock::now();
      double track_ms = std::chrono::duration<double, std::milli>(track_end - track_start).count();

      LOGD("Deep SORT tracking: %zu tracks in %.1f ms (ReID + matching + Kalman)",
           all_tracks.size(), track_ms);
    }
    else
    {
      LOGD("Tracking disabled - using detections only");
    }

    // Step 3: Store detections and tracks
    result.detections = all_detections;
    result.tracks = all_tracks;

    // Step 5: Generate annotated RGB visualization
    // Detections are in model_input_size space (640x352), so annotate on that
    cv::Mat annotated_rgb = model_input.clone();

    visualization::Annotator annotator(2); // line_width = 2 (matches Python)

    // Fixed colors for each class (matches Python visualization)
    cv::Scalar colors[3] = {
        cv::Scalar(15, 180, 15), // Green for cpb_beetle
        cv::Scalar(20, 20, 215), // Red for cpb_larva
        cv::Scalar(215, 20, 20)  // Blue for cpb_eggs
    };

    if (enable_tracking)
    {
      // Draw only confirmed Deep SORT tracks with class labels and track IDs
      int confirmed_count = 0;
      for (const auto &track : result.tracks)
      {
        // Python parity: only draw confirmed tracks matched this frame
        if (!track.is_confirmed || track.time_since_update > 0)
          continue;

        confirmed_count++;

        // Label format: "cpb_beetle ID: 5"
        std::string label = std::string(GetClassName(track.class_id)) +
                            " ID: " + std::to_string(track.track_id);

        // Use class-based color (same color for same class)
        annotator.DrawBoundingBox(annotated_rgb, track.bbox, label,
                                  colors[track.class_id]);
      }

      LOGD("Visualization: %d confirmed / %zu total tracks", confirmed_count, result.tracks.size());
    }
    else
    {
      // Draw raw YOLO detections (no track IDs)
      for (const auto &det : result.detections)
      {
        // Label format: "cpb_beetle 0.95" (class + confidence)
        char conf_str[16];
        snprintf(conf_str, sizeof(conf_str), "%.2f", det.confidence);
        std::string label = std::string(GetClassName(det.class_id)) + " " + conf_str;

        // Use class-based color (same color for same class)
        annotator.DrawBoundingBox(annotated_rgb, det.bbox, label,
                                  colors[det.class_id]);
      }

      LOGD("Visualization: %zu detections (tracking disabled)", result.detections.size());
    }

    // Copy annotated RGB to result (cv::Mat → std::vector)
    result.rgb_width = annotated_rgb.cols;
    result.rgb_height = annotated_rgb.rows;
    size_t rgb_size = result.rgb_width * result.rgb_height * 3;
    result.annotated_rgb_bgr.resize(rgb_size);
    std::memcpy(result.annotated_rgb_bgr.data(), annotated_rgb.data, rgb_size);

    return result;
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
