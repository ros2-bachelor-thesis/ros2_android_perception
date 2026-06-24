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

namespace
{

  // Scale bbox from model_input space (640x352) back to original image resolution.
  // The preprocessing chain is: resize(orig -> 640x360) then crop(y=[4,356)).
  // Inverse: x_orig = x_model * (W/640), y_orig = (y_model + 4) * (H/360)
  void ScaleBbox(const float src[4], float dst[4], int orig_w, int orig_h)
  {
    const float sx = static_cast<float>(orig_w) / 640.0f;
    const float sy = static_cast<float>(orig_h) / 360.0f;
    dst[0] = src[0] * sx;          // x1
    dst[1] = (src[1] + 4.0f) * sy; // y1 (undo 4px top crop)
    dst[2] = src[2] * sx;          // x2
    dst[3] = (src[3] + 4.0f) * sy; // y2
  }

} // anonymous namespace

namespace perception
{

  ObjectDetectionController::ObjectDetectionController(
      const std::string &yolo_param,
      const std::string &yolo_bin,
      const std::string &reid_param,
      const std::string &reid_bin)
      : ready_(false),
        use_vulkan_(false),
        gpu_instance_owned_(false)
  {
    // Runtime Vulkan probe. NCNN's GPU API is compiled in when the library
    // is built with NCNN_VULKAN=ON. create_gpu_instance loads the Vulkan
    // driver and enumerates devices; get_gpu_count returns 0 when none are
    // usable (driver missing, headless emulator, denied by SELinux, ...).
    int gpu_init = ncnn::create_gpu_instance();
    if (gpu_init == 0)
    {
      gpu_instance_owned_ = true;
      int gpu_count = ncnn::get_gpu_count();
      if (gpu_count > 0)
      {
        const ncnn::GpuInfo &info = ncnn::get_gpu_info(0);
        const uint32_t api = info.api_version();
        const uint32_t drv = info.driver_version();
        LOGI("GPU detected: %s (Vulkan API %u.%u.%u, driver 0x%08x, type=%d)",
             info.device_name(),
             (api >> 22) & 0x3ff,
             (api >> 12) & 0x3ff,
             api & 0xfff,
             drv,
             info.type());
        use_vulkan_ = true;
      }
      else
      {
        LOGI("Vulkan instance created but no GPU devices reported; using CPU NEON");
      }
    }
    else
    {
      LOGI("ncnn::create_gpu_instance() returned %d; using CPU NEON", gpu_init);
    }

    if (use_vulkan_)
    {
      LOGD("Using Vulkan GPU for perception");
    }
    else
    {
      LOGD("Using CPU (NEON) for perception");
    }

    // Load YOLOv9 detector
    detector_ = std::make_unique<NcnnDetector>(yolo_param, yolo_bin, use_vulkan_);

    if (!detector_->IsLoaded())
    {
      // Model load failed
      return;
    }

    // Load Deep SORT tracker (includes ReID model)
    DeepSortConfig config;
    // Low-FPS tuned (~2-3 FPS on this Vulkan pipeline). Python reference uses
    // 0.4 / 0.7 but those are calibrated for 30 FPS where motion blur and
    // appearance drift between frames is small.
    config.max_cosine_distance = 0.5f; // was 0.4 (Python)
    config.nn_budget = 100;            // Cap gallery size for Android memory + O(100) search bound
    config.max_age = 30;               // Python: max_age=30
    config.n_init = 3;                 // Python: n_init=3
    config.max_iou_distance = 0.9f;    // was 0.7 (Python); fallback for large displacement

    tracker_ = std::make_unique<DeepSortTracker>(
        reid_param, reid_bin, config, use_vulkan_);

    if (!tracker_->IsReady())
    {
      // ReID model load failed
      return;
    }

    ready_ = true;
  }

  ObjectDetectionController::~ObjectDetectionController()
  {
    // Net destructors must run before destroy_gpu_instance so any Vulkan
    // resources the nets hold are released first.
    tracker_.reset();
    detector_.reset();
    if (gpu_instance_owned_)
    {
      ncnn::destroy_gpu_instance();
      gpu_instance_owned_ = false;
    }
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
    // Annotate on full-resolution image for sharp UI display.
    // Detection coords (640x352) are scaled back to original resolution.
    cv::Mat annotated_rgb = rgb_bgr.clone();

    visualization::Annotator annotator(0); // auto-scale line width based on image size

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

        char conf_buf[16];
        snprintf(conf_buf, sizeof(conf_buf), "%.2f", track.confidence);
        // Label format: "cpb_beetle 5: 0.82"
        std::string label = std::string(GetClassName(track.class_id)) +
                            " " + std::to_string(track.track_id) +
                            ": " + conf_buf;
        ;

        // Scale bbox from model_input space (640x352) to original resolution
        float scaled_bbox[4];
        ScaleBbox(track.bbox, scaled_bbox, width, height);

        // Use class-based color (same color for same class)
        annotator.DrawBoundingBox(annotated_rgb, scaled_bbox, label,
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

        // Scale bbox from model_input space (640x352) to original resolution
        float scaled_bbox[4];
        ScaleBbox(det.bbox, scaled_bbox, width, height);

        // Use class-based color (same color for same class)
        annotator.DrawBoundingBox(annotated_rgb, scaled_bbox, label,
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
