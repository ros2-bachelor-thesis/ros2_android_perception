#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "perception/detection.h"
#include "perception/track.h"

// Forward declarations (full types defined in types.h, included in .cc file)
namespace perception {
  struct PerceptionResult;
}

namespace cv {
  class Mat;
}

namespace perception
{

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
    class ObjectDetectionController
    {
    public:
        /**
         * Constructor - loads NCNN models
         *
         * @param yolo_param Path to yolov9_s_pobed.ncnn.param
         * @param yolo_bin Path to yolov9_s_pobed.ncnn.bin
         * @param reid_param Path to osnet_ain_x1_0.ncnn.param
         * @param reid_bin Path to osnet_ain_x1_0.ncnn.bin
         * @param use_vulkan Use GPU acceleration (default: false, CPU NEON faster on ARM)
         */
        ObjectDetectionController(
            const std::string &yolo_param,
            const std::string &yolo_bin,
            const std::string &reid_param,
            const std::string &reid_bin,
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
            const uint8_t *rgb_data,
            int width,
            int height,
            float conf_threshold = 0.25f,
            float iou_threshold = 0.45f);

        /**
         * Process single frame with visualization
         *
         * Pipeline:
         * 1. YOLOv9 detection
         * 2. Filter detections (conf > threshold)
         * 3. Deep SORT tracking (optional, controlled by enable_tracking)
         * 4. Generate annotated RGB (YOLO boxes + track IDs if tracking enabled)
         * 5. If depth provided: generate annotated depth colormap (YOLO boxes only)
         *
         * @param bgr_data Raw BGR image buffer (interleaved, 8-bit per channel)
         * @param width RGB image width
         * @param height RGB image height
         * @param depth_data Optional depth buffer (32-bit float, meters), nullptr if not available
         * @param depth_width Depth image width (0 if depth_data is nullptr)
         * @param depth_height Depth image height (0 if depth_data is nullptr)
         * @param conf_threshold Confidence threshold (default: 0.5, matches Python)
         * @param iou_threshold NMS IoU threshold (default: 0.45)
         * @param enable_tracking Enable Deep SORT tracking (default: true). Set false for camera motion scenarios.
         * @return PerceptionResult with detections, tracks, and annotated raw BGR buffers
         */
        PerceptionResult ProcessFrame(
            const uint8_t *bgr_data,
            int width,
            int height,
            const float *depth_data = nullptr,
            int depth_width = 0,
            int depth_height = 0,
            float conf_threshold = 0.5f,
            float iou_threshold = 0.45f,
            bool enable_tracking = true);

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

        /**
         * Get class name by ID
         * @param class_id 0=cpb_beetle, 1=cpb_larva, 2=cpb_eggs
         * @return Class name string
         */
        static const char *GetClassName(int class_id);

    private:
        std::unique_ptr<NcnnDetector> detector_;
        std::unique_ptr<DeepSortTracker> tracker_;
        bool ready_;

        // Cache last results
        std::vector<Track> last_confirmed_tracks_;

        // Class names (matches Python model training)
        static constexpr const char *CLASS_NAMES[3] = {
            "cpb_beetle",  // Colorado Potato Beetle adult
            "cpb_larva",   // larvae
            "cpb_eggs"     // egg clusters
        };
    };

} // namespace perception
