#pragma once

#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <string>
#include <vector>

#include "perception/detection.h"

namespace perception {

/**
 * YOLOv9 object detector using NCNN inference framework
 *
 * Performs inference on NCNN-converted YOLOv9 model for 3-class
 * Colorado Potato Beetle detection (beetle, larva, eggs).
 *
 * Model input: 1280x736 RGB (normalized to [0,1])
 * Model output: [1, num_anchors, 5+num_classes] tensor
 * Post-processing: Confidence filtering + class-aware NMS
 */
class NcnnDetector {
 public:
  /**
   * Constructor - load NCNN model
   *
   * @param param_path Path to .ncnn.param file
   * @param bin_path Path to .ncnn.bin file
   * @param use_vulkan Enable Vulkan GPU acceleration (default: false, CPU NEON faster on ARM)
   */
  NcnnDetector(const std::string& param_path,
               const std::string& bin_path,
               bool use_vulkan = false);

  ~NcnnDetector();

  /**
   * Detect objects in image
   *
   * Pipeline:
   * 1. Preprocess: letterbox resize to 1280x736, normalize to [0,1]
   * 2. NCNN inference: YOLOv9 forward pass
   * 3. Decode output: parse bounding boxes + class scores
   * 4. Filter: confidence threshold
   * 5. NMS: class-aware non-maximum suppression
   *
   * @param image Input image (any size, BGR format from OpenCV)
   * @param conf_threshold Confidence threshold (default: 0.25)
   * @param iou_threshold IoU threshold for NMS (default: 0.45)
   * @return Vector of detections with bbox, confidence, class_id, empty feature
   */
  std::vector<Detection> Detect(const cv::Mat& image,
                                 float conf_threshold = 0.25f,
                                 float iou_threshold = 0.45f);

  /**
   * Check if model is successfully loaded
   */
  bool IsLoaded() const { return loaded_; }

  /**
   * Get model input dimensions (1280x736 for yolov9_s_pobed)
   */
  int GetInputWidth() const { return input_width_; }
  int GetInputHeight() const { return input_height_; }

 private:
  /**
   * Load NCNN model from param and bin files
   *
   * @param param_path Path to .param file
   * @param bin_path Path to .bin file
   * @return true if load successful, false otherwise
   */
  bool LoadModel(const std::string& param_path, const std::string& bin_path);

  /**
   * Parse NCNN output tensor to Detection structs
   *
   * YOLOv9 output format: [1, num_anchors, 5+num_classes]
   * Each anchor: [cx, cy, w, h, objectness, class0_score, class1_score, class2_score]
   *
   * @param output NCNN Mat output from model
   * @param img_width Original image width (for coordinate scaling)
   * @param img_height Original image height (for coordinate scaling)
   * @param gain Uniform scale factor (matches Python scale_boxes)
   * @param pad_w Letterbox padding width (for coordinate offset)
   * @param pad_h Letterbox padding height (for coordinate offset)
   * @param conf_threshold Minimum confidence to keep detection
   * @return Vector of raw detections (before NMS)
   */
  std::vector<Detection> ParseOutput(const ncnn::Mat& output,
                                      int img_width,
                                      int img_height,
                                      float gain,
                                      float pad_w,
                                      float pad_h,
                                      float conf_threshold);

  ncnn::Net net_;           ///< NCNN inference network
  bool loaded_;             ///< Model load status
  bool use_vulkan_;         ///< Vulkan GPU acceleration flag
  int input_width_;         ///< Model input width (1280 for YOLOv9-s)
  int input_height_;        ///< Model input height (736 for YOLOv9-s)
  int num_classes_;         ///< Number of classes (3: beetle, larva, eggs)

  // Layer names (may need adjustment based on actual .param file)
  const char* input_layer_ = "in0";
  const char* output_layer_ = "out0";
};

}  // namespace perception
