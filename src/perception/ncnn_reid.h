#pragma once

#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <string>
#include <vector>

namespace perception {

/**
 * mars-small128 ReID feature extractor using NCNN
 *
 * Extracts 128-dimensional appearance features from detection bounding boxes
 * for Deep SORT tracking. Features are L2-normalized for cosine distance matching.
 *
 * Model: mars-small128 (trained on Market-1501 + MARS datasets)
 * Input: 128x64 RGB uint8 (NOT normalized - expects 0-255 range)
 * Output: 128-dim float32 L2-normalized feature vector
 */
class NcnnReID {
 public:
  /**
   * Constructor - load NCNN ReID model
   *
   * @param param_path Path to mars-small128.ncnn.param file
   * @param bin_path Path to mars-small128.ncnn.bin file
   */
  NcnnReID(const std::string& param_path, const std::string& bin_path);

  ~NcnnReID();

  /**
   * Extract appearance feature from detection bounding box
   *
   * Pipeline:
   * 1. Crop bbox region from image
   * 2. Resize to 128x64 (ReID input size)
   * 3. Convert to uint8 RGB (mars-small128 expects 0-255, not normalized!)
   * 4. NCNN inference
   * 5. L2-normalize output feature
   *
   * @param image Source image (BGR format from OpenCV)
   * @param bbox Bounding box [x1, y1, x2, y2] to extract feature from
   * @return 128-dimensional L2-normalized feature vector (empty if error)
   */
  std::vector<float> Extract(const cv::Mat& image, const float bbox[4]);

  /**
   * Check if model is successfully loaded
   */
  bool IsLoaded() const { return loaded_; }

  /**
   * Get expected feature dimension (should be 128)
   */
  int GetFeatureDim() const { return feature_dim_; }

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
   * L2-normalize feature vector to unit length
   *
   * Normalized = feature / sqrt(sum(feature^2))
   * Required for cosine distance computation in Deep SORT
   *
   * @param feature Feature vector to normalize (modified in-place)
   */
  void L2Normalize(std::vector<float>& feature);

  ncnn::Net net_;           ///< NCNN inference network
  bool loaded_;             ///< Model load status
  int input_width_;         ///< ReID input width (64 for mars-small128)
  int input_height_;        ///< ReID input height (128 for mars-small128)
  int feature_dim_;         ///< Output feature dimension (128)

  // Layer names from NCNN conversion (tf2onnx → onnx2ncnn)
  const char* input_layer_ = "images:0";
  const char* output_layer_ = "features:0";
};

}  // namespace perception
