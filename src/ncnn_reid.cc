#include "perception/ncnn_reid.h"
#include "perception/image_preprocessor.h"
#include "perception/log.h"
#include <cmath>
#include <ncnn/net.h>

namespace perception {

NcnnReID::NcnnReID(const std::string& param_path, const std::string& bin_path)
    : loaded_(false),
      input_width_(128),
      input_height_(256),
      feature_dim_(512) {
  loaded_ = LoadModel(param_path, bin_path);
}

NcnnReID::~NcnnReID() {
  // NCNN Net destructor handles cleanup
}

bool NcnnReID::LoadModel(const std::string& param_path,
                          const std::string& bin_path) {

  // Configure NCNN options
  ncnn::Option opt;
  opt.lightmode = true;
  opt.num_threads = 2;  // ReID is lightweight, 2 threads sufficient
  opt.use_fp16_packed = true;
  opt.use_fp16_storage = true;
  opt.use_fp16_arithmetic = true;

  net_.opt = opt;

  // Load model files
  int ret_param = net_.load_param(param_path.c_str());
  int ret_bin = net_.load_model(bin_path.c_str());

  if (ret_param != 0 || ret_bin != 0) {
    return false;
  }

  return true;
}

std::vector<float> NcnnReID::Extract(const cv::Mat& image, const float bbox[4]) {
  std::vector<float> feature;

  if (!loaded_ || image.empty()) {
    return feature;
  }

  // Validate bbox coordinates
  int x1 = static_cast<int>(std::max(0.0f, bbox[0]));
  int y1 = static_cast<int>(std::max(0.0f, bbox[1]));
  int x2 = static_cast<int>(std::min(static_cast<float>(image.cols - 1), bbox[2]));
  int y2 = static_cast<int>(std::min(static_cast<float>(image.rows - 1), bbox[3]));

  // Check for valid bbox
  if (x2 <= x1 || y2 <= y1) {
    return feature;
  }

  // Crop bbox region
  cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
  cv::Mat cropped = image(roi).clone();

  // Resize to ReID input size (128x256 for OSNet)
  cv::Mat resized;
  cv::resize(cropped, resized, cv::Size(input_width_, input_height_));

  // Convert BGR to RGB (OSNet expects RGB)
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  // IMPORTANT: OSNet expects uint8 [0-255], NOT normalized [0-1]
  // Create NCNN Mat from pixel data
  ncnn::Mat input = ncnn::Mat::from_pixels(
      rgb.data, ncnn::Mat::PIXEL_RGB, input_width_, input_height_);

  // Create NCNN extractor
  ncnn::Extractor ex = net_.create_extractor();
  ex.set_light_mode(true);

  // Input image
  ex.input(input_layer_, input);

  // Run inference
  ncnn::Mat output;
  int ret = ex.extract(output_layer_, output);

  if (ret != 0) {
    LOGE("ReID inference failed: ex.extract returned %d", ret);
    return feature;
  }

  // Extract feature vector (should be 512-dim for OSNet)
  int feature_size = output.w * output.h * output.c;
  if (feature_size != feature_dim_) {
    LOGE("ReID output dimension mismatch: got %d (c=%d, h=%d, w=%d), expected %d",
         feature_size, output.c, output.h, output.w, feature_dim_);
    return feature;
  }

  // Copy output to vector
  feature.resize(feature_dim_);
  const float* output_data = output;
  for (int i = 0; i < feature_dim_; i++) {
    feature[i] = output_data[i];
  }

  // L2-normalize feature
  L2Normalize(feature);

  // Compute L2 norm to verify normalization
  float norm = 0.0f;
  for (float val : feature) {
    norm += val * val;
  }
  norm = std::sqrt(norm);

  // Log extraction with feature statistics
  static int extraction_count = 0;
  extraction_count++;

  if (extraction_count <= 5 || extraction_count % 10 == 0) {
    LOGI("ReID extraction #%d: bbox=[%.1f,%.1f,%.1f,%.1f], feature_norm=%.3f, first_3=[%.3f,%.3f,%.3f]",
         extraction_count, bbox[0], bbox[1], bbox[2], bbox[3], norm,
         feature[0], feature[1], feature[2]);
  }

  return feature;
}

void NcnnReID::L2Normalize(std::vector<float>& feature) {
  if (feature.empty()) {
    return;
  }

  // Compute L2 norm: sqrt(sum(x^2))
  float norm = 0.0f;
  for (float val : feature) {
    norm += val * val;
  }
  norm = std::sqrt(norm);

  // Avoid division by zero
  if (norm < 1e-6f) {
    return;
  }

  // Normalize: feature /= norm
  for (float& val : feature) {
    val /= norm;
  }
}

}  // namespace perception
