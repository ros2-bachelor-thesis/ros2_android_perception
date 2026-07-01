#include "perception/ncnn_reid.h"
#include "perception/image_preprocessor.h"
#include "perception/log.h"
#include <chrono>
#include <cmath>
#include <ncnn/net.h>

namespace perception {

NcnnReID::NcnnReID(const std::string& param_path, const std::string& bin_path,
                   bool use_vulkan)
    : loaded_(false),
      use_vulkan_(use_vulkan),
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
  opt.use_vulkan_compute = use_vulkan_;
  // FP16 disabled to match detector path: device lacks ARMv8.2 asimdhp,
  // enabling FP16 SIGSEGVs in load_model on the CPU path.
  opt.use_fp16_packed = false;
  opt.use_fp16_storage = false;
  opt.use_fp16_arithmetic = false;

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

  if (x2 <= x1 || y2 <= y1) {
    return feature;
  }

  // Step 1: crop + resize
  auto t0 = std::chrono::high_resolution_clock::now();
  cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
  cv::Mat cropped = image(roi).clone();
  cv::Mat resized;
  cv::resize(cropped, resized, cv::Size(input_width_, input_height_));
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  auto t1 = std::chrono::high_resolution_clock::now();

  // Step 2: pack pixels into ncnn::Mat
  // IMPORTANT: OSNet expects uint8 [0-255], NOT normalized [0-1]
  ncnn::Mat input = ncnn::Mat::from_pixels(
      rgb.data, ncnn::Mat::PIXEL_RGB, input_width_, input_height_);
  auto t2 = std::chrono::high_resolution_clock::now();
  LOGD("[ncnn] ReID from_pixels: %.2f ms", std::chrono::duration<double, std::milli>(t2 - t1).count());

  // Step 3: forward pass
  ncnn::Extractor ex = net_.create_extractor();
  ex.set_light_mode(true);
  ex.input(input_layer_, input);
  ncnn::Mat output;
  int ret = ex.extract(output_layer_, output);
  auto t3 = std::chrono::high_resolution_clock::now();
  LOGD("[ncnn] ReID ex.extract: %.2f ms", std::chrono::duration<double, std::milli>(t3 - t2).count());

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

  feature.resize(feature_dim_);
  const float* output_data = output;
  for (int i = 0; i < feature_dim_; i++) {
    feature[i] = output_data[i];
  }

  // Step 4: L2-normalize
  L2Normalize(feature);
  auto t4 = std::chrono::high_resolution_clock::now();

  static int extraction_count = 0;
  extraction_count++;
  LOGD("[ncnn] ReID #%d total: %.2f ms", extraction_count,
       std::chrono::duration<double, std::milli>(t4 - t0).count());

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
