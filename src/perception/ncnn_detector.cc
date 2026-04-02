#include "perception/ncnn_detector.h"
#include "perception/image_preprocessor.h"
#include "perception/nms.h"
#include <ncnn/net.h>

namespace perception {

NcnnDetector::NcnnDetector(const std::string& param_path,
                           const std::string& bin_path,
                           bool use_vulkan)
    : loaded_(false),
      use_vulkan_(use_vulkan),
      input_width_(1280),
      input_height_(736),
      num_classes_(3) {
  loaded_ = LoadModel(param_path, bin_path);
}

NcnnDetector::~NcnnDetector() {
  // NCNN Net destructor handles cleanup
}

bool NcnnDetector::LoadModel(const std::string& param_path,
                               const std::string& bin_path) {
  // Configure NCNN options
  ncnn::Option opt;
  opt.lightmode = true;  // Reduce memory usage
  opt.num_threads = 4;   // Use 4 threads for ARM big.LITTLE
  opt.use_vulkan_compute = use_vulkan_;

  net_.opt = opt;

  // Load model files
  int ret_param = net_.load_param(param_path.c_str());
  int ret_bin = net_.load_model(bin_path.c_str());

  if (ret_param != 0 || ret_bin != 0) {
    // Model load failed
    return false;
  }

  return true;
}

std::vector<Detection> NcnnDetector::Detect(const cv::Mat& image,
                                             float conf_threshold,
                                             float iou_threshold) {
  if (!loaded_ || image.empty()) {
    return {};
  }

  // Store original dimensions for coordinate scaling
  int img_width = image.cols;
  int img_height = image.rows;

  // Preprocess: simple resize (no letterbox) to match Python reference
  ncnn::Mat input = ImagePreprocessor::PrepareForYOLO(image, input_width_, input_height_);

  // Python does simple resize without maintaining aspect ratio, so:
  // - No letterbox padding (pad_w = pad_h = 0)
  // - Direct scaling from model input size to original image size
  float scale_x = static_cast<float>(img_width) / input_width_;
  float scale_y = static_cast<float>(img_height) / input_height_;

  // Create NCNN extractor
  ncnn::Extractor ex = net_.create_extractor();
  ex.set_light_mode(true);
  // Note: num_threads is set via net_.opt, not extractor

  // Input image
  ex.input(input_layer_, input);

  // Run inference
  ncnn::Mat output;
  int ret = ex.extract(output_layer_, output);

  if (ret != 0) {
    // Inference failed
    return {};
  }

  // Parse output tensor to detections (with simple resize scaling)
  std::vector<Detection> detections = ParseOutput(
      output,
      img_width,
      img_height,
      scale_x,
      scale_y,
      0,  // No padding for simple resize
      0,  // No padding for simple resize
      conf_threshold);

  // Log raw detection count before NMS
  size_t raw_count = detections.size();

  // Apply class-aware NMS
  detections = NMS::ApplyPerClass(detections, iou_threshold);

  // Log only if we have detections (avoid spam)
  if (raw_count > 0 || detections.size() > 0) {
    // Note: We don't have direct logging here, but caller will log
  }

  return detections;
}

std::vector<Detection> NcnnDetector::ParseOutput(const ncnn::Mat& output,
                                                  int img_width,
                                                  int img_height,
                                                  float scale_x,
                                                  float scale_y,
                                                  int pad_w,
                                                  int pad_h,
                                                  float conf_threshold) {
  std::vector<Detection> detections;

  // YOLOv9 output shape: [1, num_anchors, 5+num_classes]
  // Each row: [cx, cy, w, h, objectness, class0_score, class1_score, class2_score]

  int num_anchors = output.h;  // Number of anchor boxes
  int num_values = output.w;   // Should be 8 (4 bbox + 1 obj + 3 classes)

  if (num_values != 5 + num_classes_) {
    // Unexpected output format
    return detections;
  }

  // Iterate over all anchors
  for (int i = 0; i < num_anchors; i++) {
    const float* row = output.row(i);

    // Extract bbox center and size (in input image space, 1280x736)
    float cx = row[0];
    float cy = row[1];
    float w = row[2];
    float h = row[3];
    float objectness = row[4];

    // Find best class and its score
    int best_class = 0;
    float best_score = row[5];  // class 0 score
    for (int c = 1; c < num_classes_; c++) {
      float class_score = row[5 + c];
      if (class_score > best_score) {
        best_score = class_score;
        best_class = c;
      }
    }

    // Combined confidence: objectness * class_score
    float confidence = objectness * best_score;

    // Filter by confidence threshold
    if (confidence < conf_threshold) {
      continue;
    }

    // Convert center/size to corner coordinates (x1,y1,x2,y2) in model input space
    float x1 = cx - w / 2.0f;
    float y1 = cy - h / 2.0f;
    float x2 = cx + w / 2.0f;
    float y2 = cy + h / 2.0f;

    // Map from model input space (1280x736) back to original image space
    // For simple resize: just apply scale factors (no padding to remove)
    // 1. Remove padding offset (0 for simple resize)
    x1 = x1 - pad_w;
    y1 = y1 - pad_h;
    x2 = x2 - pad_w;
    y2 = y2 - pad_h;

    // 2. Scale back to original dimensions (separate X/Y scales for simple resize)
    x1 = x1 * scale_x;
    y1 = y1 * scale_y;
    x2 = x2 * scale_x;
    y2 = y2 * scale_y;

    // 3. Clamp to image boundaries
    x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_width - 1)));
    y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_height - 1)));
    x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_width - 1)));
    y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_height - 1)));

    // Create detection
    Detection det(x1, y1, x2, y2, confidence, best_class);
    detections.push_back(det);
  }

  return detections;
}

}  // namespace perception
