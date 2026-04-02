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

  // Python uses letterbox-style coordinate scaling even with simple resize:
  // gain = min(model_h / orig_h, model_w / orig_w)
  // pad = ((model_w - orig_w * gain) / 2, (model_h - orig_h * gain) / 2)
  // This matches scale_boxes() in utils/general.py line 831-832
  float gain = std::min(
      static_cast<float>(input_height_) / img_height,
      static_cast<float>(input_width_) / img_width
  );
  float pad_w = (input_width_ - img_width * gain) / 2.0f;
  float pad_h = (input_height_ - img_height * gain) / 2.0f;

  // Create NCNN extractor
  ncnn::Extractor ex = net_.create_extractor();
  ex.set_light_mode(true);
  // Note: num_threads is set via net_.opt, not extractor

  // Input image
  ex.input(input_layer_, input);

  // Run inference - YOLOv9 has dual detection heads (out0, out1)
  ncnn::Mat out0, out1;
  int ret0 = ex.extract("out0", out0);
  int ret1 = ex.extract("out1", out1);

  if (ret0 != 0 || ret1 != 0) {
    // Inference failed
    return {};
  }

  // Concatenate both detection heads along anchor dimension (height)
  // out0: [h0, w], out1: [h1, w] → output: [h0+h1, w]
  int total_anchors = out0.h + out1.h;
  int num_values = out0.w;  // Should be same for both (4 + num_classes)

  ncnn::Mat output(num_values, total_anchors);

  // Copy out0 rows first
  for (int i = 0; i < out0.h; i++) {
    const float* src = out0.row(i);
    float* dst = output.row(i);
    memcpy(dst, src, num_values * sizeof(float));
  }

  // Copy out1 rows after out0
  for (int i = 0; i < out1.h; i++) {
    const float* src = out1.row(i);
    float* dst = output.row(out0.h + i);
    memcpy(dst, src, num_values * sizeof(float));
  }

  // Parse output tensor to detections (with letterbox-style scaling)
  std::vector<Detection> detections = ParseOutput(
      output,
      img_width,
      img_height,
      gain,
      pad_w,
      pad_h,
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
                                                  float gain,
                                                  float pad_w,
                                                  float pad_h,
                                                  float conf_threshold) {
  std::vector<Detection> detections;

  // YOLOv9 output shape: [num_anchors, 4+num_classes]
  // Each row: [cx, cy, w, h, class0_score, class1_score, class2_score]
  // NOTE: No separate objectness in YOLOv9 output

  int num_anchors = output.h;  // Number of anchor boxes
  int num_values = output.w;   // Should be 7 (4 bbox + 3 classes)

  if (num_values != 4 + num_classes_) {
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

    // Find best class and its score
    // YOLOv9 output has NO separate objectness - class scores start at index 4
    // Format: [cx, cy, w, h, class0_score, class1_score, class2_score]
    int best_class = 0;
    float best_score = row[4];  // class 0 score
    for (int c = 1; c < num_classes_; c++) {
      float class_score = row[4 + c];
      if (class_score > best_score) {
        best_score = class_score;
        best_class = c;
      }
    }

    // Confidence is the max class score (no objectness multiplication)
    float confidence = best_score;

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
    // Python uses letterbox-style scaling (uniform gain) even with simple resize:
    // boxes[:, [0,2]] -= pad[0]; boxes[:, [1,3]] -= pad[1]; boxes[:, :4] /= gain
    // (utils/general.py scale_boxes() lines 837-839)
    x1 = (x1 - pad_w) / gain;
    y1 = (y1 - pad_h) / gain;
    x2 = (x2 - pad_w) / gain;
    y2 = (y2 - pad_h) / gain;

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
