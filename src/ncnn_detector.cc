#include "perception/ncnn_detector.h"
#include "perception/image_preprocessor.h"
#include "perception/nms.h"
#include "perception/log.h"
#include <chrono>
#include <ncnn/net.h>

namespace perception {

NcnnDetector::NcnnDetector(const std::string& param_path,
                           const std::string& bin_path)
    : loaded_(false),
      input_width_(640),
      input_height_(352),
      num_classes_(3) {
  loaded_ = LoadModel(param_path, bin_path);
}

NcnnDetector::~NcnnDetector() {
  // NCNN Net destructor handles cleanup
}

bool NcnnDetector::LoadModel(const std::string& param_path,
                               const std::string& bin_path) {
  // Configure NCNN options for performance
  ncnn::Option opt;
  opt.lightmode = false;      // Disable light mode for better performance (trades memory for speed)
  opt.num_threads = 8;        // Increase from 4 to 8 for modern ARM SoCs

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
  auto preprocess_start = std::chrono::high_resolution_clock::now();
  ncnn::Mat input = ImagePreprocessor::PrepareForYOLO(image, input_width_, input_height_);
  auto preprocess_end = std::chrono::high_resolution_clock::now();
  double preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
  LOGD("NCNN preprocessing: %.1f ms", preprocess_ms);

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
  ex.set_light_mode(false);  // Match net_.opt.lightmode for consistency
  // Note: num_threads is set via net_.opt, not extractor

  // Input image
  ex.input(input_layer_, input);

  // Run inference - YOLOv9 has dual detection heads (out0, out1)
  auto inference_start = std::chrono::high_resolution_clock::now();
  ncnn::Mat out0, out1;
  int ret0 = ex.extract("out0", out0);
  int ret1 = ex.extract("out1", out1);
  auto inference_end = std::chrono::high_resolution_clock::now();
  double inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
  LOGD("NCNN inference: %.1f ms", inference_ms);

  if (ret0 != 0 || ret1 != 0) {
    LOGE("Inference failed! ret0=%d, ret1=%d", ret0, ret1);
    return {};
  }

  // Log output shapes only once for debugging (not on every frame)
  static bool logged_shapes = false;
  if (!logged_shapes) {
    LOGI("NCNN output shapes: out0=[c=%d, h=%d, w=%d] (dims=%d), out1=[c=%d, h=%d, w=%d] (dims=%d)",
         out0.c, out0.h, out0.w, out0.dims, out1.c, out1.h, out1.w, out1.dims);
    logged_shapes = true;
  }

  // Check output format and reshape if needed
  // Expected: [num_anchors, 7] where 7 = 4 bbox + 3 classes
  // Common NCNN formats:
  //   - 2D: [h=num_anchors, w=7] (correct format)
  //   - 2D transposed: [h=7, w=num_anchors] (needs transpose)
  //   - 3D: [c=1, h=num_anchors, w=7] (flatten to 2D)

  auto postprocess_start = std::chrono::high_resolution_clock::now();
  ncnn::Mat output;

  // Cache format detection (log only once)
  static bool logged_format = false;

  // Case 1: 3D tensor [1, N, 7] - flatten to 2D
  if (out0.dims == 3 && out0.c == 1) {
    if (!logged_format) {
      LOGI("Detected 3D format [c=1, h=%d, w=%d], flattening to 2D", out0.h, out0.w);
      logged_format = true;
    }
    // Reshape both heads to 2D
    ncnn::Mat out0_2d(out0.w, out0.h);  // [h, w]
    memcpy(out0_2d.data, out0.data, out0.total() * sizeof(float));

    ncnn::Mat out1_2d(out1.w, out1.h);
    memcpy(out1_2d.data, out1.data, out1.total() * sizeof(float));

    // Concatenate
    int total_h = out0_2d.h + out1_2d.h;
    output = ncnn::Mat(out0_2d.w, total_h);
    memcpy(output.data, out0_2d.data, out0_2d.total() * sizeof(float));
    memcpy((float*)output.data + out0_2d.total(), out1_2d.data, out1_2d.total() * sizeof(float));
  }
  // Case 2: 2D tensor needs transpose [7, N] → [N, 7]
  else if (out0.dims == 2 && out0.h == (4 + num_classes_)) {
    if (!logged_format) {
      LOGI("Detected transposed format [h=%d, w=%d], transposing", out0.h, out0.w);
      logged_format = true;
    }
    // Transpose: out0[h=7, w=N] → [N, 7]
    int N0 = out0.w;
    int N1 = out1.w;
    int total_anchors = N0 + N1;

    output = ncnn::Mat(4 + num_classes_, total_anchors);

    // Transpose out0
    for (int i = 0; i < out0.h; i++) {
      for (int j = 0; j < out0.w; j++) {
        output.row(j)[i] = out0.row(i)[j];
      }
    }

    // Transpose out1 and append
    for (int i = 0; i < out1.h; i++) {
      for (int j = 0; j < out1.w; j++) {
        output.row(N0 + j)[i] = out1.row(i)[j];
      }
    }
  }
  // Case 3: Already correct 2D format [N, 7]
  else if (out0.dims == 2 && out0.w == (4 + num_classes_)) {
    if (!logged_format) {
      LOGI("Correct 2D format [h=%d, w=%d], concatenating", out0.h, out0.w);
      logged_format = true;
    }
    int total_anchors = out0.h + out1.h;
    int num_values = out0.w;

    output = ncnn::Mat(num_values, total_anchors);

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
  }
  // Case 4: Unknown format - error
  else {
    LOGE("Unsupported NCNN output format: out0=[c=%d, h=%d, w=%d], out1=[c=%d, h=%d, w=%d]",
         out0.c, out0.h, out0.w, out1.c, out1.h, out1.w);
    LOGE("Expected: [N, 7] or [7, N] or [1, N, 7]. Check model conversion.");
    return {};
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

  // Apply class-aware NMS
  detections = NMS::ApplyPerClass(detections, iou_threshold);

  auto postprocess_end = std::chrono::high_resolution_clock::now();
  double postprocess_ms = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
  LOGD("NCNN postprocess (format+parse+NMS): %.1f ms", postprocess_ms);

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

  LOGD("ParseOutput: num_anchors=%d, num_values=%d (expected %d)",
       num_anchors, num_values, 4 + num_classes_);

  if (num_values != 4 + num_classes_) {
    LOGE("ParseOutput: WRONG tensor width! Got %d, expected %d", num_values, 4 + num_classes_);
    return detections;
  }

  if (num_anchors > 0) {
    const float* row0 = output.row(0);
    LOGD("ParseOutput first anchor: [%.2f,%.2f,%.2f,%.2f] scores=[%.3f,%.3f,%.3f]",
         row0[0], row0[1], row0[2], row0[3], row0[4], row0[5], row0[6]);
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
