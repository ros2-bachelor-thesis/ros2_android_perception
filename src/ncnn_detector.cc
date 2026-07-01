#include "perception/ncnn_detector.h"
#include "perception/image_preprocessor.h"
#include "perception/nms.h"
#include "perception/log.h"
#include <chrono>
#include <ncnn/net.h>

namespace perception {

NcnnDetector::NcnnDetector(const std::string& param_path,
                           const std::string& bin_path,
                           bool use_vulkan)
    : loaded_(false),
      use_vulkan_(use_vulkan),
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
  opt.lightmode = true;       // Free intermediate blobs after use (reduces memory, improves cache)
  opt.num_threads = 4;        // 4 threads for big.LITTLE: use big+medium cores only (Pixel 7: 2x X1 + 2x A78)
  opt.use_vulkan_compute = use_vulkan_;
  // FP16 disabled: enabling use_fp16_arithmetic requires ARMv8.2 FP16 NEON
  // (asimdhp) on CPU. On devices without it NCNN SIGSEGVs during load_model
  // while parsing FP16 packed weights. Re-enable only after probing
  // cpu_support_arm_asimdhp() (CPU path) or gpu_info.support_fp16_*() (Vulkan).
  opt.use_fp16_packed = false;
  opt.use_fp16_storage = false;
  opt.use_fp16_arithmetic = false;
  LOGD("NCNN detector: num_threads=%d vulkan=%d", opt.num_threads, int(use_vulkan_));

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

  int img_width = image.cols;
  int img_height = image.rows;

  // Python-compatible preprocessing: resize to 640x360, crop 4px top+bottom → 640x352.
  // Matches Python reference: cv2.resize(rgb, (640,360)), then img[4:-4,:]
  const float scale = static_cast<float>(input_width_) / img_width;
  constexpr int kCropTop = 4;
  constexpr int kResizeH = 360;  // input_height_ (352) + 2*kCropTop

  auto t_pre0 = std::chrono::high_resolution_clock::now();

  cv::Mat src = image.isContinuous() ? image : image.clone();
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(input_width_, kResizeH), 0, 0, cv::INTER_LINEAR);
  cv::Mat cropped = resized(cv::Rect(0, kCropTop, input_width_, input_height_));

  ncnn::Mat input = ncnn::Mat::from_pixels(
      cropped.data,
      ncnn::Mat::PIXEL_BGR2RGB,
      input_width_, input_height_
  );
  auto t_pre1 = std::chrono::high_resolution_clock::now();
  LOGD("[ncnn] resize+crop+from_pixels: %.2f ms",
       std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

  const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  input.substract_mean_normalize(nullptr, norm_vals);
  auto t_pre2 = std::chrono::high_resolution_clock::now();
  LOGD("[ncnn] normalize: %.2f ms",
       std::chrono::duration<double, std::milli>(t_pre2 - t_pre1).count());

  ncnn::Extractor ex = net_.create_extractor();
  ex.input(input_layer_, input);

  ncnn::Mat raw_output;
  auto t_inf_start = std::chrono::high_resolution_clock::now();
  int ret = ex.extract(output_layer_, raw_output);
  auto t_inf_end = std::chrono::high_resolution_clock::now();
  LOGD("[ncnn] extract: %.1f ms ret=%d",
       std::chrono::duration<double, std::milli>(t_inf_end - t_inf_start).count(), ret);

  if (ret != 0) {
    LOGE("NCNN extract failed! ret=%d", ret);
    return {};
  }

  // Normalize output to [N, 4+num_classes_] row layout.
  // YOLOv9 DualDDetect with export=True outputs [h=7, w=N] (transposed) — transpose back.
  ncnn::Mat output;
  if (raw_output.dims == 2 && raw_output.h == (4 + num_classes_)) {
    int N = raw_output.w;
    output = ncnn::Mat(4 + num_classes_, N);
    for (int i = 0; i < raw_output.h; i++)
      for (int j = 0; j < raw_output.w; j++)
        output.row(j)[i] = raw_output.row(i)[j];
  } else if (raw_output.dims == 2 && raw_output.w == (4 + num_classes_)) {
    output = raw_output;
  } else {
    LOGE("Unexpected output shape: dims=%d h=%d w=%d", raw_output.dims, raw_output.h, raw_output.w);
    return {};
  }

  static bool logged_shape = false;
  if (!logged_shape) {
    logged_shape = true;
    LOGI("NCNN out0: raw[h=%d,w=%d,dims=%d] → [N=%d,values=%d]",
         raw_output.h, raw_output.w, raw_output.dims, output.h, output.w);
  }

  auto t_reshape_start = std::chrono::high_resolution_clock::now();

  // Inverse-map stretch+crop coordinates back to original image space.
  // stretch+crop: x_orig = x_m/scale,  y_orig = (y_m + kCropTop)/scale
  // Reused as letterbox params: gain=scale, pad_w=0, pad_h=-kCropTop
  std::vector<Detection> detections = ParseOutput(
      output,
      img_width,
      img_height,
      scale,
      0.0f,
      -static_cast<float>(kCropTop),
      conf_threshold);

  auto t_parse_end = std::chrono::high_resolution_clock::now();

  detections = NMS::ApplyPerClass(detections, iou_threshold);

  auto t_nms_end = std::chrono::high_resolution_clock::now();
  const double ms_reshape = std::chrono::duration<double, std::milli>(t_reshape_start - t_inf_end).count();
  const double ms_parse   = std::chrono::duration<double, std::milli>(t_parse_end - t_reshape_start).count();
  const double ms_nms     = std::chrono::duration<double, std::milli>(t_nms_end - t_parse_end).count();
  LOGD("Postprocess: reshape=%.2f ms  parse=%.2f ms  NMS=%.2f ms  [total=%.2f ms]",
       ms_reshape, ms_parse, ms_nms, ms_reshape + ms_parse + ms_nms);

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

  LOGD("ParseOutput: conf_threshold=%.3f", conf_threshold);

  if (num_anchors > 0) {
    const float* row0 = output.row(0);
    LOGD("ParseOutput first anchor: [%.2f,%.2f,%.2f,%.2f] scores=[%.3f,%.3f,%.3f]",
         row0[0], row0[1], row0[2], row0[3], row0[4], row0[5], row0[6]);

    // Scan all anchors to find global max score (every frame for diagnosis)
    {
      float global_max = -1e9f;
      int max_anchor = -1;
      for (int ii = 0; ii < num_anchors; ii++) {
        const float* r = output.row(ii);
        for (int c = 0; c < num_classes_; c++) {
          if (r[4 + c] > global_max) { global_max = r[4 + c]; max_anchor = ii; }
        }
      }
      const float* best_row = output.row(max_anchor >= 0 ? max_anchor : 0);
      LOGD("ParseOutput score scan: max_score=%.4f at anchor=%d bbox=[%.1f,%.1f,%.1f,%.1f]",
           global_max, max_anchor,
           best_row[0], best_row[1], best_row[2], best_row[3]);
    }
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
