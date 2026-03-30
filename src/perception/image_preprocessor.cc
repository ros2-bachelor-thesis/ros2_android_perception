#include "perception/image_preprocessor.h"
#include <algorithm>

namespace perception {

ImagePreprocessor::LetterboxParams ImagePreprocessor::GetLetterboxParams(
    int src_width, int src_height, int target_width, int target_height) {
  LetterboxParams params;

  // Calculate scale to fit within target dimensions while maintaining aspect ratio
  float scale_w = static_cast<float>(target_width) / src_width;
  float scale_h = static_cast<float>(target_height) / src_height;
  params.scale = std::min(scale_w, scale_h);

  // Calculate new dimensions after scaling
  int new_width = static_cast<int>(src_width * params.scale);
  int new_height = static_cast<int>(src_height * params.scale);

  // Calculate padding to center the image
  params.pad_w = (target_width - new_width) / 2;
  params.pad_h = (target_height - new_height) / 2;

  return params;
}

cv::Mat ImagePreprocessor::LetterboxResize(const cv::Mat& image,
                                            int target_width,
                                            int target_height,
                                            const cv::Scalar& pad_color) {
  // Calculate resize parameters
  auto params = GetLetterboxParams(image.cols, image.rows, target_width, target_height);

  // Resize image maintaining aspect ratio
  int new_width = static_cast<int>(image.cols * params.scale);
  int new_height = static_cast<int>(image.rows * params.scale);
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

  // Create canvas with padding (target dimensions)
  cv::Mat canvas(target_height, target_width, image.type(), pad_color);

  // Copy resized image to center of canvas
  cv::Rect roi(params.pad_w, params.pad_h, new_width, new_height);
  resized.copyTo(canvas(roi));

  return canvas;
}

cv::Mat ImagePreprocessor::NormalizeAndConvertRGB(const cv::Mat& image) {
  // Convert BGR to RGB
  cv::Mat rgb;
  cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

  // Convert to float and normalize to [0, 1]
  cv::Mat normalized;
  rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

  return normalized;
}

ncnn::Mat ImagePreprocessor::ToNCNN(const cv::Mat& image) {
  // Input image should be float32 in HWC format
  assert(image.type() == CV_32FC3 && "Image must be CV_32FC3 format");

  int w = image.cols;
  int h = image.rows;
  int channels = 3;

  // Create NCNN mat in CHW format
  ncnn::Mat ncnn_mat(w, h, channels);

  // Convert HWC to CHW
  for (int c = 0; c < channels; c++) {
    float* ptr = ncnn_mat.channel(c);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        ptr[y * w + x] = image.at<cv::Vec3f>(y, x)[c];
      }
    }
  }

  return ncnn_mat;
}

ncnn::Mat ImagePreprocessor::PrepareForYOLO(const cv::Mat& image,
                                             int input_width,
                                             int input_height) {
  // Step 1: Letterbox resize to target dimensions
  cv::Mat resized = LetterboxResize(image, input_width, input_height);

  // Step 2: Normalize and convert BGR to RGB
  cv::Mat normalized = NormalizeAndConvertRGB(resized);

  // Step 3: Convert to NCNN format (CHW)
  ncnn::Mat ncnn_input = ToNCNN(normalized);

  return ncnn_input;
}

ncnn::Mat ImagePreprocessor::PrepareForReID(const cv::Mat& image,
                                             const float bbox[4],
                                             int target_width,
                                             int target_height) {
  // Clamp bbox to image boundaries
  int x1 = std::max(0, static_cast<int>(bbox[0]));
  int y1 = std::max(0, static_cast<int>(bbox[1]));
  int x2 = std::min(image.cols, static_cast<int>(bbox[2]));
  int y2 = std::min(image.rows, static_cast<int>(bbox[3]));

  // Ensure valid rectangle
  if (x2 <= x1 || y2 <= y1) {
    // Invalid bbox, return empty mat
    return ncnn::Mat();
  }

  // Crop detection from image
  cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
  cv::Mat cropped = image(roi);

  // Resize to ReID input size (128x64)
  cv::Mat resized;
  cv::resize(cropped, resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);

  // Convert BGR to RGB (but keep as uint8, mars-small128 expects uint8 input)
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  // Convert to NCNN mat (CHW format, uint8)
  ncnn::Mat ncnn_mat = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB,
                                                target_width, target_height);

  return ncnn_mat;
}

}  // namespace perception
