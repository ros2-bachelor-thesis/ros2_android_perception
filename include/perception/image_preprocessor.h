#pragma once

#include <opencv2/opencv.hpp>
#include <ncnn/mat.h>

namespace perception {

/**
 * Image preprocessing utilities for NCNN inference
 *
 * Handles letterbox resizing, normalization, and format conversion
 * for YOLOv9 and mars-small128 models.
 */
class ImagePreprocessor {
 public:
  /**
   * Letterbox resize - maintain aspect ratio with padding
   *
   * @param image Input image (any size, BGR format)
   * @param target_size Target square size (e.g., 640 for YOLOv9)
   * @param pad_color Padding color (default: 114,114,114)
   * @return Resized image with padding
   */
  static cv::Mat LetterboxResize(const cv::Mat& image,
                                  int target_size,
                                  const cv::Scalar& pad_color = cv::Scalar(114, 114, 114));

  /**
   * Normalize image to [0, 1] range and convert BGR to RGB
   *
   * @param image Input image (uint8 BGR)
   * @return Normalized float RGB image
   */
  static cv::Mat NormalizeAndConvertRGB(const cv::Mat& image);

  /**
   * Convert OpenCV Mat (HWC) to NCNN Mat (CHW)
   *
   * @param image Input image (HxWxC format)
   * @return NCNN mat in CHW format ready for inference
   */
  static ncnn::Mat ToNCNN(const cv::Mat& image);

  /**
   * Full preprocessing pipeline for YOLOv9
   *
   * @param image Input image (any size, BGR uint8)
   * @param input_size Model input size (default: 640)
   * @return NCNN mat ready for YOLOv9 inference (CHW, normalized)
   */
  static ncnn::Mat PrepareForYOLO(const cv::Mat& image, int input_size = 640);

  /**
   * Preprocessing for mars-small128 ReID (crops detection bbox)
   *
   * @param image Source image
   * @param bbox Bounding box [x1, y1, x2, y2]
   * @param target_width ReID input width (default: 64)
   * @param target_height ReID input height (default: 128)
   * @return NCNN mat ready for ReID inference (uint8, NOT normalized)
   */
  static ncnn::Mat PrepareForReID(const cv::Mat& image,
                                   const float bbox[4],
                                   int target_width = 64,
                                   int target_height = 128);

  /**
   * Calculate scale and padding for letterbox resize
   * Used to map detected bbox back to original image coordinates
   */
  struct LetterboxParams {
    float scale;      // Resize scale factor
    int pad_w;        // Width padding (left)
    int pad_h;        // Height padding (top)
  };

  static LetterboxParams GetLetterboxParams(int src_width, int src_height, int target_size);
};

}  // namespace perception
