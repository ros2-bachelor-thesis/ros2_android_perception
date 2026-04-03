#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace perception {
namespace visualization {

/**
 * Image annotation helper (C++ equivalent of Python's YOLOv5 Annotator)
 *
 * Draws bounding boxes, labels, and track IDs on OpenCV images.
 * Matches Python's annotation style: box + label background + text.
 *
 * Line width auto-scales based on image size.
 */
class Annotator {
 public:
  /**
   * Constructor
   * @param line_width Line thickness (auto-calculated if 0)
   */
  explicit Annotator(int line_width = 0);

  /**
   * Draw bounding box with label
   * @param img Image to draw on (modified in-place)
   * @param bbox Bounding box [x1, y1, x2, y2]
   * @param label Text label (e.g., "cpb_beetle 0.95")
   * @param color Box + label background color (BGR)
   * @param txt_color Text color (BGR), default white
   */
  void DrawBoundingBox(cv::Mat& img,
                       const float bbox[4],
                       const std::string& label,
                       const cv::Scalar& color,
                       const cv::Scalar& txt_color = cv::Scalar(255, 255, 255)) const;

  /**
   * Draw Deep SORT track ID on bounding box
   * @param img Image to draw on (modified in-place)
   * @param track_id Unique track identifier
   * @param bbox Bounding box [x1, y1, x2, y2]
   * @param color Text + outline color (BGR)
   */
  void DrawTrackId(cv::Mat& img,
                   int track_id,
                   const float bbox[4],
                   const cv::Scalar& color) const;

  /**
   * Calculate line width based on image dimensions
   * Matches Python: max(round(sum(im.shape) / 2 * 0.003), 2)
   */
  static int CalculateLineWidth(const cv::Mat& img);

 private:
  int line_width_;  ///< Line thickness for rectangles (0 = auto)
};

}  // namespace visualization
}  // namespace perception
