#include "perception/visualization/annotator.h"
#include <algorithm>
#include <cmath>

namespace perception {
namespace visualization {

Annotator::Annotator(int line_width) : line_width_(line_width) {}

int Annotator::CalculateLineWidth(const cv::Mat& img) {
  // Python: max(round(sum(im.shape) / 2 * 0.003), 2)
  // im.shape = (height, width, channels) → sum = H + W + C
  int sum_shape = img.rows + img.cols + img.channels();
  int calculated = static_cast<int>(std::round(sum_shape / 2.0 * 0.003));
  return std::max(calculated, 2);
}

void Annotator::DrawBoundingBox(cv::Mat& img,
                                 const float bbox[4],
                                 const std::string& label,
                                 const cv::Scalar& color,
                                 const cv::Scalar& txt_color) const {
  if (img.empty()) {
    return;
  }

  // Auto-calculate line width if not specified
  int lw = (line_width_ > 0) ? line_width_ : CalculateLineWidth(img);

  // Convert bbox to integer pixel coordinates
  cv::Point p1(static_cast<int>(bbox[0]), static_cast<int>(bbox[1]));
  cv::Point p2(static_cast<int>(bbox[2]), static_cast<int>(bbox[3]));

  // Draw bounding box rectangle
  cv::rectangle(img, p1, p2, color, lw, cv::LINE_AA);

  // Draw label if provided
  if (!label.empty()) {
    // Calculate text size
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = lw / 3.0;
    int thickness = std::max(lw - 1, 1);
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);

    // Determine if label fits above box (Python: outside = p1[1] - h >= 3)
    bool outside = (p1.y - text_size.height >= 3);

    // Calculate label background rectangle
    cv::Point label_bg_p1 = p1;
    cv::Point label_bg_p2;
    if (outside) {
      // Label above box
      label_bg_p2 = cv::Point(p1.x + text_size.width, p1.y - text_size.height - 3);
    } else {
      // Label inside box (below top edge)
      label_bg_p2 = cv::Point(p1.x + text_size.width, p1.y + text_size.height + 3);
    }

    // Draw filled rectangle for label background
    cv::rectangle(img, label_bg_p1, label_bg_p2, color, cv::FILLED, cv::LINE_AA);

    // Calculate text position
    cv::Point text_origin;
    if (outside) {
      text_origin = cv::Point(p1.x, p1.y - 2);
    } else {
      text_origin = cv::Point(p1.x, p1.y + text_size.height + 2);
    }

    // Draw text
    cv::putText(img, label, text_origin, font_face, font_scale, txt_color, thickness, cv::LINE_AA);
  }
}

void Annotator::DrawTrackId(cv::Mat& img,
                             int track_id,
                             const float bbox[4],
                             const cv::Scalar& color) const {
  if (img.empty()) {
    return;
  }

  // Python code:
  // cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
  // cv2.putText(im0, f"ID: {track_id}", (int(x1), int(y1) - 10),
  //             cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

  cv::Point p1(static_cast<int>(bbox[0]), static_cast<int>(bbox[1]));
  cv::Point p2(static_cast<int>(bbox[2]), static_cast<int>(bbox[3]));

  // Draw bounding box
  cv::rectangle(img, p1, p2, color, 2, cv::LINE_AA);

  // Draw track ID text above box
  std::string id_text = "ID: " + std::to_string(track_id);
  cv::Point text_origin(p1.x, p1.y - 10);
  cv::putText(img, id_text, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv::LINE_AA);
}

}  // namespace visualization
}  // namespace perception
