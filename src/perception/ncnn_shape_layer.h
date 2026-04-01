#pragma once

#include <ncnn/layer.h>

namespace perception {

/**
 * NCNN custom Shape layer
 *
 * Implements ONNX Shape operator for mars-small128 ReID model.
 * Takes a tensor as input and outputs a 1D tensor containing its dimensions.
 *
 * Example:
 *   Input: [1, 128] tensor
 *   Output: [1, 128] as 1D int32 tensor
 *
 * Used in mars-small128 for dynamic batch size handling in BatchNorm/Reshape.
 */
class Shape : public ncnn::Layer {
 public:
  Shape();

  /**
   * Forward pass: Extract input tensor shape
   *
   * @param bottom_blob Input tensor (any dimensions)
   * @param top_blob Output 1D int32 tensor containing shape
   * @param opt NCNN options (unused)
   * @return 0 on success, -1 on error
   */
  virtual int forward(const ncnn::Mat& bottom_blob,
                      ncnn::Mat& top_blob,
                      const ncnn::Option& opt) const;
};

/**
 * Layer creator function for NCNN custom layer registration
 *
 * Usage:
 *   ncnn::Net net;
 *   net.register_custom_layer("Shape", Shape_layer_creator);
 *   net.load_param("model.param");
 */
ncnn::Layer* Shape_layer_creator(void* userdata);

}  // namespace perception
