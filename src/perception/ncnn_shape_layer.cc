#include "perception/ncnn_shape_layer.h"

namespace perception {

Shape::Shape() {
  one_blob_only = true;  // Takes exactly 1 input, produces 1 output
}

int Shape::forward(const ncnn::Mat& bottom_blob,
                   ncnn::Mat& top_blob,
                   const ncnn::Option& opt) const {
  // Determine number of dimensions
  int ndim = bottom_blob.dims;

  // Create output tensor: 1D int32 array with `ndim` elements
  // NCNN uses 4u for int32 element size
  top_blob.create(ndim, 4u, opt.blob_allocator);
  if (top_blob.empty()) {
    return -1;  // Allocation failed
  }

  // Fill output with shape dimensions
  int* outptr = (int*)top_blob;

  if (ndim == 1) {
    // 1D tensor: [w]
    outptr[0] = bottom_blob.w;
  } else if (ndim == 2) {
    // 2D tensor: [w, h]
    outptr[0] = bottom_blob.w;
    outptr[1] = bottom_blob.h;
  } else if (ndim == 3) {
    // 3D tensor: [w, h, c]
    outptr[0] = bottom_blob.w;
    outptr[1] = bottom_blob.h;
    outptr[2] = bottom_blob.c;
  } else if (ndim == 4) {
    // 4D tensor: [w, h, d, c] (NCNN supports up to 4D with d dimension)
    outptr[0] = bottom_blob.w;
    outptr[1] = bottom_blob.h;
    outptr[2] = bottom_blob.d;
    outptr[3] = bottom_blob.c;
  } else {
    // Unsupported dimension count
    return -1;
  }

  return 0;
}

ncnn::Layer* Shape_layer_creator(void* /*userdata*/) {
  return new Shape();
}

}  // namespace perception
