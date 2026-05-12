# ROS 2 Android Perception

Object detection and tracking library for ROS 2 on Android using NCNN inference framework.

> [!NOTE]
> This is an dependency for [ros2_android](https://github.com/ros2-bachelor-thesis/ros2_android).

## Features

- **YOLOv9 Object Detection**: 3-class detector for Colorado Potato Beetle (beetle, larva, eggs)
- **Deep SORT Tracking**: Multi-object tracking with OSNet-AIN appearance-based re-identification (512-dim features)
- **Per-class track IDs**: Each class maintains independent ID sequences (beetle ID 1, larva ID 1, etc.)
- **NCNN Inference**: Optimized for ARM NEON (FP16 where supported)
- **Standalone Library**: Pure C++ with no ROS dependencies
- **Full-resolution annotation**: Detection runs at 640x352 but annotations are drawn on the original full-resolution image for sharp UI display

## Build Requirements

- Android NDK 26.3
- vcstool (`pip install vcstool`)
- CMake 3.13+
- Build machine: Linux x86_64

## Quick Start

```bash
# Fetch dependencies (NCNN, Eigen, OpenCV-mobile)
make deps

# Build NCNN for Android
make ncnn

# Build perception library (default: RelWithDebInfo)
make

# Or build with specific type
make debug    # Full debug symbols, no optimization
make release  # Optimized, stripped symbols
```

## Build Output

- **Library**: `build/libros2_android_perception.so`
- **Headers**: `include/perception/`
- **Models**: Deployed separately (see models/README.md)

Library is linked into `libandroid-ros.so` and accessed via JNI.

## API

### Simple detection + tracking

```cpp
#include <perception/object_detection_controller.h>

// Initialize complete pipeline (YOLO + Deep SORT)
perception::ObjectDetectionController detector(
    "models/yolov9_s_pobed.ncnn.param",
    "models/yolov9_s_pobed.ncnn.bin",
    "models/osnet_ain_x1_0.ncnn.param",
    "models/osnet_ain_x1_0.ncnn.bin"
);

// Process frame (raw RGB buffer)
auto tracks = detector.ProcessFrame(
    rgb_data, width, height,
    0.25f,   // conf_threshold
    0.45f    // iou_threshold
);

for (const auto& track : tracks) {
  int track_id = track.track_id;      // Per-class ID (starts at 1 per class)
  int class_id = track.class_id;      // 0=beetle, 1=larva, 2=eggs
  float* bbox = track.bbox;           // [x1, y1, x2, y2]
  float confidence = track.confidence;
}
```

### Detection + tracking + visualization

```cpp
// Process frame with annotated output
perception::PerceptionResult result = detector.ProcessFrame(
    bgr_data, width, height,       // Full-resolution BGR input
    depth_data, depth_w, depth_h,  // Optional depth (32FC1, meters)
    0.5f, 0.45f,                   // conf/iou thresholds
    true                           // enable_tracking
);

// result.detections - raw YOLO detections (640x352 coords)
// result.tracks     - Deep SORT tracks (640x352 coords)
// result.annotated_rgb_bgr - full-resolution BGR with annotations
// result.rgb_width / rgb_height - annotated frame dimensions
```

> [!NOTE]
> Detection runs at 640x352 (model input size). Bounding box coordinates in `detections` and `tracks` are in model input space. The annotated visualization is drawn on the original full-resolution image with coordinates scaled back.

## Known Issues

- **0 detections**: Verify models are loaded from correct path, check logcat for errors
- **Slow inference**: Check thread count and model input size
- **Cross-class matching**: Tracker does not enforce class constraints during matching - a track may match a detection of a different class in rare cases

## License

Apache 2.0

## References

- NCNN: https://github.com/Tencent/ncnn
- YOLOv9: https://github.com/WongKinYiu/yolov9
- Deep SORT: https://github.com/nwojke/deep_sort
- OSNet-AIN: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
- Python reference: [Vermin_Collector_ROS2_3D_Object_Detection/](https://phabricator.ict.tuwien.ac.at/source/Vermin_Collector_ROS2_3D_Object_Detection/)
