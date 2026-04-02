# ROS 2 Android Perception

Object detection and tracking library for ROS 2 on Android using NCNN inference framework.

> [!NOTE]
> This is an optional package for ros2_android. Build only if you need object detection capabilities.

## Features

- **YOLOv9 Object Detection**: 3-class detector for Colorado Potato Beetle detection (beetle, larva, eggs)
- **Deep SORT Tracking**: Multi-object tracking with appearance-based re-identification
- **NCNN Inference**: Optimized for ARM NEON with optional Vulkan GPU acceleration
- **Standalone Library**: Pure C++ with no ROS dependencies - integrates via ObjectDetectionController API
- **Android Native**: Builds as shared library, integrates via JNI with ros2_android

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

- **Library**: `build/libros2_android_perception.so` (~6 MB shared library)
- **Headers**: `include/perception/`
- **Models**: Deployed separately (see models/README.md)

## Integration with ros2_android

The perception library is automatically built as a dependency via git submodule:

```bash
cd ros2_android
git submodule update --init --recursive
make all  # Builds perception library as part of native build
```

Library is linked into `libandroid-ros.so` and accessed via JNI.

## Architecture

```
ros2_android_perception/
├── include/perception/         # Public API headers
│   ├── ncnn_detector.h         # YOLOv9 inference
│   ├── ncnn_reid.h             # mars-small128 feature extractor
│   ├── deep_sort_tracker.h     # Multi-object tracking
│   ├── object_detection_controller.h  # Complete pipeline
│   ├── detection.h             # Detection data structure
│   ├── track.h                 # Track data structure
│   ├── nms.h                   # Non-maximum suppression
│   └── image_preprocessor.h    # Image preprocessing utilities
├── src/perception/             # Implementation
├── models/                     # NCNN models (user-provided, see models/README.md)
└── deps/                       # External dependencies (NCNN, Eigen, OpenCV-mobile)
```

## API Example

```cpp
#include <perception/object_detection_controller.h>

// Initialize complete pipeline (YOLO + Deep SORT)
ObjectDetectionController detector(
    "models/yolov9_s_pobed.ncnn.param",
    "models/yolov9_s_pobed.ncnn.bin",
    "models/osnet_ain_x1_0.ncnn.param",
    "models/osnet_ain_x1_0.ncnn.bin",
    use_vulkan = false  // CPU NEON is faster on mobile
);

// Process frame (raw RGB buffer)
auto tracks = detector.ProcessFrame(
    rgb_data,      // uint8_t* RGB buffer
    width,         // Image width
    height,        // Image height
    conf_threshold = 0.25f,
    iou_threshold = 0.45f
);

// Access confirmed tracks (hits >= 3 frames)
for (const auto& track : tracks) {
  int track_id = track.track_id;     // Stable ID across frames
  int class_id = track.class_id;     // 0=beetle, 1=larva, 2=eggs
  float* bbox = track.bbox;          // [x1, y1, x2, y2]
  float confidence = track.confidence;
}
```

## Preprocessing Pipeline

The implementation matches the Python reference exactly:

1. **Input**: Raw RGB buffer (uint8, interleaved RGB)
2. **Resize**: Simple resize to 1280x736 (no letterbox, matches training data)
3. **Color conversion**: BGR → RGB
4. **Normalization**: Divide by 255.0 (uint8 → float32 [0,1])
5. **Layout**: HWC → CHW (for NCNN)

> [!IMPORTANT]
> The model was trained with simple resize (not letterbox), so letterbox preprocessing will produce incorrect results. Coordinate scaling uses separate X/Y factors.

## Performance

Tested on Pixel 7 (Tensor G2 SoC):

- **YOLOv9 inference**: ~30-40 ms (CPU NEON, 1280x736 input)
- **Deep SORT tracking**: ~10-15 ms (includes ReID feature extraction)
- **Total pipeline**: ~50-60 ms per frame (~17-20 FPS)
- **Memory**: ~60 MB (models + working buffers)

**Required models**:

- `yolov9_s_pobed.ncnn.{param,bin}` (~19 MB)
- `osnet_ain_x1_0.ncnn.{param,bin}` (~5 MB)

## Known Issues

- **0 detections**: Verify models are loaded from correct path, check logcat for errors
- **Slow inference**: Ensure Vulkan is disabled on mobile (CPU NEON is faster)
- **Wrong coordinates**: Model expects 1280x736 simple resize, not letterbox

## License

Apache 2.0

## References

- NCNN: https://github.com/Tencent/ncnn
- YOLOv9: https://github.com/WongKinYiu/yolov9
- Deep SORT: https://github.com/nwojke/deep_sort
- Python reference: `~/uni_projects/ROS2/Vermin_Collector_ROS2_3D_Object_Detection/
- ReID model reference: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
