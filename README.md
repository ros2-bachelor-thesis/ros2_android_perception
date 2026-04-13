# ROS 2 Android Perception

Object detection and tracking library for ROS 2 on Android using NCNN inference framework.

> [!NOTE]
> This is an optional package for ros2_android. Build only if you need object detection capabilities.

## Features

- **YOLOv9 Object Detection**: 3-class detector for Colorado Potato Beetle (beetle, larva, eggs)
- **Deep SORT Tracking**: Multi-object tracking with OSNet-AIN appearance-based re-identification (512-dim features)
- **Per-class track IDs**: Each class maintains independent ID sequences (beetle ID 1, larva ID 1, etc.)
- **NCNN Inference**: Optimized for ARM NEON (FP16 where supported)
- **Standalone Library**: Pure C++ with no ROS dependencies - integrates via ObjectDetectionController API
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
│   ├── object_detection_controller.h  # Complete pipeline API
│   ├── ncnn_detector.h         # YOLOv9 inference
│   ├── ncnn_reid.h             # OSNet-AIN ReID feature extractor (512-dim)
│   ├── deep_sort_tracker.h     # Multi-object tracking
│   ├── kalman_filter.h         # 8D constant velocity Kalman filter
│   ├── linear_assignment.h     # Hungarian algorithm + cascade matching
│   ├── detection.h             # Detection data structure
│   ├── track.h                 # Track data structure
│   ├── types.h                 # PerceptionResult struct
│   ├── nms.h                   # Non-maximum suppression (class-aware)
│   ├── image_preprocessor.h    # Image preprocessing utilities
│   ├── log.h                   # Logging macros
│   └── visualization/
│       └── annotator.h         # Bounding box annotation (auto-scaling)
├── src/                        # Implementation
│   ├── object_detection_controller.cc
│   ├── ncnn_detector.cc
│   ├── ncnn_reid.cc
│   ├── deep_sort_tracker.cc
│   ├── kalman_filter.cc
│   ├── linear_assignment.cc
│   ├── nms.cc
│   ├── image_preprocessor.cc
│   └── visualization/
│       └── annotator.cc
├── models/                     # NCNN models (user-provided)
└── deps/                       # External dependencies (NCNN, Eigen, OpenCV-mobile)
```

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

## Preprocessing Pipeline

Matches the Python reference (`object_detection.py`):

1. **Input**: BGR buffer (any resolution, from ZED compressed JPEG)
2. **Resize**: `cv::resize` to 640x360
3. **Crop**: Remove 4px top and bottom to get 640x352 (multiple of stride 32)
4. **NCNN resize**: Internal resize from 640x352 to 1280x736 (model input)
5. **Normalization**: Divide by 255.0 (uint8 to float32 [0,1])

> [!IMPORTANT]
> The model was trained with simple resize (not letterbox). Coordinate scaling uses separate X/Y factors to map detections back from 1280x736 to 640x352 (model input space).

## Performance

Tested on Pixel 7 (Tensor G2 SoC):

- **YOLOv9 inference**: ~30-40 ms (CPU NEON, 1280x736 input)
- **Deep SORT tracking**: ~10-15 ms (includes OSNet-AIN ReID feature extraction)
- **Total pipeline**: ~50-60 ms per frame (~17-20 FPS)
- **Memory**: ~60 MB (models + working buffers)

**Required models**:

- `yolov9_s_pobed.ncnn.{param,bin}` - YOLOv9-small trained on CPB dataset (~19 MB)
- `osnet_ain_x1_0.ncnn.{param,bin}` - OSNet-AIN x1.0 ReID model (~5 MB)

## Deep SORT Tracking Details

- **Feature extractor**: OSNet-AIN x1.0 (512-dim L2-normalized features, 128x256 input)
- **Kalman filter**: 8D constant velocity model `[x, y, a, h, vx, vy, va, vh]`
- **Matching**: Two-stage cascade (appearance cosine distance + IoU fallback)
- **Track lifecycle**: Tentative (negative ID) -> Confirmed (positive per-class ID after n_init=3 hits) -> Deleted (after max_age=30 misses)
- **Per-class IDs**: Each class (beetle, larva, eggs) has its own independent ID counter starting at 1
- **Feature gallery**: Stores up to nn_budget=100 features per track for re-identification after occlusion
- **NMS**: Class-aware (applied per-class before tracking)

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
