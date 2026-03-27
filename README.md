# ROS 2 Android Perception

Object detection and tracking library for ROS 2 on Android using NCNN inference framework.

## Features

- **YOLOv9 Object Detection**: 3-class detector for Colorado Potato Beetle detection (beetle, larva, eggs)
- **Deep SORT Tracking**: Multi-object tracking with appearance-based re-identification
- **NCNN Inference**: Optimized for ARM NEON with optional Vulkan GPU acceleration
- **ROS 2 Integration**: Subscribes to camera topics, publishes detection results
- **3D Localization**: Extracts 3D object centers from depth + point cloud data

## Build Requirements

- Android NDK 25.1 or later
- vcstool (`pip install vcstool`)
- CMake 3.13+
- Build machine: Linux x86_64

## Quick Start

```bash
# Set Android NDK path
export ANDROID_NDK=~/Android/Sdk/ndk/25.1.8937393

# Fetch dependencies (NCNN, Eigen, OpenCV-mobile)
make deps

# Build NCNN for Android
make ncnn

# Build perception library
make all

# Install to build/install
make install
```

## Build Output

- **Library**: `build/libros2_android_perception.a` (~5 MB)
- **Headers**: `build/install/include/perception/`
- **Models**: `build/install/share/perception/models/` (24 MB)

## Integration with ros2_android

Add to `ros2_android/ros.repos`:

```yaml
  ros2_android_perception:
    type: git
    url: git@phabricator.ict.tuwien.ac.at:source/ros2_android_perception.git
    version: main
```

Then run `make deps` in the ros2_android repository.

## Architecture

```
ros2_android_perception/
├── include/perception/         # Public API headers
│   ├── ncnn_detector.h         # YOLOv9 inference
│   ├── ncnn_reid.h             # mars-small128 feature extractor
│   ├── deep_sort_tracker.h     # Multi-object tracking
│   └── object_detection_controller.h  # ROS 2 node interface
├── src/                        # Implementation
├── models/                     # Pre-converted NCNN models (24 MB)
└── deps/                       # External dependencies (managed by vcstool)
```

## API Example

```cpp
#include <perception/ncnn_detector.h>
#include <perception/deep_sort_tracker.h>

// Initialize detector
NcnnDetector detector("yolov9_s_pobed.ncnn.param",
                      "yolov9_s_pobed.ncnn.bin",
                      use_vulkan=true);

// Initialize tracker
DeepSortTracker tracker("mars-small128.ncnn.param",
                        "mars-small128.ncnn.bin",
                        max_cosine_distance=0.4);

// Process frame
std::vector<Detection> detections = detector.Detect(frame, 0.25, 0.45);
std::vector<Track> tracks = tracker.Update(frame, detections);

// Access results
for (const auto& track : tracks) {
  int track_id = track.track_id;
  int class_id = track.class_id;
  float* bbox = track.bbox;  // [x1, y1, x2, y2]
}
```

## ROS 2 Topics

**Subscribed** (from external ZED camera):
- `/zed/zed_node/rgb/image_rect_color/compressed` (sensor_msgs/CompressedImage)
- `/zed/zed_node/depth/depth_registered` (sensor_msgs/Image)
- `/zed/zed_node/point_cloud/cloud_registered` (sensor_msgs/PointCloud2)

**Published**:
- `/cpb_beetle_center`, `/cpb_larva_center`, `/cpb_eggs_center` (geometry_msgs/Point)
- `/cpb_beetle`, `/cpb_larva`, `/cpb_eggs` (sensor_msgs/PointCloud2)

## Performance

Tested on Pixel 7 (Tensor G2 SoC):
- **YOLOv9 inference**: ~30-40 ms (CPU NEON)
- **mars-small128**: ~5 ms per detection
- **Target**: 20 Hz processing (50 ms/frame)

## Models

See [models/README.md](models/README.md) for model specifications and rebuild instructions.

## License

Apache 2.0

## References

- NCNN: https://github.com/Tencent/ncnn
- YOLOv9: https://github.com/WongKinYiu/yolov9
- Deep SORT: https://github.com/nwojke/deep_sort
