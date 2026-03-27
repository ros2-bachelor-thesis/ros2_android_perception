# NCNN Models for Object Detection

## Models

### YOLOv9 Object Detector
- **Files**: `yolov9_s_pobed.ncnn.{param,bin}`
- **Size**: ~19 MB
- **Input**: 640x640 RGB image (NCHW format, fp32, normalized [0,1])
- **Output**: Bounding boxes, confidences, class IDs
- **Classes**:
  - 0: cpb_beetle (Colorado Potato Beetle adult)
  - 1: cpb_larva (larva stage)
  - 2: cpb_eggs (egg clusters)

### mars-small128 ReID Feature Extractor
- **Files**: `mars-small128.ncnn.{param,bin}`
- **Size**: ~5.4 MB
- **Input**: 128x64 RGB image (NCHW format, uint8)
- **Output**: 128-dimensional feature vector (L2-normalized)
- **Purpose**: Appearance-based re-identification for Deep SORT tracking

## Total Size

24 MB (19 MB + 5.4 MB)

## Rebuilding Models

These models are pre-converted from PyTorch/TensorFlow using the yolov9 Nix flake:

```bash
cd ~/projects/bachelor-thesis/yolov9
nix build .#default       # YOLOv9
nix build .#mars-small128 # Deep SORT ReID
```

Then copy to this directory:

```bash
cp ~/projects/bachelor-thesis/yolov9/result/yolov9_s_pobed.ncnn.* .
cp ~/projects/bachelor-thesis/yolov9/result-1/mars-small128.ncnn.* .
```

## Model Specifications

### YOLOv9 Detection Pipeline
1. Preprocess image: letterbox resize to 640x640, normalize to [0,1]
2. Run NCNN inference
3. Parse output: decode bounding boxes, confidences, class IDs
4. Apply NMS (conf_threshold=0.25, iou_threshold=0.45)

### mars-small128 Feature Extraction
1. Crop image to detection bounding box
2. Resize to 128x64
3. Convert to uint8 (0-255 range, not normalized)
4. Run NCNN inference
5. L2-normalize output feature vector

## Usage in Code

```cpp
#include <perception/ncnn_detector.h>
#include <perception/ncnn_reid.h>

// Load models
NcnnDetector detector("models/yolov9_s_pobed.ncnn.param",
                      "models/yolov9_s_pobed.ncnn.bin",
                      use_vulkan=true);

NcnnReID reid("models/mars-small128.ncnn.param",
              "models/mars-small128.ncnn.bin");

// Run detection
std::vector<Detection> detections = detector.Detect(image, 0.25, 0.45);

// Extract ReID features
for (auto& det : detections) {
  det.feature = reid.Extract(image, det.bbox);
}
```
