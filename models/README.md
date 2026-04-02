# NCNN Models for Object Detection

## Required Models

### YOLOv9 Object Detector

- **Files**: `yolov9_s_pobed.ncnn.{param,bin}`
- **Size**: ~19 MB
- **Input**: 1280x736 RGB image (CHW format, fp32, normalized [0,1])
- **Output**: Bounding boxes, confidences, class IDs
- **Classes**:
  - 0: cpb_beetle (Colorado Potato Beetle adult)
  - 1: cpb_larva (larva stage)
  - 2: cpb_eggs (egg clusters)

### mars-small128 ReID Feature Extractor

- **Files**: `mars-small128.ncnn.{param,bin}`
- **Size**: ~5.4 MB
- **Input**: 128x64 RGB image (CHW format, uint8)
- **Output**: 128-dimensional feature vector (L2-normalized)
- **Purpose**: Appearance-based re-identification for Deep SORT tracking

## Total Size

24 MB (19 MB + 5.4 MB)

## Model Location

Models are **not** included in this repository due to size. Download or convert them separately:

```bash
# Option 1: Copy from existing yolov9 build
cp ~/projects/bachelor-thesis/yolov9/result/yolov9_s_pobed.ncnn.* models/
cp ~/projects/bachelor-thesis/yolov9/result-1/mars-small128.ncnn.* models/

# Option 2: Rebuild from source (requires Nix)
cd ~/projects/bachelor-thesis/yolov9
nix build .#default       # YOLOv9
nix build .#mars-small128 # Deep SORT ReID
```

## Model Specifications

### YOLOv9 Detection Pipeline

1. Preprocess image: **simple resize** to 1280x736 (NOT letterbox), normalize to [0,1]
2. Run NCNN inference
3. Parse output: decode bounding boxes, confidences, class IDs
4. Apply NMS (conf_threshold=0.25, iou_threshold=0.45)

> [!IMPORTANT]
> The model was trained with simple resize (cv2.resize without aspect ratio preservation), not letterbox. Using letterbox preprocessing will produce incorrect results.

### mars-small128 Feature Extraction

1. Crop image to detection bounding box
2. Resize to 128x64
3. Convert BGR → RGB (keep as uint8, NOT normalized)
4. Run NCNN inference
5. L2-normalize output feature vector

## Preprocessing Details

### YOLOv9 Input (matches Python reference)

```cpp
// Python reference: img = cv2.resize(self.rgb_image, (1280, 736))
cv::resize(image, resized, cv::Size(1280, 736), 0, 0, cv::INTER_LINEAR);

// NOT letterbox (aspect ratio is distorted):
// LetterboxResize(image, 1280, 736);  // WRONG!
```

### Coordinate Scaling

Since the resize is non-uniform, use separate X/Y scales:

```cpp
float scale_x = orig_width / 1280.0f;
float scale_y = orig_height / 736.0f;

// Map detection coordinates back to original image
float x1_orig = x1_model * scale_x;
float y1_orig = y1_model * scale_y;
```

## Usage in Code

```cpp
#include <perception/object_detection_controller.h>

// Load models from APK cache directory
std::string models_path = android_getCacheDir();
ObjectDetectionController detector(
    models_path + "/yolov9_s_pobed.ncnn.param",
    models_path + "/yolov9_s_pobed.ncnn.bin",
    models_path + "/mars-small128.ncnn.param",
    models_path + "/mars-small128.ncnn.bin",
    use_vulkan = false
);

// Verify models loaded
if (!detector.IsReady()) {
    LOGE("Failed to load NCNN models");
    return;
}

// Process frame
auto tracks = detector.ProcessFrame(rgb_data, width, height, 0.25f, 0.45f);
```

## Troubleshooting

### 0 Detections

1. **Verify model files exist**: Check path is correct
2. **Check preprocessing**: Must be simple resize (1280x736), not letterbox
3. **Check color format**: Input should be BGR (OpenCV format)
4. **Check confidence threshold**: Default 0.25 may be too high for your data

### Wrong Bounding Boxes

1. **Coordinate scaling**: Use separate X/Y scale factors, not uniform scale
2. **Padding removal**: Should be 0 for simple resize (no letterbox padding)
3. **Coordinate format**: Output is [x1, y1, x2, y2] in original image space

### Model Conversion

If you need to rebuild the models from PyTorch:

```bash
cd ~/projects/bachelor-thesis/yolov9

# Convert YOLOv9
nix build .#default
ls result/yolov9_s_pobed.ncnn.*

# Convert mars-small128
nix build .#mars-small128
ls result-1/mars-small128.ncnn.*
```

See `yolov9/flake.nix` for conversion pipeline details.
