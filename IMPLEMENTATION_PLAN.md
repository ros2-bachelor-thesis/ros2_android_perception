# ROS2 Android Perception - Implementation Plan

**Status**: Infrastructure complete, core implementation needed
**Date**: 2026-03-29
**Estimated Effort**: ~1000-1500 LOC, 3-5 days full-time

## Current Status Summary

### ✅ Complete
- Build system (Makefile + CMake for Android NDK arm64-v8a)
- Dependencies fetched (OpenCV-mobile 4.13.0, NCNN, Eigen3)
- Pre-converted NCNN models (yolov9_s_pobed + mars-small128, 24 MB)
- Data structures (Detection, Track with Eigen state vector)
- NMS implementation (131 LOC, IoU calculation, class-aware filtering)
- Image preprocessing (132 LOC, letterbox resize, BGR→RGB, normalization)

### ⚠️ Stub Files (11 LOC each, TODO placeholders)
- `ncnn_detector.cc` - NCNN inference pipeline
- `ncnn_reid.cc` - ReID feature extraction
- `deep_sort_tracker.cc` - Main tracker orchestration
- `kalman_filter.cc` - State prediction/update
- `linear_assignment.cc` - Hungarian algorithm
- `object_detection_controller.cc` - ROS 2 integration

## Architecture Overview

```
ros2_android_perception/
├── include/perception/          # Public API (4 headers complete)
│   ├── detection.h              ✅ Complete
│   ├── track.h                  ✅ Complete
│   ├── nms.h                    ✅ Complete
│   └── image_preprocessor.h     ✅ Complete
├── src/                         # Implementation (8 files)
│   ├── nms.cc                   ✅ Complete (131 LOC)
│   ├── image_preprocessor.cc    ✅ Complete (132 LOC)
│   ├── ncnn_detector.cc         ⚠️ TODO (est. 250 LOC)
│   ├── ncnn_reid.cc             ⚠️ TODO (est. 100 LOC)
│   ├── kalman_filter.cc         ⚠️ TODO (est. 300 LOC)
│   ├── linear_assignment.cc     ⚠️ TODO (est. 200 LOC)
│   ├── deep_sort_tracker.cc     ⚠️ TODO (est. 350 LOC)
│   └── object_detection_controller.cc ⚠️ TODO (est. 300 LOC)
├── models/                      ✅ NCNN models present (24 MB)
└── deps/                        ✅ OpenCV-mobile extracted
```

## Reference Implementation

**Source**: `~/uni_projects/ROS2/Vermin_Collector_ROS2_3D_Object_Detection/`
- `object_detection.py` (449 lines) - Main ROS 2 node
- `deep_sort/` Python modules:
  - `kalman_filter.py` (229 lines) - 8D constant velocity model
  - `tracker.py` (138 lines) - Track management + matching cascade
  - `linear_assignment.py` (190 lines) - Hungarian + cascade matching
  - `nn_matching.py` (177 lines) - Cosine distance metric
  - `track.py` (166 lines) - Track state lifecycle
  - `iou_matching.py` (81 lines) - IoU-based backup matching

**Total Python**: ~1031 lines Deep SORT + 449 lines ROS node = 1480 lines

## Implementation Phases

### Phase 1: NCNN Detector (YOLOv9 Inference)
**Files**: `ncnn_detector.cc`, new header `ncnn_detector.h`
**Effort**: 250-300 LOC, 4-6 hours
**Dependencies**: NCNN, OpenCV, NMS (complete), ImagePreprocessor (complete)

#### Tasks
1. **Create public API** (`include/perception/ncnn_detector.h`)
   ```cpp
   class NcnnDetector {
    public:
     NcnnDetector(const std::string& param_path,
                  const std::string& bin_path,
                  bool use_vulkan = false);
     ~NcnnDetector();

     // Detect objects in image
     std::vector<Detection> Detect(const cv::Mat& image,
                                    float conf_threshold = 0.25f,
                                    float iou_threshold = 0.45f);

     bool IsLoaded() const;

    private:
     void LoadModel(const std::string& param, const std::string& bin);
     std::vector<Detection> ParseOutput(const ncnn::Mat& output,
                                         int img_width, int img_height,
                                         float conf_threshold);

     ncnn::Net net_;
     bool loaded_ = false;
     int input_size_ = 640;
   };
   ```

2. **Model loading**
   - `net_.load_param()` / `net_.load_model()`
   - Vulkan option (default OFF for CPU NEON)
   - Error handling for missing files

3. **Inference pipeline**
   - Preprocess with `ImagePreprocessor::PrepareForYOLO()`
   - Create NCNN extractor
   - `ex.input("in0", input_mat)`
   - `ex.extract("out0", output_mat)` (verify layer name from .param)

4. **Output parsing** (YOLOv9 format)
   - Expected shape: `[1, 8400, 8]` or `[1, num_anchors, 5+num_classes]`
   - Decode: `[cx, cy, w, h, objectness, class0_prob, class1_prob, class2_prob]`
   - Convert to Detection structs (x1,y1,x2,y2 format)
   - Apply confidence threshold

5. **NMS application**
   - Use existing `NMS::ApplyPerClass()` for class-aware suppression
   - Default IoU threshold: 0.45

6. **Coordinate scaling**
   - Account for letterbox padding in `ImagePreprocessor`
   - Map detection coords back to original image space

**Testing**:
- Load model without crash
- Inference on 640x640 test image
- Verify output shape and detection count
- Check bbox coordinates are reasonable

---

### Phase 2: ReID Feature Extractor (mars-small128)
**Files**: `ncnn_reid.cc`, new header `ncnn_reid.h`
**Effort**: 100-120 LOC, 2-3 hours
**Dependencies**: NCNN, OpenCV, ImagePreprocessor (complete)

#### Tasks
1. **Create public API** (`include/perception/ncnn_reid.h`)
   ```cpp
   class NcnnReID {
    public:
     NcnnReID(const std::string& param_path,
              const std::string& bin_path);
     ~NcnnReID();

     // Extract 128-dim feature from detection bbox
     std::vector<float> Extract(const cv::Mat& image,
                                 const float bbox[4]);

     bool IsLoaded() const;

    private:
     void LoadModel(const std::string& param, const std::string& bin);
     void L2Normalize(std::vector<float>& feature);

     ncnn::Net net_;
     bool loaded_ = false;
     int input_width_ = 64;
     int input_height_ = 128;
   };
   ```

2. **Model loading**
   - mars-small128 NCNN model
   - No Vulkan needed (small model, CPU fast enough)

3. **Feature extraction**
   - Crop bbox from image with `ImagePreprocessor::PrepareForReID()`
   - Input: 128x64 uint8 (NOT normalized - mars-small128 expects 0-255)
   - Inference: `ex.input() → ex.extract()`
   - Output: 128-dim float vector

4. **L2 normalization**
   - Normalize feature to unit length: `feature /= sqrt(sum(feature^2))`
   - Required for cosine distance matching

**Testing**:
- Extract feature from test bbox
- Verify output is 128-dim
- Check L2 norm ≈ 1.0
- Compare similar/dissimilar bbox features

---

### Phase 3: Kalman Filter (8D Constant Velocity)
**Files**: `kalman_filter.cc`, new header `kalman_filter.h`
**Effort**: 300-350 LOC, 6-8 hours
**Dependencies**: Eigen3
**Reference**: `kalman_filter.py` (229 lines)

#### Tasks
1. **Create public API** (`include/perception/kalman_filter.h`)
   ```cpp
   class KalmanFilter {
    public:
     KalmanFilter();

     // Initialize new track from measurement [x, y, a, h]
     void Initiate(const float measurement[4],
                   Eigen::VectorXf& mean,
                   Eigen::MatrixXf& covariance);

     // Predict next state
     void Predict(Eigen::VectorXf& mean,
                  Eigen::MatrixXf& covariance);

     // Update with measurement
     void Update(Eigen::VectorXf& mean,
                 Eigen::MatrixXf& covariance,
                 const float measurement[4]);

     // Project state to measurement space
     void Project(const Eigen::VectorXf& mean,
                  const Eigen::MatrixXf& covariance,
                  Eigen::VectorXf& proj_mean,
                  Eigen::MatrixXf& proj_cov);

     // Mahalanobis gating distance
     Eigen::VectorXf GatingDistance(const Eigen::VectorXf& mean,
                                     const Eigen::MatrixXf& covariance,
                                     const Eigen::MatrixXf& measurements,
                                     bool only_position = false);

    private:
     Eigen::MatrixXf motion_mat_;     // 8x8 state transition (constant velocity)
     Eigen::MatrixXf update_mat_;     // 4x8 observation matrix
     float std_weight_position_;      // 1/20
     float std_weight_velocity_;      // 1/160
   };
   ```

2. **State representation** (8D vector)
   - `[x, y, a, h, vx, vy, va, vh]`
   - `(x,y)` = bbox center
   - `a` = aspect ratio (w/h)
   - `h` = height
   - `v*` = velocities

3. **Initiate** (line 55-86 in Python)
   - Mean: `[x, y, a, h, 0, 0, 0, 0]` (velocities zero)
   - Covariance: diagonal with uncertainty proportional to height
   - `std = [2*h/20, 2*h/20, 1e-2, 2*h/20, 10*h/160, 10*h/160, 1e-5, 10*h/160]`

4. **Predict** (line 88-123)
   - State transition: `mean = F * mean` (constant velocity model)
   - Covariance: `P = F*P*F' + Q` (add process noise)
   - Motion matrix F is identity with dt=1 on off-diagonal blocks

5. **Update** (line 154-186)
   - Standard Kalman update with Cholesky decomposition
   - Measurement model: `z = H * x` (observe position only, not velocity)
   - Innovation: `y = z_measured - H*x_predicted`
   - Kalman gain: `K = P*H' * (H*P*H' + R)^-1`
   - State update: `x = x + K*y`, `P = P - K*H*P`

6. **Project** (line 125-152)
   - Project 8D state to 4D measurement space
   - Used for gating and data association
   - Add measurement noise covariance

7. **Gating distance** (line 188-229)
   - Mahalanobis distance for outlier rejection
   - `d^2 = (z - Hx)' * S^-1 * (z - Hx)` where S = H*P*H'+R
   - Chi-square threshold: 9.4877 (4 DOF, 95% confidence)
   - Option for position-only gating (2 DOF → threshold 5.9915)

**Testing**:
- Initialize track, predict 10 steps, check state propagation
- Update with measurement, verify convergence
- Compare against Python implementation on same data

---

### Phase 4: Linear Assignment (Hungarian Algorithm)
**Files**: `linear_assignment.cc`, new header `linear_assignment.h`
**Effort**: 200-250 LOC, 4-6 hours
**Dependencies**: Eigen3 (or implement Hungarian from scratch)
**Reference**: `linear_assignment.py` (190 lines), uses scipy

#### Tasks
1. **Create public API** (`include/perception/linear_assignment.h`)
   ```cpp
   struct MatchResult {
     std::vector<std::pair<int, int>> matches;  // (track_idx, detection_idx)
     std::vector<int> unmatched_tracks;
     std::vector<int> unmatched_detections;
   };

   class LinearAssignment {
    public:
     // Hungarian algorithm on cost matrix
     static MatchResult MinCostMatching(
         const Eigen::MatrixXf& cost_matrix,
         float max_distance,
         const std::vector<int>& track_indices,
         const std::vector<int>& detection_indices);

     // Matching cascade (prioritize recent tracks)
     static MatchResult MatchingCascade(
         const Eigen::MatrixXf& cost_matrix,
         float max_distance,
         int cascade_depth,
         const std::vector<Track>& tracks,
         const std::vector<int>& track_indices,
         const std::vector<int>& detection_indices);

     // Gate cost matrix with Mahalanobis distance
     static void GateCostMatrix(
         Eigen::MatrixXf& cost_matrix,
         const KalmanFilter& kf,
         const std::vector<Track>& tracks,
         const std::vector<Detection>& detections,
         const std::vector<int>& track_indices,
         const std::vector<int>& detection_indices,
         float gating_threshold = 9.4877f);
   };
   ```

2. **Hungarian algorithm options**
   - **Option A**: Use existing C++ library (e.g., `munkres-cpp`)
     - Pros: Tested, ~100 LOC saved
     - Cons: Add dependency
   - **Option B**: Implement from scratch
     - Pros: No dependency, educational
     - Cons: ~150 LOC, complex, error-prone
   - **Recommendation**: Option A for speed, Option B if avoiding deps

3. **MinCostMatching** (line 11-75 in Python)
   - Build cost matrix (tracks × detections)
   - Mask entries > max_distance (set to INFTY)
   - Run Hungarian to get optimal assignment
   - Filter matches where cost > max_distance

4. **MatchingCascade** (line 78-149)
   - Prioritize tracks by `time_since_update` (recent first)
   - Iterate cascade levels 0 to max_age
   - At each level, match tracks with `time_since_update == level`
   - Removes matched detections from pool for next level

5. **GateCostMatrix** (line 151-177)
   - Compute Mahalanobis distance for each (track, detection) pair
   - Set cost to INFTY if distance > chi2_threshold
   - Prevents physically impossible associations

**Testing**:
- Small cost matrix (3x3), verify Hungarian output
- Cascade with mixed track ages
- Gating with outlier detections

---

### Phase 5: Deep SORT Tracker (Main Orchestration)
**Files**: `deep_sort_tracker.cc`, new header `deep_sort_tracker.h`
**Effort**: 350-400 LOC, 8-10 hours
**Dependencies**: All above components
**Reference**: `tracker.py` (138 lines), `track.py` (166 lines), `nn_matching.py` (177 lines)

#### Tasks
1. **Create public API** (`include/perception/deep_sort_tracker.h`)
   ```cpp
   struct DeepSortConfig {
     float max_cosine_distance = 0.4f;
     int nn_budget = 100;
     int max_age = 30;
     int n_init = 3;
     float max_iou_distance = 0.7f;
   };

   class DeepSortTracker {
    public:
     DeepSortTracker(const std::string& reid_param,
                     const std::string& reid_bin,
                     const DeepSortConfig& config = DeepSortConfig());
     ~DeepSortTracker();

     // Main update: predict + match + update tracks
     std::vector<Track> Update(const cv::Mat& image,
                               const std::vector<Detection>& detections);

     // Get confirmed tracks only
     std::vector<Track> GetConfirmedTracks() const;

    private:
     void Predict();
     void MatchDetections(const std::vector<Detection>& detections);
     void InitiateTrack(const Detection& detection);
     void UpdateTrack(Track& track, const Detection& detection);
     void MarkMissed(Track& track);
     void DeleteOldTracks();

     float CosineDistance(const std::vector<float>& f1,
                          const std::vector<float>& f2);

     Eigen::MatrixXf BuildCostMatrix(
         const std::vector<Track>& tracks,
         const std::vector<Detection>& detections,
         const std::vector<int>& track_indices,
         const std::vector<int>& detection_indices);

     std::vector<Track> tracks_;
     int next_id_ = 1;
     KalmanFilter kf_;
     NcnnReID reid_;
     DeepSortConfig config_;

     // Feature gallery for cosine distance metric
     std::map<int, std::vector<std::vector<float>>> track_features_;
   };
   ```

2. **Track lifecycle** (from `track.py`)
   - **Tentative**: New track, not yet confirmed (hits < n_init)
   - **Confirmed**: Matched n_init consecutive times
   - **Deleted**: Missed for max_age frames

3. **Update pipeline** (tracker.py line 58-91)
   ```
   1. Predict all tracks (Kalman predict step)
   2. Match detections to tracks (cascade + IoU fallback)
   3. Update matched tracks (Kalman update + feature smoothing)
   4. Mark unmatched tracks as missed
   5. Initiate new tracks from unmatched detections
   6. Delete old tracks (age > max_age)
   7. Update feature gallery
   ```

4. **Matching strategy** (tracker.py line 93-131)
   - **Stage 1**: Cascade matching on confirmed tracks (appearance features)
     - Use cosine distance on ReID features
     - Gate with Mahalanobis distance
     - Prioritize recently seen tracks
   - **Stage 2**: IoU matching on remaining + tentative tracks
     - Use IoU distance (cheaper, no features needed)
     - Fallback for occluded/unmatched tracks

5. **Feature gallery** (nn_matching.py)
   - Store last `nn_budget` features per track
   - Compute distance as min over all stored features
   - Allows re-identification after occlusion
   - Update gallery on each successful match

6. **Cosine distance metric**
   - `distance = 1 - dot(f1, f2) / (||f1|| * ||f2||)`
   - Features are L2-normalized, so simplifies to `1 - dot(f1, f2)`
   - Threshold: max_cosine_distance = 0.4

7. **IoU-based matching** (iou_matching.py)
   - Compute IoU between predicted track bbox and detection bbox
   - Cost = 1 - IoU
   - Threshold: max_iou_distance = 0.7 → min IoU = 0.3

**Testing**:
- Single track with 10 consecutive detections → confirm
- Track with 3 misses → tentative, then delete
- Two overlapping objects → verify stable IDs
- Occlusion simulation → feature gallery re-ID

---

### Phase 6: ROS 2 Controller Integration
**Files**: `object_detection_controller.cc`, new header `object_detection_controller.h`
**Effort**: 300-350 LOC, 6-8 hours
**Dependencies**: All perception components + ROS 2 (sensor_msgs, geometry_msgs)
**Reference**: `object_detection.py` (449 lines), ros2_android controllers

#### Tasks
1. **Create public API** (`include/perception/object_detection_controller.h`)
   ```cpp
   class ObjectDetectionController {
    public:
     ObjectDetectionController(const std::string& models_path);
     ~ObjectDetectionController();

     void Enable();
     void Disable();
     bool IsEnabled() const;

     // Set callbacks for ROS 2 integration
     void SetImageCallback(
         std::function<void(const sensor_msgs::msg::CompressedImage::SharedPtr)> cb);

     void SetPublishCallback(
         std::function<void(const std::vector<Track>&)> cb);

    private:
     void OnCompressedImage(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
     void PublishDetections(const std::vector<Track>& tracks);

     NcnnDetector detector_;
     DeepSortTracker tracker_;
     std::string models_path_;
     bool enabled_ = false;
   };
   ```

2. **Model loading**
   - Load from `models_path + "/yolov9_s_pobed.ncnn.{param,bin}"`
   - ReID model loaded by DeepSortTracker

3. **Image callback** (object_detection.py line 168-184, 186-386)
   - Receive `sensor_msgs::msg::CompressedImage`
   - Decode JPEG with OpenCV: `cv::imdecode()`
   - Run detection: `detector_.Detect(image, 0.25, 0.45)`
   - Run tracking: `tracker_.Update(image, detections)`

4. **Publishing** (object_detection.py line 82-93, 328-360)
   - Filter confirmed tracks only
   - Publish per class:
     - `/cpb_beetle_center` → geometry_msgs::msg::Point
     - `/cpb_larva_center` → geometry_msgs::msg::Point
     - `/cpb_eggs_center` → geometry_msgs::msg::Point
   - Point format: `(center_x, center_y, 0.0)` (2D only, no depth)

5. **QoS settings**
   - Subscription: BEST_EFFORT, KeepLast(1) (match camera publisher)
   - Publications: RELIABLE, KeepLast(10) (ensure delivery)

6. **Error handling**
   - Model load failures → log error, disable controller
   - Decode failures → skip frame, log warning
   - Inference failures → skip frame, log error

**Testing**:
- Subscribe to rosbag CompressedImage topic
- Verify detections publish to correct topics
- Check frame rate (target >5 Hz)
- Monitor CPU/memory usage

---

### Phase 7: Integration into ros2_android Main App
**Files**: Modify `ros2_android/` build system and app code
**Effort**: 100-150 LOC modifications, 2-4 hours
**Dependencies**: Completed ros2_android_perception library

#### Tasks
1. **Add perception as ExternalProject** (`ros2_android/dependencies.cmake`)
   ```cmake
   ExternalProject_Add(deps-ros2_android_perception
     SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ros2_android_perception
     CMAKE_ARGS
       ${android_cmake_args}
       -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/deps-install
       -Dncnn_DIR=<ncnn_install_dir>/lib/cmake/ncnn
     BUILD_ALWAYS OFF
     INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
   )
   ```

2. **Link perception library** (`ros2_android/src/CMakeLists.txt`)
   ```cmake
   find_package(ros2_android_perception REQUIRED)
   target_link_libraries(android-ros PRIVATE ros2_android_perception)
   ```

3. **Deploy models to APK** (`ros2_android/app/build.gradle.kts`)
   ```kotlin
   android {
     sourceSets {
       getByName("main") {
         assets.srcDirs("../ros2_android_perception/models")
       }
     }
   }
   ```

4. **Extract models at runtime** (in JNI initialization)
   ```cpp
   // In AndroidApp constructor
   std::string models_path = ExtractAssets(cache_dir, "models/");
   detection_controller_ = std::make_unique<ObjectDetectionController>(models_path);
   ```

5. **Add UI controls** (`app/src/main/kotlin/.../ui/screens/DetectionScreen.kt`)
   - Toggle detection on/off
   - Display current detection rate
   - Show confirmed tracks count per class
   - (Optional) Display annotated image

6. **Update NativeBridge** (add JNI methods)
   ```kotlin
   external fun enableDetection()
   external fun disableDetection()
   external fun isDetectionEnabled(): Boolean
   external fun getDetectionStats(): DetectionStats
   ```

**Testing**:
- Build APK with perception included
- Check APK size increase (~25 MB for models)
- Launch app, enable detection
- Verify no crashes on model loading
- Check logcat for detection messages

---

## Detailed TODO List

### Pre-Implementation Checks
- [x] Verify NCNN models present (`models/*.ncnn.*`) - 24 MB total
- [x] Check OpenCV-mobile extracted (`deps/opencv-mobile-4.13.0-android/`)
- [ ] Run `make deps` to fetch NCNN + Eigen source
- [ ] Run `make ncnn` to build NCNN for Android (test build)
- [ ] Verify Eigen3 headers available

### Phase 1: NCNN Detector
- [ ] Create `include/perception/ncnn_detector.h` with class definition
- [ ] Implement `ncnn_detector.cc` constructor (model loading)
- [ ] Implement `LoadModel()` with error handling
- [ ] Implement `Detect()` inference pipeline
- [ ] Implement `ParseOutput()` for YOLOv9 tensor decoding
- [ ] Integrate `NMS::ApplyPerClass()` for post-processing
- [ ] Handle letterbox coordinate scaling
- [ ] Add unit test for model loading
- [ ] Add unit test for inference on test image
- [ ] Verify detection bboxes are in correct format

### Phase 2: ReID Extractor
- [ ] Create `include/perception/ncnn_reid.h` with class definition
- [ ] Implement `ncnn_reid.cc` constructor (model loading)
- [ ] Implement `Extract()` for feature extraction from bbox
- [ ] Implement `L2Normalize()` for feature normalization
- [ ] Add unit test for feature extraction
- [ ] Verify feature dimension is 128
- [ ] Verify L2 norm ≈ 1.0

### Phase 3: Kalman Filter
- [ ] Create `include/perception/kalman_filter.h` with class definition
- [ ] Implement `kalman_filter.cc` constructor (initialize matrices)
- [ ] Implement `Initiate()` for track initialization
- [ ] Implement `Predict()` for state propagation
- [ ] Implement `Update()` for measurement correction
- [ ] Implement `Project()` for measurement space projection
- [ ] Implement `GatingDistance()` for Mahalanobis distance
- [ ] Add chi-square thresholds (9.4877 for 4 DOF, 5.9915 for 2 DOF)
- [ ] Add unit test comparing against Python implementation
- [ ] Test predict-update cycle convergence

### Phase 4: Linear Assignment
- [ ] Create `include/perception/linear_assignment.h` with structs
- [ ] Research Hungarian algorithm libraries (munkres-cpp vs custom)
- [ ] Implement or integrate Hungarian solver
- [ ] Implement `MinCostMatching()` with distance gating
- [ ] Implement `MatchingCascade()` with age prioritization
- [ ] Implement `GateCostMatrix()` with Mahalanobis gating
- [ ] Add unit test for small cost matrix
- [ ] Test cascade with mixed track ages
- [ ] Test gating with outliers

### Phase 5: Deep SORT Tracker
- [ ] Create `include/perception/deep_sort_tracker.h` with config struct
- [ ] Implement `deep_sort_tracker.cc` constructor (init ReID, KF)
- [ ] Implement `Predict()` for all tracks
- [ ] Implement `MatchDetections()` with cascade + IoU fallback
- [ ] Implement `BuildCostMatrix()` with cosine distance
- [ ] Implement `CosineDistance()` metric
- [ ] Implement `InitiateTrack()` for new detections
- [ ] Implement `UpdateTrack()` with Kalman update + feature smoothing
- [ ] Implement `MarkMissed()` for unmatched tracks
- [ ] Implement `DeleteOldTracks()` with max_age threshold
- [ ] Implement feature gallery management (nn_budget)
- [ ] Add track lifecycle state machine (tentative/confirmed/deleted)
- [ ] Add unit test for single track lifecycle
- [ ] Test multi-object tracking with crossings
- [ ] Test occlusion handling with feature gallery

### Phase 6: ROS 2 Controller
- [ ] Create `include/perception/object_detection_controller.h`
- [ ] Implement `object_detection_controller.cc` constructor
- [ ] Implement model asset extraction helper
- [ ] Implement `OnCompressedImage()` callback
- [ ] Integrate JPEG decoding with OpenCV
- [ ] Call detector + tracker in callback
- [ ] Implement `PublishDetections()` for Point messages
- [ ] Add QoS configuration (BEST_EFFORT sub, RELIABLE pub)
- [ ] Add Enable/Disable lifecycle
- [ ] Add error handling for model load failures
- [ ] Test with rosbag playback on desktop
- [ ] Test subscription QoS compatibility

### Phase 7: ros2_android Integration
- [ ] Add perception to `dependencies.cmake` as ExternalProject
- [ ] Link perception library in `src/CMakeLists.txt`
- [ ] Stage models to APK assets in `app/build.gradle.kts`
- [ ] Implement asset extraction in JNI bridge
- [ ] Instantiate ObjectDetectionController in AndroidApp
- [ ] Add JNI methods for enable/disable
- [ ] Add Kotlin UI screen for detection control
- [ ] Update NativeBridge with detection methods
- [ ] Build APK and test on device
- [ ] Profile CPU/memory usage during inference
- [ ] Verify frame rate >5 Hz

### Testing & Validation
- [ ] End-to-end test: rosbag → detections → ROS topics
- [ ] Performance benchmark: measure inference time per frame
- [ ] Memory profiling: check for leaks over 5+ minutes
- [ ] Accuracy check: compare detections vs Python reference
- [ ] Stability test: sustained 10+ minute run without crash
- [ ] Multi-object test: verify stable track IDs across frames
- [ ] Occlusion test: verify re-identification after occlusion

### Documentation
- [ ] Document NCNN model conversion process (reference flake.nix)
- [ ] Document build instructions in ros2_android_perception/README.md
- [ ] Add API documentation in header comments
- [ ] Document integration steps in ros2_android/README.md
- [ ] Update implementation/IMPLEMENTATION.md with outcomes
- [ ] Add troubleshooting section for common build issues

## Success Criteria

### Functional
- [ ] YOLOv9 detects 3 classes (beetle, larva, eggs) on test images
- [ ] Deep SORT assigns stable track IDs across frames
- [ ] ROS 2 topics publish detection centers at >5 Hz
- [ ] No crashes during sustained operation (10+ minutes)
- [ ] Track IDs survive brief occlusions (feature gallery works)

### Performance
- [ ] Inference latency <200ms per frame on Pixel 7
- [ ] Memory usage <100 MB for perception (models + buffers)
- [ ] No memory leaks (verified with Android Studio Profiler)
- [ ] CPU usage <50% average during inference

### Integration
- [ ] Builds successfully with Android NDK 25.1
- [ ] APK size increase <30 MB (models + library)
- [ ] Compatible with existing ros2_android sensors/camera
- [ ] UI controls work (enable/disable, stats display)

## Known Limitations (Deferred)

- **No 3D fusion**: Publishing 2D centers only (Z=0), no depth/point cloud
- **No PointCloud2 output**: Only Point messages for centers
- **Model input mismatch**: 640x640 vs Python 1280x736 (lower accuracy expected)
- **No executor isolation**: Inference blocks SingleThreadedExecutor (acceptable <200ms)
- **No custom messages**: Using geometry_msgs::Point instead of TargetCoordinates.msg

## References

- **Python implementation**: `~/uni_projects/ROS2/Vermin_Collector_ROS2_3D_Object_Detection/`
- **NCNN models**: `ros2_android_perception/models/`
- **Deep SORT paper**: Wojke et al. 2017 (arXiv:1703.07402)
- **YOLOv9 paper**: Wang et al. 2024
- **Session recovery doc**: `implementation/object_detection_context.txt`
- **Project architecture**: `CLAUDE.md`

## Estimated Timeline

| Phase | Description | LOC | Time | Dependencies |
|-------|-------------|-----|------|--------------|
| 1 | NCNN Detector | 250 | 4-6h | NCNN, OpenCV |
| 2 | ReID Extractor | 100 | 2-3h | NCNN, OpenCV |
| 3 | Kalman Filter | 300 | 6-8h | Eigen3 |
| 4 | Linear Assignment | 200 | 4-6h | Eigen3, Hungarian lib |
| 5 | Deep SORT Tracker | 350 | 8-10h | Phases 1-4 |
| 6 | ROS 2 Controller | 300 | 6-8h | Phase 5, ROS 2 |
| 7 | Integration | 150 | 2-4h | Phase 6, ros2_android |
| Testing | E2E + profiling | - | 4-6h | All phases |
| **Total** | | **~1650** | **36-51h** | |

**Realistic estimate**: 5-7 full working days (8h/day) for complete implementation + testing.

## Next Steps

1. **Run pre-implementation checks** (fetch deps, build NCNN)
2. **Start with Phase 1** (NCNN Detector) - most critical path
3. **Iterate quickly** - build + test each phase before moving on
4. **Profile early** - check memory/CPU after Phase 1
5. **Document issues** - capture blockers in implementation/IMPLEMENTATION.md

---

**Last Updated**: 2026-03-29
**Status**: Ready to begin implementation
