#pragma once

#include <opencv2/opencv.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "perception/detection.h"
#include "perception/track.h"
#include "perception/kalman_filter.h"
#include "perception/ncnn_reid.h"

namespace perception
{

  /**
   * Deep SORT tracker configuration
   */
  struct DeepSortConfig
  {
    float max_cosine_distance = 0.4f; ///< Max cosine distance for appearance matching

    /// Max features stored per track (gallery size)
    /// Python reference uses unlimited (nn_budget=None), C++ uses 100 for Android memory constraints.
    /// Trade-off: Lower value reduces memory but may degrade re-identification after long occlusions.
    int nn_budget = 100;

    int max_age = 30;              ///< Max frames before deleting lost track
    int n_init = 3;                ///< Min consecutive hits to confirm track
    float max_iou_distance = 0.7f; ///< Max IoU distance for fallback matching
  };

  /**
   * Deep SORT multi-object tracker
   *
   * Combines YOLOv9 detection with appearance-based tracking using
   * Kalman filtering and ReID features. Implements cascade matching
   * strategy from Deep SORT paper (Wojke et al. 2017).
   *
   * Pipeline:
   * 1. Predict all track states (Kalman predict)
   * 2. Extract ReID features from detections
   * 3. Cascade matching (appearance + Mahalanobis gating)
   * 4. IoU matching for unmatched tracks/detections
   * 5. Update matched tracks (Kalman update + feature smoothing)
   * 6. Mark missed tracks, initiate new tracks
   * 7. Delete old tracks (age > max_age)
   */
  class DeepSortTracker
  {
  public:
    /**
     * Constructor
     *
     * @param reid_param Path to mars-small128.ncnn.param
     * @param reid_bin Path to mars-small128.ncnn.bin
     * @param config Tracker configuration parameters
     */
    DeepSortTracker(const std::string &reid_param,
                    const std::string &reid_bin,
                    const DeepSortConfig &config = DeepSortConfig());

    ~DeepSortTracker();

    /**
     * Update tracker with new detections
     *
     * Main tracking pipeline: predict, match, update, manage lifecycle.
     *
     * @param image Source image (for ReID feature extraction)
     * @param detections Detected objects from YOLOv9
     * @return Current list of tracks (includes tentative and confirmed)
     */
    std::vector<Track> Update(const cv::Mat &image,
                              const std::vector<Detection> &detections);

    /**
     * Get only confirmed tracks (hits >= n_init)
     *
     * @return Confirmed tracks ready for downstream use
     */
    std::vector<Track> GetConfirmedTracks() const;

    /**
     * Get total number of active tracks
     */
    size_t GetTrackCount() const { return tracks_.size(); }

    /**
     * Check if ReID model loaded successfully
     */
    bool IsReady() const;

  private:
    /**
     * Predict all track states (Kalman filter prediction)
     */
    void Predict();

    /**
     * Match detections to tracks using cascade + IoU strategy
     *
     * @param image Source image for ReID
     * @param detections Input detections
     * @return Indices of (track, detection) matches
     */
    std::vector<std::pair<int, int>> MatchDetections(
        const cv::Mat &image,
        std::vector<Detection> &detections);

    /**
     * Update matched track with detection
     *
     * @param track_idx Index into tracks_ vector
     * @param detection Matched detection
     */
    void UpdateTrack(int track_idx, const Detection &detection);

    /**
     * Mark track as missed (no detection match)
     *
     * @param track_idx Index into tracks_ vector
     */
    void MarkMissed(int track_idx);

    /**
     * Initiate new track from unmatched detection
     *
     * @param detection Detection to create track from
     */
    void InitiateTrack(const Detection &detection);

    /**
     * Delete tracks that exceed max_age
     */
    void DeleteOldTracks();

    /**
     * Build cost matrix using cosine distance on ReID features
     *
     * @param track_indices Indices of tracks to include
     * @param detections Detections with features
     * @param detection_indices Indices of detections to include
     * @return Cost matrix (tracks x detections)
     */
    Eigen::MatrixXf BuildCostMatrix(
        const std::vector<int> &track_indices,
        const std::vector<Detection> &detections,
        const std::vector<int> &detection_indices);

    /**
     * Compute cosine distance between two L2-normalized features
     *
     * Distance = 1 - dot(f1, f2) / (||f1|| * ||f2||)
     * For normalized features: distance = 1 - dot(f1, f2)
     *
     * @return Cosine distance [0, 2] (0 = identical, 2 = opposite)
     */
    float CosineDistance(const std::vector<float> &f1,
                         const std::vector<float> &f2);

    /**
     * Compute minimum cosine distance to feature gallery
     *
     * @param feature Query feature
     * @param track_id Track whose gallery to search
     * @return Minimum distance to any feature in gallery
     */
    float MinCosineDistanceToGallery(const std::vector<float> &feature,
                                     int track_id);

    std::vector<Track> tracks_;                              ///< Active tracks
    std::map<int, std::vector<std::vector<float>>> gallery_; ///< Feature gallery per track
    int next_id_;                                            ///< Next track ID to assign

    KalmanFilter kf_;                ///< Kalman filter instance
    std::unique_ptr<NcnnReID> reid_; ///< ReID feature extractor
    DeepSortConfig config_;          ///< Configuration parameters
  };

} // namespace perception
