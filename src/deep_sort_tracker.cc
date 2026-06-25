#include "perception/deep_sort_tracker.h"
#include "perception/linear_assignment.h"
#include "perception/log.h"
#include <algorithm>
#include <cmath>
#include <set>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace perception
{

  DeepSortTracker::DeepSortTracker(const std::string &reid_param,
                                   const std::string &reid_bin,
                                   const DeepSortConfig &config,
                                   bool use_vulkan)
      : next_id_{1, 1, 1}, next_tentative_id_(1), config_(config)
  {
    reid_ = std::make_unique<NcnnReID>(reid_param, reid_bin, use_vulkan);
  }

  DeepSortTracker::~DeepSortTracker() = default;

  bool DeepSortTracker::IsReady() const
  {
    return reid_ && reid_->IsLoaded();
  }

  std::vector<Track> DeepSortTracker::Update(
      const cv::Mat &image,
      const std::vector<Detection> &detections)
  {

    if (!IsReady() || image.empty())
    {
      return tracks_;
    }

    // Compute dt in 30 FPS frame equivalents from wall clock. At the first
    // call, fall back to dt=1 (Python parity). Clamp to [0.5, 30] so a long
    // pause (UI off, model load) doesn't blow up the Kalman covariance.
    float dt_frames = 1.0f;
    auto now = std::chrono::steady_clock::now();
    if (has_last_update_time_)
    {
      std::chrono::duration<float> elapsed = now - last_update_time_;
      dt_frames = elapsed.count() * 30.0f;
      if (dt_frames < 0.5f) dt_frames = 0.5f;
      if (dt_frames > 30.0f) dt_frames = 30.0f;
    }
    last_update_time_ = now;
    has_last_update_time_ = true;

    // Reset per-frame diagnostic counters.
    diag_mahal_rejected_ = 0;
    diag_cosine_rejected_ = 0;

    // Python behavior: If no detections, run predict-only cycle
    if (detections.empty())
    {
      Predict(dt_frames);
      for (size_t i = 0; i < tracks_.size(); i++)
      {
        MarkMissed(i);
      }
      DeleteOldTracks();
      return tracks_;
    }

    // Step 1: Predict all track states
    Predict(dt_frames);

    // Step 2: Extract ReID features from detections
    std::vector<Detection> detections_with_features = detections;
    for (auto &det : detections_with_features)
    {
      det.feature = reid_->Extract(image, det.bbox);
    }

    // Step 3: Match detections to tracks
    auto matches = MatchDetections(image, detections_with_features);

    // Track which tracks and detections are matched
    std::set<int> matched_tracks;
    std::set<int> matched_detections;

    // Step 4: Update matched tracks
    static int frame_count = 0;
    frame_count++;

    for (const auto &[track_idx, det_idx] : matches)
    {
      UpdateTrack(track_idx, detections_with_features[det_idx]);
      matched_tracks.insert(track_idx);
      matched_detections.insert(det_idx);
    }

    // Log matching statistics
    int num_unmatched_tracks = tracks_.size() - matched_tracks.size();
    int num_unmatched_dets = detections_with_features.size() - matched_detections.size();

    if (frame_count <= 10 || frame_count % 5 == 0)
    {
      LOGD("Frame #%d: %zu matches, %d unmatched tracks, %d unmatched dets, %zu total tracks, dt=%.2f, cos_rej=%d",
           frame_count, matches.size(), num_unmatched_tracks, num_unmatched_dets,
           tracks_.size(), dt_frames, diag_cosine_rejected_);
    }

    // Step 5: Mark unmatched tracks as missed
    for (size_t i = 0; i < tracks_.size(); i++)
    {
      if (!matched_tracks.count(i))
      {
        MarkMissed(i);
      }
    }

    // Step 6: Initiate new tracks from unmatched detections (with filtering)
    for (size_t i = 0; i < detections_with_features.size(); i++)
    {
      if (!matched_detections.count(i))
      {
        // Filter out tiny detections (likely noise or artifacts)
        const auto &det = detections_with_features[i];
        float width = det.bbox[2] - det.bbox[0];
        float height = det.bbox[3] - det.bbox[1];
        float area = width * height;

        // Minimum bbox area threshold (100 pixels = 10x10 bbox)
        // Prevents creating tracks for noise/artifacts
        if (area >= 100.0f)
        {
          InitiateTrack(detections_with_features[i]);
        }
      }
    }

    // Step 7: Delete old tracks
    DeleteOldTracks();

    return tracks_;
  }

  void DeepSortTracker::Predict(float dt)
  {
    for (auto &track : tracks_)
    {
      kf_.Predict(track.state, track.covariance, dt);
      track.UpdateBboxFromState();
      track.age += 1;              // Python parity - track.py line 123
      track.time_since_update += 1; // Python parity - track.py line 124
    }
  }

  std::vector<std::pair<int, int>> DeepSortTracker::MatchDetections(
      const cv::Mat &image,
      std::vector<Detection> &detections)
  {

    // Build indices
    std::vector<int> track_indices;
    std::vector<int> detection_indices;
    for (size_t i = 0; i < tracks_.size(); i++)
    {
      track_indices.push_back(i);
    }
    for (size_t i = 0; i < detections.size(); i++)
    {
      detection_indices.push_back(i);
    }

    // Separate confirmed and unconfirmed tracks
    std::vector<int> confirmed_tracks;
    std::vector<int> unconfirmed_tracks;
    for (size_t i = 0; i < tracks_.size(); i++)
    {
      if (tracks_[i].is_confirmed)
      {
        confirmed_tracks.push_back(i);
      }
      else
      {
        unconfirmed_tracks.push_back(i);
      }
    }

    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_detections = detection_indices;
    MatchResult cascade_result;

    // Stage 1: Cascade matching on confirmed tracks (appearance-based)
    if (!confirmed_tracks.empty() && !detections.empty())
    {
      // Build cost matrix using cosine distance
      Eigen::MatrixXf cost_matrix = BuildCostMatrix(
          track_indices, detections, detection_indices);

      // Apply Mahalanobis gating
      LinearAssignment::GateCostMatrix(
          cost_matrix, kf_, tracks_, detections,
          track_indices, detection_indices);

      // Run cascade matching
      cascade_result = LinearAssignment::MatchingCascade(
          cost_matrix, config_.max_cosine_distance, config_.max_age,
          tracks_, confirmed_tracks, unmatched_detections);

      matches = cascade_result.matches;
      unmatched_detections = cascade_result.unmatched_detections;
    }

    // Stage 2: IoU matching for unconfirmed tracks + cascade-unmatched confirmed with TSU==1
    // Python parity (tracker.py:118-123):
    //   iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if tracks[k].time_since_update == 1]
    //   unmatched_tracks_a = [k for k in unmatched_tracks_a if tracks[k].time_since_update != 1]
    if (!unmatched_detections.empty())
    {
      std::vector<int> iou_track_candidates = unconfirmed_tracks;
      for (int idx : cascade_result.unmatched_tracks)
      {
        if (tracks_[idx].time_since_update == 1)
        {
          iou_track_candidates.push_back(idx);
        }
      }

      if (!iou_track_candidates.empty())
      {
        // Build IoU cost matrix (locally indexed: rows=0..N-1, cols=0..M-1)
        Eigen::MatrixXf iou_cost = LinearAssignment::IoUCostMatrix(
            tracks_, detections, iou_track_candidates, unmatched_detections);

        // Match using IoU with local indexing
        auto iou_result = LinearAssignment::MinCostMatchingLocal(
            iou_cost, config_.max_iou_distance,
            iou_track_candidates, unmatched_detections);

        // Merge matches
        matches.insert(matches.end(),
                       iou_result.matches.begin(),
                       iou_result.matches.end());
      }
    }

    return matches;
  }

  void DeepSortTracker::UpdateTrack(int track_idx, const Detection &detection)
  {
    Track &track = tracks_[track_idx];

    // Convert detection bbox to measurement [x, y, a, h]
    float cx = (detection.bbox[0] + detection.bbox[2]) / 2.0f;
    float cy = (detection.bbox[1] + detection.bbox[3]) / 2.0f;
    float w = detection.bbox[2] - detection.bbox[0];
    float h = detection.bbox[3] - detection.bbox[1];
    float a = (h > 0.0f) ? (w / h) : 1.0f;

    float measurement[4] = {cx, cy, a, h};

    // Kalman update
    kf_.Update(track.state, track.covariance, measurement);

    // Update bbox from state
    track.UpdateBboxFromState();

    // Python parity (track.py:140): self.features.append(detection.feature)
    // Store raw features in gallery for nearest-neighbor matching (no EMA)
    if (!detection.feature.empty())
    {
      track.feature = detection.feature; // Keep latest for reference
      gallery_[track.track_id].push_back(detection.feature);
      // Trim gallery if budget is set (nn_budget <= 0 means unlimited, matching Python nn_budget=None)
      if (config_.nn_budget > 0 &&
          gallery_[track.track_id].size() > static_cast<size_t>(config_.nn_budget))
      {
        gallery_[track.track_id].erase(gallery_[track.track_id].begin());
      }
    }

    // Update track state
    track.confidence = detection.confidence;
    track.time_since_update = 0;
    track.hits += 1;

    // Confirm track if criteria met - assign real ID on confirmation
    if (!track.is_confirmed && track.ShouldConfirm(config_.n_init))
    {
      int old_id = track.track_id;
      track.track_id = next_id_[track.class_id]++;
      track.is_confirmed = true;
      // Migrate gallery from tentative ID to confirmed ID
      if (gallery_.count(old_id))
      {
        gallery_[track.track_id] = std::move(gallery_[old_id]);
        gallery_.erase(old_id);
      }
      LOGD("Track confirmed: tentative_id=%d -> confirmed_id=%d (hits=%d)",
           old_id, track.track_id, track.hits);
    }
  }

  void DeepSortTracker::MarkMissed(int track_idx)
  {
    Track &track = tracks_[track_idx];
    // Departs from Python Deep SORT, which deletes tentative tracks on the
    // first miss. At low FPS (e.g. 2-3 Hz) the n_init=3 confirmation window
    // is impossible to reach if one Mahalanobis-gated miss kills the track,
    // so we allow a tentative track to survive up to n_init misses before
    // being deleted. Confirmed tracks still respect max_age unchanged.
    if (!track.is_confirmed)
    {
      if (track.time_since_update > config_.n_init)
      {
        track.is_deleted = true;
      }
    }
    else if (track.time_since_update > config_.max_age)
    {
      // Confirmed tracks deleted after exceeding max_age
      track.is_deleted = true;
    }
  }

  void DeepSortTracker::InitiateTrack(const Detection &detection)
  {
    // Use negative temporary ID for tentative tracks. Real ID assigned on confirmation.
    // This prevents ID counter inflation from short-lived false detections.
    Track track(-(next_tentative_id_++), detection.bbox, detection.class_id);
    track.confidence = detection.confidence;

    // Initialize Kalman filter
    float cx = (detection.bbox[0] + detection.bbox[2]) / 2.0f;
    float cy = (detection.bbox[1] + detection.bbox[3]) / 2.0f;
    float w = detection.bbox[2] - detection.bbox[0];
    float h = detection.bbox[3] - detection.bbox[1];
    float a = (h > 0.0f) ? (w / h) : 1.0f;

    float measurement[4] = {cx, cy, a, h};
    kf_.Initiate(measurement, track.state, track.covariance);

    // Initialize feature
    if (!detection.feature.empty())
    {
      track.feature = detection.feature;
      gallery_[track.track_id].push_back(detection.feature);
    }

    tracks_.push_back(track);
  }

  void DeepSortTracker::Reset()
  {
    tracks_.clear();
    gallery_.clear();
    next_id_[0] = next_id_[1] = next_id_[2] = 1;
    next_tentative_id_ = 1;
    has_last_update_time_ = false;
    diag_mahal_rejected_ = 0;
    diag_cosine_rejected_ = 0;
  }

  void DeepSortTracker::DeleteOldTracks()
  {
    // Python parity (tracker.py:79): self.tracks = [t for t in self.tracks if not t.is_deleted()]
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [this](const Track &t)
                       {
                         if (t.is_deleted)
                         {
                           gallery_.erase(t.track_id);
                           return true;
                         }
                         return false;
                       }),
        tracks_.end());
  }

  Eigen::MatrixXf DeepSortTracker::BuildCostMatrix(
      const std::vector<int> &track_indices,
      const std::vector<Detection> &detections,
      const std::vector<int> &detection_indices)
  {

    int num_tracks = track_indices.size();
    int num_detections = detection_indices.size();

    Eigen::MatrixXf cost_matrix(num_tracks, num_detections);

    for (int i = 0; i < num_tracks; i++)
    {
      const Track &track = tracks_[track_indices[i]];

      for (int j = 0; j < num_detections; j++)
      {
        const Detection &det = detections[detection_indices[j]];

        if (det.feature.empty())
        {
          cost_matrix(i, j) = 1e5f; // Invalid if no feature
          continue;
        }

        // Class-aware gate: forbid matching detections of class A to tracks
        // of class B. With 3 visually-distinct classes (beetle / larva / eggs)
        // a low cosine distance across classes is almost always a false match.
        if (track.class_id != det.class_id)
        {
          cost_matrix(i, j) = 1e5f;
          continue;
        }

        // Compute minimum cosine distance to gallery
        float min_dist = MinCosineDistanceToGallery(det.feature, track.track_id);
        cost_matrix(i, j) = min_dist;
        if (min_dist > config_.max_cosine_distance)
        {
          diag_cosine_rejected_++;
        }
      }
    }

    return cost_matrix;
  }

  float DeepSortTracker::CosineDistance(const std::vector<float> &f1,
                                        const std::vector<float> &f2)
  {
    if (f1.empty() || f2.empty() || f1.size() != f2.size())
    {
      return 1e5f;
    }

    // Compute dot product (features should be L2-normalized)
    float dot = 0.0f;
#ifdef __ARM_NEON
    size_t i = 0;
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (; i + 4 <= f1.size(); i += 4)
    {
      float32x4_t a = vld1q_f32(&f1[i]);
      float32x4_t b = vld1q_f32(&f2[i]);
      sum = vfmaq_f32(sum, a, b);
    }
    dot = vaddvq_f32(sum);
    for (; i < f1.size(); i++)
    {
      dot += f1[i] * f2[i];
    }
#else
    for (size_t i = 0; i < f1.size(); i++)
    {
      dot += f1[i] * f2[i];
    }
#endif

    // Cosine distance = 1 - similarity
    return 1.0f - dot;
  }

  float DeepSortTracker::MinCosineDistanceToGallery(
      const std::vector<float> &feature,
      int track_id)
  {

    if (!gallery_.count(track_id) || gallery_[track_id].empty())
    {
      return 1e5f;
    }

    // Find minimum distance over all stored features
    float min_dist = std::numeric_limits<float>::max();
    for (const auto &gallery_feature : gallery_[track_id])
    {
      float dist = CosineDistance(feature, gallery_feature);
      min_dist = std::min(min_dist, dist);
    }

    // Log first few distance calculations
    static int dist_calc_count = 0;
    dist_calc_count++;
    if (dist_calc_count <= 20)
    {
      LOGD("Cosine distance: track_id=%d, min_dist=%.3f (threshold=%.2f), gallery_size=%zu",
           track_id, min_dist, config_.max_cosine_distance, gallery_[track_id].size());
    }

    return min_dist;
  }

  std::vector<Track> DeepSortTracker::GetConfirmedTracks() const
  {
    std::vector<Track> confirmed;
    for (const auto &track : tracks_)
    {
      // Return confirmed tracks missed for at most 3 frames (~1s at 2.7 FPS).
      // Python uses <=1 (30 FPS, so ~33ms gap). At 2.7 FPS one missed YOLO
      // detection = ~370ms gap; <=3 prevents blinking during brief occlusions
      // while still pruning truly lost tracks.
      if (track.is_confirmed && track.time_since_update <= 3)
      {
        confirmed.push_back(track);
      }
    }
    return confirmed;
  }

} // namespace perception
