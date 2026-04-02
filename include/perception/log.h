#pragma once

#include <android/log.h>

// Shared logging macros for perception module
// LOGD is only enabled in Debug builds to reduce logging overhead in Release

// LOGD is only enabled in Debug builds
#if ROS2_ANDROID_DEBUG
#define LOGD(...) \
  ((void)__android_log_print(ANDROID_LOG_DEBUG, "Perception", __VA_ARGS__))
#else
#define LOGD(...) ((void)0)
#endif

// Always enabled
#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "Perception", __VA_ARGS__))
#define LOGW(...) \
  ((void)__android_log_print(ANDROID_LOG_WARN, "Perception", __VA_ARGS__))
#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, "Perception", __VA_ARGS__))
