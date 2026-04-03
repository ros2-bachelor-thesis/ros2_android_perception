cmake_minimum_required(VERSION 3.13)
project(ros2_android_perception VERSION 0.1.0)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include ExternalProject for dependency building
include(ExternalProject)

# ============================================================================
# Auto-build dependencies if not present
# ============================================================================

set(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")
set(OPENCV_VERSION "4.10.0")
set(OPENCV_RELEASE_TAG "v29")  # opencv-mobile release tag for 4.10.0
set(OPENCV_DIR "${DEPS_DIR}/opencv-mobile-${OPENCV_VERSION}-android")
set(OPENCV_CHECK_FILE "${OPENCV_DIR}/sdk/native/staticlibs/arm64-v8a/libopencv_core.a")
set(NCNN_DIR "${DEPS_DIR}/ncnn/build-android")
set(NCNN_CHECK_FILE "${NCNN_DIR}/install/lib/libncnn.a")

# Download opencv-mobile prebuilt binary if not present
if(NOT EXISTS "${OPENCV_CHECK_FILE}")
  message(STATUS "opencv-mobile not found, downloading...")
  file(DOWNLOAD
    "https://github.com/nihui/opencv-mobile/releases/download/${OPENCV_RELEASE_TAG}/opencv-mobile-${OPENCV_VERSION}-android.zip"
    "${DEPS_DIR}/opencv-mobile-android.zip"
    SHOW_PROGRESS
    STATUS DOWNLOAD_STATUS
  )
  list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
  if(NOT STATUS_CODE EQUAL 0)
    message(FATAL_ERROR "Failed to download opencv-mobile")
  endif()

  message(STATUS "Extracting opencv-mobile...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${DEPS_DIR}/opencv-mobile-android.zip"
    WORKING_DIRECTORY "${DEPS_DIR}"
    RESULT_VARIABLE EXTRACT_RESULT
  )
  if(NOT EXTRACT_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to extract opencv-mobile")
  endif()

  file(REMOVE "${DEPS_DIR}/opencv-mobile-android.zip")
  message(STATUS "opencv-mobile download complete")
endif()

# Build NCNN if not present
if(NOT EXISTS "${NCNN_CHECK_FILE}")
  message(STATUS "NCNN not built, building...")

  # Initialize NCNN submodules
  execute_process(
    COMMAND git submodule update --init --recursive
    WORKING_DIRECTORY "${DEPS_DIR}/ncnn"
    RESULT_VARIABLE SUBMODULE_RESULT
  )
  if(NOT SUBMODULE_RESULT EQUAL 0)
    message(WARNING "Failed to initialize NCNN submodules, trying to continue...")
  endif()

  # Configure NCNN
  file(MAKE_DIRECTORY "${NCNN_DIR}")
  execute_process(
    COMMAND ${CMAKE_COMMAND}
      -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
      -DANDROID_ABI=${ANDROID_ABI}
      -DANDROID_PLATFORM=${ANDROID_PLATFORM}
      -DCMAKE_BUILD_TYPE=Release
      -DNCNN_VULKAN=ON
      -DNCNN_STRING=ON
      -DNCNN_BUILD_TOOLS=OFF
      -DNCNN_BUILD_BENCHMARK=OFF
      -DNCNN_BUILD_EXAMPLES=OFF
      -DNCNN_SHARED_LIB=OFF
      -DCMAKE_INSTALL_PREFIX=${NCNN_DIR}/install
      ..
    WORKING_DIRECTORY "${NCNN_DIR}"
    RESULT_VARIABLE CONFIGURE_RESULT
  )
  if(NOT CONFIGURE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to configure NCNN")
  endif()

  # Build NCNN
  include(ProcessorCount)
  ProcessorCount(N)
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build . --parallel ${N} --target install
    WORKING_DIRECTORY "${NCNN_DIR}"
    RESULT_VARIABLE BUILD_RESULT
  )
  if(NOT BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build NCNN")
  endif()

  message(STATUS "NCNN build complete")
endif()

# ============================================================================
# Find dependencies (now guaranteed to exist)
# ============================================================================

# OpenCV-mobile (prebuilt binary - manually create targets)
set(OPENCV_INCLUDE_DIR "${OPENCV_DIR}/sdk/native/jni/include")
set(OPENCV_LIB_DIR "${OPENCV_DIR}/sdk/native/staticlibs/${ANDROID_ABI}")

add_library(opencv_core STATIC IMPORTED)
set_target_properties(opencv_core PROPERTIES
  IMPORTED_LOCATION "${OPENCV_LIB_DIR}/libopencv_core.a"
  INTERFACE_INCLUDE_DIRECTORIES "${OPENCV_INCLUDE_DIR}"
)

add_library(opencv_imgproc STATIC IMPORTED)
set_target_properties(opencv_imgproc PROPERTIES
  IMPORTED_LOCATION "${OPENCV_LIB_DIR}/libopencv_imgproc.a"
  INTERFACE_INCLUDE_DIRECTORIES "${OPENCV_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES opencv_core
)

# Eigen3 (header-only library)
set(EIGEN3_INCLUDE_DIR "${DEPS_DIR}/eigen")
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
)

# NCNN (provides CMake config)
set(ncnn_DIR "${NCNN_DIR}/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

# Perception library (shared - bundles NCNN, OpenCV, glslang internally)
add_library(${PROJECT_NAME} SHARED
  src/image_preprocessor.cc
  src/ncnn_detector.cc
  src/ncnn_reid.cc
  src/nms.cc
  src/deep_sort_tracker.cc
  src/kalman_filter.cc
  src/linear_assignment.cc
  src/object_detection_controller.cc
  src/visualization/annotator.cc
)

# Define ROS2_ANDROID_DEBUG for Debug/RelWithDebInfo builds
# This controls LOGD() macro availability (matches ros2_android/src/core/log.h)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  target_compile_definitions(${PROJECT_NAME} PRIVATE ROS2_ANDROID_DEBUG=1)
  message(STATUS "ROS2_ANDROID_DEBUG enabled (BUILD_TYPE=${CMAKE_BUILD_TYPE})")
else()
  message(STATUS "ROS2_ANDROID_DEBUG disabled (BUILD_TYPE=${CMAKE_BUILD_TYPE})")
endif()

# Enable OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)
  target_link_options(${PROJECT_NAME} PRIVATE -fopenmp)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${NCNN_DIR}/install/include>
  $<BUILD_INTERFACE:${OPENCV_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ncnn
  opencv_imgproc
  opencv_core
  Eigen3::Eigen
  log  # Android logging
)

# Install targets
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(DIRECTORY include/perception/
  DESTINATION include/perception
  FILES_MATCHING PATTERN "*.h"
)

# Install models
install(FILES
  models/yolov9_s_pobed.ncnn.param
  models/yolov9_s_pobed.ncnn.bin
  models/osnet_ain_x1_0.ncnn.param
  models/osnet_ain_x1_0.ncnn.bin
  DESTINATION share/perception/models
)

# Export CMake config
install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION lib/cmake/${PROJECT_NAME}
)

# Generate config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

# Install config
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME}
)
