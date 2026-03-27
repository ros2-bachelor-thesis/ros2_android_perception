cmake_minimum_required(VERSION 3.13)
project(ros2_android_perception VERSION 0.1.0)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/opencv-mobile-4.13.0-android/sdk/native/jni")
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# Eigen3 (header-only library)
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen")
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
)

find_package(ncnn REQUIRED)

# Perception library (static for Android)
add_library(${PROJECT_NAME} STATIC
  src/image_preprocessor.cc
  src/ncnn_detector.cc
  src/ncnn_reid.cc
  src/nms.cc
  src/deep_sort_tracker.cc
  src/kalman_filter.cc
  src/linear_assignment.cc
  src/object_detection_controller.cc
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/ncnn/build-android/build/install/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ncnn
  opencv_core
  opencv_imgproc
  Eigen3::Eigen
)

# Install targets
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include
)

# Install models
install(FILES
  models/yolov9_s_pobed.ncnn.param
  models/yolov9_s_pobed.ncnn.bin
  models/mars-small128.ncnn.param
  models/mars-small128.ncnn.bin
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
