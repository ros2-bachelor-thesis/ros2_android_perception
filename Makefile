# ROS 2 Android Perception Build System

SHELL := bash
.DEFAULT_GOAL := all

# Directories
BUILD_DIR := build
INSTALL_DIR := $(BUILD_DIR)/install
DEPS_DIR := deps

# Build type: RelWithDebInfo (default), Debug, or Release
BUILD_TYPE ?= RelWithDebInfo

# Convenience aliases
.PHONY: debug release
debug: BUILD_TYPE := Debug
debug: all
release: BUILD_TYPE := Release
release: all

# Parallelism
NPROC := $(shell nproc)

# Android SDK/NDK configuration (checked only when building)
ANDROID_ABI := arm64-v8a
ANDROID_PLATFORM := android-21

ifdef ANDROID_HOME
TOOLCHAIN_FILE := $(shell ls $(ANDROID_HOME)/ndk/*/build/cmake/android.toolchain.cmake 2>/dev/null | head -1)
endif

# CMake arguments for Android cross-compilation
CMAKE_ARGS := \
	-DCMAKE_TOOLCHAIN_FILE=$(TOOLCHAIN_FILE) \
	-DANDROID_ABI=$(ANDROID_ABI) \
	-DANDROID_PLATFORM=$(ANDROID_PLATFORM) \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR)

# Marker files for incremental builds
DEPS_STAMP := $(DEPS_DIR)/.deps-fetched
NCNN_STAMP := $(DEPS_DIR)/.ncnn-built
OPENCV_STAMP := $(DEPS_DIR)/.opencv-built
LIB_STAMP := $(BUILD_DIR)/.lib-built

.PHONY: all deps ncnn opencv lib install clean clean-deps test help

# ============================================================================
# Main targets
# ============================================================================

## Build perception library (default)
all: lib

## Fetch dependencies via vcstool
deps: $(DEPS_STAMP)

$(DEPS_STAMP): perception.repos
	@echo "==> Fetching dependencies via vcs..."
	vcs import --input perception.repos $(DEPS_DIR)/
	@touch $(DEPS_STAMP)
	@echo "==> Dependencies fetched"

## Build NCNN library for Android
ncnn: $(NCNN_STAMP)

$(NCNN_STAMP): $(DEPS_STAMP)
ifndef ANDROID_HOME
	$(error ANDROID_HOME not set. Run: export ANDROID_HOME=~/Android/Sdk)
endif
ifeq ($(TOOLCHAIN_FILE),)
	$(error Android NDK not found in $(ANDROID_HOME)/ndk/)
endif
	@echo "==> Building NCNN for Android arm64-v8a..."
	@mkdir -p $(DEPS_DIR)/ncnn/build-android
	cd $(DEPS_DIR)/ncnn/build-android && cmake .. \
		-DCMAKE_TOOLCHAIN_FILE=$(TOOLCHAIN_FILE) \
		-DANDROID_ABI=$(ANDROID_ABI) \
		-DANDROID_PLATFORM=$(ANDROID_PLATFORM) \
		-DCMAKE_BUILD_TYPE=Release \
		-DNCNN_VULKAN=ON \
		-DNCNN_BUILD_TOOLS=OFF \
		-DNCNN_BUILD_BENCHMARK=OFF \
		-DNCNN_BUILD_EXAMPLES=OFF \
		-DNCNN_SHARED_LIB=OFF \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR)
	cd $(DEPS_DIR)/ncnn/build-android && $(MAKE) -j$(NPROC) install
	@touch $(NCNN_STAMP)
	@echo "==> NCNN build complete"

## Setup OpenCV-mobile for Android
opencv: $(OPENCV_STAMP)

$(OPENCV_STAMP): $(DEPS_STAMP)
	@echo "==> Setting up OpenCV-mobile..."
	@# opencv-mobile provides prebuilt binaries
	@if [ -d "$(DEPS_DIR)/opencv-mobile/sdk" ]; then \
		echo "Using opencv-mobile prebuilt"; \
	else \
		echo "WARNING: opencv-mobile not found, will use system OpenCV"; \
	fi
	@touch $(OPENCV_STAMP)
	@echo "==> OpenCV setup complete"

## Build perception library
lib: $(LIB_STAMP)

LIB_SOURCES := $(shell find src include -type f \( -name '*.cc' -o -name '*.h' \) 2>/dev/null)

$(LIB_STAMP): $(NCNN_STAMP) $(OPENCV_STAMP) CMakeLists.txt $(LIB_SOURCES)
	@echo "==> Building perception library..."
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. $(CMAKE_ARGS) \
		-Dncnn_DIR=$(INSTALL_DIR)/lib/cmake/ncnn
	cd $(BUILD_DIR) && $(MAKE) -j$(NPROC)
	@touch $(LIB_STAMP)
	@echo "==> Library built: $(BUILD_DIR)/libros2_android_perception.a"

## Install library to build/install
install: $(LIB_STAMP)
	@echo "==> Installing library..."
	cd $(BUILD_DIR) && $(MAKE) install
	@echo "==> Install complete"
	@ls -lh $(INSTALL_DIR)/lib/libros2_android_perception.a
	@ls -lh $(INSTALL_DIR)/share/perception/models/ 2>/dev/null || true

# ============================================================================
# Testing
# ============================================================================

## Run unit tests
test: lib
	@echo "==> Running tests..."
	@if [ -d "test" ] && [ -f "$(BUILD_DIR)/Makefile" ]; then \
		cd $(BUILD_DIR) && ctest --output-on-failure; \
	else \
		echo "No tests configured"; \
	fi

# ============================================================================
# Cleaning
# ============================================================================

## Clean build artifacts
clean:
	@echo "==> Cleaning build..."
	rm -rf $(BUILD_DIR)

## Clean fetched dependencies (re-fetch required after this)
clean-deps:
	@echo "==> Cleaning fetched dependencies..."
	rm -f $(DEPS_STAMP) $(NCNN_STAMP) $(OPENCV_STAMP)
	rm -rf $(DEPS_DIR)/ncnn/build-android
	find $(DEPS_DIR) -mindepth 1 ! -name 'COLCON_IGNORE' -exec rm -rf {} + 2>/dev/null || true

# ============================================================================
# Help
# ============================================================================

## Show help
help:
	@echo "ROS 2 Android Perception Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all          Build perception library, RelWithDebInfo [default]"
	@echo "  debug        Build library with debug symbols (no optimization)"
	@echo "  release      Build library optimized (no debug symbols)"
	@echo "  deps         Fetch dependencies via vcstool"
	@echo "  ncnn         Build NCNN for Android"
	@echo "  opencv       Setup OpenCV-mobile"
	@echo "  lib          Build perception library"
	@echo "  install      Install library to build/install"
	@echo "  test         Run unit tests"
	@echo "  clean        Clean build artifacts"
	@echo "  clean-deps   Clean fetched dependencies"
	@echo "  help         Show this help"
