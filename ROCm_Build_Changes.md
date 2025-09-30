# ROCm Build Configuration Changes

## Overview

This document describes the changes made to support ROCm/HIP compilation for the adGraph project. All source files have already been converted from CUDA to HIP, and these changes update the CMakeLists.txt files to support ROCm compilation.

## Files Modified

### 1. Main CMakeLists.txt (`cpp/CMakeLists.txt`)

**Changes Made:**

- Changed project language from `CUDA` to `HIP`
- Replaced `find_package(CUDA)` with `enable_language(HIP)`
- Updated compiler flags:
  - `CMAKE_CUDA_FLAGS` → `CMAKE_HIP_FLAGS`
  - `CMAKE_CUDA_STANDARD` → `CMAKE_HIP_STANDARD`
- Updated GPU architecture flags:
  - Replaced CUDA `-gencode` flags with ROCm `--offload-arch` flags
  - Default architectures: gfx906, gfx908, gfx90a
- Updated include directories:
  - Removed `CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES`
  - Added `$ENV{DTK_ROOT}/include` and `$ENV{DTK_ROOT}/include/hip`
- Updated library directories:
  - Removed `CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES`
  - Added `$ENV{DTK_ROOT}/lib` and `$ENV{DTK_ROOT}/lib64`
- Updated linked libraries:
  - `cublas cusparse curand cusolver cudart` → `hipblas rocsparse rocrand rocsolver hip_hcc`
- Updated debug flags:
  - CUDA `-G` → HIP `-g -O0`

### 2. Tests CMakeLists.txt (`cpp/tests/CMakeLists.txt`)

**Changes Made:**

- Changed project language from `CUDA` to `HIP`
- Added `enable_language(HIP)`
- Updated include directories (same as main CMakeLists.txt)
- Updated library directories (same as main CMakeLists.txt)
- Updated test library linking:
  - `cublas cusparse curand cusolver cudart` → `hipblas rocsparse rocrand rocsolver hip_hcc`

### 3. CNMem CMakeLists.txt (`cpp/thirdparty/cnmem/CMakeLists.txt`)

**Changes Made:**

- Removed `find_package(CUDA QUIET REQUIRED)`
- Added `enable_language(HIP)`
- Updated include directories to use ROCm paths
- Updated library linking:
  - `CUDA_LIBRARIES` → `hip_hcc`
- Updated test executables:
  - Replaced `cuda_add_executable` with `add_executable`
  - Updated library linking for tests

## Build Configuration

The build system expects the following environment variables:

- `DTK_ROOT`: Path to ROCm installation (e.g., `/public/software/compiler/rocm/dtk-22.04.2`)
- `CC`: C compiler (should be set to ROCm clang)
- `CXX`: C++ compiler (should be set to hipcc)

## Build Script Compatibility

The existing `build_hip.sh` script is compatible with these changes and provides:

- Environment validation
- ROCm compiler configuration
- CMake configuration with proper HIP settings

## Usage

1. Set up the environment:

   ```bash
   source setup_hip_env.sh
   ```

2. Build the project:

   ```bash
   ./build_hip.sh
   ```

## Notes

- All source files maintain their original extensions (.cu, .cpp, .h, etc.)
- No file extensions were changed as per requirements
- The build system now uses HIP instead of CUDA for GPU compilation
- ROCm libraries replace CUDA libraries in the linking stage
