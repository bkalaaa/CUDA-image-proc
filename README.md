# CUDA Image Processing

A high-performance GPU-accelerated image processing application implemented in C++ with CUDA. This project aims to achieve at least 70% throughput improvement over CPU-only implementations.

## Features

- **Gaussian Smoothing**: Separable convolution implementation for efficient memory usage
- **Edge Detection**: Sobel and Canny edge detection with shared memory optimization
- **Histogram Equalization**: GPU-accelerated histogram computation and equalization
- **Multi-format Support**: Both grayscale and RGB image processing
- **Performance Benchmarking**: Comprehensive CPU vs GPU performance comparison

## Technical Implementation

- CUDA C++ kernels for GPU computation
- OpenCV for image I/O operations
- CUDA streams for overlapping data transfers and computation
- Optimized memory coalescing and shared memory usage
- Automatic kernel launch parameter tuning

## Build Requirements

- CMake 3.18+
- CUDA Toolkit 11.0+
- OpenCV 4.0+
- C++17 compatible compiler
- GPU with compute capability 7.5+ (RTX 20xx series or newer recommended)

## Building

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Usage

```bash
./cuda-image-proc --mode gpu --ops gaussian,sobel --batch input_dir/
```

## Performance Goals

Target: ≥70% throughput improvement over CPU implementations on 1080p image batches.

## Project Structure

```
├── src/           # Source code
├── tests/         # Unit tests
├── cmake/         # CMake modules
└── docs/          # Documentation
```