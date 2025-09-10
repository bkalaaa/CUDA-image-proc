#include "histogram_kernels.h"
#include <iostream>
#include <algorithm>
#include <cstring>

HistogramConfig::HistogramConfig(int block_sz) 
    : block_size(block_sz), shared_bins(HIST_SIZE), 
      shared_memory_size(HIST_SIZE * sizeof(int)),
      use_shared_memory(true), apply_rgb_separately(false) {}

void HistogramConfig::print_config() const {
    std::cout << "\nHistogram Configuration:" << std::endl;
    std::cout << "  Block size: " << block_size << std::endl;
    std::cout << "  Histogram bins: " << shared_bins << std::endl;
    std::cout << "  Shared memory size: " << shared_memory_size << " bytes" << std::endl;
    std::cout << "  Use shared memory: " << (use_shared_memory ? "yes" : "no") << std::endl;
    std::cout << "  RGB separate channels: " << (apply_rgb_separately ? "yes" : "no") << std::endl;
}

bool HistogramConfig::validate() const {
    return block_size > 0 && shared_bins == HIST_SIZE && 
           shared_memory_size >= HIST_SIZE * sizeof(int);
}

HistogramKernelManager::HistogramKernelManager() : initialized_(false) {}

HistogramKernelManager::~HistogramKernelManager() {
    cleanup();
}

bool HistogramKernelManager::initialize(int block_size) {
    config_ = HistogramConfig(block_size);
    
    if (!config_.validate()) {
        std::cerr << "Invalid histogram configuration" << std::endl;
        return false;
    }
    
    initialized_ = true;
    return true;
}

void HistogramKernelManager::cleanup() {
    initialized_ = false;
}

bool HistogramKernelManager::compute_histogram(const ImageBuffer& input, int* histogram,
                                              cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "Histogram kernel manager not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    CUDA_CHECK(cudaMemsetAsync(histogram, 0, HIST_SIZE * sizeof(int), stream));
    
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(metadata.width * metadata.height);
    
    histogram_kernel<<<grid_size, block_size, 0, stream>>>(
        input.get_device_ptr(), histogram, 
        metadata.width, metadata.height, metadata.channels);
    
    CUDA_CHECK_LAST();
    return true;
}

bool HistogramKernelManager::compute_histogram_shared(const ImageBuffer& input, int* histogram,
                                                     cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "Histogram kernel manager not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    CUDA_CHECK(cudaMemsetAsync(histogram, 0, HIST_SIZE * sizeof(int), stream));
    
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(metadata.width * metadata.height);
    
    histogram_shared_kernel<<<grid_size, block_size, config_.shared_memory_size, stream>>>(
        input.get_device_ptr(), histogram,
        metadata.width, metadata.height, metadata.channels);
    
    CUDA_CHECK_LAST();
    return true;
}

bool HistogramKernelManager::compute_cdf(const int* histogram, float* cdf, int total_pixels) const {
    dim3 block_size(256);
    dim3 grid_size(1);
    
    compute_cdf_kernel<<<grid_size, block_size>>>(histogram, cdf, total_pixels);
    
    CUDA_CHECK_LAST();
    return true;
}

bool HistogramKernelManager::apply_histogram_transform(const ImageBuffer& input, ImageBuffer& output,
                                                      const float* cdf, cudaStream_t stream) const {
    const ImageMetadata& metadata = input.get_metadata();
    
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(metadata.width * metadata.height);
    
    histogram_equalization_kernel<<<grid_size, block_size, 0, stream>>>(
        input.get_device_ptr(), output.get_device_ptr(), cdf,
        metadata.width, metadata.height, metadata.channels);
    
    CUDA_CHECK_LAST();
    return true;
}

bool HistogramKernelManager::apply_histogram_equalization(const ImageBuffer& input, ImageBuffer& output,
                                                         cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "Histogram kernel manager not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    if (!output.allocate_device(metadata)) {
        return false;
    }
    
    DeviceMemory<int> histogram_buffer;
    DeviceMemory<float> cdf_buffer;
    
    if (!histogram_buffer.allocate(HIST_SIZE) || !cdf_buffer.allocate(HIST_SIZE)) {
        return false;
    }
    
    if (!compute_histogram_shared(input, histogram_buffer.get_ptr(), stream)) {
        return false;
    }
    
    int total_pixels = metadata.width * metadata.height;
    if (!compute_cdf(histogram_buffer.get_ptr(), cdf_buffer.get_ptr(), total_pixels)) {
        return false;
    }
    
    return apply_histogram_transform(input, output, cdf_buffer.get_ptr(), stream);
}

bool HistogramKernelManager::apply_histogram_equalization_rgb(const ImageBuffer& input, ImageBuffer& output,
                                                             cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "Histogram kernel manager not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    if (metadata.channels != 3) {
        return apply_histogram_equalization(input, output, stream);
    }
    
    if (!output.allocate_device(metadata)) {
        return false;
    }
    
    DeviceMemory<float> cdf_r_buffer, cdf_g_buffer, cdf_b_buffer;
    
    if (!cdf_r_buffer.allocate(HIST_SIZE) || 
        !cdf_g_buffer.allocate(HIST_SIZE) || 
        !cdf_b_buffer.allocate(HIST_SIZE)) {
        return false;
    }
    
    std::cout << "    RGB histogram equalization not fully implemented, using grayscale" << std::endl;
    return apply_histogram_equalization(input, output, stream);
}

void HistogramKernelManager::benchmark_histogram_kernels(const ImageBuffer& input, int iterations) const {
    if (!initialized_) {
        std::cerr << "Cannot benchmark: Histogram kernel manager not initialized" << std::endl;
        return;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    DeviceMemory<int> histogram_buffer;
    if (!histogram_buffer.allocate(HIST_SIZE)) {
        std::cerr << "Failed to allocate histogram buffer for benchmarking" << std::endl;
        return;
    }
    
    CudaTimer timer;
    
    std::cout << "\nBenchmarking Histogram Kernels:" << std::endl;
    std::cout << "Image size: " << metadata.width << "x" << metadata.height 
              << "x" << metadata.channels << std::endl;
    
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        compute_histogram(input, histogram_buffer.get_ptr());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    float global_time = timer.elapsed_ms();
    
    timer.reset();
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        compute_histogram_shared(input, histogram_buffer.get_ptr());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    float shared_time = timer.elapsed_ms();
    
    HistogramUtils::print_histogram_performance(global_time, shared_time,
                                               metadata.width, metadata.height, iterations);
}

dim3 HistogramKernelManager::calculate_grid_size(int total_pixels) const {
    int grid_x = (total_pixels + config_.block_size - 1) / config_.block_size;
    return dim3(grid_x, 1, 1);
}

dim3 HistogramKernelManager::calculate_block_size() const {
    return dim3(config_.block_size, 1, 1);
}

__global__ void histogram_kernel(const unsigned char* input, int* histogram,
                                int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    for (int c = 0; c < channels; ++c) {
        int pixel_idx = idx * channels + c;
        unsigned char pixel_value = input[pixel_idx];
        atomicAdd(&histogram[pixel_value], 1);
    }
}

__global__ void histogram_shared_kernel(const unsigned char* input, int* histogram,
                                       int width, int height, int channels) {
    extern __shared__ int shared_hist[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    for (int i = tid; i < HIST_SIZE; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    
    __syncthreads();
    
    if (idx < total_pixels) {
        for (int c = 0; c < channels; ++c) {
            int pixel_idx = idx * channels + c;
            unsigned char pixel_value = input[pixel_idx];
            atomicAdd(&shared_hist[pixel_value], 1);
        }
    }
    
    __syncthreads();
    
    for (int i = tid; i < HIST_SIZE; i += blockDim.x) {
        if (shared_hist[i] > 0) {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

__global__ void compute_cdf_kernel(const int* histogram, float* cdf, int total_pixels) {
    int tid = threadIdx.x;
    
    __shared__ float shared_cdf[HIST_SIZE];
    
    if (tid < HIST_SIZE) {
        shared_cdf[tid] = static_cast<float>(histogram[tid]) / total_pixels;
    } else {
        shared_cdf[tid] = 0.0f;
    }
    
    __syncthreads();
    
    for (int stride = 1; stride < HIST_SIZE; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < HIST_SIZE) {
            shared_cdf[index] += shared_cdf[index - stride];
        }
        __syncthreads();
    }
    
    for (int stride = HIST_SIZE / 4; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < HIST_SIZE) {
            shared_cdf[index + stride] += shared_cdf[index];
        }
        __syncthreads();
    }
    
    if (tid < HIST_SIZE) {
        cdf[tid] = shared_cdf[tid];
    }
}

__global__ void histogram_equalization_kernel(const unsigned char* input,
                                             unsigned char* output,
                                             const float* cdf,
                                             int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    for (int c = 0; c < channels; ++c) {
        int pixel_idx = idx * channels + c;
        unsigned char pixel_value = input[pixel_idx];
        float normalized_value = cdf[pixel_value];
        output[pixel_idx] = static_cast<unsigned char>(normalized_value * 255.0f + 0.5f);
    }
}

__global__ void histogram_equalization_rgb_kernel(const unsigned char* input,
                                                  unsigned char* output,
                                                  const float* cdf_r,
                                                  const float* cdf_g,
                                                  const float* cdf_b,
                                                  int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    int pixel_idx = idx * 3;
    
    unsigned char r = input[pixel_idx];
    unsigned char g = input[pixel_idx + 1];
    unsigned char b = input[pixel_idx + 2];
    
    output[pixel_idx] = static_cast<unsigned char>(cdf_r[r] * 255.0f + 0.5f);
    output[pixel_idx + 1] = static_cast<unsigned char>(cdf_g[g] * 255.0f + 0.5f);
    output[pixel_idx + 2] = static_cast<unsigned char>(cdf_b[b] * 255.0f + 0.5f);
}

namespace HistogramUtils {
    dim3 calculate_grid_size(int total_elements, int block_size) {
        int grid_x = (total_elements + block_size - 1) / block_size;
        return dim3(grid_x, 1, 1);
    }
    
    dim3 calculate_block_size(int block_size) {
        return dim3(block_size, 1, 1);
    }
    
    __device__ void atomic_add_shared_histogram(int* shared_hist, int bin, int value) {
        atomicAdd(&shared_hist[bin], value);
    }
    
    void print_histogram_performance(float global_time, float shared_time,
                                   int width, int height, int iterations) {
        float global_avg = global_time / iterations;
        float shared_avg = shared_time / iterations;
        float speedup = global_time / shared_time;
        
        std::cout << "\nHistogram Performance Results:" << std::endl;
        std::cout << "  Global memory: " << global_avg << " ms/iteration" << std::endl;
        std::cout << "  Shared memory: " << shared_avg << " ms/iteration" << std::endl;
        std::cout << "  Speedup: " << speedup << "x" << std::endl;
        
        float pixels_per_sec = (width * height * iterations) / (shared_time / 1000.0f);
        std::cout << "  Throughput: " << pixels_per_sec / 1000000.0f << " Mpixels/sec" << std::endl;
    }
    
    bool validate_histogram(const int* histogram, int expected_total) {
        int total = 0;
        for (int i = 0; i < HIST_SIZE; ++i) {
            total += histogram[i];
        }
        return total == expected_total;
    }
    
    void print_histogram_stats(const int* histogram) {
        int total = 0;
        int min_val = 256, max_val = -1;
        
        for (int i = 0; i < HIST_SIZE; ++i) {
            total += histogram[i];
            if (histogram[i] > 0) {
                if (i < min_val) min_val = i;
                if (i > max_val) max_val = i;
            }
        }
        
        std::cout << "Histogram stats: total=" << total 
                  << ", range=[" << min_val << "," << max_val << "]" << std::endl;
    }
}