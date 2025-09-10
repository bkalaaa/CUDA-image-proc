#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "image_io.h"

constexpr int HIST_SIZE = 256;
constexpr int HIST_BLOCK_SIZE = 256;
constexpr int HIST_THREADS_PER_BLOCK = 256;

struct HistogramConfig {
    int block_size;
    int shared_bins;
    size_t shared_memory_size;
    bool use_shared_memory;
    bool apply_rgb_separately;
    
    HistogramConfig(int block_sz = HIST_BLOCK_SIZE);
    void print_config() const;
    bool validate() const;
};

class HistogramKernelManager {
public:
    HistogramKernelManager();
    ~HistogramKernelManager();
    
    bool initialize(int block_size = HIST_BLOCK_SIZE);
    void cleanup();
    
    bool apply_histogram_equalization(const ImageBuffer& input, ImageBuffer& output,
                                     cudaStream_t stream = 0) const;
    
    bool apply_histogram_equalization_rgb(const ImageBuffer& input, ImageBuffer& output,
                                         cudaStream_t stream = 0) const;
    
    bool compute_histogram(const ImageBuffer& input, int* histogram,
                          cudaStream_t stream = 0) const;
    
    bool compute_histogram_shared(const ImageBuffer& input, int* histogram,
                                 cudaStream_t stream = 0) const;
    
    void benchmark_histogram_kernels(const ImageBuffer& input, int iterations = 10) const;
    
    const HistogramConfig& get_config() const { return config_; }
    
private:
    HistogramConfig config_;
    bool initialized_;
    
    bool compute_cdf(const int* histogram, float* cdf, int total_pixels) const;
    bool apply_histogram_transform(const ImageBuffer& input, ImageBuffer& output,
                                  const float* cdf, cudaStream_t stream) const;
    
    dim3 calculate_grid_size(int total_pixels) const;
    dim3 calculate_block_size() const;
};

__global__ void histogram_kernel(const unsigned char* input, int* histogram,
                                int width, int height, int channels);

__global__ void histogram_shared_kernel(const unsigned char* input, int* histogram,
                                       int width, int height, int channels);

__global__ void compute_cdf_kernel(const int* histogram, float* cdf, int total_pixels);

__global__ void histogram_equalization_kernel(const unsigned char* input,
                                             unsigned char* output,
                                             const float* cdf,
                                             int width, int height, int channels);

__global__ void histogram_equalization_rgb_kernel(const unsigned char* input,
                                                  unsigned char* output,
                                                  const float* cdf_r,
                                                  const float* cdf_g,
                                                  const float* cdf_b,
                                                  int width, int height);

namespace HistogramUtils {
    dim3 calculate_grid_size(int total_elements, int block_size);
    dim3 calculate_block_size(int block_size);
    
    __device__ void atomic_add_shared_histogram(int* shared_hist, int bin, int value);
    
    void print_histogram_performance(float global_time, float shared_time,
                                   int width, int height, int iterations);
    
    bool validate_histogram(const int* histogram, int expected_total);
    
    void print_histogram_stats(const int* histogram);
}