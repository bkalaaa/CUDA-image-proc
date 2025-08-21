#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "image_io.h"

constexpr int SOBEL_RADIUS = 1;
constexpr int SOBEL_SIZE = 3;
constexpr int SOBEL_TILE_SIZE = 16;

struct SobelConfig {
    int tile_width;
    int tile_height;
    int shared_width;
    int shared_height;
    int shared_size_bytes;
    bool compute_magnitude;
    bool apply_nms;
    
    SobelConfig(int tile_w = SOBEL_TILE_SIZE, int tile_h = SOBEL_TILE_SIZE);
    void print_config() const;
    bool validate() const;
};

class SobelKernelManager {
public:
    SobelKernelManager();
    ~SobelKernelManager();
    
    bool initialize(int tile_size = SOBEL_TILE_SIZE);
    void cleanup();
    
    bool apply_sobel_edge_detection(const ImageBuffer& input, ImageBuffer& output, 
                                   cudaStream_t stream = 0) const;
    
    bool apply_sobel_shared(const ImageBuffer& input, ImageBuffer& output,
                           cudaStream_t stream = 0) const;
    
    bool apply_sobel_fused_vectorized(const ImageBuffer& input, ImageBuffer& output,
                                     cudaStream_t stream = 0) const;
    
    bool apply_sobel_shared_fused(const ImageBuffer& input, ImageBuffer& output,
                                 cudaStream_t stream = 0) const;
    
    bool apply_sobel_gradients(const ImageBuffer& input, 
                              ImageBuffer& gx_output, ImageBuffer& gy_output,
                              cudaStream_t stream = 0) const;
    
    bool apply_sobel_magnitude(const ImageBuffer& gx, const ImageBuffer& gy,
                              ImageBuffer& magnitude, cudaStream_t stream = 0) const;
    
    bool apply_canny_nms(const ImageBuffer& magnitude, const ImageBuffer& gx, const ImageBuffer& gy,
                        ImageBuffer& output, float low_threshold = 50.0f, float high_threshold = 150.0f,
                        cudaStream_t stream = 0) const;
    
    void benchmark_sobel_kernels(const ImageBuffer& input, int iterations = 10) const;
    
    void benchmark_sobel_optimizations(const ImageBuffer& input, int iterations = 10) const;
    
    const SobelConfig& get_config() const { return config_; }
    
private:
    SobelConfig config_;
    bool initialized_;
    
    dim3 calculate_grid_size(int width, int height) const;
    dim3 calculate_block_size() const;
};

extern __constant__ float d_sobel_gx[SOBEL_SIZE * SOBEL_SIZE];
extern __constant__ float d_sobel_gy[SOBEL_SIZE * SOBEL_SIZE];

__global__ void sobel_edge_detection_kernel(const unsigned char* input,
                                           unsigned char* output,
                                           int width, int height, int channels);

__global__ void sobel_shared_kernel(const unsigned char* input,
                                   unsigned char* output,
                                   int width, int height, int channels,
                                   int tile_width, int shared_width);

__global__ void sobel_gradients_kernel(const unsigned char* input,
                                      float* gx_output, float* gy_output,
                                      int width, int height, int channels);

__global__ void sobel_gradients_shared_kernel(const unsigned char* input,
                                             float* gx_output, float* gy_output,
                                             int width, int height, int channels,
                                             int tile_width, int shared_width);

__global__ void sobel_magnitude_kernel(const float* gx, const float* gy,
                                      unsigned char* magnitude,
                                      int width, int height, int channels);

__global__ void sobel_magnitude_fused_kernel(const unsigned char* input,
                                            unsigned char* output,
                                            int width, int height, int channels);

__global__ void sobel_magnitude_fused_vectorized_kernel(const unsigned char* input,
                                                       unsigned char* output,
                                                       int width, int height, int channels);

__global__ void sobel_shared_fused_kernel(const unsigned char* input,
                                         unsigned char* output,
                                         int width, int height, int channels,
                                         int tile_width, int shared_width);

__global__ void canny_non_max_suppression_kernel(const unsigned char* magnitude,
                                                const float* gx, const float* gy,
                                                unsigned char* output,
                                                int width, int height,
                                                float low_threshold, float high_threshold);

namespace SobelUtils {
    dim3 calculate_grid_size(int width, int height, int block_size);
    dim3 calculate_block_size(int block_size);
    
    void upload_sobel_coefficients();
    
    __device__ float compute_gradient_magnitude(float gx, float gy);
    __device__ float compute_gradient_direction(float gx, float gy);
    
    __device__ void load_sobel_tile_with_halo(const unsigned char* input,
                                             unsigned char* shared_mem,
                                             int width, int height, int channels,
                                             int tile_x, int tile_y,
                                             int shared_width, int shared_height);
    
    __device__ float apply_sobel_filter_x(const unsigned char* shared_mem,
                                         int shared_x, int shared_y, int shared_width,
                                         int channel);
    
    __device__ float apply_sobel_filter_y(const unsigned char* shared_mem,
                                         int shared_x, int shared_y, int shared_width,
                                         int channel);
    
    void print_sobel_performance(float global_time_ms, float shared_time_ms, 
                                int width, int height, int iterations);
    
    void print_optimization_performance(float basic_time, float shared_time, 
                                       float shared_fused_time, float vectorized_time,
                                       const ImageMetadata& metadata, int iterations);
}