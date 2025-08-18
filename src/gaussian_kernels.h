#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "image_io.h"

constexpr int MAX_GAUSSIAN_RADIUS = 15;
constexpr int MAX_GAUSSIAN_SIZE = 2 * MAX_GAUSSIAN_RADIUS + 1;

class GaussianKernelManager {
public:
    GaussianKernelManager();
    ~GaussianKernelManager();
    
    bool initialize(float sigma, int radius = -1);
    void cleanup();
    
    int get_radius() const { return radius_; }
    int get_size() const { return size_; }
    float get_sigma() const { return sigma_; }
    
    const float* get_host_weights() const { return host_weights_; }
    
    bool apply_gaussian_separable(const ImageBuffer& input, ImageBuffer& output, 
                                 cudaStream_t stream = 0) const;
    
    bool apply_gaussian_horizontal(const unsigned char* input, unsigned char* output,
                                  int width, int height, int channels,
                                  cudaStream_t stream = 0) const;
    
    bool apply_gaussian_vertical(const unsigned char* input, unsigned char* output,
                                int width, int height, int channels,
                                cudaStream_t stream = 0) const;
    
    void print_kernel_info() const;
    
private:
    float sigma_;
    int radius_;
    int size_;
    float* host_weights_;
    bool initialized_;
    
    bool generate_weights();
    bool upload_weights_to_constant_memory() const;
    
    static int calculate_radius_from_sigma(float sigma);
    static void normalize_weights(float* weights, int size);
};

extern __constant__ float d_gaussian_weights[MAX_GAUSSIAN_SIZE];

__global__ void gaussian_horizontal_kernel(const unsigned char* input, 
                                         unsigned char* output,
                                         int width, int height, int channels,
                                         int radius);

__global__ void gaussian_vertical_kernel(const unsigned char* input,
                                        unsigned char* output, 
                                        int width, int height, int channels,
                                        int radius);

__global__ void gaussian_horizontal_coalesced_kernel(const unsigned char* input,
                                                    unsigned char* output,
                                                    int width, int height, int channels,
                                                    int radius);

__global__ void gaussian_vertical_coalesced_kernel(const unsigned char* input,
                                                  unsigned char* output,
                                                  int width, int height, int channels,
                                                  int radius);

namespace GaussianUtils {
    dim3 calculate_grid_size(int width, int height, int block_size);
    dim3 calculate_block_size(int block_size);
    
    bool validate_gaussian_parameters(float sigma, int radius);
    
    void benchmark_gaussian_kernels(const ImageBuffer& input, 
                                   const GaussianKernelManager& kernel_manager,
                                   int iterations = 10);
}