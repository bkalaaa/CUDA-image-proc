#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "image_io.h"
#include "gaussian_kernels.h"

constexpr int TILE_SIZE = 16;
constexpr int SHARED_MEM_PADDING = 1;

struct TileConfig {
    int tile_width;
    int tile_height;
    int halo_radius;
    int shared_width;
    int shared_height;
    int shared_size_bytes;
    
    TileConfig(int tile_w = TILE_SIZE, int tile_h = TILE_SIZE, int radius = 7);
    void print_config() const;
    bool validate() const;
};

class SharedMemoryGaussian {
public:
    SharedMemoryGaussian();
    ~SharedMemoryGaussian();
    
    bool initialize(float sigma, int radius = -1, int tile_size = TILE_SIZE);
    void cleanup();
    
    bool apply_gaussian_shared(const ImageBuffer& input, ImageBuffer& output, 
                              cudaStream_t stream = 0) const;
    
    bool apply_gaussian_horizontal_shared(const unsigned char* input, unsigned char* output,
                                         int width, int height, int channels,
                                         cudaStream_t stream = 0) const;
    
    bool apply_gaussian_vertical_shared(const unsigned char* input, unsigned char* output,
                                       int width, int height, int channels,
                                       cudaStream_t stream = 0) const;
    
    void benchmark_shared_vs_global(const ImageBuffer& input, int iterations = 10) const;
    
    const TileConfig& get_tile_config() const { return tile_config_; }
    int get_radius() const { return radius_; }
    float get_sigma() const { return sigma_; }
    
private:
    float sigma_;
    int radius_;
    TileConfig tile_config_;
    GaussianKernelManager* global_kernel_;
    bool initialized_;
    
    dim3 calculate_grid_size(int width, int height) const;
    dim3 calculate_block_size() const;
};

__global__ void gaussian_horizontal_shared_kernel(const unsigned char* input,
                                                 unsigned char* output,
                                                 int width, int height, int channels,
                                                 int radius, int tile_width, int shared_width);

__global__ void gaussian_vertical_shared_kernel(const unsigned char* input,
                                               unsigned char* output,
                                               int width, int height, int channels,
                                               int radius, int tile_height, int shared_height);

__global__ void gaussian_horizontal_shared_padded_kernel(const unsigned char* input,
                                                        unsigned char* output,
                                                        int width, int height, int channels,
                                                        int radius, int tile_width, 
                                                        int shared_width);

__global__ void gaussian_vertical_shared_padded_kernel(const unsigned char* input,
                                                      unsigned char* output,
                                                      int width, int height, int channels,
                                                      int radius, int tile_height,
                                                      int shared_height);

namespace SharedMemoryUtils {
    size_t calculate_shared_memory_size(int tile_size, int radius, int channels, bool with_padding = true);
    
    int calculate_padded_width(int width, int bank_size = 32);
    
    bool validate_shared_memory_usage(size_t required_bytes);
    
    void print_shared_memory_info(const TileConfig& config, int channels);
    
    __device__ int get_padded_index(int row, int col, int padded_width);
    
    __device__ void load_tile_with_halo(const unsigned char* input, 
                                       unsigned char* shared_mem,
                                       int width, int height, int channels,
                                       int tile_x, int tile_y, int radius,
                                       int shared_width, int shared_height);
    
    __device__ void load_tile_with_halo_padded(const unsigned char* input,
                                              unsigned char* shared_mem,
                                              int width, int height, int channels,
                                              int tile_x, int tile_y, int radius,
                                              int shared_width, int padded_width);
}