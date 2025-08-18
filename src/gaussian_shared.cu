#include "gaussian_shared.h"
#include <iostream>
#include <cmath>
#include <algorithm>

TileConfig::TileConfig(int tile_w, int tile_h, int radius) 
    : tile_width(tile_w), tile_height(tile_h), halo_radius(radius) {
    shared_width = tile_width + 2 * halo_radius;
    shared_height = tile_height + 2 * halo_radius;
    shared_size_bytes = shared_width * shared_height * sizeof(unsigned char);
}

void TileConfig::print_config() const {
    std::cout << "Tile Configuration:" << std::endl;
    std::cout << "  Tile size: " << tile_width << "x" << tile_height << std::endl;
    std::cout << "  Halo radius: " << halo_radius << std::endl;
    std::cout << "  Shared memory size: " << shared_width << "x" << shared_height << std::endl;
    std::cout << "  Memory per tile: " << shared_size_bytes << " bytes" << std::endl;
}

bool TileConfig::validate() const {
    if (tile_width <= 0 || tile_height <= 0) {
        std::cerr << "Error: Invalid tile dimensions" << std::endl;
        return false;
    }
    
    if (halo_radius < 0 || halo_radius > MAX_GAUSSIAN_RADIUS) {
        std::cerr << "Error: Invalid halo radius" << std::endl;
        return false;
    }
    
    const size_t max_shared_memory = 48 * 1024;
    if (shared_size_bytes > max_shared_memory) {
        std::cerr << "Error: Shared memory requirement (" << shared_size_bytes 
                  << " bytes) exceeds maximum (" << max_shared_memory << " bytes)" << std::endl;
        return false;
    }
    
    return true;
}

SharedMemoryGaussian::SharedMemoryGaussian() 
    : sigma_(0.0f), radius_(0), global_kernel_(nullptr), initialized_(false) {}

SharedMemoryGaussian::~SharedMemoryGaussian() {
    cleanup();
}

bool SharedMemoryGaussian::initialize(float sigma, int radius, int tile_size) {
    cleanup();
    
    sigma_ = sigma;
    
    if (radius < 0) {
        radius_ = GaussianKernelManager::calculate_radius_from_sigma(sigma);
    } else {
        radius_ = std::min(radius, MAX_GAUSSIAN_RADIUS);
    }
    
    tile_config_ = TileConfig(tile_size, tile_size, radius_);
    
    if (!tile_config_.validate()) {
        return false;
    }
    
    global_kernel_ = new GaussianKernelManager();
    if (!global_kernel_->initialize(sigma_, radius_)) {
        delete global_kernel_;
        global_kernel_ = nullptr;
        return false;
    }
    
    initialized_ = true;
    
    std::cout << "Shared memory Gaussian initialized:" << std::endl;
    std::cout << "  Sigma: " << sigma_ << ", Radius: " << radius_ << std::endl;
    tile_config_.print_config();
    
    return true;
}

void SharedMemoryGaussian::cleanup() {
    if (global_kernel_) {
        global_kernel_->cleanup();
        delete global_kernel_;
        global_kernel_ = nullptr;
    }
    initialized_ = false;
}

bool SharedMemoryGaussian::apply_gaussian_shared(const ImageBuffer& input, 
                                                 ImageBuffer& output, 
                                                 cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "Shared memory Gaussian not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    if (!output.allocate_device(metadata)) {
        return false;
    }
    
    DeviceMemory<unsigned char> temp_buffer;
    if (!temp_buffer.allocate(metadata.total_bytes)) {
        return false;
    }
    
    if (!apply_gaussian_horizontal_shared(input.get_device_ptr(), temp_buffer.get(),
                                         metadata.width, metadata.height, 
                                         metadata.channels, stream)) {
        return false;
    }
    
    if (!apply_gaussian_vertical_shared(temp_buffer.get(), output.get_device_ptr(),
                                       metadata.width, metadata.height,
                                       metadata.channels, stream)) {
        return false;
    }
    
    return true;
}

bool SharedMemoryGaussian::apply_gaussian_horizontal_shared(const unsigned char* input,
                                                           unsigned char* output,
                                                           int width, int height, int channels,
                                                           cudaStream_t stream) const {
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(width, height);
    
    size_t shared_mem_size = SharedMemoryUtils::calculate_shared_memory_size(
        tile_config_.tile_width, radius_, channels, true);
    
    gaussian_horizontal_shared_padded_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, output, width, height, channels, radius_, 
        tile_config_.tile_width, tile_config_.shared_width);
    
    CUDA_CHECK_LAST();
    
    return true;
}

bool SharedMemoryGaussian::apply_gaussian_vertical_shared(const unsigned char* input,
                                                         unsigned char* output,
                                                         int width, int height, int channels,
                                                         cudaStream_t stream) const {
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(width, height);
    
    size_t shared_mem_size = SharedMemoryUtils::calculate_shared_memory_size(
        tile_config_.tile_height, radius_, channels, true);
    
    gaussian_vertical_shared_padded_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, output, width, height, channels, radius_,
        tile_config_.tile_height, tile_config_.shared_height);
    
    CUDA_CHECK_LAST();
    
    return true;
}

void SharedMemoryGaussian::benchmark_shared_vs_global(const ImageBuffer& input, int iterations) const {
    if (!initialized_ || !global_kernel_) {
        std::cerr << "Cannot benchmark: not properly initialized" << std::endl;
        return;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    DeviceMemory<unsigned char> output_buffer;
    output_buffer.allocate(metadata.total_bytes);
    
    CudaTimer timer;
    
    std::cout << "\nBenchmarking Shared Memory vs Global Memory Gaussian:" << std::endl;
    std::cout << "Image size: " << metadata.width << "x" << metadata.height 
              << "x" << metadata.channels << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        global_kernel_->apply_gaussian_separable(input, output_buffer);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    
    float global_time = timer.elapsed_ms();
    
    timer.reset();
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        apply_gaussian_shared(input, output_buffer);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    
    float shared_time = timer.elapsed_ms();
    
    float speedup = global_time / shared_time;
    
    std::cout << "Results:" << std::endl;
    std::cout << "  Global memory: " << global_time / iterations << " ms/iteration" << std::endl;
    std::cout << "  Shared memory: " << shared_time / iterations << " ms/iteration" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
}

dim3 SharedMemoryGaussian::calculate_grid_size(int width, int height) const {
    int grid_x = (width + tile_config_.tile_width - 1) / tile_config_.tile_width;
    int grid_y = (height + tile_config_.tile_height - 1) / tile_config_.tile_height;
    return dim3(grid_x, grid_y, 1);
}

dim3 SharedMemoryGaussian::calculate_block_size() const {
    return dim3(tile_config_.tile_width, tile_config_.tile_height, 1);
}

__global__ void gaussian_horizontal_shared_kernel(const unsigned char* input,
                                                 unsigned char* output,
                                                 int width, int height, int channels,
                                                 int radius, int tile_width, int shared_width) {
    extern __shared__ unsigned char shared_mem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int global_x = bx * tile_width + tx;
    int global_y = by * blockDim.y + ty;
    
    if (global_y >= height) return;
    
    SharedMemoryUtils::load_tile_with_halo(input, shared_mem, width, height, channels,
                                          bx * tile_width, by * blockDim.y, radius,
                                          shared_width, blockDim.y + 2 * radius);
    
    __syncthreads();
    
    if (global_x >= width) return;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        
        for (int i = -radius; i <= radius; ++i) {
            int shared_x = tx + radius + i;
            int shared_idx = ((ty + radius) * shared_width + shared_x) * channels + c;
            sum += shared_mem[shared_idx] * d_gaussian_weights[i + radius];
        }
        
        int output_idx = (global_y * width + global_x) * channels + c;
        output[output_idx] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));
    }
}

__global__ void gaussian_vertical_shared_kernel(const unsigned char* input,
                                               unsigned char* output,
                                               int width, int height, int channels,
                                               int radius, int tile_height, int shared_height) {
    extern __shared__ unsigned char shared_mem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int global_x = bx * blockDim.x + tx;
    int global_y = by * tile_height + ty;
    
    if (global_x >= width) return;
    
    SharedMemoryUtils::load_tile_with_halo(input, shared_mem, width, height, channels,
                                          bx * blockDim.x, by * tile_height, radius,
                                          blockDim.x + 2 * radius, shared_height);
    
    __syncthreads();
    
    if (global_y >= height) return;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        
        for (int i = -radius; i <= radius; ++i) {
            int shared_y = ty + radius + i;
            int shared_idx = (shared_y * (blockDim.x + 2 * radius) + (tx + radius)) * channels + c;
            sum += shared_mem[shared_idx] * d_gaussian_weights[i + radius];
        }
        
        int output_idx = (global_y * width + global_x) * channels + c;
        output[output_idx] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));
    }
}

__global__ void gaussian_horizontal_shared_padded_kernel(const unsigned char* input,
                                                        unsigned char* output,
                                                        int width, int height, int channels,
                                                        int radius, int tile_width, 
                                                        int shared_width) {
    extern __shared__ unsigned char shared_mem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int global_x = bx * tile_width + tx;
    int global_y = by * blockDim.y + ty;
    
    if (global_y >= height) return;
    
    int padded_width = SharedMemoryUtils::calculate_padded_width(shared_width);
    
    SharedMemoryUtils::load_tile_with_halo_padded(input, shared_mem, width, height, channels,
                                                 bx * tile_width, by * blockDim.y, radius,
                                                 shared_width, padded_width);
    
    __syncthreads();
    
    if (global_x >= width) return;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        
        for (int i = -radius; i <= radius; ++i) {
            int shared_x = tx + radius + i;
            int shared_idx = SharedMemoryUtils::get_padded_index(ty + radius, shared_x, padded_width) * channels + c;
            sum += shared_mem[shared_idx] * d_gaussian_weights[i + radius];
        }
        
        int output_idx = (global_y * width + global_x) * channels + c;
        output[output_idx] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));
    }
}

__global__ void gaussian_vertical_shared_padded_kernel(const unsigned char* input,
                                                      unsigned char* output,
                                                      int width, int height, int channels,
                                                      int radius, int tile_height,
                                                      int shared_height) {
    extern __shared__ unsigned char shared_mem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int global_x = bx * blockDim.x + tx;
    int global_y = by * tile_height + ty;
    
    if (global_x >= width) return;
    
    int shared_width = blockDim.x + 2 * radius;
    int padded_width = SharedMemoryUtils::calculate_padded_width(shared_width);
    
    SharedMemoryUtils::load_tile_with_halo_padded(input, shared_mem, width, height, channels,
                                                 bx * blockDim.x, by * tile_height, radius,
                                                 shared_width, padded_width);
    
    __syncthreads();
    
    if (global_y >= height) return;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        
        for (int i = -radius; i <= radius; ++i) {
            int shared_y = ty + radius + i;
            int shared_idx = SharedMemoryUtils::get_padded_index(shared_y, tx + radius, padded_width) * channels + c;
            sum += shared_mem[shared_idx] * d_gaussian_weights[i + radius];
        }
        
        int output_idx = (global_y * width + global_x) * channels + c;
        output[output_idx] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));
    }
}

namespace SharedMemoryUtils {
    size_t calculate_shared_memory_size(int tile_size, int radius, int channels, bool with_padding) {
        int shared_dim = tile_size + 2 * radius;
        
        if (with_padding) {
            int padded_width = calculate_padded_width(shared_dim);
            return padded_width * shared_dim * channels * sizeof(unsigned char);
        } else {
            return shared_dim * shared_dim * channels * sizeof(unsigned char);
        }
    }
    
    int calculate_padded_width(int width, int bank_size) {
        int remainder = width % bank_size;
        return (remainder == 0) ? width + SHARED_MEM_PADDING : width;
    }
    
    bool validate_shared_memory_usage(size_t required_bytes) {
        const size_t max_shared_memory = 48 * 1024;
        return required_bytes <= max_shared_memory;
    }
    
    void print_shared_memory_info(const TileConfig& config, int channels) {
        size_t regular_size = calculate_shared_memory_size(config.tile_width, config.halo_radius, channels, false);
        size_t padded_size = calculate_shared_memory_size(config.tile_width, config.halo_radius, channels, true);
        
        std::cout << "Shared Memory Usage:" << std::endl;
        std::cout << "  Regular: " << regular_size << " bytes" << std::endl;
        std::cout << "  Padded: " << padded_size << " bytes" << std::endl;
        std::cout << "  Overhead: " << (padded_size - regular_size) << " bytes" << std::endl;
    }
    
    __device__ int get_padded_index(int row, int col, int padded_width) {
        return row * padded_width + col;
    }
    
    __device__ void load_tile_with_halo(const unsigned char* input, 
                                       unsigned char* shared_mem,
                                       int width, int height, int channels,
                                       int tile_x, int tile_y, int radius,
                                       int shared_width, int shared_height) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tile_width = blockDim.x;
        int tile_height = blockDim.y;
        
        int start_x = tile_x - radius;
        int start_y = tile_y - radius;
        
        for (int sy = ty; sy < shared_height; sy += tile_height) {
            for (int sx = tx; sx < shared_width; sx += tile_width) {
                int global_x = start_x + sx;
                int global_y = start_y + sy;
                
                global_x = max(0, min(global_x, width - 1));
                global_y = max(0, min(global_y, height - 1));
                
                for (int c = 0; c < channels; ++c) {
                    int global_idx = (global_y * width + global_x) * channels + c;
                    int shared_idx = (sy * shared_width + sx) * channels + c;
                    shared_mem[shared_idx] = input[global_idx];
                }
            }
        }
    }
    
    __device__ void load_tile_with_halo_padded(const unsigned char* input,
                                              unsigned char* shared_mem,
                                              int width, int height, int channels,
                                              int tile_x, int tile_y, int radius,
                                              int shared_width, int padded_width) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tile_width = blockDim.x;
        int tile_height = blockDim.y;
        
        int start_x = tile_x - radius;
        int start_y = tile_y - radius;
        int shared_height = tile_height + 2 * radius;
        
        for (int sy = ty; sy < shared_height; sy += tile_height) {
            for (int sx = tx; sx < shared_width; sx += tile_width) {
                int global_x = start_x + sx;
                int global_y = start_y + sy;
                
                global_x = max(0, min(global_x, width - 1));
                global_y = max(0, min(global_y, height - 1));
                
                for (int c = 0; c < channels; ++c) {
                    int global_idx = (global_y * width + global_x) * channels + c;
                    int shared_idx = get_padded_index(sy, sx, padded_width) * channels + c;
                    shared_mem[shared_idx] = input[global_idx];
                }
            }
        }
    }
}