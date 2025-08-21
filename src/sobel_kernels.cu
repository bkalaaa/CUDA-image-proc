#include "sobel_kernels.h"
#include <iostream>
#include <cmath>
#include <algorithm>

__constant__ float d_sobel_gx[SOBEL_SIZE * SOBEL_SIZE] = {
    -1.0f, 0.0f, 1.0f,
    -2.0f, 0.0f, 2.0f,
    -1.0f, 0.0f, 1.0f
};

__constant__ float d_sobel_gy[SOBEL_SIZE * SOBEL_SIZE] = {
    -1.0f, -2.0f, -1.0f,
     0.0f,  0.0f,  0.0f,
     1.0f,  2.0f,  1.0f
};

SobelConfig::SobelConfig(int tile_w, int tile_h) 
    : tile_width(tile_w), tile_height(tile_h), compute_magnitude(true), apply_nms(false) {
    shared_width = tile_width + 2 * SOBEL_RADIUS;
    shared_height = tile_height + 2 * SOBEL_RADIUS;
    shared_size_bytes = shared_width * shared_height * sizeof(unsigned char);
}

void SobelConfig::print_config() const {
    std::cout << "Sobel Configuration:" << std::endl;
    std::cout << "  Tile size: " << tile_width << "x" << tile_height << std::endl;
    std::cout << "  Shared memory size: " << shared_width << "x" << shared_height << std::endl;
    std::cout << "  Memory per tile: " << shared_size_bytes << " bytes" << std::endl;
    std::cout << "  Compute magnitude: " << (compute_magnitude ? "Yes" : "No") << std::endl;
    std::cout << "  Apply NMS: " << (apply_nms ? "Yes" : "No") << std::endl;
}

bool SobelConfig::validate() const {
    if (tile_width <= 0 || tile_height <= 0) {
        std::cerr << "Error: Invalid tile dimensions" << std::endl;
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

SobelKernelManager::SobelKernelManager() : initialized_(false) {}

SobelKernelManager::~SobelKernelManager() {
    cleanup();
}

bool SobelKernelManager::initialize(int tile_size) {
    cleanup();
    
    config_ = SobelConfig(tile_size, tile_size);
    
    if (!config_.validate()) {
        return false;
    }
    
    SobelUtils::upload_sobel_coefficients();
    
    initialized_ = true;
    
    std::cout << "Sobel kernel manager initialized:" << std::endl;
    config_.print_config();
    
    return true;
}

void SobelKernelManager::cleanup() {
    initialized_ = false;
}

bool SobelKernelManager::apply_sobel_edge_detection(const ImageBuffer& input, 
                                                   ImageBuffer& output, 
                                                   cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "Sobel kernel manager not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    if (!output.allocate_device(metadata)) {
        return false;
    }
    
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(metadata.width, metadata.height);
    
    sobel_magnitude_fused_kernel<<<grid_size, block_size, 0, stream>>>(
        input.get_device_ptr(), output.get_device_ptr(), 
        metadata.width, metadata.height, metadata.channels);
    
    CUDA_CHECK_LAST();
    
    return true;
}

bool SobelKernelManager::apply_sobel_shared(const ImageBuffer& input, 
                                           ImageBuffer& output,
                                           cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "Sobel kernel manager not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    if (!output.allocate_device(metadata)) {
        return false;
    }
    
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(metadata.width, metadata.height);
    
    size_t shared_mem_size = config_.shared_size_bytes * metadata.channels;
    
    sobel_shared_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input.get_device_ptr(), output.get_device_ptr(),
        metadata.width, metadata.height, metadata.channels,
        config_.tile_width, config_.shared_width);
    
    CUDA_CHECK_LAST();
    
    return true;
}

bool SobelKernelManager::apply_sobel_gradients(const ImageBuffer& input, 
                                              ImageBuffer& gx_output, ImageBuffer& gy_output,
                                              cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "Sobel kernel manager not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    ImageMetadata float_metadata(metadata.width, metadata.height, metadata.channels);
    float_metadata.total_bytes = metadata.total_pixels * metadata.channels * sizeof(float);
    
    DeviceMemory<float> gx_buffer, gy_buffer;
    if (!gx_buffer.allocate(metadata.total_pixels * metadata.channels) ||
        !gy_buffer.allocate(metadata.total_pixels * metadata.channels)) {
        return false;
    }
    
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(metadata.width, metadata.height);
    
    size_t shared_mem_size = config_.shared_size_bytes * metadata.channels;
    
    sobel_gradients_shared_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input.get_device_ptr(), gx_buffer.get(), gy_buffer.get(),
        metadata.width, metadata.height, metadata.channels,
        config_.tile_width, config_.shared_width);
    
    CUDA_CHECK_LAST();
    
    return true;
}

bool SobelKernelManager::apply_sobel_magnitude(const ImageBuffer& gx, const ImageBuffer& gy,
                                              ImageBuffer& magnitude, cudaStream_t stream) const {
    const ImageMetadata& metadata = gx.get_metadata();
    
    if (!magnitude.allocate_device(metadata)) {
        return false;
    }
    
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(metadata.width, metadata.height);
    
    sobel_magnitude_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(gx.get_device_ptr()),
        reinterpret_cast<const float*>(gy.get_device_ptr()),
        magnitude.get_device_ptr(),
        metadata.width, metadata.height, metadata.channels);
    
    CUDA_CHECK_LAST();
    
    return true;
}

bool SobelKernelManager::apply_canny_nms(const ImageBuffer& magnitude, 
                                        const ImageBuffer& gx, const ImageBuffer& gy,
                                        ImageBuffer& output, 
                                        float low_threshold, float high_threshold,
                                        cudaStream_t stream) const {
    const ImageMetadata& metadata = magnitude.get_metadata();
    
    if (!output.allocate_device(metadata)) {
        return false;
    }
    
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(metadata.width, metadata.height);
    
    canny_non_max_suppression_kernel<<<grid_size, block_size, 0, stream>>>(
        magnitude.get_device_ptr(),
        reinterpret_cast<const float*>(gx.get_device_ptr()),
        reinterpret_cast<const float*>(gy.get_device_ptr()),
        output.get_device_ptr(),
        metadata.width, metadata.height,
        low_threshold, high_threshold);
    
    CUDA_CHECK_LAST();
    
    return true;
}

void SobelKernelManager::benchmark_sobel_kernels(const ImageBuffer& input, int iterations) const {
    if (!initialized_) {
        std::cerr << "Cannot benchmark: Sobel kernel manager not initialized" << std::endl;
        return;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    DeviceMemory<unsigned char> output_buffer;
    output_buffer.allocate(metadata.total_bytes);
    
    CudaTimer timer;
    
    std::cout << "\nBenchmarking Sobel Global vs Shared Memory:" << std::endl;
    std::cout << "Image size: " << metadata.width << "x" << metadata.height 
              << "x" << metadata.channels << std::endl;
    
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        apply_sobel_edge_detection(input, output_buffer);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    
    float global_time = timer.elapsed_ms();
    
    timer.reset();
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        apply_sobel_shared(input, output_buffer);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    
    float shared_time = timer.elapsed_ms();
    
    SobelUtils::print_sobel_performance(global_time, shared_time, 
                                       metadata.width, metadata.height, iterations);
}

void SobelKernelManager::benchmark_sobel_optimizations(const ImageBuffer& input, int iterations) const {
    if (!initialized_) {
        std::cerr << "Cannot benchmark: Sobel kernel manager not initialized" << std::endl;
        return;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    DeviceMemory<unsigned char> output_buffer;
    output_buffer.allocate(metadata.total_bytes);
    
    CudaTimer timer;
    
    std::cout << "\nBenchmarking Sobel Optimization Levels:" << std::endl;
    std::cout << "Image size: " << metadata.width << "x" << metadata.height 
              << "x" << metadata.channels << std::endl;
    
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        apply_sobel_edge_detection(input, output_buffer);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    float basic_time = timer.elapsed_ms();
    
    timer.reset();
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        apply_sobel_shared(input, output_buffer);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    float shared_time = timer.elapsed_ms();
    
    timer.reset();
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        apply_sobel_shared_fused(input, output_buffer);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    float shared_fused_time = timer.elapsed_ms();
    
    float vectorized_time = 0.0f;
    if (metadata.width % 4 == 0) {
        timer.reset();
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            apply_sobel_fused_vectorized(input, output_buffer);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        vectorized_time = timer.elapsed_ms();
    }
    
    SobelUtils::print_optimization_performance(basic_time, shared_time, shared_fused_time, 
                                              vectorized_time, metadata, iterations);
}

bool SobelKernelManager::apply_sobel_fused_vectorized(const ImageBuffer& input, 
                                                     ImageBuffer& output,
                                                     cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "Sobel kernel manager not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    if (!output.allocate_device(metadata)) {
        return false;
    }
    
    if (metadata.width % 4 != 0) {
        return apply_sobel_edge_detection(input, output, stream);
    }
    
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(metadata.width / 4, metadata.height);
    
    sobel_magnitude_fused_vectorized_kernel<<<grid_size, block_size, 0, stream>>>(
        input.get_device_ptr(), output.get_device_ptr(), 
        metadata.width, metadata.height, metadata.channels);
    
    CUDA_CHECK_LAST();
    
    return true;
}

bool SobelKernelManager::apply_sobel_shared_fused(const ImageBuffer& input, 
                                                 ImageBuffer& output,
                                                 cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "Sobel kernel manager not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    if (!output.allocate_device(metadata)) {
        return false;
    }
    
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(metadata.width, metadata.height);
    
    size_t shared_mem_size = config_.shared_size_bytes * metadata.channels;
    
    sobel_shared_fused_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input.get_device_ptr(), output.get_device_ptr(),
        metadata.width, metadata.height, metadata.channels,
        config_.tile_width, config_.shared_width);
    
    CUDA_CHECK_LAST();
    
    return true;
}

dim3 SobelKernelManager::calculate_grid_size(int width, int height) const {
    int grid_x = (width + config_.tile_width - 1) / config_.tile_width;
    int grid_y = (height + config_.tile_height - 1) / config_.tile_height;
    return dim3(grid_x, grid_y, 1);
}

dim3 SobelKernelManager::calculate_block_size() const {
    return dim3(config_.tile_width, config_.tile_height, 1);
}

__global__ void sobel_edge_detection_kernel(const unsigned char* input,
                                           unsigned char* output,
                                           int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int c = 0; c < channels; ++c) {
        float gx = 0.0f, gy = 0.0f;
        
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                
                int input_idx = (py * width + px) * channels + c;
                float pixel_val = input[input_idx];
                
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                gx += pixel_val * d_sobel_gx[kernel_idx];
                gy += pixel_val * d_sobel_gy[kernel_idx];
            }
        }
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = static_cast<unsigned char>(min(255.0f, magnitude));
    }
}

__global__ void sobel_shared_kernel(const unsigned char* input,
                                   unsigned char* output,
                                   int width, int height, int channels,
                                   int tile_width, int shared_width) {
    extern __shared__ unsigned char shared_mem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int global_x = bx * tile_width + tx;
    int global_y = by * blockDim.y + ty;
    
    if (global_y >= height) return;
    
    SobelUtils::load_sobel_tile_with_halo(input, shared_mem, width, height, channels,
                                         bx * tile_width, by * blockDim.y,
                                         shared_width, blockDim.y + 2);
    
    __syncthreads();
    
    if (global_x >= width) return;
    
    for (int c = 0; c < channels; ++c) {
        float gx = SobelUtils::apply_sobel_filter_x(shared_mem, tx + 1, ty + 1, shared_width, c);
        float gy = SobelUtils::apply_sobel_filter_y(shared_mem, tx + 1, ty + 1, shared_width, c);
        
        float magnitude = SobelUtils::compute_gradient_magnitude(gx, gy);
        
        int output_idx = (global_y * width + global_x) * channels + c;
        output[output_idx] = static_cast<unsigned char>(min(255.0f, magnitude));
    }
}

__global__ void sobel_gradients_kernel(const unsigned char* input,
                                      float* gx_output, float* gy_output,
                                      int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int c = 0; c < channels; ++c) {
        float gx = 0.0f, gy = 0.0f;
        
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                
                int input_idx = (py * width + px) * channels + c;
                float pixel_val = input[input_idx];
                
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                gx += pixel_val * d_sobel_gx[kernel_idx];
                gy += pixel_val * d_sobel_gy[kernel_idx];
            }
        }
        
        int output_idx = (y * width + x) * channels + c;
        gx_output[output_idx] = gx;
        gy_output[output_idx] = gy;
    }
}

__global__ void sobel_gradients_shared_kernel(const unsigned char* input,
                                             float* gx_output, float* gy_output,
                                             int width, int height, int channels,
                                             int tile_width, int shared_width) {
    extern __shared__ unsigned char shared_mem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int global_x = bx * tile_width + tx;
    int global_y = by * blockDim.y + ty;
    
    if (global_y >= height) return;
    
    SobelUtils::load_sobel_tile_with_halo(input, shared_mem, width, height, channels,
                                         bx * tile_width, by * blockDim.y,
                                         shared_width, blockDim.y + 2);
    
    __syncthreads();
    
    if (global_x >= width) return;
    
    for (int c = 0; c < channels; ++c) {
        float gx = SobelUtils::apply_sobel_filter_x(shared_mem, tx + 1, ty + 1, shared_width, c);
        float gy = SobelUtils::apply_sobel_filter_y(shared_mem, tx + 1, ty + 1, shared_width, c);
        
        int output_idx = (global_y * width + global_x) * channels + c;
        gx_output[output_idx] = gx;
        gy_output[output_idx] = gy;
    }
}

__global__ void sobel_magnitude_kernel(const float* gx, const float* gy,
                                      unsigned char* magnitude,
                                      int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int c = 0; c < channels; ++c) {
        int idx = (y * width + x) * channels + c;
        float mag = SobelUtils::compute_gradient_magnitude(gx[idx], gy[idx]);
        magnitude[idx] = static_cast<unsigned char>(min(255.0f, mag));
    }
}

__global__ void sobel_magnitude_fused_kernel(const unsigned char* input,
                                            unsigned char* output,
                                            int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int c = 0; c < channels; ++c) {
        float gx = 0.0f, gy = 0.0f;
        
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                
                int input_idx = (py * width + px) * channels + c;
                float pixel_val = input[input_idx];
                
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                gx += pixel_val * d_sobel_gx[kernel_idx];
                gy += pixel_val * d_sobel_gy[kernel_idx];
            }
        }
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = static_cast<unsigned char>(min(255.0f, magnitude));
    }
}

__global__ void canny_non_max_suppression_kernel(const unsigned char* magnitude,
                                                const float* gx, const float* gy,
                                                unsigned char* output,
                                                int width, int height,
                                                float low_threshold, float high_threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height || x == 0 || y == 0 || x == width-1 || y == height-1) {
        if (x < width && y < height) {
            output[y * width + x] = 0;
        }
        return;
    }
    
    int idx = y * width + x;
    float mag = magnitude[idx];
    
    if (mag < low_threshold) {
        output[idx] = 0;
        return;
    }
    
    float gradient_x = gx[idx];
    float gradient_y = gy[idx];
    float angle = atan2f(gradient_y, gradient_x) * 180.0f / M_PI;
    
    if (angle < 0) angle += 180.0f;
    
    float mag1, mag2;
    
    if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
        mag1 = magnitude[y * width + (x - 1)];
        mag2 = magnitude[y * width + (x + 1)];
    } else if (angle >= 22.5 && angle < 67.5) {
        mag1 = magnitude[(y - 1) * width + (x + 1)];
        mag2 = magnitude[(y + 1) * width + (x - 1)];
    } else if (angle >= 67.5 && angle < 112.5) {
        mag1 = magnitude[(y - 1) * width + x];
        mag2 = magnitude[(y + 1) * width + x];
    } else {
        mag1 = magnitude[(y - 1) * width + (x - 1)];
        mag2 = magnitude[(y + 1) * width + (x + 1)];
    }
    
    if (mag >= mag1 && mag >= mag2) {
        output[idx] = (mag >= high_threshold) ? 255 : 128;
    } else {
        output[idx] = 0;
    }
}

__global__ void sobel_magnitude_fused_vectorized_kernel(const unsigned char* input,
                                                       unsigned char* output,
                                                       int width, int height, int channels) {
    int vec_x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = vec_x * 4;
    
    if (x + 3 >= width || y >= height) {
        if (x < width && y < height) {
            for (int i = 0; i < min(4, width - x); ++i) {
                int px = x + i;
                for (int c = 0; c < channels; ++c) {
                    float gx = 0.0f, gy = 0.0f;
                    
                    for (int ky = -1; ky <= 1; ++ky) {
                        for (int kx = -1; kx <= 1; ++kx) {
                            int px_sample = min(max(px + kx, 0), width - 1);
                            int py_sample = min(max(y + ky, 0), height - 1);
                            
                            int input_idx = (py_sample * width + px_sample) * channels + c;
                            float pixel_val = input[input_idx];
                            
                            int kernel_idx = (ky + 1) * 3 + (kx + 1);
                            gx += pixel_val * d_sobel_gx[kernel_idx];
                            gy += pixel_val * d_sobel_gy[kernel_idx];
                        }
                    }
                    
                    float magnitude = sqrtf(gx * gx + gy * gy);
                    int output_idx = (y * width + px) * channels + c;
                    output[output_idx] = static_cast<unsigned char>(min(255.0f, magnitude));
                }
            }
        }
        return;
    }
    
    uchar4 pixels[4];
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < 4; ++i) {
            int px = x + i;
            
            float gx = 0.0f, gy = 0.0f;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int px_sample = min(max(px + kx, 0), width - 1);
                    int py_sample = min(max(y + ky, 0), height - 1);
                    
                    int input_idx = (py_sample * width + px_sample) * channels + c;
                    float pixel_val = input[input_idx];
                    
                    int kernel_idx = (ky + 1) * 3 + (kx + 1);
                    gx += pixel_val * d_sobel_gx[kernel_idx];
                    gy += pixel_val * d_sobel_gy[kernel_idx];
                }
            }
            
            float magnitude = sqrtf(gx * gx + gy * gy);
            int output_idx = (y * width + px) * channels + c;
            output[output_idx] = static_cast<unsigned char>(min(255.0f, magnitude));
        }
    }
}

__global__ void sobel_shared_fused_kernel(const unsigned char* input,
                                         unsigned char* output,
                                         int width, int height, int channels,
                                         int tile_width, int shared_width) {
    extern __shared__ unsigned char shared_mem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int global_x = bx * tile_width + tx;
    int global_y = by * blockDim.y + ty;
    
    if (global_y >= height) return;
    
    SobelUtils::load_sobel_tile_with_halo(input, shared_mem, width, height, channels,
                                         bx * tile_width, by * blockDim.y,
                                         shared_width, blockDim.y + 2);
    
    __syncthreads();
    
    if (global_x >= width) return;
    
    for (int c = 0; c < channels; ++c) {
        float gx = 0.0f, gy = 0.0f;
        
        #pragma unroll
        for (int ky = -1; ky <= 1; ++ky) {
            #pragma unroll
            for (int kx = -1; kx <= 1; ++kx) {
                int sx = tx + 1 + kx;
                int sy = ty + 1 + ky;
                int shared_idx = (sy * shared_width + sx) * channels + c;
                float pixel_val = shared_mem[shared_idx];
                
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                gx += pixel_val * d_sobel_gx[kernel_idx];
                gy += pixel_val * d_sobel_gy[kernel_idx];
            }
        }
        
        float magnitude = __fsqrt_rn(gx * gx + gy * gy);
        
        int output_idx = (global_y * width + global_x) * channels + c;
        output[output_idx] = __float2uint_rn(fminf(255.0f, magnitude));
    }
}

namespace SobelUtils {
    dim3 calculate_grid_size(int width, int height, int block_size) {
        int grid_x = (width + block_size - 1) / block_size;
        int grid_y = (height + block_size - 1) / block_size;
        return dim3(grid_x, grid_y, 1);
    }
    
    dim3 calculate_block_size(int block_size) {
        return dim3(block_size, block_size, 1);
    }
    
    void upload_sobel_coefficients() {
        static bool uploaded = false;
        if (!uploaded) {
            uploaded = true;
        }
    }
    
    __device__ float compute_gradient_magnitude(float gx, float gy) {
        return sqrtf(gx * gx + gy * gy);
    }
    
    __device__ float compute_gradient_direction(float gx, float gy) {
        return atan2f(gy, gx);
    }
    
    __device__ void load_sobel_tile_with_halo(const unsigned char* input,
                                             unsigned char* shared_mem,
                                             int width, int height, int channels,
                                             int tile_x, int tile_y,
                                             int shared_width, int shared_height) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tile_width = blockDim.x;
        int tile_height = blockDim.y;
        
        int start_x = tile_x - SOBEL_RADIUS;
        int start_y = tile_y - SOBEL_RADIUS;
        
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
    
    __device__ float apply_sobel_filter_x(const unsigned char* shared_mem,
                                         int shared_x, int shared_y, int shared_width,
                                         int channel) {
        float gx = 0.0f;
        
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int sx = shared_x + kx;
                int sy = shared_y + ky;
                int shared_idx = (sy * shared_width + sx) * 1 + channel;
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                gx += shared_mem[shared_idx] * d_sobel_gx[kernel_idx];
            }
        }
        
        return gx;
    }
    
    __device__ float apply_sobel_filter_y(const unsigned char* shared_mem,
                                         int shared_x, int shared_y, int shared_width,
                                         int channel) {
        float gy = 0.0f;
        
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int sx = shared_x + kx;
                int sy = shared_y + ky;
                int shared_idx = (sy * shared_width + sx) * 1 + channel;
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                gy += shared_mem[shared_idx] * d_sobel_gy[kernel_idx];
            }
        }
        
        return gy;
    }
    
    void print_sobel_performance(float global_time_ms, float shared_time_ms, 
                                int width, int height, int iterations) {
        float speedup = global_time_ms / shared_time_ms;
        float global_avg = global_time_ms / iterations;
        float shared_avg = shared_time_ms / iterations;
        
        std::cout << "Results:" << std::endl;
        std::cout << "  Global memory: " << global_avg << " ms/iteration" << std::endl;
        std::cout << "  Shared memory: " << shared_avg << " ms/iteration" << std::endl;
        std::cout << "  Speedup: " << speedup << "x" << std::endl;
        
        float pixels_per_sec = (width * height * iterations) / (shared_time_ms / 1000.0f);
        std::cout << "  Throughput: " << pixels_per_sec / 1000000.0f << " Mpixels/sec" << std::endl;
    }
    
    void print_optimization_performance(float basic_time, float shared_time, 
                                       float shared_fused_time, float vectorized_time,
                                       const ImageMetadata& metadata, int iterations) {
        float basic_avg = basic_time / iterations;
        float shared_avg = shared_time / iterations;
        float shared_fused_avg = shared_fused_time / iterations;
        
        std::cout << "\nOptimization Results:" << std::endl;
        std::cout << "  Basic global memory: " << basic_avg << " ms/iteration (baseline)" << std::endl;
        std::cout << "  Shared memory: " << shared_avg << " ms/iteration" << std::endl;
        std::cout << "  Shared + fused: " << shared_fused_avg << " ms/iteration" << std::endl;
        
        if (vectorized_time > 0.0f) {
            float vectorized_avg = vectorized_time / iterations;
            std::cout << "  Vectorized loads: " << vectorized_avg << " ms/iteration" << std::endl;
            
            float vec_speedup = basic_time / vectorized_time;
            std::cout << "  Best speedup: " << vec_speedup << "x (vectorized)" << std::endl;
        }
        
        float shared_speedup = basic_time / shared_time;
        float fused_speedup = basic_time / shared_fused_time;
        
        std::cout << "  Shared memory speedup: " << shared_speedup << "x" << std::endl;
        std::cout << "  Fused kernel speedup: " << fused_speedup << "x" << std::endl;
        
        float best_time = (vectorized_time > 0.0f) ? 
                         min(min(shared_fused_time, vectorized_time), shared_time) : 
                         min(shared_fused_time, shared_time);
        
        float pixels_per_sec = (metadata.width * metadata.height * iterations) / (best_time / 1000.0f);
        std::cout << "  Peak throughput: " << pixels_per_sec / 1000000.0f << " Mpixels/sec" << std::endl;
    }
}