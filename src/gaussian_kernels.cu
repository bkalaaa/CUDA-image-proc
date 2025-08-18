#include "gaussian_kernels.h"
#include <iostream>
#include <cmath>
#include <algorithm>

__constant__ float d_gaussian_weights[MAX_GAUSSIAN_SIZE];

GaussianKernelManager::GaussianKernelManager() 
    : sigma_(0.0f), radius_(0), size_(0), host_weights_(nullptr), initialized_(false) {}

GaussianKernelManager::~GaussianKernelManager() {
    cleanup();
}

bool GaussianKernelManager::initialize(float sigma, int radius) {
    cleanup();
    
    sigma_ = sigma;
    
    if (radius < 0) {
        radius_ = calculate_radius_from_sigma(sigma);
    } else {
        radius_ = std::min(radius, MAX_GAUSSIAN_RADIUS);
    }
    
    size_ = 2 * radius_ + 1;
    
    if (!GaussianUtils::validate_gaussian_parameters(sigma_, radius_)) {
        return false;
    }
    
    if (!generate_weights()) {
        return false;
    }
    
    if (!upload_weights_to_constant_memory()) {
        return false;
    }
    
    initialized_ = true;
    
    std::cout << "Gaussian kernel initialized: sigma=" << sigma_ 
              << ", radius=" << radius_ << ", size=" << size_ << std::endl;
    
    return true;
}

void GaussianKernelManager::cleanup() {
    if (host_weights_) {
        delete[] host_weights_;
        host_weights_ = nullptr;
    }
    initialized_ = false;
}

bool GaussianKernelManager::generate_weights() {
    host_weights_ = new float[size_];
    
    float sum = 0.0f;
    float two_sigma_squared = 2.0f * sigma_ * sigma_;
    
    for (int i = 0; i < size_; ++i) {
        int x = i - radius_;
        host_weights_[i] = std::exp(-(x * x) / two_sigma_squared);
        sum += host_weights_[i];
    }
    
    normalize_weights(host_weights_, size_);
    
    return true;
}

bool GaussianKernelManager::upload_weights_to_constant_memory() const {
    cudaError_t error = cudaMemcpyToSymbol(d_gaussian_weights, host_weights_, 
                                          size_ * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to upload Gaussian weights to constant memory: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

bool GaussianKernelManager::apply_gaussian_separable(const ImageBuffer& input, 
                                                    ImageBuffer& output, 
                                                    cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "Gaussian kernel manager not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = input.get_metadata();
    
    if (!output.allocate_device(metadata)) {
        std::cerr << "Failed to allocate output buffer for Gaussian" << std::endl;
        return false;
    }
    
    DeviceMemory<unsigned char> temp_buffer;
    if (!temp_buffer.allocate(metadata.total_bytes)) {
        std::cerr << "Failed to allocate temporary buffer for Gaussian" << std::endl;
        return false;
    }
    
    if (!apply_gaussian_horizontal(input.get_device_ptr(), temp_buffer.get(),
                                  metadata.width, metadata.height, 
                                  metadata.channels, stream)) {
        return false;
    }
    
    if (!apply_gaussian_vertical(temp_buffer.get(), output.get_device_ptr(),
                                metadata.width, metadata.height,
                                metadata.channels, stream)) {
        return false;
    }
    
    return true;
}

bool GaussianKernelManager::apply_gaussian_horizontal(const unsigned char* input,
                                                     unsigned char* output,
                                                     int width, int height, int channels,
                                                     cudaStream_t stream) const {
    dim3 block_size = GaussianUtils::calculate_block_size(16);
    dim3 grid_size = GaussianUtils::calculate_grid_size(width, height, 16);
    
    gaussian_horizontal_coalesced_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, width, height, channels, radius_);
    
    CUDA_CHECK_LAST();
    
    return true;
}

bool GaussianKernelManager::apply_gaussian_vertical(const unsigned char* input,
                                                   unsigned char* output,
                                                   int width, int height, int channels,
                                                   cudaStream_t stream) const {
    dim3 block_size = GaussianUtils::calculate_block_size(16);
    dim3 grid_size = GaussianUtils::calculate_grid_size(width, height, 16);
    
    gaussian_vertical_coalesced_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, width, height, channels, radius_);
    
    CUDA_CHECK_LAST();
    
    return true;
}

void GaussianKernelManager::print_kernel_info() const {
    std::cout << "Gaussian Kernel Information:" << std::endl;
    std::cout << "  Sigma: " << sigma_ << std::endl;
    std::cout << "  Radius: " << radius_ << std::endl;
    std::cout << "  Size: " << size_ << std::endl;
    std::cout << "  Weights: ";
    for (int i = 0; i < size_; ++i) {
        std::cout << host_weights_[i];
        if (i < size_ - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

int GaussianKernelManager::calculate_radius_from_sigma(float sigma) {
    int radius = static_cast<int>(std::ceil(3.0f * sigma));
    return std::min(radius, MAX_GAUSSIAN_RADIUS);
}

void GaussianKernelManager::normalize_weights(float* weights, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += weights[i];
    }
    
    if (sum > 0.0f) {
        for (int i = 0; i < size; ++i) {
            weights[i] /= sum;
        }
    }
}

__global__ void gaussian_horizontal_kernel(const unsigned char* input,
                                         unsigned char* output,
                                         int width, int height, int channels,
                                         int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        
        for (int i = -radius; i <= radius; ++i) {
            int px = x + i;
            px = max(0, min(px, width - 1));
            
            int input_idx = (y * width + px) * channels + c;
            sum += input[input_idx] * d_gaussian_weights[i + radius];
        }
        
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));
    }
}

__global__ void gaussian_vertical_kernel(const unsigned char* input,
                                        unsigned char* output,
                                        int width, int height, int channels,
                                        int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        
        for (int i = -radius; i <= radius; ++i) {
            int py = y + i;
            py = max(0, min(py, height - 1));
            
            int input_idx = (py * width + x) * channels + c;
            sum += input[input_idx] * d_gaussian_weights[i + radius];
        }
        
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));
    }
}

__global__ void gaussian_horizontal_coalesced_kernel(const unsigned char* input,
                                                    unsigned char* output,
                                                    int width, int height, int channels,
                                                    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int base_idx = y * width * channels + x * channels;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        
        for (int i = -radius; i <= radius; ++i) {
            int px = x + i;
            px = max(0, min(px, width - 1));
            
            int input_idx = y * width * channels + px * channels + c;
            sum += input[input_idx] * d_gaussian_weights[i + radius];
        }
        
        output[base_idx + c] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));
    }
}

__global__ void gaussian_vertical_coalesced_kernel(const unsigned char* input,
                                                  unsigned char* output,
                                                  int width, int height, int channels,
                                                  int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int base_idx = y * width * channels + x * channels;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        
        for (int i = -radius; i <= radius; ++i) {
            int py = y + i;
            py = max(0, min(py, height - 1));
            
            int input_idx = py * width * channels + x * channels + c;
            sum += input[input_idx] * d_gaussian_weights[i + radius];
        }
        
        output[base_idx + c] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));
    }
}

namespace GaussianUtils {
    dim3 calculate_grid_size(int width, int height, int block_size) {
        int grid_x = (width + block_size - 1) / block_size;
        int grid_y = (height + block_size - 1) / block_size;
        return dim3(grid_x, grid_y, 1);
    }
    
    dim3 calculate_block_size(int block_size) {
        return dim3(block_size, block_size, 1);
    }
    
    bool validate_gaussian_parameters(float sigma, int radius) {
        if (sigma <= 0.0f) {
            std::cerr << "Error: Gaussian sigma must be positive" << std::endl;
            return false;
        }
        
        if (radius < 1 || radius > MAX_GAUSSIAN_RADIUS) {
            std::cerr << "Error: Gaussian radius must be between 1 and " 
                      << MAX_GAUSSIAN_RADIUS << std::endl;
            return false;
        }
        
        return true;
    }
    
    void benchmark_gaussian_kernels(const ImageBuffer& input,
                                   const GaussianKernelManager& kernel_manager,
                                   int iterations) {
        const ImageMetadata& metadata = input.get_metadata();
        
        DeviceMemory<unsigned char> temp_buffer;
        DeviceMemory<unsigned char> output_buffer;
        
        temp_buffer.allocate(metadata.total_bytes);
        output_buffer.allocate(metadata.total_bytes);
        
        CudaTimer timer;
        
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            kernel_manager.apply_gaussian_horizontal(input.get_device_ptr(),
                                                   temp_buffer.get(),
                                                   metadata.width, metadata.height,
                                                   metadata.channels);
            
            kernel_manager.apply_gaussian_vertical(temp_buffer.get(),
                                                 output_buffer.get(),
                                                 metadata.width, metadata.height,
                                                 metadata.channels);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        
        float avg_time = timer.elapsed_ms() / iterations;
        float throughput = (metadata.total_pixels * iterations) / (timer.elapsed_ms() / 1000.0f);
        
        std::cout << "Gaussian GPU Benchmark Results:" << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Average time: " << avg_time << " ms" << std::endl;
        std::cout << "  Throughput: " << throughput / 1000000.0f << " Mpixels/sec" << std::endl;
    }
}