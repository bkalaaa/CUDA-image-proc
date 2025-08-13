#include "cuda_utils.h"
#include <vector>
#include <iomanip>

void DeviceInfo::print() const {
    std::cout << "=== GPU Device Information ===" << std::endl;
    std::cout << "Device ID: " << device_id << std::endl;
    std::cout << "Name: " << name << std::endl;
    std::cout << "Compute Capability: " << compute_major << "." << compute_minor << std::endl;
    std::cout << "Total Memory: " << (total_memory / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Free Memory: " << (free_memory / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << multiprocessor_count << std::endl;
    std::cout << "Max Threads per Block: " << max_threads_per_block << std::endl;
    std::cout << "Max Block Dimensions: " << max_block_dim_x << " x " << max_block_dim_y << std::endl;
    std::cout << "Max Grid Dimensions: " << max_grid_dim_x << " x " << max_grid_dim_y << std::endl;
    std::cout << "Shared Memory per Block: " << (shared_memory_per_block / 1024) << " KB" << std::endl;
    std::cout << "Warp Size: " << warp_size << std::endl;
    std::cout << "Unified Memory: " << (supports_unified_memory ? "Yes" : "No") << std::endl;
    std::cout << "Compatible: " << (is_compatible() ? "Yes" : "No") << std::endl;
}

bool DeviceInfo::is_compatible() const {
    return (compute_major >= 7) && (compute_minor >= 5);
}

CudaContext::CudaContext() : initialized_(false), current_device_(-1), device_count_(0) {}

CudaContext::~CudaContext() {
    cleanup();
}

bool CudaContext::initialize() {
    if (initialized_) {
        return true;
    }
    
    cudaError_t error = cudaGetDeviceCount(&device_count_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    if (device_count_ == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    error = cudaGetDevice(&current_device_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get current CUDA device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    DeviceInfo info = get_device_info(current_device_);
    if (!info.is_compatible()) {
        std::cerr << "Current GPU (compute " << info.compute_major << "." 
                  << info.compute_minor << ") is not compatible. Requires compute 7.5+" << std::endl;
        return false;
    }
    
    error = cudaSetDevice(current_device_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    initialized_ = true;
    
    std::cout << "CUDA initialized successfully" << std::endl;
    info.print();
    
    return true;
}

void CudaContext::cleanup() {
    if (initialized_) {
        cudaDeviceReset();
        initialized_ = false;
    }
}

int CudaContext::get_device_count() const {
    return device_count_;
}

DeviceInfo CudaContext::get_device_info(int device_id) const {
    DeviceInfo info = {};
    
    int target_device = (device_id >= 0) ? device_id : current_device_;
    if (target_device < 0 || target_device >= device_count_) {
        return info;
    }
    
    info.device_id = target_device;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, target_device));
    
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_block_dim_x = prop.maxThreadsDim[0];
    info.max_block_dim_y = prop.maxThreadsDim[1];
    info.max_grid_dim_x = prop.maxGridSize[0];
    info.max_grid_dim_y = prop.maxGridSize[1];
    info.shared_memory_per_block = prop.sharedMemPerBlock;
    info.warp_size = prop.warpSize;
    info.supports_unified_memory = prop.managedMemory;
    
    size_t free, total;
    if (target_device == current_device_) {
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        info.free_memory = free;
        info.total_memory = total;
    } else {
        info.total_memory = prop.totalGlobalMem;
        info.free_memory = prop.totalGlobalMem;
    }
    
    return info;
}

bool CudaContext::set_device(int device_id) {
    if (device_id < 0 || device_id >= device_count_) {
        std::cerr << "Invalid device ID: " << device_id << std::endl;
        return false;
    }
    
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set device " << device_id << ": " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    current_device_ = device_id;
    return true;
}

int CudaContext::get_current_device() const {
    return current_device_;
}

void CudaContext::synchronize() const {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaContext::check_errors() const {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error detected: " << cudaGetErrorString(error) << std::endl;
    }
}

size_t CudaContext::get_free_memory() const {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

size_t CudaContext::get_total_memory() const {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return total;
}

CudaTimer::CudaTimer() : events_created_(false), timing_started_(false) {
    cudaError_t error1 = cudaEventCreate(&start_event_);
    cudaError_t error2 = cudaEventCreate(&stop_event_);
    
    if (error1 == cudaSuccess && error2 == cudaSuccess) {
        events_created_ = true;
    } else {
        std::cerr << "Failed to create CUDA events for timing" << std::endl;
    }
}

CudaTimer::~CudaTimer() {
    if (events_created_) {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }
}

void CudaTimer::start() {
    if (events_created_) {
        CUDA_CHECK(cudaEventRecord(start_event_));
        timing_started_ = true;
    }
}

void CudaTimer::stop() {
    if (events_created_ && timing_started_) {
        CUDA_CHECK(cudaEventRecord(stop_event_));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
    }
}

float CudaTimer::elapsed_ms() const {
    if (!events_created_ || !timing_started_) {
        return 0.0f;
    }
    
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event_, stop_event_));
    return elapsed;
}

void CudaTimer::reset() {
    timing_started_ = false;
}

dim3 calculate_grid_size(int width, int height, int block_size) {
    int grid_x = (width + block_size - 1) / block_size;
    int grid_y = (height + block_size - 1) / block_size;
    return dim3(grid_x, grid_y, 1);
}

dim3 calculate_block_size(int block_size) {
    return dim3(block_size, block_size, 1);
}

bool check_cuda_capability() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        std::cerr << "CUDA runtime error: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found" << std::endl;
        return false;
    }
    
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, current_device));
    
    bool compatible = (prop.major >= 7) && (prop.minor >= 5);
    
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Compatible: " << (compatible ? "Yes" : "No") << std::endl;
    
    return compatible;
}