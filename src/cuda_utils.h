#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <memory>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_CHECK_RETURN(call, retval) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            return retval; \
        } \
    } while(0)

#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA kernel error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_major;
    int compute_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_block_dim_x;
    int max_block_dim_y;
    int max_grid_dim_x;
    int max_grid_dim_y;
    size_t shared_memory_per_block;
    int warp_size;
    bool supports_unified_memory;
    
    void print() const;
    bool is_compatible() const;
};

class CudaContext {
public:
    CudaContext();
    ~CudaContext();
    
    bool initialize();
    void cleanup();
    
    int get_device_count() const;
    DeviceInfo get_device_info(int device_id = -1) const;
    bool set_device(int device_id);
    int get_current_device() const;
    
    void synchronize() const;
    void check_errors() const;
    
    size_t get_free_memory() const;
    size_t get_total_memory() const;
    
private:
    bool initialized_;
    int current_device_;
    int device_count_;
};

template<typename T>
class PinnedHostMemory {
public:
    PinnedHostMemory() : ptr_(nullptr), size_(0) {}
    
    explicit PinnedHostMemory(size_t count) : ptr_(nullptr), size_(0) {
        allocate(count);
    }
    
    ~PinnedHostMemory() {
        free();
    }
    
    PinnedHostMemory(const PinnedHostMemory&) = delete;
    PinnedHostMemory& operator=(const PinnedHostMemory&) = delete;
    
    PinnedHostMemory(PinnedHostMemory&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    PinnedHostMemory& operator=(PinnedHostMemory&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    bool allocate(size_t count) {
        free();
        size_ = count * sizeof(T);
        cudaError_t error = cudaHostAlloc(&ptr_, size_, cudaHostAllocDefault);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate pinned memory: " 
                      << cudaGetErrorString(error) << std::endl;
            ptr_ = nullptr;
            size_ = 0;
            return false;
        }
        return true;
    }
    
    void free() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }
    
    T* get() { return static_cast<T*>(ptr_); }
    const T* get() const { return static_cast<const T*>(ptr_); }
    
    T& operator[](size_t index) { return static_cast<T*>(ptr_)[index]; }
    const T& operator[](size_t index) const { return static_cast<const T*>(ptr_)[index]; }
    
    size_t size_bytes() const { return size_; }
    size_t size() const { return size_ / sizeof(T); }
    
    bool is_allocated() const { return ptr_ != nullptr; }
    
private:
    void* ptr_;
    size_t size_;
};

template<typename T>
class DeviceMemory {
public:
    DeviceMemory() : ptr_(nullptr), size_(0) {}
    
    explicit DeviceMemory(size_t count) : ptr_(nullptr), size_(0) {
        allocate(count);
    }
    
    ~DeviceMemory() {
        free();
    }
    
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    DeviceMemory(DeviceMemory&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    bool allocate(size_t count) {
        free();
        size_ = count * sizeof(T);
        cudaError_t error = cudaMalloc(&ptr_, size_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate device memory: " 
                      << cudaGetErrorString(error) << std::endl;
            ptr_ = nullptr;
            size_ = 0;
            return false;
        }
        return true;
    }
    
    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }
    
    T* get() { return static_cast<T*>(ptr_); }
    const T* get() const { return static_cast<const T*>(ptr_); }
    
    size_t size_bytes() const { return size_; }
    size_t size() const { return size_ / sizeof(T); }
    
    bool is_allocated() const { return ptr_ != nullptr; }
    
    bool copy_from_host(const T* host_ptr, size_t count) {
        if (!is_allocated() || count * sizeof(T) > size_) {
            return false;
        }
        cudaError_t error = cudaMemcpy(ptr_, host_ptr, count * sizeof(T), 
                                      cudaMemcpyHostToDevice);
        return error == cudaSuccess;
    }
    
    bool copy_to_host(T* host_ptr, size_t count) const {
        if (!is_allocated() || count * sizeof(T) > size_) {
            return false;
        }
        cudaError_t error = cudaMemcpy(host_ptr, ptr_, count * sizeof(T), 
                                      cudaMemcpyDeviceToHost);
        return error == cudaSuccess;
    }
    
    bool copy_from_host_async(const T* host_ptr, size_t count, cudaStream_t stream) {
        if (!is_allocated() || count * sizeof(T) > size_) {
            return false;
        }
        cudaError_t error = cudaMemcpyAsync(ptr_, host_ptr, count * sizeof(T), 
                                           cudaMemcpyHostToDevice, stream);
        return error == cudaSuccess;
    }
    
    bool copy_to_host_async(T* host_ptr, size_t count, cudaStream_t stream) const {
        if (!is_allocated() || count * sizeof(T) > size_) {
            return false;
        }
        cudaError_t error = cudaMemcpyAsync(host_ptr, ptr_, count * sizeof(T), 
                                           cudaMemcpyDeviceToHost, stream);
        return error == cudaSuccess;
    }
    
private:
    void* ptr_;
    size_t size_;
};

class CudaTimer {
public:
    CudaTimer();
    ~CudaTimer();
    
    void start();
    void stop();
    float elapsed_ms() const;
    
    void reset();
    
private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    bool events_created_;
    bool timing_started_;
};

dim3 calculate_grid_size(int width, int height, int block_size);
dim3 calculate_block_size(int block_size);

bool check_cuda_capability();