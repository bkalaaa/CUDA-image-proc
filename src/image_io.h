#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <memory>

struct ImageMetadata {
    int width;
    int height;
    int channels;
    int step;
    size_t total_pixels;
    size_t total_bytes;
    
    ImageMetadata() : width(0), height(0), channels(0), step(0), total_pixels(0), total_bytes(0) {}
    
    ImageMetadata(int w, int h, int c) 
        : width(w), height(h), channels(c), step(w * c), 
          total_pixels(w * h), total_bytes(w * h * c) {}
    
    bool is_valid() const {
        return width > 0 && height > 0 && channels > 0;
    }
    
    void print() const;
};

class ImageBuffer {
public:
    ImageBuffer();
    explicit ImageBuffer(const ImageMetadata& metadata);
    ~ImageBuffer() = default;
    
    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator=(const ImageBuffer&) = delete;
    
    ImageBuffer(ImageBuffer&& other) noexcept;
    ImageBuffer& operator=(ImageBuffer&& other) noexcept;
    
    bool allocate_host(const ImageMetadata& metadata);
    bool allocate_device(const ImageMetadata& metadata);
    bool allocate_pinned(const ImageMetadata& metadata);
    
    void free_host();
    void free_device();
    void free_pinned();
    void free_all();
    
    unsigned char* get_host_ptr() { return host_data_.get(); }
    const unsigned char* get_host_ptr() const { return host_data_.get(); }
    
    unsigned char* get_device_ptr() { return device_data_.get(); }
    const unsigned char* get_device_ptr() const { return device_data_.get(); }
    
    unsigned char* get_pinned_ptr() { return pinned_data_.get(); }
    const unsigned char* get_pinned_ptr() const { return pinned_data_.get(); }
    
    const ImageMetadata& get_metadata() const { return metadata_; }
    
    bool has_host_data() const { return host_data_.is_allocated(); }
    bool has_device_data() const { return device_data_.is_allocated(); }
    bool has_pinned_data() const { return pinned_data_.is_allocated(); }
    
    bool copy_host_to_device();
    bool copy_device_to_host();
    bool copy_host_to_pinned();
    bool copy_pinned_to_host();
    bool copy_pinned_to_device();
    bool copy_device_to_pinned();
    
    bool copy_host_to_device_async(cudaStream_t stream);
    bool copy_device_to_host_async(cudaStream_t stream);
    bool copy_pinned_to_device_async(cudaStream_t stream);
    bool copy_device_to_pinned_async(cudaStream_t stream);
    
private:
    ImageMetadata metadata_;
    std::unique_ptr<unsigned char[]> host_data_;
    DeviceMemory<unsigned char> device_data_;
    PinnedHostMemory<unsigned char> pinned_data_;
};

class ImageConverter {
public:
    static ImageMetadata extract_metadata(const cv::Mat& mat);
    
    static bool mat_to_host_buffer(const cv::Mat& mat, ImageBuffer& buffer);
    static bool host_buffer_to_mat(const ImageBuffer& buffer, cv::Mat& mat);
    
    static bool mat_to_device_buffer(const cv::Mat& mat, ImageBuffer& buffer);
    static bool device_buffer_to_mat(const ImageBuffer& buffer, cv::Mat& mat);
    
    static bool mat_to_pinned_buffer(const cv::Mat& mat, ImageBuffer& buffer);
    static bool pinned_buffer_to_mat(const ImageBuffer& buffer, cv::Mat& mat);
    
    static unsigned char* mat_to_raw_ptr(const cv::Mat& mat);
    static cv::Mat raw_ptr_to_mat(unsigned char* data, const ImageMetadata& metadata);
    
    static bool ensure_continuous(cv::Mat& mat);
    static cv::Mat clone_continuous(const cv::Mat& mat);
    
private:
    static bool validate_mat(const cv::Mat& mat);
    static bool validate_buffer(const ImageBuffer& buffer, const ImageMetadata& expected);
};

class PitchedImageBuffer {
public:
    PitchedImageBuffer();
    explicit PitchedImageBuffer(const ImageMetadata& metadata);
    ~PitchedImageBuffer();
    
    PitchedImageBuffer(const PitchedImageBuffer&) = delete;
    PitchedImageBuffer& operator=(const PitchedImageBuffer&) = delete;
    
    bool allocate_pitched(const ImageMetadata& metadata);
    void free_pitched();
    
    unsigned char* get_pitched_ptr() { return pitched_ptr_; }
    const unsigned char* get_pitched_ptr() const { return pitched_ptr_; }
    
    size_t get_pitch() const { return pitch_; }
    const ImageMetadata& get_metadata() const { return metadata_; }
    
    bool copy_from_host(const unsigned char* host_data);
    bool copy_to_host(unsigned char* host_data) const;
    
    bool copy_from_host_async(const unsigned char* host_data, cudaStream_t stream);
    bool copy_to_host_async(unsigned char* host_data, cudaStream_t stream) const;
    
    bool is_allocated() const { return pitched_ptr_ != nullptr; }
    
private:
    ImageMetadata metadata_;
    unsigned char* pitched_ptr_;
    size_t pitch_;
};

namespace ImageIO {
    bool transfer_mat_to_device(const cv::Mat& src, ImageBuffer& dst);
    bool transfer_device_to_mat(const ImageBuffer& src, cv::Mat& dst);
    
    bool transfer_mat_to_device_async(const cv::Mat& src, ImageBuffer& dst, cudaStream_t stream);
    bool transfer_device_to_mat_async(const ImageBuffer& src, cv::Mat& dst, cudaStream_t stream);
    
    ImageBuffer create_buffer_from_mat(const cv::Mat& mat, bool allocate_device = true, bool allocate_pinned = false);
    cv::Mat create_mat_from_buffer(const ImageBuffer& buffer);
    
    void print_transfer_info(const ImageMetadata& metadata, const std::string& operation);
}