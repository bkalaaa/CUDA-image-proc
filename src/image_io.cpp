#include "image_io.h"
#include <iostream>
#include <iomanip>
#include <cstring>

void ImageMetadata::print() const {
    std::cout << "Image Metadata:" << std::endl;
    std::cout << "  Dimensions: " << width << "x" << height << std::endl;
    std::cout << "  Channels: " << channels << std::endl;
    std::cout << "  Step: " << step << std::endl;
    std::cout << "  Total pixels: " << total_pixels << std::endl;
    std::cout << "  Total bytes: " << total_bytes << std::endl;
}

ImageBuffer::ImageBuffer() {}

ImageBuffer::ImageBuffer(const ImageMetadata& metadata) : metadata_(metadata) {}

ImageBuffer::ImageBuffer(ImageBuffer&& other) noexcept 
    : metadata_(other.metadata_),
      host_data_(std::move(other.host_data_)),
      device_data_(std::move(other.device_data_)),
      pinned_data_(std::move(other.pinned_data_)) {
    other.metadata_ = ImageMetadata();
}

ImageBuffer& ImageBuffer::operator=(ImageBuffer&& other) noexcept {
    if (this != &other) {
        metadata_ = other.metadata_;
        host_data_ = std::move(other.host_data_);
        device_data_ = std::move(other.device_data_);
        pinned_data_ = std::move(other.pinned_data_);
        other.metadata_ = ImageMetadata();
    }
    return *this;
}

bool ImageBuffer::allocate_host(const ImageMetadata& metadata) {
    metadata_ = metadata;
    if (!metadata_.is_valid()) {
        return false;
    }
    
    try {
        host_data_ = std::make_unique<unsigned char[]>(metadata_.total_bytes);
        return true;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate host memory: " << e.what() << std::endl;
        return false;
    }
}

bool ImageBuffer::allocate_device(const ImageMetadata& metadata) {
    metadata_ = metadata;
    if (!metadata_.is_valid()) {
        return false;
    }
    
    return device_data_.allocate(metadata_.total_bytes);
}

bool ImageBuffer::allocate_pinned(const ImageMetadata& metadata) {
    metadata_ = metadata;
    if (!metadata_.is_valid()) {
        return false;
    }
    
    return pinned_data_.allocate(metadata_.total_bytes);
}

void ImageBuffer::free_host() {
    host_data_.reset();
}

void ImageBuffer::free_device() {
    device_data_.free();
}

void ImageBuffer::free_pinned() {
    pinned_data_.free();
}

void ImageBuffer::free_all() {
    free_host();
    free_device();
    free_pinned();
}

bool ImageBuffer::copy_host_to_device() {
    if (!has_host_data() || !has_device_data()) {
        return false;
    }
    
    return device_data_.copy_from_host(host_data_.get(), metadata_.total_bytes);
}

bool ImageBuffer::copy_device_to_host() {
    if (!has_host_data() || !has_device_data()) {
        return false;
    }
    
    return device_data_.copy_to_host(host_data_.get(), metadata_.total_bytes);
}

bool ImageBuffer::copy_host_to_pinned() {
    if (!has_host_data() || !has_pinned_data()) {
        return false;
    }
    
    std::memcpy(pinned_data_.get(), host_data_.get(), metadata_.total_bytes);
    return true;
}

bool ImageBuffer::copy_pinned_to_host() {
    if (!has_host_data() || !has_pinned_data()) {
        return false;
    }
    
    std::memcpy(host_data_.get(), pinned_data_.get(), metadata_.total_bytes);
    return true;
}

bool ImageBuffer::copy_pinned_to_device() {
    if (!has_pinned_data() || !has_device_data()) {
        return false;
    }
    
    return device_data_.copy_from_host(pinned_data_.get(), metadata_.total_bytes);
}

bool ImageBuffer::copy_device_to_pinned() {
    if (!has_pinned_data() || !has_device_data()) {
        return false;
    }
    
    return device_data_.copy_to_host(pinned_data_.get(), metadata_.total_bytes);
}

bool ImageBuffer::copy_host_to_device_async(cudaStream_t stream) {
    if (!has_host_data() || !has_device_data()) {
        return false;
    }
    
    return device_data_.copy_from_host_async(host_data_.get(), metadata_.total_bytes, stream);
}

bool ImageBuffer::copy_device_to_host_async(cudaStream_t stream) {
    if (!has_host_data() || !has_device_data()) {
        return false;
    }
    
    return device_data_.copy_to_host_async(host_data_.get(), metadata_.total_bytes, stream);
}

bool ImageBuffer::copy_pinned_to_device_async(cudaStream_t stream) {
    if (!has_pinned_data() || !has_device_data()) {
        return false;
    }
    
    return device_data_.copy_from_host_async(pinned_data_.get(), metadata_.total_bytes, stream);
}

bool ImageBuffer::copy_device_to_pinned_async(cudaStream_t stream) {
    if (!has_pinned_data() || !has_device_data()) {
        return false;
    }
    
    return device_data_.copy_to_host_async(pinned_data_.get(), metadata_.total_bytes, stream);
}

ImageMetadata ImageConverter::extract_metadata(const cv::Mat& mat) {
    if (!validate_mat(mat)) {
        return ImageMetadata();
    }
    
    ImageMetadata metadata;
    metadata.width = mat.cols;
    metadata.height = mat.rows;
    metadata.channels = mat.channels();
    metadata.step = mat.step;
    metadata.total_pixels = mat.total();
    metadata.total_bytes = mat.total() * mat.elemSize();
    
    return metadata;
}

bool ImageConverter::mat_to_host_buffer(const cv::Mat& mat, ImageBuffer& buffer) {
    if (!validate_mat(mat)) {
        return false;
    }
    
    ImageMetadata metadata = extract_metadata(mat);
    
    if (!buffer.allocate_host(metadata)) {
        return false;
    }
    
    if (mat.isContinuous()) {
        std::memcpy(buffer.get_host_ptr(), mat.data, metadata.total_bytes);
    } else {
        unsigned char* dst = buffer.get_host_ptr();
        for (int row = 0; row < mat.rows; ++row) {
            const unsigned char* src_row = mat.ptr<unsigned char>(row);
            std::memcpy(dst + row * metadata.width * metadata.channels, 
                       src_row, metadata.width * metadata.channels);
        }
    }
    
    return true;
}

bool ImageConverter::host_buffer_to_mat(const ImageBuffer& buffer, cv::Mat& mat) {
    if (!buffer.has_host_data()) {
        return false;
    }
    
    const ImageMetadata& metadata = buffer.get_metadata();
    if (!metadata.is_valid()) {
        return false;
    }
    
    int cv_type = (metadata.channels == 1) ? CV_8UC1 : 
                  (metadata.channels == 3) ? CV_8UC3 : CV_8UC4;
    
    mat = cv::Mat(metadata.height, metadata.width, cv_type);
    
    if (mat.isContinuous()) {
        std::memcpy(mat.data, buffer.get_host_ptr(), metadata.total_bytes);
    } else {
        const unsigned char* src = buffer.get_host_ptr();
        for (int row = 0; row < mat.rows; ++row) {
            unsigned char* dst_row = mat.ptr<unsigned char>(row);
            std::memcpy(dst_row, src + row * metadata.width * metadata.channels,
                       metadata.width * metadata.channels);
        }
    }
    
    return true;
}

bool ImageConverter::mat_to_device_buffer(const cv::Mat& mat, ImageBuffer& buffer) {
    ImageMetadata metadata = extract_metadata(mat);
    
    if (!buffer.allocate_host(metadata) || !buffer.allocate_device(metadata)) {
        return false;
    }
    
    if (!mat_to_host_buffer(mat, buffer)) {
        return false;
    }
    
    return buffer.copy_host_to_device();
}

bool ImageConverter::device_buffer_to_mat(const ImageBuffer& buffer, cv::Mat& mat) {
    if (!buffer.has_device_data()) {
        return false;
    }
    
    ImageBuffer temp_buffer;
    const ImageMetadata& metadata = buffer.get_metadata();
    
    if (!temp_buffer.allocate_host(metadata)) {
        return false;
    }
    
    temp_buffer.device_data_ = std::move(const_cast<ImageBuffer&>(buffer).device_data_);
    
    if (!temp_buffer.copy_device_to_host()) {
        return false;
    }
    
    return host_buffer_to_mat(temp_buffer, mat);
}

bool ImageConverter::mat_to_pinned_buffer(const cv::Mat& mat, ImageBuffer& buffer) {
    if (!validate_mat(mat)) {
        return false;
    }
    
    ImageMetadata metadata = extract_metadata(mat);
    
    if (!buffer.allocate_pinned(metadata)) {
        return false;
    }
    
    if (mat.isContinuous()) {
        std::memcpy(buffer.get_pinned_ptr(), mat.data, metadata.total_bytes);
    } else {
        unsigned char* dst = buffer.get_pinned_ptr();
        for (int row = 0; row < mat.rows; ++row) {
            const unsigned char* src_row = mat.ptr<unsigned char>(row);
            std::memcpy(dst + row * metadata.width * metadata.channels, 
                       src_row, metadata.width * metadata.channels);
        }
    }
    
    return true;
}

bool ImageConverter::pinned_buffer_to_mat(const ImageBuffer& buffer, cv::Mat& mat) {
    if (!buffer.has_pinned_data()) {
        return false;
    }
    
    const ImageMetadata& metadata = buffer.get_metadata();
    if (!metadata.is_valid()) {
        return false;
    }
    
    int cv_type = (metadata.channels == 1) ? CV_8UC1 : 
                  (metadata.channels == 3) ? CV_8UC3 : CV_8UC4;
    
    mat = cv::Mat(metadata.height, metadata.width, cv_type);
    
    std::memcpy(mat.data, buffer.get_pinned_ptr(), metadata.total_bytes);
    
    return true;
}

unsigned char* ImageConverter::mat_to_raw_ptr(const cv::Mat& mat) {
    if (!validate_mat(mat)) {
        return nullptr;
    }
    
    if (mat.isContinuous()) {
        return mat.data;
    }
    
    return nullptr;
}

cv::Mat ImageConverter::raw_ptr_to_mat(unsigned char* data, const ImageMetadata& metadata) {
    if (!data || !metadata.is_valid()) {
        return cv::Mat();
    }
    
    int cv_type = (metadata.channels == 1) ? CV_8UC1 : 
                  (metadata.channels == 3) ? CV_8UC3 : CV_8UC4;
    
    return cv::Mat(metadata.height, metadata.width, cv_type, data);
}

bool ImageConverter::ensure_continuous(cv::Mat& mat) {
    if (mat.isContinuous()) {
        return true;
    }
    
    mat = clone_continuous(mat);
    return mat.isContinuous();
}

cv::Mat ImageConverter::clone_continuous(const cv::Mat& mat) {
    if (mat.isContinuous()) {
        return mat.clone();
    }
    
    cv::Mat continuous_mat = cv::Mat(mat.size(), mat.type());
    mat.copyTo(continuous_mat);
    return continuous_mat;
}

bool ImageConverter::validate_mat(const cv::Mat& mat) {
    if (mat.empty()) {
        std::cerr << "Error: Input Mat is empty" << std::endl;
        return false;
    }
    
    if (mat.type() != CV_8UC1 && mat.type() != CV_8UC3 && mat.type() != CV_8UC4) {
        std::cerr << "Error: Unsupported Mat type. Only CV_8UC1, CV_8UC3, CV_8UC4 supported" << std::endl;
        return false;
    }
    
    return true;
}

bool ImageConverter::validate_buffer(const ImageBuffer& buffer, const ImageMetadata& expected) {
    const ImageMetadata& actual = buffer.get_metadata();
    
    return actual.width == expected.width &&
           actual.height == expected.height &&
           actual.channels == expected.channels;
}

PitchedImageBuffer::PitchedImageBuffer() : pitched_ptr_(nullptr), pitch_(0) {}

PitchedImageBuffer::PitchedImageBuffer(const ImageMetadata& metadata) 
    : metadata_(metadata), pitched_ptr_(nullptr), pitch_(0) {
    allocate_pitched(metadata);
}

PitchedImageBuffer::~PitchedImageBuffer() {
    free_pitched();
}

bool PitchedImageBuffer::allocate_pitched(const ImageMetadata& metadata) {
    metadata_ = metadata;
    if (!metadata_.is_valid()) {
        return false;
    }
    
    cudaError_t error = cudaMallocPitch(&pitched_ptr_, &pitch_, 
                                       metadata_.width * metadata_.channels, 
                                       metadata_.height);
    
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate pitched memory: " 
                  << cudaGetErrorString(error) << std::endl;
        pitched_ptr_ = nullptr;
        pitch_ = 0;
        return false;
    }
    
    return true;
}

void PitchedImageBuffer::free_pitched() {
    if (pitched_ptr_) {
        cudaFree(pitched_ptr_);
        pitched_ptr_ = nullptr;
        pitch_ = 0;
    }
}

bool PitchedImageBuffer::copy_from_host(const unsigned char* host_data) {
    if (!is_allocated() || !host_data) {
        return false;
    }
    
    cudaError_t error = cudaMemcpy2D(pitched_ptr_, pitch_, host_data, 
                                    metadata_.width * metadata_.channels,
                                    metadata_.width * metadata_.channels,
                                    metadata_.height, cudaMemcpyHostToDevice);
    
    return error == cudaSuccess;
}

bool PitchedImageBuffer::copy_to_host(unsigned char* host_data) const {
    if (!is_allocated() || !host_data) {
        return false;
    }
    
    cudaError_t error = cudaMemcpy2D(host_data, metadata_.width * metadata_.channels,
                                    pitched_ptr_, pitch_,
                                    metadata_.width * metadata_.channels,
                                    metadata_.height, cudaMemcpyDeviceToHost);
    
    return error == cudaSuccess;
}

bool PitchedImageBuffer::copy_from_host_async(const unsigned char* host_data, cudaStream_t stream) {
    if (!is_allocated() || !host_data) {
        return false;
    }
    
    cudaError_t error = cudaMemcpy2DAsync(pitched_ptr_, pitch_, host_data,
                                         metadata_.width * metadata_.channels,
                                         metadata_.width * metadata_.channels,
                                         metadata_.height, cudaMemcpyHostToDevice, stream);
    
    return error == cudaSuccess;
}

bool PitchedImageBuffer::copy_to_host_async(unsigned char* host_data, cudaStream_t stream) const {
    if (!is_allocated() || !host_data) {
        return false;
    }
    
    cudaError_t error = cudaMemcpy2DAsync(host_data, metadata_.width * metadata_.channels,
                                         pitched_ptr_, pitch_,
                                         metadata_.width * metadata_.channels,
                                         metadata_.height, cudaMemcpyDeviceToHost, stream);
    
    return error == cudaSuccess;
}

namespace ImageIO {
    bool transfer_mat_to_device(const cv::Mat& src, ImageBuffer& dst) {
        return ImageConverter::mat_to_device_buffer(src, dst);
    }
    
    bool transfer_device_to_mat(const ImageBuffer& src, cv::Mat& dst) {
        return ImageConverter::device_buffer_to_mat(src, dst);
    }
    
    bool transfer_mat_to_device_async(const cv::Mat& src, ImageBuffer& dst, cudaStream_t stream) {
        if (!ImageConverter::mat_to_pinned_buffer(src, dst)) {
            return false;
        }
        
        ImageMetadata metadata = ImageConverter::extract_metadata(src);
        if (!dst.allocate_device(metadata)) {
            return false;
        }
        
        return dst.copy_pinned_to_device_async(stream);
    }
    
    bool transfer_device_to_mat_async(const ImageBuffer& src, cv::Mat& dst, cudaStream_t stream) {
        ImageBuffer temp_buffer;
        const ImageMetadata& metadata = src.get_metadata();
        
        if (!temp_buffer.allocate_pinned(metadata)) {
            return false;
        }
        
        temp_buffer.device_data_ = std::move(const_cast<ImageBuffer&>(src).device_data_);
        
        if (!temp_buffer.copy_device_to_pinned_async(stream)) {
            return false;
        }
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        return ImageConverter::pinned_buffer_to_mat(temp_buffer, dst);
    }
    
    ImageBuffer create_buffer_from_mat(const cv::Mat& mat, bool allocate_device, bool allocate_pinned) {
        ImageBuffer buffer;
        ImageMetadata metadata = ImageConverter::extract_metadata(mat);
        
        if (allocate_device) {
            buffer.allocate_device(metadata);
        }
        
        if (allocate_pinned) {
            ImageConverter::mat_to_pinned_buffer(mat, buffer);
        } else {
            ImageConverter::mat_to_host_buffer(mat, buffer);
        }
        
        if (allocate_device) {
            if (allocate_pinned) {
                buffer.copy_pinned_to_device();
            } else {
                buffer.copy_host_to_device();
            }
        }
        
        return buffer;
    }
    
    cv::Mat create_mat_from_buffer(const ImageBuffer& buffer) {
        cv::Mat mat;
        ImageConverter::host_buffer_to_mat(buffer, mat);
        return mat;
    }
    
    void print_transfer_info(const ImageMetadata& metadata, const std::string& operation) {
        std::cout << "Transfer: " << operation << std::endl;
        std::cout << "  Size: " << metadata.width << "x" << metadata.height 
                  << "x" << metadata.channels << std::endl;
        std::cout << "  Bytes: " << metadata.total_bytes << " (" 
                  << std::fixed << std::setprecision(2) 
                  << metadata.total_bytes / (1024.0 * 1024.0) << " MB)" << std::endl;
    }
}