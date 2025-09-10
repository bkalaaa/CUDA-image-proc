#include "color_space.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace ColorSpace {

__device__ HSVPixel rgb_to_hsv(const RGBPixel& rgb) {
    float r = rgb.r / 255.0f;
    float g = rgb.g / 255.0f;
    float b = rgb.b / 255.0f;
    
    float max_val = fmaxf(r, fmaxf(g, b));
    float min_val = fminf(r, fminf(g, b));
    float delta = max_val - min_val;
    
    HSVPixel hsv;
    hsv.v = max_val;
    
    if (max_val < 1e-6f) {
        hsv.s = 0.0f;
        hsv.h = 0.0f;
        return hsv;
    }
    
    hsv.s = delta / max_val;
    
    if (delta < 1e-6f) {
        hsv.h = 0.0f;
    } else if (max_val == r) {
        hsv.h = fmodf(60.0f * ((g - b) / delta) + 360.0f, 360.0f);
    } else if (max_val == g) {
        hsv.h = 60.0f * ((b - r) / delta) + 120.0f;
    } else {
        hsv.h = 60.0f * ((r - g) / delta) + 240.0f;
    }
    
    return hsv;
}

__device__ RGBPixel hsv_to_rgb(const HSVPixel& hsv) {
    float c = hsv.v * hsv.s;
    float x = c * (1.0f - fabsf(fmodf(hsv.h / 60.0f, 2.0f) - 1.0f));
    float m = hsv.v - c;
    
    float r, g, b;
    
    if (hsv.h < 60.0f) {
        r = c; g = x; b = 0.0f;
    } else if (hsv.h < 120.0f) {
        r = x; g = c; b = 0.0f;
    } else if (hsv.h < 180.0f) {
        r = 0.0f; g = c; b = x;
    } else if (hsv.h < 240.0f) {
        r = 0.0f; g = x; b = c;
    } else if (hsv.h < 300.0f) {
        r = x; g = 0.0f; b = c;
    } else {
        r = c; g = 0.0f; b = x;
    }
    
    return RGBPixel(
        ColorSpaceUtils::clamp_uchar((r + m) * 255.0f),
        ColorSpaceUtils::clamp_uchar((g + m) * 255.0f),
        ColorSpaceUtils::clamp_uchar((b + m) * 255.0f)
    );
}

__device__ YUVPixel rgb_to_yuv(const RGBPixel& rgb) {
    float r = rgb.r / 255.0f;
    float g = rgb.g / 255.0f;
    float b = rgb.b / 255.0f;
    
    YUVPixel yuv;
    yuv.y = 0.299f * r + 0.587f * g + 0.114f * b;
    yuv.u = -0.147f * r - 0.289f * g + 0.436f * b;
    yuv.v = 0.615f * r - 0.515f * g - 0.100f * b;
    
    return yuv;
}

__device__ RGBPixel yuv_to_rgb(const YUVPixel& yuv) {
    float r = yuv.y + 1.140f * yuv.v;
    float g = yuv.y - 0.395f * yuv.u - 0.581f * yuv.v;
    float b = yuv.y + 2.032f * yuv.u;
    
    return RGBPixel(
        ColorSpaceUtils::clamp_uchar(r * 255.0f),
        ColorSpaceUtils::clamp_uchar(g * 255.0f),
        ColorSpaceUtils::clamp_uchar(b * 255.0f)
    );
}

__device__ unsigned char rgb_to_grayscale(const RGBPixel& rgb) {
    float gray = 0.299f * rgb.r + 0.587f * rgb.g + 0.114f * rgb.b;
    return ColorSpaceUtils::clamp_uchar(gray);
}

__device__ RGBPixel grayscale_to_rgb(unsigned char gray) {
    return RGBPixel(gray, gray, gray);
}

__global__ void convert_rgb_to_hsv_kernel(const unsigned char* rgb_input,
                                         float* hsv_output,
                                         int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    int pixel_idx = idy * width + idx;
    int rgb_idx = pixel_idx * 3;
    
    RGBPixel rgb(rgb_input[rgb_idx], rgb_input[rgb_idx + 1], rgb_input[rgb_idx + 2]);
    HSVPixel hsv = rgb_to_hsv(rgb);
    
    int hsv_idx = pixel_idx * 3;
    hsv_output[hsv_idx] = hsv.h;
    hsv_output[hsv_idx + 1] = hsv.s;
    hsv_output[hsv_idx + 2] = hsv.v;
}

__global__ void convert_hsv_to_rgb_kernel(const float* hsv_input,
                                         unsigned char* rgb_output,
                                         int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    int pixel_idx = idy * width + idx;
    int hsv_idx = pixel_idx * 3;
    
    HSVPixel hsv(hsv_input[hsv_idx], hsv_input[hsv_idx + 1], hsv_input[hsv_idx + 2]);
    RGBPixel rgb = hsv_to_rgb(hsv);
    
    int rgb_idx = pixel_idx * 3;
    rgb_output[rgb_idx] = rgb.r;
    rgb_output[rgb_idx + 1] = rgb.g;
    rgb_output[rgb_idx + 2] = rgb.b;
}

__global__ void convert_rgb_to_yuv_kernel(const unsigned char* rgb_input,
                                         float* yuv_output,
                                         int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    int pixel_idx = idy * width + idx;
    int rgb_idx = pixel_idx * 3;
    
    RGBPixel rgb(rgb_input[rgb_idx], rgb_input[rgb_idx + 1], rgb_input[rgb_idx + 2]);
    YUVPixel yuv = rgb_to_yuv(rgb);
    
    int yuv_idx = pixel_idx * 3;
    yuv_output[yuv_idx] = yuv.y;
    yuv_output[yuv_idx + 1] = yuv.u;
    yuv_output[yuv_idx + 2] = yuv.v;
}

__global__ void convert_yuv_to_rgb_kernel(const float* yuv_input,
                                         unsigned char* rgb_output,
                                         int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    int pixel_idx = idy * width + idx;
    int yuv_idx = pixel_idx * 3;
    
    YUVPixel yuv(yuv_input[yuv_idx], yuv_input[yuv_idx + 1], yuv_input[yuv_idx + 2]);
    RGBPixel rgb = yuv_to_rgb(yuv);
    
    int rgb_idx = pixel_idx * 3;
    rgb_output[rgb_idx] = rgb.r;
    rgb_output[rgb_idx + 1] = rgb.g;
    rgb_output[rgb_idx + 2] = rgb.b;
}

__global__ void convert_rgb_to_grayscale_kernel(const unsigned char* rgb_input,
                                               unsigned char* gray_output,
                                               int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    int pixel_idx = idy * width + idx;
    int rgb_idx = pixel_idx * 3;
    
    RGBPixel rgb(rgb_input[rgb_idx], rgb_input[rgb_idx + 1], rgb_input[rgb_idx + 2]);
    gray_output[pixel_idx] = rgb_to_grayscale(rgb);
}

__global__ void convert_grayscale_to_rgb_kernel(const unsigned char* gray_input,
                                               unsigned char* rgb_output,
                                               int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    int pixel_idx = idy * width + idx;
    
    unsigned char gray = gray_input[pixel_idx];
    RGBPixel rgb = grayscale_to_rgb(gray);
    
    int rgb_idx = pixel_idx * 3;
    rgb_output[rgb_idx] = rgb.r;
    rgb_output[rgb_idx + 1] = rgb.g;
    rgb_output[rgb_idx + 2] = rgb.b;
}

ColorSpaceConverter::ColorSpaceConverter() : initialized_(false) {}

ColorSpaceConverter::~ColorSpaceConverter() {
    cleanup();
}

bool ColorSpaceConverter::initialize() {
    initialized_ = true;
    return true;
}

void ColorSpaceConverter::cleanup() {
    initialized_ = false;
}

bool ColorSpaceConverter::convert_rgb_to_grayscale(const ImageBuffer& rgb_input, ImageBuffer& gray_output,
                                                  cudaStream_t stream) const {
    if (!initialized_) {
        std::cerr << "ColorSpaceConverter not initialized" << std::endl;
        return false;
    }
    
    const ImageMetadata& metadata = rgb_input.get_metadata();
    
    if (metadata.channels != 3) {
        std::cerr << "RGB input must have 3 channels" << std::endl;
        return false;
    }
    
    ImageMetadata gray_metadata = metadata;
    gray_metadata.channels = 1;
    gray_metadata.total_bytes = metadata.width * metadata.height;
    
    if (!gray_output.allocate_device(gray_metadata)) {
        return false;
    }
    
    dim3 block_size = calculate_block_size();
    dim3 grid_size = calculate_grid_size(metadata.width, metadata.height);
    
    convert_rgb_to_grayscale_kernel<<<grid_size, block_size, 0, stream>>>(
        rgb_input.get_device_ptr(), gray_output.get_device_ptr(),
        metadata.width, metadata.height);
    
    CUDA_CHECK_LAST();
    return true;
}

dim3 ColorSpaceConverter::calculate_grid_size(int width, int height) const {
    int block_dim = 16;
    int grid_x = (width + block_dim - 1) / block_dim;
    int grid_y = (height + block_dim - 1) / block_dim;
    return dim3(grid_x, grid_y, 1);
}

dim3 ColorSpaceConverter::calculate_block_size() const {
    return dim3(16, 16, 1);
}

namespace ColorSpaceUtils {
    void print_color_conversion_performance(const std::string& conversion_name,
                                          float time_ms, int width, int height, int iterations) {
        float avg_time = time_ms / iterations;
        float pixels_per_sec = (width * height * iterations) / (time_ms / 1000.0f);
        
        std::cout << conversion_name << " Performance:" << std::endl;
        std::cout << "  Average time: " << avg_time << " ms/iteration" << std::endl;
        std::cout << "  Throughput: " << pixels_per_sec / 1000000.0f << " Mpixels/sec" << std::endl;
    }
    
    bool validate_color_range(const ImageBuffer& buffer, float min_val, float max_val) {
        return true;
    }
    
    __device__ float clamp_float(float value, float min_val, float max_val) {
        return fminf(fmaxf(value, min_val), max_val);
    }
    
    __device__ unsigned char clamp_uchar(float value) {
        return static_cast<unsigned char>(fminf(fmaxf(value, 0.0f), 255.0f));
    }
}

}