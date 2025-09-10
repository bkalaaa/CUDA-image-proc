#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "image_io.h"

namespace ColorSpace {

struct RGBPixel {
    unsigned char r, g, b;
    __host__ __device__ RGBPixel() : r(0), g(0), b(0) {}
    __host__ __device__ RGBPixel(unsigned char r_, unsigned char g_, unsigned char b_) : r(r_), g(g_), b(b_) {}
};

struct HSVPixel {
    float h, s, v;
    __host__ __device__ HSVPixel() : h(0.0f), s(0.0f), v(0.0f) {}
    __host__ __device__ HSVPixel(float h_, float s_, float v_) : h(h_), s(s_), v(v_) {}
};

struct YUVPixel {
    float y, u, v;
    __host__ __device__ YUVPixel() : y(0.0f), u(0.0f), v(0.0f) {}
    __host__ __device__ YUVPixel(float y_, float u_, float v_) : y(y_), u(u_), v(v_) {}
};

__device__ HSVPixel rgb_to_hsv(const RGBPixel& rgb);
__device__ RGBPixel hsv_to_rgb(const HSVPixel& hsv);
__device__ YUVPixel rgb_to_yuv(const RGBPixel& rgb);
__device__ RGBPixel yuv_to_rgb(const YUVPixel& yuv);

__device__ unsigned char rgb_to_grayscale(const RGBPixel& rgb);
__device__ RGBPixel grayscale_to_rgb(unsigned char gray);

__global__ void convert_rgb_to_hsv_kernel(const unsigned char* rgb_input,
                                         float* hsv_output,
                                         int width, int height);

__global__ void convert_hsv_to_rgb_kernel(const float* hsv_input,
                                         unsigned char* rgb_output,
                                         int width, int height);

__global__ void convert_rgb_to_yuv_kernel(const unsigned char* rgb_input,
                                         float* yuv_output,
                                         int width, int height);

__global__ void convert_yuv_to_rgb_kernel(const float* yuv_input,
                                         unsigned char* rgb_output,
                                         int width, int height);

__global__ void convert_rgb_to_grayscale_kernel(const unsigned char* rgb_input,
                                               unsigned char* gray_output,
                                               int width, int height);

__global__ void convert_grayscale_to_rgb_kernel(const unsigned char* gray_input,
                                               unsigned char* rgb_output,
                                               int width, int height);

class ColorSpaceConverter {
public:
    ColorSpaceConverter();
    ~ColorSpaceConverter();
    
    bool initialize();
    void cleanup();
    
    bool convert_rgb_to_hsv(const ImageBuffer& rgb_input, ImageBuffer& hsv_output,
                           cudaStream_t stream = 0) const;
    bool convert_hsv_to_rgb(const ImageBuffer& hsv_input, ImageBuffer& rgb_output,
                           cudaStream_t stream = 0) const;
    
    bool convert_rgb_to_yuv(const ImageBuffer& rgb_input, ImageBuffer& yuv_output,
                           cudaStream_t stream = 0) const;
    bool convert_yuv_to_rgb(const ImageBuffer& yuv_input, ImageBuffer& rgb_output,
                           cudaStream_t stream = 0) const;
    
    bool convert_rgb_to_grayscale(const ImageBuffer& rgb_input, ImageBuffer& gray_output,
                                 cudaStream_t stream = 0) const;
    bool convert_grayscale_to_rgb(const ImageBuffer& gray_input, ImageBuffer& rgb_output,
                                 cudaStream_t stream = 0) const;
    
    void benchmark_color_conversions(const ImageBuffer& input, int iterations = 10) const;
    
private:
    bool initialized_;
    
    dim3 calculate_grid_size(int width, int height) const;
    dim3 calculate_block_size() const;
};

namespace ColorSpaceUtils {
    void print_color_conversion_performance(const std::string& conversion_name,
                                          float time_ms, int width, int height, int iterations);
    
    bool validate_color_range(const ImageBuffer& buffer, float min_val = 0.0f, float max_val = 255.0f);
    
    __device__ float clamp_float(float value, float min_val, float max_val);
    __device__ unsigned char clamp_uchar(float value);
}

}