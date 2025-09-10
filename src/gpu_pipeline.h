#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include "cli.h"
#include "benchmark.h"
#include "cuda_utils.h"
#include "image_io.h"
#include "cuda_streams.h"
#include "gaussian_kernels.h"
#include "sobel_kernels.h"
#include "histogram_kernels.h"

class GPUPipeline {
public:
    explicit GPUPipeline(bool rgb_mode = false, int block_size = 16);
    ~GPUPipeline();
    
    bool initialize();
    void cleanup();
    
    bool process_single_image(const std::string& input_path, 
                             const std::set<Operation>& operations);
    
    bool process_batch(const std::string& batch_dir, 
                      const std::set<Operation>& operations);
    
    bool process_single_image_with_benchmark(const std::string& input_path, 
                                            const std::set<Operation>& operations,
                                            BenchmarkRunner& benchmark);
    
    bool process_batch_with_benchmark(const std::string& batch_dir, 
                                     const std::set<Operation>& operations,
                                     BenchmarkRunner& benchmark);
    
    bool process_batch_with_streaming(const std::string& batch_dir,
                                     const std::set<Operation>& operations,
                                     BenchmarkRunner& benchmark);
    
    void set_output_directory(const std::string& output_dir);
    void set_block_size(int block_size);
    void enable_streaming(bool enable) { streaming_enabled_ = enable; }
    
private:
    bool rgb_mode_;
    int block_size_;
    bool streaming_enabled_;
    std::string output_directory_;
    std::unique_ptr<CudaContext> cuda_context_;
    std::unique_ptr<GaussianKernelManager> gaussian_kernel_;
    std::unique_ptr<SobelKernelManager> sobel_kernel_;
    std::unique_ptr<HistogramKernelManager> histogram_kernel_;
    
    bool process_single_operation(const cv::Mat& input_image, 
                                 const std::string& input_path,
                                 Operation operation,
                                 BenchmarkRunner* benchmark = nullptr);
    
    cv::Mat load_and_prepare_image(const std::string& input_path);
    bool save_processed_image(const cv::Mat& image, const std::string& output_path);
    
    std::string generate_output_filename(const std::string& input_path, 
                                       Operation operation);
    
    std::vector<std::string> get_image_files(const std::string& directory);
    bool is_supported_image_format(const std::string& filename);
    
    bool apply_operation_gpu(const ImageBuffer& input, ImageBuffer& output, Operation operation);
    
    bool apply_gaussian_smoothing_gpu(const ImageBuffer& input, ImageBuffer& output);
    bool apply_sobel_edge_detection_gpu(const ImageBuffer& input, ImageBuffer& output);
    bool apply_canny_edge_detection_gpu(const ImageBuffer& input, ImageBuffer& output);
    bool apply_histogram_equalization_gpu(const ImageBuffer& input, ImageBuffer& output);
};