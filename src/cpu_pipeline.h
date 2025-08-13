#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "cli.h"
#include "benchmark.h"

class CPUPipeline {
public:
    explicit CPUPipeline(bool rgb_mode = false);
    
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
    
    void set_output_directory(const std::string& output_dir);
    
private:
    bool rgb_mode_;
    std::string output_directory_;
    
    cv::Mat apply_gaussian_smoothing(const cv::Mat& input);
    cv::Mat apply_sobel_edge_detection(const cv::Mat& input);
    cv::Mat apply_canny_edge_detection(const cv::Mat& input);
    cv::Mat apply_histogram_equalization(const cv::Mat& input);
    
    std::string generate_output_filename(const std::string& input_path, 
                                       Operation operation);
    
    std::vector<std::string> get_image_files(const std::string& directory);
    
    bool is_supported_image_format(const std::string& filename);
    
    cv::Mat load_and_prepare_image(const std::string& input_path);
    bool save_processed_image(const cv::Mat& image, const std::string& output_path);
};