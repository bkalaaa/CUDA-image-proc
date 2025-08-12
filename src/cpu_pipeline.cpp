#include "cpu_pipeline.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

CPUPipeline::CPUPipeline(bool rgb_mode) : rgb_mode_(rgb_mode) {
    output_directory_ = "output";
    fs::create_directories(output_directory_);
}

void CPUPipeline::set_output_directory(const std::string& output_dir) {
    output_directory_ = output_dir;
    fs::create_directories(output_directory_);
}

bool CPUPipeline::process_single_image(const std::string& input_path, 
                                      const std::set<Operation>& operations) {
    std::cout << "Processing: " << input_path << std::endl;
    
    cv::Mat input_image = load_and_prepare_image(input_path);
    if (input_image.empty()) {
        std::cerr << "Error: Could not load image " << input_path << std::endl;
        return false;
    }
    
    bool success = true;
    for (const auto& operation : operations) {
        cv::Mat result;
        std::string op_name;
        
        switch (operation) {
            case Operation::GAUSSIAN:
                result = apply_gaussian_smoothing(input_image);
                op_name = "Gaussian smoothing";
                break;
            case Operation::SOBEL:
                result = apply_sobel_edge_detection(input_image);
                op_name = "Sobel edge detection";
                break;
            case Operation::CANNY:
                result = apply_canny_edge_detection(input_image);
                op_name = "Canny edge detection";
                break;
            case Operation::HISTOGRAM:
                result = apply_histogram_equalization(input_image);
                op_name = "Histogram equalization";
                break;
        }
        
        if (!result.empty()) {
            std::string output_path = generate_output_filename(input_path, operation);
            if (save_processed_image(result, output_path)) {
                std::cout << "  " << op_name << " -> " << output_path << std::endl;
            } else {
                std::cerr << "  Error saving " << op_name << " result" << std::endl;
                success = false;
            }
        } else {
            std::cerr << "  Error applying " << op_name << std::endl;
            success = false;
        }
    }
    
    return success;
}

bool CPUPipeline::process_batch(const std::string& batch_dir, 
                               const std::set<Operation>& operations) {
    std::vector<std::string> image_files = get_image_files(batch_dir);
    
    if (image_files.empty()) {
        std::cerr << "No supported image files found in " << batch_dir << std::endl;
        return false;
    }
    
    std::cout << "Found " << image_files.size() << " image(s) to process" << std::endl;
    
    bool all_success = true;
    for (const auto& image_path : image_files) {
        if (!process_single_image(image_path, operations)) {
            all_success = false;
        }
    }
    
    return all_success;
}

cv::Mat CPUPipeline::apply_gaussian_smoothing(const cv::Mat& input) {
    cv::Mat result;
    
    int kernel_size = 15;
    double sigma = 3.0;
    
    cv::GaussianBlur(input, result, cv::Size(kernel_size, kernel_size), sigma, sigma);
    
    return result;
}

cv::Mat CPUPipeline::apply_sobel_edge_detection(const cv::Mat& input) {
    cv::Mat gray, grad_x, grad_y, result;
    
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3);
    
    cv::convertScaleAbs(grad_x, grad_x);
    cv::convertScaleAbs(grad_y, grad_y);
    
    cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, result);
    
    if (rgb_mode_ && input.channels() == 3) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
    
    return result;
}

cv::Mat CPUPipeline::apply_canny_edge_detection(const cv::Mat& input) {
    cv::Mat gray, result;
    
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.0);
    
    double low_threshold = 50.0;
    double high_threshold = 150.0;
    
    cv::Canny(gray, result, low_threshold, high_threshold, 3);
    
    if (rgb_mode_ && input.channels() == 3) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
    
    return result;
}

cv::Mat CPUPipeline::apply_histogram_equalization(const cv::Mat& input) {
    cv::Mat result;
    
    if (input.channels() == 1) {
        cv::equalizeHist(input, result);
    } else if (input.channels() == 3) {
        if (rgb_mode_) {
            std::vector<cv::Mat> channels;
            cv::split(input, channels);
            
            for (auto& channel : channels) {
                cv::equalizeHist(channel, channel);
            }
            
            cv::merge(channels, result);
        } else {
            cv::Mat yuv;
            cv::cvtColor(input, yuv, cv::COLOR_BGR2YUV);
            
            std::vector<cv::Mat> channels;
            cv::split(yuv, channels);
            
            cv::equalizeHist(channels[0], channels[0]);
            
            cv::merge(channels, yuv);
            cv::cvtColor(yuv, result, cv::COLOR_YUV2BGR);
        }
    }
    
    return result;
}

std::string CPUPipeline::generate_output_filename(const std::string& input_path, 
                                                 Operation operation) {
    fs::path input_file(input_path);
    std::string base_name = input_file.stem().string();
    std::string extension = input_file.extension().string();
    
    std::string suffix;
    switch (operation) {
        case Operation::GAUSSIAN: suffix = "_gaussian"; break;
        case Operation::SOBEL: suffix = "_sobel"; break;
        case Operation::CANNY: suffix = "_canny"; break;
        case Operation::HISTOGRAM: suffix = "_hist"; break;
    }
    
    std::string output_filename = base_name + suffix + extension;
    return (fs::path(output_directory_) / output_filename).string();
}

std::vector<std::string> CPUPipeline::get_image_files(const std::string& directory) {
    std::vector<std::string> image_files;
    
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file() && 
                is_supported_image_format(entry.path().string())) {
                image_files.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error reading directory " << directory << ": " << e.what() << std::endl;
    }
    
    std::sort(image_files.begin(), image_files.end());
    return image_files;
}

bool CPUPipeline::is_supported_image_format(const std::string& filename) {
    fs::path file_path(filename);
    std::string extension = file_path.extension().string();
    
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    return extension == ".jpg" || extension == ".jpeg" || 
           extension == ".png" || extension == ".bmp" || 
           extension == ".tiff" || extension == ".tif";
}

cv::Mat CPUPipeline::load_and_prepare_image(const std::string& input_path) {
    cv::Mat image;
    
    if (rgb_mode_) {
        image = cv::imread(input_path, cv::IMREAD_COLOR);
    } else {
        cv::Mat color_image = cv::imread(input_path, cv::IMREAD_COLOR);
        if (!color_image.empty()) {
            cv::cvtColor(color_image, image, cv::COLOR_BGR2GRAY);
        }
    }
    
    return image;
}

bool CPUPipeline::save_processed_image(const cv::Mat& image, const std::string& output_path) {
    std::vector<int> compression_params;
    
    fs::path output_file(output_path);
    std::string extension = output_file.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == ".jpg" || extension == ".jpeg") {
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(95);
    } else if (extension == ".png") {
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(3);
    }
    
    return cv::imwrite(output_path, image, compression_params);
}