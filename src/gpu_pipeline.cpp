#include "gpu_pipeline.h"
#include <filesystem>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

GPUPipeline::GPUPipeline(bool rgb_mode, int block_size) 
    : rgb_mode_(rgb_mode), block_size_(block_size), streaming_enabled_(true) {
    output_directory_ = "output";
    fs::create_directories(output_directory_);
    cuda_context_ = std::make_unique<CudaContext>();
    gaussian_kernel_ = std::make_unique<GaussianKernelManager>();
    sobel_kernel_ = std::make_unique<SobelKernelManager>();
}

GPUPipeline::~GPUPipeline() {
    cleanup();
}

bool GPUPipeline::initialize() {
    if (!cuda_context_->initialize()) {
        std::cerr << "Failed to initialize CUDA context for GPU pipeline" << std::endl;
        return false;
    }
    
    if (!gaussian_kernel_->initialize(3.0f)) {
        std::cerr << "Failed to initialize Gaussian kernel" << std::endl;
        return false;
    }
    
    if (!sobel_kernel_->initialize(block_size_)) {
        std::cerr << "Failed to initialize Sobel kernel" << std::endl;
        return false;
    }
    
    std::cout << "GPU Pipeline initialized successfully" << std::endl;
    std::cout << "Block size: " << block_size_ << "x" << block_size_ << std::endl;
    std::cout << "RGB mode: " << (rgb_mode_ ? "enabled" : "disabled") << std::endl;
    gaussian_kernel_->print_kernel_info();
    sobel_kernel_->get_config().print_config();
    
    return true;
}

void GPUPipeline::cleanup() {
    if (gaussian_kernel_) {
        gaussian_kernel_->cleanup();
    }
    if (sobel_kernel_) {
        sobel_kernel_->cleanup();
    }
    if (cuda_context_) {
        cuda_context_->cleanup();
    }
}

void GPUPipeline::set_output_directory(const std::string& output_dir) {
    output_directory_ = output_dir;
    fs::create_directories(output_directory_);
}

void GPUPipeline::set_block_size(int block_size) {
    if (block_size == 16 || block_size == 32) {
        block_size_ = block_size;
    }
}

bool GPUPipeline::process_single_image(const std::string& input_path, 
                                      const std::set<Operation>& operations) {
    std::cout << "Processing (GPU): " << input_path << std::endl;
    
    cv::Mat input_image = load_and_prepare_image(input_path);
    if (input_image.empty()) {
        std::cerr << "Error: Could not load image " << input_path << std::endl;
        return false;
    }
    
    bool success = true;
    for (const auto& operation : operations) {
        if (!process_single_operation(input_image, input_path, operation)) {
            success = false;
        }
    }
    
    return success;
}

bool GPUPipeline::process_batch(const std::string& batch_dir, 
                               const std::set<Operation>& operations) {
    std::vector<std::string> image_files = get_image_files(batch_dir);
    
    if (image_files.empty()) {
        std::cerr << "No supported image files found in " << batch_dir << std::endl;
        return false;
    }
    
    std::cout << "Found " << image_files.size() << " image(s) to process on GPU" << std::endl;
    
    bool all_success = true;
    for (const auto& image_path : image_files) {
        if (!process_single_image(image_path, operations)) {
            all_success = false;
        }
    }
    
    return all_success;
}

bool GPUPipeline::process_single_image_with_benchmark(const std::string& input_path, 
                                                    const std::set<Operation>& operations,
                                                    BenchmarkRunner& benchmark) {
    std::cout << "Processing (GPU): " << input_path << std::endl;
    
    cv::Mat input_image = load_and_prepare_image(input_path);
    if (input_image.empty()) {
        std::cerr << "Error: Could not load image " << input_path << std::endl;
        return false;
    }
    
    benchmark.increment_image_count();
    bool success = true;
    
    for (const auto& operation : operations) {
        if (!process_single_operation(input_image, input_path, operation, &benchmark)) {
            success = false;
        }
    }
    
    return success;
}

bool GPUPipeline::process_batch_with_benchmark(const std::string& batch_dir, 
                                              const std::set<Operation>& operations,
                                              BenchmarkRunner& benchmark) {
    std::vector<std::string> image_files = get_image_files(batch_dir);
    
    if (image_files.empty()) {
        std::cerr << "No supported image files found in " << batch_dir << std::endl;
        return false;
    }
    
    std::cout << "Found " << image_files.size() << " image(s) to process on GPU" << std::endl;
    
    benchmark.start_total_timer();
    
    bool all_success = true;
    for (const auto& image_path : image_files) {
        if (!process_single_image_with_benchmark(image_path, operations, benchmark)) {
            all_success = false;
        }
    }
    
    benchmark.end_total_timer();
    
    return all_success;
}

bool GPUPipeline::process_batch_with_streaming(const std::string& batch_dir,
                                              const std::set<Operation>& operations,
                                              BenchmarkRunner& benchmark) {
    std::vector<std::string> image_files = get_image_files(batch_dir);
    
    if (image_files.empty()) {
        std::cerr << "No supported image files found in " << batch_dir << std::endl;
        return false;
    }
    
    if (!streaming_enabled_ || image_files.size() < 3) {
        std::cout << "Using standard batch processing (streaming disabled or too few images)" << std::endl;
        return process_batch_with_benchmark(batch_dir, operations, benchmark);
    }
    
    std::cout << "Using streaming pipeline for batch processing" << std::endl;
    
    StreamingPipeline streaming_pipeline(rgb_mode_, block_size_);
    streaming_pipeline.set_output_directory(output_directory_);
    
    if (!streaming_pipeline.initialize()) {
        std::cerr << "Failed to initialize streaming pipeline, falling back to standard processing" << std::endl;
        return process_batch_with_benchmark(batch_dir, operations, benchmark);
    }
    
    bool success = streaming_pipeline.process_batch_streamed(image_files, operations, benchmark);
    
    streaming_pipeline.cleanup();
    
    return success;
}

bool GPUPipeline::process_single_operation(const cv::Mat& input_image, 
                                         const std::string& input_path,
                                         Operation operation,
                                         BenchmarkRunner* benchmark) {
    std::string op_name;
    switch (operation) {
        case Operation::GAUSSIAN: op_name = "Gaussian smoothing"; break;
        case Operation::SOBEL: op_name = "Sobel edge detection"; break;
        case Operation::CANNY: op_name = "Canny edge detection"; break;
        case Operation::HISTOGRAM: op_name = "Histogram equalization"; break;
    }
    
    if (benchmark) {
        benchmark->start_operation(operation);
    }
    
    ImageBuffer input_buffer = ImageIO::create_buffer_from_mat(input_image, true, true);
    ImageBuffer output_buffer;
    
    bool operation_success = apply_operation_gpu(input_buffer, output_buffer, operation);
    
    if (benchmark) {
        benchmark->end_operation(operation);
    }
    
    if (!operation_success) {
        std::cerr << "  Error applying " << op_name << " on GPU" << std::endl;
        return false;
    }
    
    cv::Mat result_image;
    if (!ImageIO::transfer_device_to_mat(output_buffer, result_image)) {
        std::cerr << "  Error transferring result from GPU for " << op_name << std::endl;
        return false;
    }
    
    std::string output_path = generate_output_filename(input_path, operation);
    if (save_processed_image(result_image, output_path)) {
        std::cout << "  " << op_name << " (GPU) -> " << output_path << std::endl;
        return true;
    } else {
        std::cerr << "  Error saving " << op_name << " result" << std::endl;
        return false;
    }
}

cv::Mat GPUPipeline::load_and_prepare_image(const std::string& input_path) {
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

bool GPUPipeline::save_processed_image(const cv::Mat& image, const std::string& output_path) {
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

std::string GPUPipeline::generate_output_filename(const std::string& input_path, 
                                                 Operation operation) {
    fs::path input_file(input_path);
    std::string base_name = input_file.stem().string();
    std::string extension = input_file.extension().string();
    
    std::string suffix;
    switch (operation) {
        case Operation::GAUSSIAN: suffix = "_gaussian_gpu"; break;
        case Operation::SOBEL: suffix = "_sobel_gpu"; break;
        case Operation::CANNY: suffix = "_canny_gpu"; break;
        case Operation::HISTOGRAM: suffix = "_hist_gpu"; break;
    }
    
    std::string output_filename = base_name + suffix + extension;
    return (fs::path(output_directory_) / output_filename).string();
}

std::vector<std::string> GPUPipeline::get_image_files(const std::string& directory) {
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

bool GPUPipeline::is_supported_image_format(const std::string& filename) {
    fs::path file_path(filename);
    std::string extension = file_path.extension().string();
    
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    return extension == ".jpg" || extension == ".jpeg" || 
           extension == ".png" || extension == ".bmp" || 
           extension == ".tiff" || extension == ".tif";
}

bool GPUPipeline::apply_operation_gpu(const ImageBuffer& input, ImageBuffer& output, Operation operation) {
    const ImageMetadata& metadata = input.get_metadata();
    
    if (!output.allocate_device(metadata)) {
        return false;
    }
    
    switch (operation) {
        case Operation::GAUSSIAN:
            return apply_gaussian_smoothing_gpu(input, output);
        case Operation::SOBEL:
            return apply_sobel_edge_detection_gpu(input, output);
        case Operation::CANNY:
            return apply_canny_edge_detection_gpu(input, output);
        case Operation::HISTOGRAM:
            return apply_histogram_equalization_gpu(input, output);
        default:
            return false;
    }
}

bool GPUPipeline::apply_gaussian_smoothing_gpu(const ImageBuffer& input, ImageBuffer& output) {
    if (!gaussian_kernel_) {
        std::cerr << "Gaussian kernel not initialized" << std::endl;
        return false;
    }
    
    return gaussian_kernel_->apply_gaussian_separable_shared(input, output);
}

bool GPUPipeline::apply_sobel_edge_detection_gpu(const ImageBuffer& input, ImageBuffer& output) {
    if (!sobel_kernel_) {
        std::cerr << "Sobel kernel not initialized" << std::endl;
        return false;
    }
    
    return sobel_kernel_->apply_sobel_shared_fused(input, output);
}

bool GPUPipeline::apply_canny_edge_detection_gpu(const ImageBuffer& input, ImageBuffer& output) {
    std::cout << "    GPU Canny edge detection not yet implemented, using placeholder" << std::endl;
    
    const ImageMetadata& metadata = input.get_metadata();
    
    if (!output.allocate_host(metadata)) {
        return false;
    }
    
    if (!input.copy_device_to_host()) {
        return false;
    }
    
    std::memcpy(output.get_host_ptr(), input.get_host_ptr(), metadata.total_bytes);
    
    return output.copy_host_to_device();
}

bool GPUPipeline::apply_histogram_equalization_gpu(const ImageBuffer& input, ImageBuffer& output) {
    std::cout << "    GPU Histogram equalization not yet implemented, using placeholder" << std::endl;
    
    const ImageMetadata& metadata = input.get_metadata();
    
    if (!output.allocate_host(metadata)) {
        return false;
    }
    
    if (!input.copy_device_to_host()) {
        return false;
    }
    
    std::memcpy(output.get_host_ptr(), input.get_host_ptr(), metadata.total_bytes);
    
    return output.copy_host_to_device();
}