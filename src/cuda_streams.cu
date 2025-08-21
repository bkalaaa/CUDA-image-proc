#include "cuda_streams.h"
#include "gaussian_kernels.h"
#include "sobel_kernels.h"
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

CudaStream::CudaStream() : stream_(nullptr) {}

CudaStream::~CudaStream() {
    destroy();
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        destroy();
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

bool CudaStream::create() {
    if (stream_ != nullptr) {
        return true;
    }
    
    cudaError_t error = cudaStreamCreate(&stream_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(error) << std::endl;
        stream_ = nullptr;
        return false;
    }
    
    return true;
}

void CudaStream::destroy() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void CudaStream::synchronize() const {
    if (stream_ != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
}

bool CudaStream::is_complete() const {
    if (stream_ == nullptr) {
        return true;
    }
    
    cudaError_t error = cudaStreamQuery(stream_);
    return error == cudaSuccess;
}

void CudaStream::record_event(cudaEvent_t event) const {
    if (stream_ != nullptr) {
        CUDA_CHECK(cudaEventRecord(event, stream_));
    }
}

void CudaStream::wait_event(cudaEvent_t event) const {
    if (stream_ != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(stream_, event, 0));
    }
}

StreamManager::StreamManager(int num_streams) : initialized_(false) {
    streams_.reserve(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        streams_.push_back(std::make_unique<CudaStream>());
    }
}

StreamManager::~StreamManager() {
    cleanup();
}

bool StreamManager::initialize() {
    if (initialized_) {
        return true;
    }
    
    for (auto& stream : streams_) {
        if (!stream->create()) {
            std::cerr << "Failed to create stream in StreamManager" << std::endl;
            cleanup();
            return false;
        }
    }
    
    initialized_ = true;
    std::cout << "Created " << streams_.size() << " CUDA streams" << std::endl;
    return true;
}

void StreamManager::cleanup() {
    for (auto& stream : streams_) {
        if (stream) {
            stream->destroy();
        }
    }
    initialized_ = false;
}

CudaStream& StreamManager::get_stream(int index) {
    if (index < 0 || index >= static_cast<int>(streams_.size())) {
        std::cerr << "Invalid stream index: " << index << std::endl;
        return *streams_[0];
    }
    return *streams_[index];
}

const CudaStream& StreamManager::get_stream(int index) const {
    if (index < 0 || index >= static_cast<int>(streams_.size())) {
        std::cerr << "Invalid stream index: " << index << std::endl;
        return *streams_[0];
    }
    return *streams_[index];
}

void StreamManager::synchronize_all() {
    for (const auto& stream : streams_) {
        if (stream && stream->is_valid()) {
            stream->synchronize();
        }
    }
}

bool StreamManager::all_complete() const {
    for (const auto& stream : streams_) {
        if (stream && stream->is_valid() && !stream->is_complete()) {
            return false;
        }
    }
    return true;
}

PipelineStage::PipelineStage() : type(TRANSFER_H2D), operation(Operation::GAUSSIAN), 
                                start_event(nullptr), end_event(nullptr), completed(false) {}

PipelineStage::~PipelineStage() {
    destroy_events();
}

void PipelineStage::create_events() {
    if (start_event == nullptr) {
        CUDA_CHECK(cudaEventCreate(&start_event));
    }
    if (end_event == nullptr) {
        CUDA_CHECK(cudaEventCreate(&end_event));
    }
}

void PipelineStage::destroy_events() {
    if (start_event != nullptr) {
        cudaEventDestroy(start_event);
        start_event = nullptr;
    }
    if (end_event != nullptr) {
        cudaEventDestroy(end_event);
        end_event = nullptr;
    }
}

void PipelineStage::record_start(cudaStream_t stream) {
    if (start_event != nullptr) {
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }
}

void PipelineStage::record_end(cudaStream_t stream) {
    if (end_event != nullptr) {
        CUDA_CHECK(cudaEventRecord(end_event, stream));
    }
}

float PipelineStage::elapsed_time() const {
    if (start_event == nullptr || end_event == nullptr) {
        return 0.0f;
    }
    
    float elapsed = 0.0f;
    cudaError_t error = cudaEventElapsedTime(&elapsed, start_event, end_event);
    return (error == cudaSuccess) ? elapsed : 0.0f;
}

StreamingPipeline::StreamingPipeline(bool rgb_mode, int block_size) 
    : rgb_mode_(rgb_mode), block_size_(block_size) {
    output_directory_ = "output";
    fs::create_directories(output_directory_);
    
    stream_manager_ = std::make_unique<StreamManager>(3);
    cuda_context_ = std::make_unique<CudaContext>();
}

StreamingPipeline::~StreamingPipeline() {
    cleanup();
}

bool StreamingPipeline::initialize() {
    if (!cuda_context_->initialize()) {
        std::cerr << "Failed to initialize CUDA context for streaming pipeline" << std::endl;
        return false;
    }
    
    if (!stream_manager_->initialize()) {
        std::cerr << "Failed to initialize stream manager" << std::endl;
        return false;
    }
    
    std::cout << "Streaming pipeline initialized successfully" << std::endl;
    std::cout << "Using 3-stage pipeline: H2D -> Compute -> D2H" << std::endl;
    return true;
}

void StreamingPipeline::cleanup() {
    wait_for_completion();
    cleanup_buffer_pool();
    
    if (stream_manager_) {
        stream_manager_->cleanup();
    }
    if (cuda_context_) {
        cuda_context_->cleanup();
    }
}

void StreamingPipeline::set_output_directory(const std::string& output_dir) {
    output_directory_ = output_dir;
    fs::create_directories(output_directory_);
}

bool StreamingPipeline::process_batch_streamed(const std::vector<std::string>& image_paths,
                                              const std::set<Operation>& operations,
                                              BenchmarkRunner& benchmark) {
    if (image_paths.empty() || operations.empty()) {
        return false;
    }
    
    cv::Mat first_image = load_and_prepare_image(image_paths[0]);
    if (first_image.empty()) {
        std::cerr << "Failed to load first image for metadata extraction" << std::endl;
        return false;
    }
    
    ImageMetadata metadata = ImageConverter::extract_metadata(first_image);
    
    size_t buffer_pool_size = std::min(static_cast<size_t>(4), image_paths.size());
    if (!create_buffer_pool(buffer_pool_size, metadata)) {
        std::cerr << "Failed to create buffer pool" << std::endl;
        return false;
    }
    
    std::cout << "Processing " << image_paths.size() << " images with " 
              << operations.size() << " operations using streaming pipeline" << std::endl;
    
    benchmark.start_total_timer();
    
    bool all_success = true;
    
    for (const auto& image_path : image_paths) {
        benchmark.increment_image_count();
        
        for (const auto& operation : operations) {
            BufferPair* buffers = acquire_buffer_pair();
            if (!buffers) {
                process_completed_stages(benchmark);
                buffers = acquire_buffer_pair();
                if (!buffers) {
                    std::cerr << "Failed to acquire buffer pair" << std::endl;
                    all_success = false;
                    continue;
                }
            }
            
            PipelineStage h2d_stage;
            h2d_stage.type = PipelineStage::TRANSFER_H2D;
            h2d_stage.image_path = image_path;
            h2d_stage.operation = operation;
            h2d_stage.input_buffer = buffers->input;
            h2d_stage.output_buffer = buffers->output;
            h2d_stage.create_events();
            
            if (!launch_h2d_transfer(image_path, buffers, h2d_stage)) {
                std::cerr << "Failed to launch H2D transfer for " << image_path << std::endl;
                release_buffer_pair(buffers);
                all_success = false;
                continue;
            }
            
            active_stages_.push_back(std::move(h2d_stage));
        }
        
        process_completed_stages(benchmark);
    }
    
    wait_for_completion();
    process_completed_stages(benchmark);
    
    benchmark.end_total_timer();
    
    std::cout << "Streaming pipeline processing completed" << std::endl;
    return all_success;
}

bool StreamingPipeline::create_buffer_pool(size_t pool_size, const ImageMetadata& metadata) {
    cleanup_buffer_pool();
    
    buffer_pool_.resize(pool_size);
    
    for (auto& pair : buffer_pool_) {
        pair.input = std::make_shared<ImageBuffer>();
        pair.output = std::make_shared<ImageBuffer>();
        
        if (!pair.input->allocate_device(metadata) || 
            !pair.input->allocate_pinned(metadata) ||
            !pair.output->allocate_device(metadata) ||
            !pair.output->allocate_pinned(metadata)) {
            std::cerr << "Failed to allocate buffer pair in pool" << std::endl;
            cleanup_buffer_pool();
            return false;
        }
        
        pair.in_use = false;
    }
    
    std::cout << "Created buffer pool with " << pool_size << " pairs" << std::endl;
    return true;
}

void StreamingPipeline::cleanup_buffer_pool() {
    for (auto& pair : buffer_pool_) {
        if (pair.input) {
            pair.input->free_all();
        }
        if (pair.output) {
            pair.output->free_all();
        }
    }
    buffer_pool_.clear();
}

StreamingPipeline::BufferPair* StreamingPipeline::acquire_buffer_pair() {
    for (auto& pair : buffer_pool_) {
        if (!pair.in_use) {
            pair.in_use = true;
            return &pair;
        }
    }
    return nullptr;
}

void StreamingPipeline::release_buffer_pair(BufferPair* pair) {
    if (pair) {
        pair->in_use = false;
    }
}

bool StreamingPipeline::launch_h2d_transfer(const std::string& image_path, 
                                           BufferPair* buffers,
                                           PipelineStage& stage) {
    cv::Mat image = load_and_prepare_image(image_path);
    if (image.empty()) {
        return false;
    }
    
    if (!ImageConverter::mat_to_pinned_buffer(image, *buffers->input)) {
        return false;
    }
    
    stage.record_start(stream_manager_->get_h2d_stream().get());
    
    bool success = buffers->input->copy_pinned_to_device_async(
        stream_manager_->get_h2d_stream().get());
    
    stage.record_end(stream_manager_->get_h2d_stream().get());
    
    return success;
}

bool StreamingPipeline::launch_compute_operation(Operation operation,
                                                BufferPair* buffers,
                                                PipelineStage& stage) {
    stage.record_start(stream_manager_->get_compute_stream().get());
    
    bool success = apply_operation_gpu_streamed(*buffers->input, *buffers->output, 
                                               operation, 
                                               stream_manager_->get_compute_stream().get());
    
    stage.record_end(stream_manager_->get_compute_stream().get());
    
    return success;
}

bool StreamingPipeline::launch_d2h_transfer(const std::string& output_path,
                                           BufferPair* buffers,
                                           PipelineStage& stage) {
    stage.record_start(stream_manager_->get_d2h_stream().get());
    
    bool success = buffers->output->copy_device_to_pinned_async(
        stream_manager_->get_d2h_stream().get());
    
    stage.record_end(stream_manager_->get_d2h_stream().get());
    
    return success;
}

void StreamingPipeline::process_completed_stages(BenchmarkRunner& benchmark) {
    auto it = active_stages_.begin();
    while (it != active_stages_.end()) {
        bool stage_complete = false;
        
        switch (it->type) {
            case PipelineStage::TRANSFER_H2D:
                stage_complete = stream_manager_->get_h2d_stream().is_complete();
                break;
            case PipelineStage::COMPUTE:
                stage_complete = stream_manager_->get_compute_stream().is_complete();
                break;
            case PipelineStage::TRANSFER_D2H:
                stage_complete = stream_manager_->get_d2h_stream().is_complete();
                break;
        }
        
        if (stage_complete) {
            benchmark.start_operation(it->operation);
            
            float elapsed = it->elapsed_time();
            
            benchmark.end_operation(it->operation);
            
            BufferPair* buffers = nullptr;
            for (auto& pair : buffer_pool_) {
                if (pair.input == it->input_buffer && pair.output == it->output_buffer) {
                    buffers = &pair;
                    break;
                }
            }
            
            if (buffers) {
                release_buffer_pair(buffers);
            }
            
            it = active_stages_.erase(it);
        } else {
            ++it;
        }
    }
}

void StreamingPipeline::wait_for_completion() {
    stream_manager_->synchronize_all();
    
    for (auto& stage : active_stages_) {
        for (auto& pair : buffer_pool_) {
            if (pair.input == stage.input_buffer && pair.output == stage.output_buffer) {
                release_buffer_pair(&pair);
                break;
            }
        }
    }
    
    active_stages_.clear();
}

cv::Mat StreamingPipeline::load_and_prepare_image(const std::string& input_path) {
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

bool StreamingPipeline::save_processed_image(const cv::Mat& image, const std::string& output_path) {
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

std::string StreamingPipeline::generate_output_filename(const std::string& input_path, 
                                                       Operation operation) {
    fs::path input_file(input_path);
    std::string base_name = input_file.stem().string();
    std::string extension = input_file.extension().string();
    
    std::string suffix;
    switch (operation) {
        case Operation::GAUSSIAN: suffix = "_gaussian_stream"; break;
        case Operation::SOBEL: suffix = "_sobel_stream"; break;
        case Operation::CANNY: suffix = "_canny_stream"; break;
        case Operation::HISTOGRAM: suffix = "_hist_stream"; break;
    }
    
    std::string output_filename = base_name + suffix + extension;
    return (fs::path(output_directory_) / output_filename).string();
}

bool StreamingPipeline::apply_operation_gpu_streamed(const ImageBuffer& input, 
                                                    ImageBuffer& output, 
                                                    Operation operation,
                                                    cudaStream_t stream) {
    const ImageMetadata& metadata = input.get_metadata();
    
    switch (operation) {
        case Operation::GAUSSIAN: {
            GaussianKernelManager gaussian_kernel;
            if (!gaussian_kernel.initialize(3.0f)) {
                return false;
            }
            return gaussian_kernel.apply_gaussian_separable_shared(input, output, stream);
        }
        case Operation::SOBEL: {
            SobelKernelManager sobel_kernel;
            if (!sobel_kernel.initialize()) {
                return false;
            }
            return sobel_kernel.apply_sobel_shared_fused(input, output, stream);
        }
        case Operation::CANNY:
        case Operation::HISTOGRAM:
        default:
            std::cout << "      Stream GPU operation placeholder for " 
                      << operation_to_string(operation) << std::endl;
            
            cudaError_t error = cudaMemcpyAsync(output.get_device_ptr(), 
                                               input.get_device_ptr(),
                                               metadata.total_bytes,
                                               cudaMemcpyDeviceToDevice,
                                               stream);
            
            return error == cudaSuccess;
    }
}

StreamBenchmark::StreamBenchmark() : current_stage_(PipelineStage::TRANSFER_H2D) {}

StreamBenchmark::~StreamBenchmark() {}

void StreamBenchmark::start_stage_timing(PipelineStage::Type stage_type) {
    current_stage_ = stage_type;
    stage_timer_.start();
}

void StreamBenchmark::end_stage_timing(PipelineStage::Type stage_type) {
    if (current_stage_ == stage_type) {
        stage_timer_.stop();
    }
}

void StreamBenchmark::record_h2d_transfer(float time_ms, size_t bytes) {
    h2d_stats_.total_time_ms += time_ms;
    h2d_stats_.total_bytes += bytes;
    h2d_stats_.transfer_count++;
}

void StreamBenchmark::record_compute_operation(Operation op, float time_ms) {
    compute_stats_[op].total_time_ms += time_ms;
    compute_stats_[op].operation_count++;
}

void StreamBenchmark::record_d2h_transfer(float time_ms, size_t bytes) {
    d2h_stats_.total_time_ms += time_ms;
    d2h_stats_.total_bytes += bytes;
    d2h_stats_.transfer_count++;
}

void StreamBenchmark::print_streaming_summary() const {
    std::cout << "\n=== Streaming Pipeline Performance ===\n";
    
    std::cout << "H2D Transfers:\n";
    std::cout << "  Count: " << h2d_stats_.transfer_count << "\n";
    std::cout << "  Total time: " << h2d_stats_.total_time_ms << " ms\n";
    std::cout << "  Bandwidth: " << h2d_stats_.get_bandwidth_gbps() << " GB/s\n";
    
    std::cout << "D2H Transfers:\n";
    std::cout << "  Count: " << d2h_stats_.transfer_count << "\n";
    std::cout << "  Total time: " << d2h_stats_.total_time_ms << " ms\n";
    std::cout << "  Bandwidth: " << d2h_stats_.get_bandwidth_gbps() << " GB/s\n";
    
    std::cout << "Compute Operations:\n";
    for (const auto& pair : compute_stats_) {
        const auto& stats = pair.second;
        std::cout << "  " << operation_to_string(pair.first) << ": "
                  << stats.operation_count << " ops, "
                  << stats.total_time_ms << " ms total\n";
    }
}

namespace StreamUtils {
    bool transfer_async_with_fallback(const ImageBuffer& src, ImageBuffer& dst, 
                                     cudaStream_t stream) {
        if (src.has_pinned_data() && dst.has_device_data()) {
            return dst.copy_from_host_async(src.get_pinned_ptr(), 
                                           src.get_metadata().total_bytes, 
                                           stream);
        } else if (src.has_device_data() && dst.has_pinned_data()) {
            return src.copy_to_host_async(dst.get_pinned_ptr(), 
                                         dst.get_metadata().total_bytes, 
                                         stream);
        }
        
        return false;
    }
    
    float measure_transfer_bandwidth(size_t bytes, float time_ms) {
        if (time_ms <= 0.0f) return 0.0f;
        return (bytes / (1024.0f * 1024.0f * 1024.0f)) / (time_ms / 1000.0f);
    }
    
    void print_stream_info(const StreamManager& manager) {
        std::cout << "Stream Manager Info:\n";
        std::cout << "  Total streams: " << manager.get_stream_count() << "\n";
        std::cout << "  All complete: " << (manager.all_complete() ? "Yes" : "No") << "\n";
    }
}