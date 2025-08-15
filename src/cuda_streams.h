#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <queue>
#include <functional>
#include "cuda_utils.h"
#include "image_io.h"
#include "benchmark.h"

class CudaStream {
public:
    CudaStream();
    ~CudaStream();
    
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;
    
    bool create();
    void destroy();
    
    cudaStream_t get() const { return stream_; }
    bool is_valid() const { return stream_ != nullptr; }
    
    void synchronize() const;
    bool is_complete() const;
    
    void record_event(cudaEvent_t event) const;
    void wait_event(cudaEvent_t event) const;
    
private:
    cudaStream_t stream_;
};

class StreamManager {
public:
    explicit StreamManager(int num_streams = 3);
    ~StreamManager();
    
    bool initialize();
    void cleanup();
    
    CudaStream& get_stream(int index);
    const CudaStream& get_stream(int index) const;
    
    int get_stream_count() const { return streams_.size(); }
    
    void synchronize_all();
    bool all_complete() const;
    
    CudaStream& get_h2d_stream() { return get_stream(0); }
    CudaStream& get_compute_stream() { return get_stream(1); }
    CudaStream& get_d2h_stream() { return get_stream(2); }
    
private:
    std::vector<std::unique_ptr<CudaStream>> streams_;
    bool initialized_;
};

struct PipelineStage {
    enum Type { TRANSFER_H2D, COMPUTE, TRANSFER_D2H };
    
    Type type;
    std::string image_path;
    Operation operation;
    std::shared_ptr<ImageBuffer> input_buffer;
    std::shared_ptr<ImageBuffer> output_buffer;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    bool completed;
    
    PipelineStage();
    ~PipelineStage();
    
    void create_events();
    void destroy_events();
    
    void record_start(cudaStream_t stream);
    void record_end(cudaStream_t stream);
    float elapsed_time() const;
};

class StreamingPipeline {
public:
    explicit StreamingPipeline(bool rgb_mode = false, int block_size = 16);
    ~StreamingPipeline();
    
    bool initialize();
    void cleanup();
    
    bool process_batch_streamed(const std::vector<std::string>& image_paths,
                               const std::set<Operation>& operations,
                               BenchmarkRunner& benchmark);
    
    void set_output_directory(const std::string& output_dir);
    
private:
    bool rgb_mode_;
    int block_size_;
    std::string output_directory_;
    
    std::unique_ptr<StreamManager> stream_manager_;
    std::unique_ptr<CudaContext> cuda_context_;
    
    struct BufferPair {
        std::shared_ptr<ImageBuffer> input;
        std::shared_ptr<ImageBuffer> output;
        bool in_use;
        
        BufferPair() : in_use(false) {}
    };
    
    std::vector<BufferPair> buffer_pool_;
    std::queue<PipelineStage> pending_stages_;
    std::vector<PipelineStage> active_stages_;
    
    bool create_buffer_pool(size_t pool_size, const ImageMetadata& metadata);
    void cleanup_buffer_pool();
    
    BufferPair* acquire_buffer_pair();
    void release_buffer_pair(BufferPair* pair);
    
    bool launch_h2d_transfer(const std::string& image_path, 
                            BufferPair* buffers,
                            PipelineStage& stage);
    
    bool launch_compute_operation(Operation operation,
                                 BufferPair* buffers,
                                 PipelineStage& stage);
    
    bool launch_d2h_transfer(const std::string& output_path,
                            BufferPair* buffers,
                            PipelineStage& stage);
    
    void process_completed_stages(BenchmarkRunner& benchmark);
    void wait_for_completion();
    
    cv::Mat load_and_prepare_image(const std::string& input_path);
    bool save_processed_image(const cv::Mat& image, const std::string& output_path);
    
    std::string generate_output_filename(const std::string& input_path, 
                                       Operation operation);
    
    bool apply_operation_gpu_streamed(const ImageBuffer& input, 
                                     ImageBuffer& output, 
                                     Operation operation,
                                     cudaStream_t stream);
};

class StreamBenchmark {
public:
    StreamBenchmark();
    ~StreamBenchmark();
    
    void start_stage_timing(PipelineStage::Type stage_type);
    void end_stage_timing(PipelineStage::Type stage_type);
    
    void record_h2d_transfer(float time_ms, size_t bytes);
    void record_compute_operation(Operation op, float time_ms);
    void record_d2h_transfer(float time_ms, size_t bytes);
    
    void print_streaming_summary() const;
    
private:
    struct TransferStats {
        float total_time_ms;
        size_t total_bytes;
        int transfer_count;
        
        TransferStats() : total_time_ms(0.0f), total_bytes(0), transfer_count(0) {}
        
        float get_bandwidth_gbps() const {
            if (total_time_ms <= 0.0f) return 0.0f;
            return (total_bytes / (1024.0f * 1024.0f * 1024.0f)) / (total_time_ms / 1000.0f);
        }
    };
    
    struct ComputeStats {
        float total_time_ms;
        int operation_count;
        
        ComputeStats() : total_time_ms(0.0f), operation_count(0) {}
    };
    
    TransferStats h2d_stats_;
    TransferStats d2h_stats_;
    std::map<Operation, ComputeStats> compute_stats_;
    
    Timer stage_timer_;
    PipelineStage::Type current_stage_;
};

namespace StreamUtils {
    bool transfer_async_with_fallback(const ImageBuffer& src, ImageBuffer& dst, 
                                     cudaStream_t stream);
    
    void insert_stream_callback(cudaStream_t stream, 
                               std::function<void()> callback);
    
    float measure_transfer_bandwidth(size_t bytes, float time_ms);
    
    void print_stream_info(const StreamManager& manager);
}