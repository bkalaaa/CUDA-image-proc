#include "benchmark.h"
#include <iomanip>
#include <algorithm>

Timer::Timer() : is_running_(false) {}

void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = true;
}

void Timer::stop() {
    if (is_running_) {
        end_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = false;
    }
}

double Timer::elapsed_ms() const {
    auto end = is_running_ ? std::chrono::high_resolution_clock::now() : end_time_;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_);
    return duration.count() / 1000.0;
}

double Timer::elapsed_seconds() const {
    return elapsed_ms() / 1000.0;
}

void Timer::reset() {
    is_running_ = false;
}

void BenchmarkResults::add_operation_time(Operation op, double time_ms) {
    operation_times_[op] += time_ms;
    operation_counts_[op]++;
}

void BenchmarkResults::finalize_results() {
    operation_results.clear();
    
    for (const auto& pair : operation_times_) {
        OperationBenchmark benchmark;
        benchmark.operation = pair.first;
        benchmark.total_time_ms = pair.second;
        benchmark.image_count = operation_counts_[pair.first];
        
        if (benchmark.image_count > 0) {
            benchmark.avg_time_per_image_ms = benchmark.total_time_ms / benchmark.image_count;
            benchmark.throughput_images_per_sec = 1000.0 / benchmark.avg_time_per_image_ms;
        }
        
        operation_results.push_back(benchmark);
    }
    
    std::sort(operation_results.begin(), operation_results.end(),
              [](const OperationBenchmark& a, const OperationBenchmark& b) {
                  return static_cast<int>(a.operation) < static_cast<int>(b.operation);
              });
    
    if (total_processing_time_ms > 0 && total_images > 0) {
        total_throughput_images_per_sec = (total_images * 1000.0) / total_processing_time_ms;
    }
}

void BenchmarkResults::print_summary() const {
    std::cout << "\n=== Benchmark Summary ===\n";
    std::cout << "Mode: " << (mode == ProcessingMode::GPU ? "GPU" : "CPU") << "\n";
    std::cout << "Color: " << (rgb_mode ? "RGB" : "Grayscale") << "\n";
    std::cout << "Total Images: " << total_images << "\n";
    std::cout << "Total Time: " << format_time(total_processing_time_ms) << "\n";
    std::cout << "Overall Throughput: " << format_throughput(total_throughput_images_per_sec) << "\n";
    
    std::cout << "\nPer-Operation Results:\n";
    std::cout << std::left << std::setw(12) << "Operation" 
              << std::setw(12) << "Avg Time" 
              << std::setw(15) << "Throughput" << "\n";
    std::cout << std::string(40, '-') << "\n";
    
    for (const auto& result : operation_results) {
        std::cout << std::left << std::setw(12) << operation_to_string(result.operation)
                  << std::setw(12) << format_time(result.avg_time_per_image_ms)
                  << std::setw(15) << format_throughput(result.throughput_images_per_sec) << "\n";
    }
}

void BenchmarkResults::print_detailed() const {
    std::cout << "\n=== Detailed Benchmark Results ===\n";
    std::cout << "Processing Mode: " << (mode == ProcessingMode::GPU ? "GPU" : "CPU") << "\n";
    std::cout << "Color Mode: " << (rgb_mode ? "RGB" : "Grayscale") << "\n";
    std::cout << "Total Images Processed: " << total_images << "\n";
    std::cout << "Total Processing Time: " << format_time(total_processing_time_ms) << "\n";
    std::cout << "Overall Throughput: " << format_throughput(total_throughput_images_per_sec) << "\n\n";
    
    for (const auto& result : operation_results) {
        std::cout << "Operation: " << operation_to_string(result.operation) << "\n";
        std::cout << "  Images processed: " << result.image_count << "\n";
        std::cout << "  Total time: " << format_time(result.total_time_ms) << "\n";
        std::cout << "  Average time per image: " << format_time(result.avg_time_per_image_ms) << "\n";
        std::cout << "  Throughput: " << format_throughput(result.throughput_images_per_sec) << "\n\n";
    }
}

BenchmarkRunner::BenchmarkRunner(ProcessingMode mode, bool rgb_mode) 
    : mode_(mode), rgb_mode_(rgb_mode), operation_in_progress_(false) {
    results_.mode = mode;
    results_.rgb_mode = rgb_mode;
    results_.total_images = 0;
    results_.total_processing_time_ms = 0.0;
    results_.total_throughput_images_per_sec = 0.0;
}

void BenchmarkRunner::start_operation(Operation op) {
    if (operation_in_progress_) {
        std::cerr << "Warning: Starting new operation while another is in progress\n";
    }
    
    current_operation_ = op;
    operation_timer_.start();
    operation_in_progress_ = true;
}

void BenchmarkRunner::end_operation(Operation op) {
    if (!operation_in_progress_) {
        std::cerr << "Warning: Ending operation that was not started\n";
        return;
    }
    
    if (current_operation_ != op) {
        std::cerr << "Warning: Operation mismatch in timing\n";
    }
    
    operation_timer_.stop();
    double elapsed = operation_timer_.elapsed_ms();
    results_.add_operation_time(op, elapsed);
    operation_in_progress_ = false;
}

void BenchmarkRunner::increment_image_count() {
    results_.total_images++;
}

void BenchmarkRunner::start_total_timer() {
    total_timer_.start();
}

void BenchmarkRunner::end_total_timer() {
    total_timer_.stop();
    results_.total_processing_time_ms = total_timer_.elapsed_ms();
    results_.finalize_results();
}

BenchmarkResults BenchmarkRunner::get_results() const {
    return results_;
}

std::string operation_to_string(Operation op) {
    switch (op) {
        case Operation::GAUSSIAN: return "Gaussian";
        case Operation::SOBEL: return "Sobel";
        case Operation::CANNY: return "Canny";
        case Operation::HISTOGRAM: return "Histogram";
        default: return "Unknown";
    }
}

std::string format_time(double ms) {
    std::ostringstream oss;
    if (ms < 1.0) {
        oss << std::fixed << std::setprecision(3) << ms << " ms";
    } else if (ms < 1000.0) {
        oss << std::fixed << std::setprecision(1) << ms << " ms";
    } else {
        oss << std::fixed << std::setprecision(2) << ms / 1000.0 << " s";
    }
    return oss.str();
}

std::string format_throughput(double images_per_sec) {
    std::ostringstream oss;
    if (images_per_sec < 1.0) {
        oss << std::fixed << std::setprecision(3) << images_per_sec << " img/s";
    } else {
        oss << std::fixed << std::setprecision(1) << images_per_sec << " img/s";
    }
    return oss.str();
}

void BenchmarkResults::print_rgb_comparison(const BenchmarkResults& grayscale_results) const {
    if (!rgb_mode || grayscale_results.rgb_mode) {
        std::cout << "RGB comparison requires RGB results vs grayscale results" << std::endl;
        return;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "RGB vs GRAYSCALE PERFORMANCE COMPARISON" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "Mode: " << (mode == ProcessingMode::CPU ? "CPU" : "GPU") << std::endl;
    std::cout << "Total Images: " << total_images << std::endl;
    
    std::cout << "\nOverall Performance:" << std::endl;
    std::cout << "  RGB Processing:       " << total_processing_time_ms << " ms" << std::endl;
    std::cout << "  Grayscale Processing: " << grayscale_results.total_processing_time_ms << " ms" << std::endl;
    
    double rgb_overhead = (total_processing_time_ms / grayscale_results.total_processing_time_ms - 1.0) * 100.0;
    std::cout << "  RGB Overhead:         " << rgb_overhead << "%" << std::endl;
    
    std::cout << "\nThroughput Comparison:" << std::endl;
    std::cout << "  RGB:                  " << total_throughput_images_per_sec << " images/sec" << std::endl;
    std::cout << "  Grayscale:            " << grayscale_results.total_throughput_images_per_sec << " images/sec" << std::endl;
    
    std::cout << "\nPer-Operation Performance:" << std::endl;
    for (size_t i = 0; i < operation_results.size() && i < grayscale_results.operation_results.size(); ++i) {
        const auto& rgb_op = operation_results[i];
        const auto& gray_op = grayscale_results.operation_results[i];
        
        if (rgb_op.operation == gray_op.operation) {
            std::string op_name = operation_to_string(rgb_op.operation);
            double performance_ratio = rgb_op.avg_time_per_image_ms / gray_op.avg_time_per_image_ms;
            
            std::cout << "  " << op_name << ":" << std::endl;
            std::cout << "    RGB:       " << rgb_op.avg_time_per_image_ms << " ms/image" << std::endl;
            std::cout << "    Grayscale: " << gray_op.avg_time_per_image_ms << " ms/image" << std::endl;
            std::cout << "    Ratio:     " << performance_ratio << "x" << std::endl;
        }
    }
    
    std::cout << "\nMemory and Channel Analysis:" << std::endl;
    std::cout << "  RGB uses 3x memory bandwidth compared to grayscale" << std::endl;
    std::cout << "  Expected theoretical slowdown: ~3x for memory-bound operations" << std::endl;
    std::cout << "  Actual average slowdown: " << (total_processing_time_ms / grayscale_results.total_processing_time_ms) << "x" << std::endl;
    
    std::cout << std::string(60, '=') << std::endl;
}