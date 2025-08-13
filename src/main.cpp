#include <iostream>
#include <opencv2/opencv.hpp>
#include "cli.h"
#include "cpu_pipeline.h"
#include "gpu_pipeline.h"
#include "benchmark.h"
#include "cuda_utils.h"
#include "image_io.h"

int main(int argc, char* argv[]) {
    CLIArgs args = CLIParser::parse(argc, argv);
    
    if (args.show_help) {
        args.print_usage();
        return 0;
    }
    
    if (!args.validate()) {
        std::cerr << "\nUse --help for usage information.\n";
        return 1;
    }
    
    std::cout << "CUDA Image Processing Application\n";
    std::cout << "=================================\n";
    
    std::cout << "Processing Mode: " << (args.mode == ProcessingMode::GPU ? "GPU" : "CPU") << "\n";
    std::cout << "Color Mode: " << (args.rgb_mode ? "RGB" : "Grayscale") << "\n";
    
    std::cout << "Operations: ";
    bool first = true;
    for (const auto& op : args.operations) {
        if (!first) std::cout << ", ";
        switch (op) {
            case Operation::GAUSSIAN: std::cout << "Gaussian"; break;
            case Operation::SOBEL: std::cout << "Sobel"; break;
            case Operation::CANNY: std::cout << "Canny"; break;
            case Operation::HISTOGRAM: std::cout << "Histogram"; break;
        }
        first = false;
    }
    std::cout << "\n";
    
    if (!args.batch_directory.empty()) {
        std::cout << "Batch Directory: " << args.batch_directory << "\n";
    } else {
        std::cout << "Single Image: " << args.single_image << "\n";
    }
    
    if (args.mode == ProcessingMode::GPU) {
        std::cout << "CUDA Block Size: " << args.block_size << "x" << args.block_size << "\n";
    }
    
    std::cout << "\nOpenCV Version: " << cv::getVersionString() << "\n\n";
    
    bool processing_success = false;
    
    if (args.mode == ProcessingMode::CPU) {
        std::cout << "Starting CPU processing with benchmarking...\n";
        
        CPUPipeline cpu_pipeline(args.rgb_mode);
        BenchmarkRunner benchmark(args.mode, args.rgb_mode);
        
        if (!args.batch_directory.empty()) {
            processing_success = cpu_pipeline.process_batch_with_benchmark(
                args.batch_directory, args.operations, benchmark);
        } else {
            benchmark.start_total_timer();
            processing_success = cpu_pipeline.process_single_image_with_benchmark(
                args.single_image, args.operations, benchmark);
            benchmark.end_total_timer();
        }
        
        if (processing_success) {
            std::cout << "\nCPU processing completed successfully!\n";
            std::cout << "Output files saved to: output/\n";
            
            BenchmarkResults results = benchmark.get_results();
            results.print_summary();
        } else {
            std::cout << "\nCPU processing completed with errors.\n";
            BenchmarkResults results = benchmark.get_results();
            if (results.total_images > 0) {
                results.print_summary();
            }
        }
    } else {
        std::cout << "Initializing CUDA for GPU processing...\n";
        
        CudaContext cuda_context;
        if (!cuda_context.initialize()) {
            std::cerr << "Failed to initialize CUDA. Falling back to CPU mode.\n";
            
            CPUPipeline cpu_pipeline(args.rgb_mode);
            BenchmarkRunner benchmark(ProcessingMode::CPU, args.rgb_mode);
            
            if (!args.batch_directory.empty()) {
                processing_success = cpu_pipeline.process_batch_with_benchmark(
                    args.batch_directory, args.operations, benchmark);
            } else {
                benchmark.start_total_timer();
                processing_success = cpu_pipeline.process_single_image_with_benchmark(
                    args.single_image, args.operations, benchmark);
                benchmark.end_total_timer();
            }
            
            if (processing_success) {
                std::cout << "\nCPU fallback processing completed successfully!\n";
                std::cout << "Output files saved to: output/\n";
                BenchmarkResults results = benchmark.get_results();
                results.print_summary();
            }
        } else {
            std::cout << "Starting GPU processing with I/O pipeline...\n";
            
            GPUPipeline gpu_pipeline(args.rgb_mode, args.block_size);
            BenchmarkRunner benchmark(args.mode, args.rgb_mode);
            
            if (!gpu_pipeline.initialize()) {
                std::cerr << "Failed to initialize GPU pipeline\n";
                cuda_context.cleanup();
                return 1;
            }
            
            if (!args.batch_directory.empty()) {
                processing_success = gpu_pipeline.process_batch_with_benchmark(
                    args.batch_directory, args.operations, benchmark);
            } else {
                benchmark.start_total_timer();
                processing_success = gpu_pipeline.process_single_image_with_benchmark(
                    args.single_image, args.operations, benchmark);
                benchmark.end_total_timer();
            }
            
            if (processing_success) {
                std::cout << "\nGPU processing completed successfully!\n";
                std::cout << "Output files saved to: output/\n";
                
                BenchmarkResults results = benchmark.get_results();
                results.print_summary();
            } else {
                std::cout << "\nGPU processing completed with errors.\n";
                BenchmarkResults results = benchmark.get_results();
                if (results.total_images > 0) {
                    results.print_summary();
                }
            }
            
            gpu_pipeline.cleanup();
        }
    }
    
    return processing_success ? 0 : 1;
}