#pragma once

#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include "cli.h"

class Timer {
public:
    Timer();
    
    void start();
    void stop();
    double elapsed_ms() const;
    double elapsed_seconds() const;
    
    void reset();
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool is_running_;
};

struct OperationBenchmark {
    Operation operation;
    double total_time_ms;
    int image_count;
    double avg_time_per_image_ms;
    double throughput_images_per_sec;
    
    OperationBenchmark() : operation(Operation::GAUSSIAN), total_time_ms(0.0), 
                          image_count(0), avg_time_per_image_ms(0.0), 
                          throughput_images_per_sec(0.0) {}
};

struct BenchmarkResults {
    ProcessingMode mode;
    bool rgb_mode;
    int total_images;
    double total_processing_time_ms;
    double total_throughput_images_per_sec;
    std::vector<OperationBenchmark> operation_results;
    
    void add_operation_time(Operation op, double time_ms);
    void finalize_results();
    void print_summary() const;
    void print_detailed() const;
    
private:
    std::map<Operation, double> operation_times_;
    std::map<Operation, int> operation_counts_;
};

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(ProcessingMode mode, bool rgb_mode);
    
    void start_operation(Operation op);
    void end_operation(Operation op);
    
    void increment_image_count();
    void start_total_timer();
    void end_total_timer();
    
    BenchmarkResults get_results() const;
    
private:
    ProcessingMode mode_;
    bool rgb_mode_;
    Timer total_timer_;
    Timer operation_timer_;
    Operation current_operation_;
    BenchmarkResults results_;
    bool operation_in_progress_;
};

std::string operation_to_string(Operation op);
std::string format_time(double ms);
std::string format_throughput(double images_per_sec);