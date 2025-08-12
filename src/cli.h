#pragma once

#include <string>
#include <vector>
#include <set>

enum class ProcessingMode { CPU, GPU };

enum class Operation { GAUSSIAN, SOBEL, CANNY, HISTOGRAM };

struct CLIArgs {
    ProcessingMode mode = ProcessingMode::GPU;
    std::set<Operation> operations;
    std::string batch_directory;
    std::string single_image;
    bool rgb_mode = false;
    bool show_help = false;
    int block_size = 16;
    
    bool validate() const;
    void print_usage() const;
};

class CLIParser {
public:
    static CLIArgs parse(int argc, char* argv[]);
    
private:
    static ProcessingMode parse_mode(const std::string& mode_str);
    static std::set<Operation> parse_operations(const std::string& ops_str);
    static void print_help();
};