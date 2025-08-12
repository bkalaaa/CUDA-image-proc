#include "cli.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;

bool CLIArgs::validate() const {
    if (show_help) {
        return true;
    }
    
    if (operations.empty()) {
        std::cerr << "Error: No operations specified. Use --ops to specify operations.\n";
        return false;
    }
    
    if (batch_directory.empty() && single_image.empty()) {
        std::cerr << "Error: No input specified. Use --batch <dir> or provide a single image path.\n";
        return false;
    }
    
    if (!batch_directory.empty() && !single_image.empty()) {
        std::cerr << "Error: Cannot specify both --batch and single image. Choose one.\n";
        return false;
    }
    
    if (!batch_directory.empty()) {
        if (!fs::exists(batch_directory) || !fs::is_directory(batch_directory)) {
            std::cerr << "Error: Batch directory '" << batch_directory << "' does not exist or is not a directory.\n";
            return false;
        }
    }
    
    if (!single_image.empty()) {
        if (!fs::exists(single_image) || !fs::is_regular_file(single_image)) {
            std::cerr << "Error: Single image '" << single_image << "' does not exist or is not a file.\n";
            return false;
        }
    }
    
    if (block_size != 16 && block_size != 32) {
        std::cerr << "Error: Block size must be 16 or 32.\n";
        return false;
    }
    
    return true;
}

void CLIArgs::print_usage() const {
    CLIParser::print_help();
}

CLIArgs CLIParser::parse(int argc, char* argv[]) {
    CLIArgs args;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            args.show_help = true;
            return args;
        }
        else if (arg == "--mode") {
            if (i + 1 < argc) {
                args.mode = parse_mode(argv[++i]);
            } else {
                std::cerr << "Error: --mode requires an argument (cpu|gpu)\n";
                args.show_help = true;
                return args;
            }
        }
        else if (arg == "--ops") {
            if (i + 1 < argc) {
                args.operations = parse_operations(argv[++i]);
            } else {
                std::cerr << "Error: --ops requires an argument (gaussian,sobel,hist)\n";
                args.show_help = true;
                return args;
            }
        }
        else if (arg == "--batch") {
            if (i + 1 < argc) {
                args.batch_directory = argv[++i];
            } else {
                std::cerr << "Error: --batch requires a directory argument\n";
                args.show_help = true;
                return args;
            }
        }
        else if (arg == "--rgb") {
            args.rgb_mode = true;
        }
        else if (arg == "--block") {
            if (i + 1 < argc) {
                args.block_size = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --block requires an argument (16|32)\n";
                args.show_help = true;
                return args;
            }
        }
        else if (arg.front() != '-') {
            if (args.single_image.empty()) {
                args.single_image = arg;
            } else {
                std::cerr << "Error: Multiple image files specified. Use --batch for multiple images.\n";
                args.show_help = true;
                return args;
            }
        }
        else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            args.show_help = true;
            return args;
        }
    }
    
    return args;
}

ProcessingMode CLIParser::parse_mode(const std::string& mode_str) {
    std::string mode_lower = mode_str;
    std::transform(mode_lower.begin(), mode_lower.end(), mode_lower.begin(), ::tolower);
    
    if (mode_lower == "cpu") {
        return ProcessingMode::CPU;
    } else if (mode_lower == "gpu") {
        return ProcessingMode::GPU;
    } else {
        std::cerr << "Warning: Unknown mode '" << mode_str << "', defaulting to GPU\n";
        return ProcessingMode::GPU;
    }
}

std::set<Operation> CLIParser::parse_operations(const std::string& ops_str) {
    std::set<Operation> operations;
    std::stringstream ss(ops_str);
    std::string op;
    
    static const std::unordered_map<std::string, Operation> op_map = {
        {"gaussian", Operation::GAUSSIAN},
        {"sobel", Operation::SOBEL},
        {"canny", Operation::CANNY},
        {"hist", Operation::HISTOGRAM},
        {"histogram", Operation::HISTOGRAM}
    };
    
    while (std::getline(ss, op, ',')) {
        std::transform(op.begin(), op.end(), op.begin(), ::tolower);
        
        auto it = op_map.find(op);
        if (it != op_map.end()) {
            operations.insert(it->second);
        } else {
            std::cerr << "Warning: Unknown operation '" << op << "', skipping\n";
        }
    }
    
    return operations;
}

void CLIParser::print_help() {
    std::cout << R"(CUDA Image Processing Application

Usage: cuda-image-proc [OPTIONS] [IMAGE_FILE]

OPTIONS:
    --mode <cpu|gpu>        Processing mode (default: gpu)
    --ops <operations>      Comma-separated list of operations:
                           gaussian,sobel,canny,hist (required)
    --batch <directory>     Process all images in directory
    --rgb                   Enable RGB processing (default: grayscale)
    --block <16|32>         CUDA block size for GPU mode (default: 16)
    --help, -h              Show this help message

EXAMPLES:
    # Single image with Gaussian smoothing on GPU
    cuda-image-proc --ops gaussian input.jpg
    
    # Batch processing with multiple operations
    cuda-image-proc --mode gpu --ops gaussian,sobel,hist --batch images/
    
    # CPU processing with RGB mode
    cuda-image-proc --mode cpu --ops sobel --rgb image.png
    
    # Custom CUDA block size
    cuda-image-proc --ops gaussian --block 32 --batch input_dir/

SUPPORTED OPERATIONS:
    gaussian    - Gaussian smoothing (separable convolution)
    sobel       - Sobel edge detection
    canny       - Canny edge detection
    hist        - Histogram equalization

SUPPORTED FORMATS:
    Input: JPEG, PNG, BMP, TIFF (via OpenCV)
    Output: Same format as input, saved with operation suffix

PERFORMANCE:
    Target: ≥70% throughput improvement (GPU vs CPU) on 1080p images
)";
}