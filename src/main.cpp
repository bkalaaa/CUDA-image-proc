#include <iostream>
#include <opencv2/opencv.hpp>
#include "cli.h"

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
    
    std::cout << "\nOpenCV Version: " << cv::getVersionString() << "\n";
    
    std::cout << "\nCLI parsing successful. Implementation pending in future commits.\n";
    
    return 0;
}