#!/usr/bin/env zsh

set -e

PROJECT_ROOT="$(cd "$(dirname "${0:A}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo "=== CUDA Image Processing Build Validation ==="

# Test configuration for a specific build type
test_config() {
    local build_type=$1
    local build_subdir="${BUILD_DIR}/${build_type:l}"
    
    echo "=== Testing ${build_type} Configuration ==="
    
    rm -rf "${build_subdir}"
    mkdir -p "${build_subdir}"
    cd "${build_subdir}"
    
    cmake "${PROJECT_ROOT}" -DCMAKE_BUILD_TYPE=${build_type}
    make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    if [ -f "src/cuda-image-proc" ]; then
        echo "SUCCESS: ${build_type} build successful"
        return 0
    else
        echo "ERROR: ${build_type} build failed"
        return 1
    fi
}

case "${1:-}" in
    release|Release) test_config "Release" ;;
    debug|Debug) test_config "Debug" ;;
    *) 
        test_config "Release" && test_config "Debug"
        echo "All configurations validated successfully"
        ;;
esac