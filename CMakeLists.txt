cmake_minimum_required(VERSION 3.18)
project(cuda-image-proc LANGUAGES CXX CUDA)

# Add custom CMake modules path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enhanced compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -flto")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# Platform-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -funroll-loops")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ffast-math")
endif()

# Find dependencies using custom scripts
include(FindCUDAExtended)
find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV or set OpenCV_ROOT.")
endif()

# Verify minimum versions
if(OpenCV_VERSION VERSION_LESS "4.0")
    message(WARNING "OpenCV version ${OpenCV_VERSION} detected. Version 4.0+ recommended.")
endif()

if(CUDAToolkit_VERSION VERSION_LESS "11.0")
    message(WARNING "CUDA version ${CUDAToolkit_VERSION} detected. Version 11.0+ recommended.")
endif()

# Add subdirectories
add_subdirectory(src)

enable_testing()
add_subdirectory(tests)

# Build configuration summary
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Compiler flags summary
message(STATUS "=== Compiler Flags ===")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS_DEBUG}")
    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()
message(STATUS "=====================================")