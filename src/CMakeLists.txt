set(SOURCES
    main.cpp
    cli.cpp
    cpu_pipeline.cpp
)

set(HEADERS
    cli.h
    cpu_pipeline.h
)

# Create main executable
add_executable(cuda-image-proc ${SOURCES} ${HEADERS})

# Set C++ and CUDA properties
set_target_properties(cuda-image-proc PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(cuda-image-proc PRIVATE
    ${OpenCV_LIBS}
    CUDA::extended
)

# Include directories
target_include_directories(cuda-image-proc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Compile definitions
target_compile_definitions(cuda-image-proc PRIVATE
    $<$<CONFIG:Debug>:DEBUG=1>
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(cuda-image-proc PRIVATE
        $<$<CONFIG:Release>:-ffast-math -funroll-loops -march=native>
        $<$<CONFIG:Debug>:-fsanitize=address -fno-omit-frame-pointer>
    )
    
    target_link_options(cuda-image-proc PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(cuda-image-proc PRIVATE
        $<$<CONFIG:Release>:-ffast-math -funroll-loops -march=native>
    )
endif()

# CUDA-specific compile options
target_compile_options(cuda-image-proc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--use_fast_math>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--device-debug>
)