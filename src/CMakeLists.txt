set(SOURCES
    main.cpp
)

set(HEADERS
)

add_executable(cuda-image-proc ${SOURCES} ${HEADERS})

target_link_libraries(cuda-image-proc 
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

set_target_properties(cuda-image-proc PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(cuda-image-proc PRIVATE DEBUG=1)
endif()