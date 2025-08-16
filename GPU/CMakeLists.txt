cmake_minimum_required(VERSION 3.18)
project(GPU_AAD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")

# Source files
set(SOURCES
    main.cpp
    GPUAADTape.cpp
    GPUAADNumber.cpp
)

set(CUDA_SOURCES
    cuda_kernels.cu
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

# Set properties for CUDA files
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RUNTIME_LIBRARY Shared)

# Link CUDA libraries
target_link_libraries(${PROJECT_NAME} CUDA::cudart)

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})