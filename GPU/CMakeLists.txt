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
    blackscholes_aad_kernels_fixed.cu
    numerical_stability.cu
)

# Test source files
set(TEST_SOURCES
    test_gpu_aad.cpp
    GPUAADTape.cpp
    GPUAADNumber.cpp
)

# Create main executable
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

# Create test executables
add_executable(test_gpu_aad ${TEST_SOURCES} ${CUDA_SOURCES})
add_executable(test_blackscholes_aad test_blackscholes_aad.cpp ${CUDA_SOURCES})

# Set properties for CUDA files
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RUNTIME_LIBRARY Shared)
set_property(TARGET test_gpu_aad PROPERTY CUDA_RUNTIME_LIBRARY Shared)
set_property(TARGET test_blackscholes_aad PROPERTY CUDA_RUNTIME_LIBRARY Shared)

# Link CUDA libraries
target_link_libraries(${PROJECT_NAME} CUDA::cudart)
target_link_libraries(test_gpu_aad CUDA::cudart)
target_link_libraries(test_blackscholes_aad CUDA::cudart)

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_compile_options(test_gpu_aad PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_compile_options(test_blackscholes_aad PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(test_gpu_aad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(test_blackscholes_aad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})