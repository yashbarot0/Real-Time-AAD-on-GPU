cmake_minimum_required(VERSION 3.15)
project(AAD)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Original AAD test
add_executable(aad_test main.cpp)

# Real-time AAD test
add_executable(realtime_test test_realtime_cpu.cpp)

# Debug AAD test
add_executable(debug_aad debug_aad.cpp)

# Add compiler flags for optimization
target_compile_options(aad_test PRIVATE -O3)
target_compile_options(realtime_test PRIVATE -O3)
target_compile_options(debug_aad PRIVATE -O0 -g)