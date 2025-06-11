// ===== main.cpp =====
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "GPUAADNumber.h"

// Black-Scholes Call Option Formula
GPUAADNumber BlackScholesCall(
    const GPUAADNumber& S, const GPUAADNumber& K,
    const GPUAADNumber& T, const GPUAADNumber& r,
    const GPUAADNumber& sigma)
{
    GPUAADNumber sqrtT = sqrt(T);
    GPUAADNumber d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    GPUAADNumber d2 = d1 - sigma * sqrtT;
    GPUAADNumber call = S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2);
    return call;
}

void test_single_evaluation() {
    std::cout << "=== Single Evaluation Test ===" << std::endl;
    
    GPUAADTape tape;
    GPUAADNumber::set_active_tape(&tape);
    
    GPUAADNumber S(100.0);
    GPUAADNumber K(100.0);
    GPUAADNumber T(1.0);
    GPUAADNumber r(0.05);
    GPUAADNumber sigma(0.2);
    
    GPUAADNumber price = BlackScholesCall(S, K, T, r, sigma);
    
    // Set seed for backpropagation
    tape.set_adjoint(price.index(), 1.0);
    tape.propagate_gpu();
    
    std::cout << "Price: " << price.val() << std::endl;
    std::cout << "Delta (dP/dS): " << S.adj() << std::endl;
    std::cout << "Vega (dP/dsigma): " << sigma.adj() << std::endl;
    std::cout << "Rho (dP/dr): " << r.adj() << std::endl;
    std::cout << "Theta (dP/dT): " << T.adj() << std::endl;
}

void benchmark_gpu_aad() {
    std::cout << "\n=== GPU AAD Benchmark ===" << std::endl;
    
    constexpr int NUM_RUNS = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    
    double total_price = 0.0;
    double total_delta = 0.0;
    
    for (int i = 0; i < NUM_RUNS; ++i) {
        GPUAADTape tape;
        GPUAADNumber::set_active_tape(&tape);
        
        GPUAADNumber S(100.0);
        GPUAADNumber K(100.0);
        GPUAADNumber T(1.0);
        GPUAADNumber r(0.05);
        GPUAADNumber sigma(0.2);
        
        GPUAADNumber price = BlackScholesCall(S, K, T, r, sigma);
        
        tape.set_adjoint(price.index(), 1.0);
        tape.propagate_gpu();
        
        total_price += price.val();
        total_delta += S.adj();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Total runs: " << NUM_RUNS << std::endl;
    std::cout << "Average price: " << total_price / NUM_RUNS << std::endl;
    std::cout << "Average delta: " << total_delta / NUM_RUNS << std::endl;
    std::cout << "Total time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Avg time per evaluation: " << (elapsed.count() / NUM_RUNS * 1e6) << " µs" << std::endl;
    std::cout << "Evaluations per second: " << (NUM_RUNS / elapsed.count()) << std::endl;
}

int main() {
    try {
        test_single_evaluation();
        benchmark_gpu_aad();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

// // ===== CMakeLists.txt =====
// cmake_minimum_required(VERSION 3.18)
// project(GPU_AAD LANGUAGES CXX CUDA)

// set(CMAKE_CXX_STANDARD 17)
// set(CMAKE_CXX_STANDARD_REQUIRED ON)
// set(CMAKE_CUDA_STANDARD 17)
// set(CMAKE_CUDA_STANDARD_REQUIRED ON)

// # Find CUDA
// find_package(CUDA REQUIRED)

// # Set CUDA architecture (adjust for your GPU)
// set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")

// # Source files
// set(SOURCES
//     main.cpp
//     GPUAADTape.cpp
//     GPUAADNumber.cpp
// )

// set(CUDA_SOURCES
//     cuda_kernels.cu
// )

// # Create executable
// add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

// # Set properties for CUDA files
// set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RUNTIME_LIBRARY Shared)

// # Link CUDA libraries
// target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

// # Compiler flags
// target_compile_options(${PROJECT_NAME} PRIVATE 
//     $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
//     $<$<COMPILE_LANGUAGE:CXX>:-O3>
// )

// # Include directories
// target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})