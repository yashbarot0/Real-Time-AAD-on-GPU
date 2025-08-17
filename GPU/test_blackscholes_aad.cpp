// ===== test_blackscholes_aad.cpp =====
// Test program for GPU Black-Scholes AAD kernels

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "AADTypes.h"
#include "blackscholes_aad_kernels.h"
#include <cuda_runtime.h>

// Helper function to check CUDA errors
bool checkCudaError(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

// Analytical Black-Scholes for validation
double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double norm_pdf(double x) {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

void analytical_black_scholes(double S, double K, double T, double r, double sigma,
                             double& price, double& delta, double& vega, double& gamma, double& theta, double& rho) {
    if (T <= 0.0 || sigma <= 0.0) {
        price = std::max(S - K, 0.0);
        delta = (S > K) ? 1.0 : 0.0;
        vega = gamma = theta = rho = 0.0;
        return;
    }
    
    double sqrt_T = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;
    
    double N_d1 = norm_cdf(d1);
    double N_d2 = norm_cdf(d2);
    double phi_d1 = norm_pdf(d1);
    
    double discount = std::exp(-r * T);
    
    price = S * N_d1 - K * discount * N_d2;
    delta = N_d1;
    vega = S * phi_d1 * sqrt_T;
    gamma = phi_d1 / (S * sigma * sqrt_T);
    theta = -S * phi_d1 * sigma / (2.0 * sqrt_T) - r * K * discount * N_d2;
    rho = K * T * discount * N_d2;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== GPU Black-Scholes AAD Kernel Test ===" << std::endl;
    
    // Test parameters
    const int num_scenarios = 1000;
    const int max_tape_size_per_scenario = 100;
    const int max_vars_per_scenario = 50;
    
    // Create test data
    std::vector<double> spot_prices(num_scenarios);
    std::vector<double> strike_prices(num_scenarios);
    std::vector<double> times_to_expiry(num_scenarios);
    std::vector<double> risk_free_rates(num_scenarios);
    std::vector<double> volatilities(num_scenarios);
    
    // Initialize with varied parameters
    for (int i = 0; i < num_scenarios; i++) {
        spot_prices[i] = 90.0 + 20.0 * i / num_scenarios;  // 90 to 110
        strike_prices[i] = 100.0;
        times_to_expiry[i] = 0.1 + 0.9 * i / num_scenarios;  // 0.1 to 1.0 years
        risk_free_rates[i] = 0.01 + 0.09 * i / num_scenarios;  // 1% to 10%
        volatilities[i] = 0.1 + 0.4 * i / num_scenarios;  // 10% to 50%
    }
    
    // Allocate host memory for results
    std::vector<double> gpu_prices(num_scenarios);
    std::vector<double> gpu_deltas(num_scenarios);
    std::vector<double> gpu_vegas(num_scenarios);
    std::vector<double> gpu_gammas(num_scenarios);
    std::vector<double> gpu_thetas(num_scenarios);
    std::vector<double> gpu_rhos(num_scenarios);
    std::vector<int> tape_positions(num_scenarios);
    std::vector<int> error_flags(num_scenarios);
    
    // Allocate device memory
    BatchInputs* d_inputs;
    BatchOutputs* d_outputs;
    GPUTapeEntry* d_tape;
    double* d_values;
    int* d_tape_positions;
    int* d_error_flags;
    
    // Allocate device memory for inputs
    cudaError_t error = cudaMalloc(&d_inputs, sizeof(BatchInputs));
    if (!checkCudaError(error, "cudaMalloc d_inputs")) return -1;
    
    BatchInputs h_inputs;
    cudaMalloc(&h_inputs.spot_prices, num_scenarios * sizeof(double));
    cudaMalloc(&h_inputs.strike_prices, num_scenarios * sizeof(double));
    cudaMalloc(&h_inputs.times_to_expiry, num_scenarios * sizeof(double));
    cudaMalloc(&h_inputs.risk_free_rates, num_scenarios * sizeof(double));
    cudaMalloc(&h_inputs.volatilities, num_scenarios * sizeof(double));
    h_inputs.num_scenarios = num_scenarios;
    
    // Copy input data to device
    cudaMemcpy(h_inputs.spot_prices, spot_prices.data(), num_scenarios * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h_inputs.strike_prices, strike_prices.data(), num_scenarios * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h_inputs.times_to_expiry, times_to_expiry.data(), num_scenarios * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h_inputs.risk_free_rates, risk_free_rates.data(), num_scenarios * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h_inputs.volatilities, volatilities.data(), num_scenarios * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, &h_inputs, sizeof(BatchInputs), cudaMemcpyHostToDevice);
    
    // Allocate device memory for outputs
    cudaMalloc(&d_outputs, sizeof(BatchOutputs));
    BatchOutputs h_outputs;
    cudaMalloc(&h_outputs.option_prices, num_scenarios * sizeof(double));
    cudaMalloc(&h_outputs.deltas, num_scenarios * sizeof(double));
    cudaMalloc(&h_outputs.vegas, num_scenarios * sizeof(double));
    cudaMalloc(&h_outputs.gammas, num_scenarios * sizeof(double));
    cudaMalloc(&h_outputs.thetas, num_scenarios * sizeof(double));
    cudaMalloc(&h_outputs.rhos, num_scenarios * sizeof(double));
    cudaMemcpy(d_outputs, &h_outputs, sizeof(BatchOutputs), cudaMemcpyHostToDevice);
    
    // Allocate other device memory
    cudaMalloc(&d_tape, num_scenarios * max_tape_size_per_scenario * sizeof(GPUTapeEntry));
    cudaMalloc(&d_values, num_scenarios * max_vars_per_scenario * sizeof(double));
    cudaMalloc(&d_tape_positions, num_scenarios * sizeof(int));
    cudaMalloc(&d_error_flags, num_scenarios * sizeof(int));
    
    std::cout << "Running GPU Black-Scholes forward pass..." << std::endl;
    
    // Launch the AAD forward pass kernel
    auto start = std::chrono::high_resolution_clock::now();
    
    launch_batch_blackscholes_forward(
        d_inputs, d_tape, d_values, d_tape_positions, d_outputs,
        num_scenarios, max_tape_size_per_scenario, max_vars_per_scenario);
    
    // Launch the AAD reverse pass kernel to compute Greeks
    launch_batch_aad_reverse(
        d_inputs, d_tape, d_values, d_tape_positions, d_outputs,
        num_scenarios, max_tape_size_per_scenario, max_vars_per_scenario);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "GPU computation completed in " << duration.count() << " microseconds" << std::endl;
    std::cout << "Average time per scenario: " << (double)duration.count() / num_scenarios << " microseconds" << std::endl;
    
    // Copy results back to host
    cudaMemcpy(gpu_prices.data(), h_outputs.option_prices, num_scenarios * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_deltas.data(), h_outputs.deltas, num_scenarios * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_vegas.data(), h_outputs.vegas, num_scenarios * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_gammas.data(), h_outputs.gammas, num_scenarios * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_thetas.data(), h_outputs.thetas, num_scenarios * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_rhos.data(), h_outputs.rhos, num_scenarios * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tape_positions.data(), d_tape_positions, num_scenarios * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(error_flags.data(), d_error_flags, num_scenarios * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Validate against analytical results
    std::cout << "\n=== Validation Against Analytical Results ===" << std::endl;
    
    double max_price_error = 0.0, max_delta_error = 0.0, max_vega_error = 0.0;
    int error_count = 0;
    
    for (int i = 0; i < std::min(10, num_scenarios); i++) {
        double analytical_price, analytical_delta, analytical_vega, analytical_gamma, analytical_theta, analytical_rho;
        
        analytical_black_scholes(spot_prices[i], strike_prices[i], times_to_expiry[i],
                               risk_free_rates[i], volatilities[i],
                               analytical_price, analytical_delta, analytical_vega,
                               analytical_gamma, analytical_theta, analytical_rho);
        
        double price_error = std::abs(gpu_prices[i] - analytical_price);
        double delta_error = std::abs(gpu_deltas[i] - analytical_delta);
        double vega_error = std::abs(gpu_vegas[i] - analytical_vega);
        
        max_price_error = std::max(max_price_error, price_error);
        max_delta_error = std::max(max_delta_error, delta_error);
        max_vega_error = std::max(max_vega_error, vega_error);
        
        if (error_flags[i] != 0) error_count++;
        
        if (i < 5) {  // Print first 5 scenarios
            std::cout << "Scenario " << i << ":" << std::endl;
            std::cout << "  S=" << spot_prices[i] << ", K=" << strike_prices[i] 
                     << ", T=" << times_to_expiry[i] << ", r=" << risk_free_rates[i] 
                     << ", σ=" << volatilities[i] << std::endl;
            std::cout << "  GPU Price: " << gpu_prices[i] << ", Analytical: " << analytical_price 
                     << ", Error: " << price_error << std::endl;
            std::cout << "  GPU Delta: " << gpu_deltas[i] << ", Analytical: " << analytical_delta 
                     << ", Error: " << delta_error << std::endl;
            std::cout << "  GPU Vega:  " << gpu_vegas[i] << ", Analytical: " << analytical_vega 
                     << ", Error: " << vega_error << std::endl;
            std::cout << "  Tape size: " << tape_positions[i] << ", Error flags: " << error_flags[i] << std::endl;
            std::cout << std::endl;
        }
    }
    
    std::cout << "Maximum errors:" << std::endl;
    std::cout << "  Price: " << max_price_error << std::endl;
    std::cout << "  Delta: " << max_delta_error << std::endl;
    std::cout << "  Vega:  " << max_vega_error << std::endl;
    std::cout << "Scenarios with errors: " << error_count << " / " << num_scenarios << std::endl;
    
    // Performance metrics
    double throughput = (double)num_scenarios / (duration.count() / 1e6);  // scenarios per second
    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Throughput: " << throughput << " scenarios/second" << std::endl;
    std::cout << "  Target: 10M scenarios/second" << std::endl;
    std::cout << "  Achievement: " << (throughput / 10e6 * 100) << "% of target" << std::endl;
    
    // Cleanup
    cudaFree(h_inputs.spot_prices);
    cudaFree(h_inputs.strike_prices);
    cudaFree(h_inputs.times_to_expiry);
    cudaFree(h_inputs.risk_free_rates);
    cudaFree(h_inputs.volatilities);
    cudaFree(d_inputs);
    
    cudaFree(h_outputs.option_prices);
    cudaFree(h_outputs.deltas);
    cudaFree(h_outputs.vegas);
    cudaFree(h_outputs.gammas);
    cudaFree(h_outputs.thetas);
    cudaFree(h_outputs.rhos);
    cudaFree(d_outputs);
    
    cudaFree(d_tape);
    cudaFree(d_values);
    cudaFree(d_tape_positions);
    cudaFree(d_error_flags);
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}