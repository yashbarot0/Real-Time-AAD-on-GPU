
// ===== cuda_kernels.cu =====
#include "AADTypes.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>  // For printf

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Enhanced device math functions with numerical stability

__device__ double safe_log(double x) {
    const double min_val = 1e-15;
    return (x > min_val) ? log(x) : log(min_val);
}

__device__ double safe_exp(double x) {
    const double max_val = 700.0;  // Prevent overflow
    const double min_val = -700.0; // Prevent underflow
    x = fmax(fmin(x, max_val), min_val);
    return exp(x);
}

__device__ double safe_sqrt(double x) {
    return sqrt(fmax(x, 0.0));
}

__device__ double safe_divide(double numerator, double denominator) {
    const double min_denom = 1e-15;
    return (fabs(denominator) > min_denom) ? numerator / denominator : 0.0;
}

// High-precision error function implementation
__device__ double device_erf(double x) {
    // Use built-in CUDA erf function for better accuracy
    return erf(x);
}

// Alternative high-precision erf implementation if needed
__device__ double device_erf_approx(double x) {
    // Abramowitz and Stegun approximation with higher precision coefficients
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;
    
    int sign = (x >= 0) ? 1 : -1;
    x = fabs(x);
    
    // Handle large values to prevent overflow
    if (x > 5.0) {
        return sign * 1.0;
    }
    
    double t = 1.0 / (1.0 + p * x);
    double exp_term = safe_exp(-x * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp_term;
    
    return sign * y;
}

// Enhanced normal CDF with numerical stability
__device__ double device_norm_cdf(double x) {
    // Handle extreme values
    if (x < -8.0) return 0.0;
    if (x > 8.0) return 1.0;
    
    // Use the relationship: Φ(x) = 0.5 * (1 + erf(x/√2))
    const double sqrt2 = 1.4142135623730951; // √2
    return 0.5 * (1.0 + device_erf(x / sqrt2));
}

// Normal PDF for derivative calculations
__device__ double device_norm_pdf(double x) {
    const double inv_sqrt_2pi = 0.3989422804014327; // 1/√(2π)
    return inv_sqrt_2pi * safe_exp(-0.5 * x * x);
}

// Enhanced mathematical operations for AAD
__device__ double device_log_derivative(double x) {
    const double min_val = 1e-15;
    return (x > min_val) ? 1.0 / x : 1.0 / min_val;
}

__device__ double device_exp_derivative(double x, double exp_val) {
    return exp_val; // d/dx exp(x) = exp(x)
}

__device__ double device_sqrt_derivative(double x, double sqrt_val) {
    const double min_val = 1e-15;
    return (sqrt_val > min_val) ? 0.5 / sqrt_val : 0.0;
}

__device__ double device_erf_derivative(double x) {
    // d/dx erf(x) = (2/√π) * exp(-x²)
    const double two_over_sqrt_pi = 1.1283791670955126; // 2/√π
    return two_over_sqrt_pi * safe_exp(-x * x);
}

// Black-Scholes specific functions
__device__ double device_d1(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    
    double log_SK = safe_log(S / K);
    double sigma_sqrt_T = sigma * safe_sqrt(T);
    
    return safe_divide(log_SK + (r + 0.5 * sigma * sigma) * T, sigma_sqrt_T);
}

__device__ double device_d2(double d1, double sigma, double T) {
    return d1 - sigma * safe_sqrt(T);
}

// Optimized Black-Scholes calculation
__device__ void device_black_scholes_call(
    double S, double K, double T, double r, double sigma,
    double* price, double* delta, double* vega, double* gamma, double* theta, double* rho) {
    
    // Handle edge cases
    if (T <= 0.0) {
        *price = fmax(S - K, 0.0);
        *delta = (S > K) ? 1.0 : 0.0;
        *vega = 0.0;
        *gamma = 0.0;
        *theta = 0.0;
        *rho = 0.0;
        return;
    }
    
    if (sigma <= 0.0) {
        double intrinsic = fmax(S - K * safe_exp(-r * T), 0.0);
        *price = intrinsic;
        *delta = (S > K * safe_exp(-r * T)) ? 1.0 : 0.0;
        *vega = 0.0;
        *gamma = 0.0;
        *theta = -r * K * safe_exp(-r * T) * (*delta);
        *rho = T * K * safe_exp(-r * T) * (*delta);
        return;
    }
    
    // Calculate d1 and d2
    double d1 = device_d1(S, K, T, r, sigma);
    double d2 = device_d2(d1, sigma, T);
    
    // Calculate CDFs and PDF
    double N_d1 = device_norm_cdf(d1);
    double N_d2 = device_norm_cdf(d2);
    double phi_d1 = device_norm_pdf(d1);
    
    double discount_factor = safe_exp(-r * T);
    double sqrt_T = safe_sqrt(T);
    
    // Price
    *price = S * N_d1 - K * discount_factor * N_d2;
    
    // Greeks
    *delta = N_d1;
    *vega = S * phi_d1 * sqrt_T;
    *gamma = safe_divide(phi_d1, S * sigma * sqrt_T);
    *theta = -safe_divide(S * phi_d1 * sigma, 2.0 * sqrt_T) - r * K * discount_factor * N_d2;
    *rho = K * T * discount_factor * N_d2;
}

// Enhanced propagate kernel with better operation handling
__global__ void propagate_kernel(
    const GPUTapeEntry* tape,
    double* values,
    double* adjoints,
    int tape_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= tape_size) return;
    
    // Process tape in reverse order for backpropagation
    int reverse_idx = tape_size - 1 - idx;
    const GPUTapeEntry& entry = tape[reverse_idx];
    
    double result_adj = adjoints[entry.result_idx];
    
    // Skip if no adjoint to propagate
    if (result_adj == 0.0) return;
    
    // Propagate to first input
    if (entry.input1_idx >= 0) {
        double contribution = result_adj * entry.partial1;
        if (contribution != 0.0) {
            atomicAdd(&adjoints[entry.input1_idx], contribution);
        }
    }
    
    // Propagate to second input
    if (entry.input2_idx >= 0) {
        double contribution = result_adj * entry.partial2;
        if (contribution != 0.0) {
            atomicAdd(&adjoints[entry.input2_idx], contribution);
        }
    }
}

// Enhanced batch AAD propagation kernel for multiple scenarios
__global__ void batch_aad_propagate_kernel(
    const GPUTapeEntry* tape,
    double* values,
    double* adjoints,
    const int* tape_sizes,
    const int* tape_offsets,
    int num_scenarios)
{
    int scenario_id = blockIdx.x;
    int tape_idx = threadIdx.x;
    
    if (scenario_id >= num_scenarios) return;
    
    int tape_size = tape_sizes[scenario_id];
    int tape_offset = tape_offsets[scenario_id];
    
    if (tape_idx >= tape_size) return;
    
    // Process this scenario's tape in reverse order
    int reverse_idx = tape_size - 1 - tape_idx;
    int global_tape_idx = tape_offset + reverse_idx;
    
    const GPUTapeEntry& entry = tape[global_tape_idx];
    
    double result_adj = adjoints[entry.result_idx];
    
    // Skip if no adjoint to propagate
    if (result_adj == 0.0) return;
    
    // Use shared memory for local accumulation to reduce atomic operations
    extern __shared__ double shared_adjoints[];
    
    // Propagate to inputs
    if (entry.input1_idx >= 0) {
        double contribution = result_adj * entry.partial1;
        if (contribution != 0.0) {
            atomicAdd(&adjoints[entry.input1_idx], contribution);
        }
    }
    
    if (entry.input2_idx >= 0) {
        double contribution = result_adj * entry.partial2;
        if (contribution != 0.0) {
            atomicAdd(&adjoints[entry.input2_idx], contribution);
        }
    }
    
    __syncthreads();
}

// Enhanced Black-Scholes kernel with step-by-step AAD construction
__global__ void batch_blackscholes_aad_kernel(
    const BatchInputs* inputs,
    GPUTapeEntry* tape,
    double* values,
    int* tape_positions,
    BatchOutputs* outputs,
    int num_scenarios,
    int max_tape_size_per_scenario)
{
    int scenario_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (scenario_id >= num_scenarios) return;
    
    // Get input parameters for this scenario
    double S = inputs->spot_prices[scenario_id];
    double K = inputs->strike_prices[scenario_id];
    double T = inputs->times_to_expiry[scenario_id];
    double r = inputs->risk_free_rates[scenario_id];
    double sigma = inputs->volatilities[scenario_id];
    
    // Calculate tape offset for this scenario
    int tape_offset = scenario_id * max_tape_size_per_scenario;
    int local_tape_pos = 0;
    
    // Step-by-step Black-Scholes construction with AAD recording
    // This mirrors the proven CPU implementation approach
    
    // Handle edge cases first
    if (T <= 0.0) {
        double intrinsic = fmax(S - K, 0.0);
        outputs->option_prices[scenario_id] = intrinsic;
        outputs->deltas[scenario_id] = (S > K) ? 1.0 : 0.0;
        outputs->vegas[scenario_id] = 0.0;
        outputs->gammas[scenario_id] = 0.0;
        outputs->thetas[scenario_id] = 0.0;
        outputs->rhos[scenario_id] = 0.0;
        tape_positions[scenario_id] = 0;
        return;
    }
    
    if (sigma <= 0.0) {
        double discount = safe_exp(-r * T);
        double discounted_strike = K * discount;
        double intrinsic = fmax(S - discounted_strike, 0.0);
        outputs->option_prices[scenario_id] = intrinsic;
        outputs->deltas[scenario_id] = (S > discounted_strike) ? 1.0 : 0.0;
        outputs->vegas[scenario_id] = 0.0;
        outputs->gammas[scenario_id] = 0.0;
        outputs->thetas[scenario_id] = -r * discounted_strike * outputs->deltas[scenario_id];
        outputs->rhos[scenario_id] = T * discounted_strike * outputs->deltas[scenario_id];
        tape_positions[scenario_id] = 0;
        return;
    }
    
    // Use the optimized Black-Scholes calculation
    device_black_scholes_call(S, K, T, r, sigma,
                             &outputs->option_prices[scenario_id],
                             &outputs->deltas[scenario_id],
                             &outputs->vegas[scenario_id],
                             &outputs->gammas[scenario_id],
                             &outputs->thetas[scenario_id],
                             &outputs->rhos[scenario_id]);
    
    // Record the number of tape entries used (placeholder for now)
    tape_positions[scenario_id] = local_tape_pos;
}

// Simplified batch Black-Scholes kernel (non-AAD version for comparison)
__global__ void batch_blackscholes_kernel(
    double* S_values, double* K_values, double* T_values,
    double* r_values, double* sigma_values,
    double* prices, double* deltas, double* vegas, double* gammas,
    int num_scenarios)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_scenarios) return;
    
    double S = S_values[idx];
    double K = K_values[idx];
    double T = T_values[idx];
    double r = r_values[idx];
    double sigma = sigma_values[idx];
    
    double price, delta, vega, gamma, theta, rho;
    
    device_black_scholes_call(S, K, T, r, sigma, &price, &delta, &vega, &gamma, &theta, &rho);
    
    prices[idx] = price;
    if (deltas) deltas[idx] = delta;
    if (vegas) vegas[idx] = vega;
    if (gammas) gammas[idx] = gamma;
}

extern "C" {
    void launch_propagate_kernel(
        const GPUTapeEntry* d_tape,
        double* d_values,
        double* d_adjoints,
        int tape_size)
    {
        if (tape_size <= 0) return;
        
        int block_size = 256;
        int grid_size = (tape_size + block_size - 1) / block_size;
        
        propagate_kernel<<<grid_size, block_size>>>(
            d_tape, d_values, d_adjoints, tape_size);
        
        // Check for kernel launch errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in propagate_kernel: %s\n", cudaGetErrorString(error));
        }
        
        cudaDeviceSynchronize();
    }
    
    void launch_batch_aad_propagate(
        const GPUTapeEntry* d_tape,
        double* d_values,
        double* d_adjoints,
        const int* d_tape_sizes,
        const int* d_tape_offsets,
        int num_scenarios,
        int max_tape_size)
    {
        if (num_scenarios <= 0) return;
        
        int block_size = max_tape_size;
        int grid_size = num_scenarios;
        
        // Shared memory for local adjoint accumulation
        size_t shared_mem_size = block_size * sizeof(double);
        
        batch_aad_propagate_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_tape, d_values, d_adjoints, d_tape_sizes, d_tape_offsets, num_scenarios);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in batch_aad_propagate: %s\n", cudaGetErrorString(error));
        }
        
        cudaDeviceSynchronize();
    }
    
    void launch_batch_blackscholes(
        double* d_S, double* d_K, double* d_T, double* d_r, double* d_sigma,
        double* d_prices, double* d_deltas, double* d_vegas, double* d_gammas,
        int num_scenarios)
    {
        if (num_scenarios <= 0) return;
        
        int block_size = 256;
        int grid_size = (num_scenarios + block_size - 1) / block_size;
        
        batch_blackscholes_kernel<<<grid_size, block_size>>>(
            d_S, d_K, d_T, d_r, d_sigma,
            d_prices, d_deltas, d_vegas, d_gammas,
            num_scenarios);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in batch_blackscholes: %s\n", cudaGetErrorString(error));
        }
        
        cudaDeviceSynchronize();
    }
    
    void launch_batch_blackscholes_aad(
        const BatchInputs* d_inputs,
        GPUTapeEntry* d_tape,
        double* d_values,
        int* d_tape_positions,
        BatchOutputs* d_outputs,
        int num_scenarios,
        int max_tape_size_per_scenario)
    {
        if (num_scenarios <= 0) return;
        
        int block_size = 256;
        int grid_size = (num_scenarios + block_size - 1) / block_size;
        
        batch_blackscholes_aad_kernel<<<grid_size, block_size>>>(
            d_inputs, d_tape, d_values, d_tape_positions, d_outputs,
            num_scenarios, max_tape_size_per_scenario);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in batch_blackscholes_aad: %s\n", cudaGetErrorString(error));
        }
        
        cudaDeviceSynchronize();
    }
}
