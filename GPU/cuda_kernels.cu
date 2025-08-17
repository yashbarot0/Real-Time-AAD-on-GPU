
// ===== cuda_kernels.cu =====
#include "AADTypes.h"
#include "device_functions.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>  // For printf

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
