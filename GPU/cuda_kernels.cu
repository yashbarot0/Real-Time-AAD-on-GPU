
// ===== cuda_kernels.cu =====
#include "AADTypes.h"
#include "device_functions.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>  // For printf

// Forward declarations for AAD recording functions from blackscholes_aad_kernels_fixed.cu
__device__ int record_constant(double value, double* values, int* next_var_idx);
__device__ int record_unary_op(AADOpType op_type, int input_idx, double result_val, double partial,
                               GPUTapeEntry* tape, double* values, int* tape_pos, int* next_var_idx, int max_tape_size);
__device__ int record_binary_op(AADOpType op_type, int input1_idx, int input2_idx, double result_val, 
                                double partial1, double partial2, GPUTapeEntry* tape, double* values, 
                                int* tape_pos, int* next_var_idx, int max_tape_size);
__device__ int aad_add(int a_idx, int b_idx, double* values, GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size);
__device__ int aad_sub(int a_idx, int b_idx, double* values, GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size);
__device__ int aad_mul(int a_idx, int b_idx, double* values, GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size);
__device__ int aad_div(int a_idx, int b_idx, double* values, GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size);
__device__ int aad_log(int x_idx, double* values, GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size);
__device__ int aad_exp(int x_idx, double* values, GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size);
__device__ int aad_sqrt(int x_idx, double* values, GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size);
__device__ int aad_norm_cdf(int x_idx, double* values, GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size);
__device__ int aad_mul_const(int x_idx, double constant, double* values, GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size);
__device__ int aad_neg(int x_idx, double* values, GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size);

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

// AAD reverse pass kernel to compute Greeks
__global__ void batch_aad_reverse_kernel(
    const GPUTapeEntry* tape,
    double* values,
    double* adjoints,
    const int* tape_positions,
    const int* variable_indices,
    BatchOutputs* outputs,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario)
{
    int scenario_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (scenario_id >= num_scenarios) return;
    
    // Calculate offsets for this scenario
    int tape_offset = scenario_id * max_tape_size_per_scenario;
    int values_offset = scenario_id * max_vars_per_scenario;
    int tape_size = tape_positions[scenario_id];
    
    if (tape_size <= 0) return;
    
    // Local pointers for this scenario
    const GPUTapeEntry* local_tape = tape + tape_offset;
    double* local_adjoints = adjoints + values_offset;
    
    // Clear adjoints
    for (int i = 0; i < max_vars_per_scenario; i++) {
        local_adjoints[i] = 0.0;
    }
    
    // Set seed adjoint for the option price (last computed variable)
    int price_idx = tape_size > 0 ? local_tape[tape_size - 1].result_idx : 0;
    local_adjoints[price_idx] = 1.0;
    
    // Reverse pass through the tape
    for (int i = tape_size - 1; i >= 0; i--) {
        const GPUTapeEntry& entry = local_tape[i];
        double result_adj = local_adjoints[entry.result_idx];
        
        if (result_adj == 0.0) continue;
        
        // Propagate to first input
        if (entry.input1_idx >= 0 && entry.input1_idx < max_vars_per_scenario) {
            atomicAdd(&local_adjoints[entry.input1_idx], result_adj * entry.partial1);
        }
        
        // Propagate to second input
        if (entry.input2_idx >= 0 && entry.input2_idx < max_vars_per_scenario) {
            atomicAdd(&local_adjoints[entry.input2_idx], result_adj * entry.partial2);
        }
    }
    
    // Extract Greeks from adjoints
    // Assuming variable order: S(0), K(1), T(2), r(3), sigma(4)
    if (variable_indices) {
        int S_idx = variable_indices[scenario_id * 5 + 0];
        int K_idx = variable_indices[scenario_id * 5 + 1];
        int T_idx = variable_indices[scenario_id * 5 + 2];
        int r_idx = variable_indices[scenario_id * 5 + 3];
        int sigma_idx = variable_indices[scenario_id * 5 + 4];
        
        outputs->deltas[scenario_id] = (S_idx >= 0 && S_idx < max_vars_per_scenario) ? local_adjoints[S_idx] : 0.0;
        outputs->vegas[scenario_id] = (sigma_idx >= 0 && sigma_idx < max_vars_per_scenario) ? local_adjoints[sigma_idx] : 0.0;
        outputs->rhos[scenario_id] = (r_idx >= 0 && r_idx < max_vars_per_scenario) ? local_adjoints[r_idx] : 0.0;
        outputs->thetas[scenario_id] = (T_idx >= 0 && T_idx < max_vars_per_scenario) ? -local_adjoints[T_idx] : 0.0; // Negative for time decay
    } else {
        // Fallback: assume standard ordering
        outputs->deltas[scenario_id] = local_adjoints[0];  // S is first variable
        outputs->vegas[scenario_id] = local_adjoints[4];   // sigma is fifth variable
        outputs->rhos[scenario_id] = local_adjoints[3];    // r is fourth variable
        outputs->thetas[scenario_id] = -local_adjoints[2]; // T is third variable (negative for time decay)
    }
    
    // Gamma computation would require second-order derivatives (not implemented yet)
    outputs->gammas[scenario_id] = 0.0;
}

// True AAD Black-Scholes kernel with step-by-step tape recording
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
    double S_val = inputs->spot_prices[scenario_id];
    double K_val = inputs->strike_prices[scenario_id];
    double T_val = inputs->times_to_expiry[scenario_id];
    double r_val = inputs->risk_free_rates[scenario_id];
    double sigma_val = inputs->volatilities[scenario_id];
    
    // Calculate tape and values offset for this scenario
    int tape_offset = scenario_id * max_tape_size_per_scenario;
    int values_offset = scenario_id * 50; // Assuming max 50 variables per scenario
    
    // Local pointers for this scenario's tape and values
    GPUTapeEntry* local_tape = tape + tape_offset;
    double* local_values = values + values_offset;
    
    int local_tape_pos = 0;
    int next_var_idx = 0;
    
    // Handle edge cases first (no AAD needed)
    if (T_val <= 0.0) {
        double intrinsic = fmax(S_val - K_val, 0.0);
        outputs->option_prices[scenario_id] = intrinsic;
        outputs->deltas[scenario_id] = (S_val > K_val) ? 1.0 : 0.0;
        outputs->vegas[scenario_id] = 0.0;
        outputs->gammas[scenario_id] = 0.0;
        outputs->thetas[scenario_id] = 0.0;
        outputs->rhos[scenario_id] = 0.0;
        tape_positions[scenario_id] = 0;
        return;
    }
    
    if (sigma_val <= 0.0) {
        double discount = safe_exp(-r_val * T_val);
        double discounted_strike = K_val * discount;
        double intrinsic = fmax(S_val - discounted_strike, 0.0);
        outputs->option_prices[scenario_id] = intrinsic;
        outputs->deltas[scenario_id] = (S_val > discounted_strike) ? 1.0 : 0.0;
        outputs->vegas[scenario_id] = 0.0;
        outputs->gammas[scenario_id] = 0.0;
        outputs->thetas[scenario_id] = -r_val * discounted_strike * outputs->deltas[scenario_id];
        outputs->rhos[scenario_id] = T_val * discounted_strike * outputs->deltas[scenario_id];
        tape_positions[scenario_id] = 0;
        return;
    }
    
    // Step-by-step Black-Scholes construction with AAD recording
    // Record input variables
    int S_idx = record_constant(S_val, local_values, &next_var_idx);
    int K_idx = record_constant(K_val, local_values, &next_var_idx);
    int T_idx = record_constant(T_val, local_values, &next_var_idx);
    int r_idx = record_constant(r_val, local_values, &next_var_idx);
    int sigma_idx = record_constant(sigma_val, local_values, &next_var_idx);
    
    // Calculate d1 step by step with AAD
    // d1 = (log(S/K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    
    // S/K
    int S_over_K_idx = aad_div(S_idx, K_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // log(S/K)
    int log_S_over_K_idx = aad_log(S_over_K_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // sigma^2
    int sigma_squared_idx = aad_mul(sigma_idx, sigma_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // 0.5 * sigma^2
    int half_sigma_squared_idx = aad_mul_const(sigma_squared_idx, 0.5, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // r + 0.5 * sigma^2
    int r_plus_half_sigma_squared_idx = aad_add(r_idx, half_sigma_squared_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // (r + 0.5 * sigma^2) * T
    int drift_times_T_idx = aad_mul(r_plus_half_sigma_squared_idx, T_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // log(S/K) + (r + 0.5 * sigma^2) * T
    int numerator_d1_idx = aad_add(log_S_over_K_idx, drift_times_T_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // sqrt(T)
    int sqrt_T_idx = aad_sqrt(T_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // sigma * sqrt(T)
    int sigma_sqrt_T_idx = aad_mul(sigma_idx, sqrt_T_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // d1 = numerator / (sigma * sqrt(T))
    int d1_idx = aad_div(numerator_d1_idx, sigma_sqrt_T_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // d2 = d1 - sigma * sqrt(T)
    int d2_idx = aad_sub(d1_idx, sigma_sqrt_T_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // N(d1) and N(d2)
    int N_d1_idx = aad_norm_cdf(d1_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    int N_d2_idx = aad_norm_cdf(d2_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // -r * T
    int neg_r_idx = aad_neg(r_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    int neg_r_times_T_idx = aad_mul(neg_r_idx, T_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // exp(-r * T)
    int discount_factor_idx = aad_exp(neg_r_times_T_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // K * exp(-r * T)
    int discounted_K_idx = aad_mul(K_idx, discount_factor_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // K * exp(-r * T) * N(d2)
    int second_term_idx = aad_mul(discounted_K_idx, N_d2_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // S * N(d1)
    int first_term_idx = aad_mul(S_idx, N_d1_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // Price = S * N(d1) - K * exp(-r * T) * N(d2)
    int price_idx = aad_sub(first_term_idx, second_term_idx, local_values, local_tape, &local_tape_pos, &next_var_idx, max_tape_size_per_scenario);
    
    // Store the option price
    outputs->option_prices[scenario_id] = local_values[price_idx];
    
    // Store tape position for reverse pass
    tape_positions[scenario_id] = local_tape_pos;
    
    // For now, set Greeks to 0 - they will be computed by reverse pass
    outputs->deltas[scenario_id] = 0.0;
    outputs->vegas[scenario_id] = 0.0;
    outputs->gammas[scenario_id] = 0.0;
    outputs->thetas[scenario_id] = 0.0;
    outputs->rhos[scenario_id] = 0.0;
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
    
    void launch_batch_aad_reverse(
        const GPUTapeEntry* d_tape,
        double* d_values,
        double* d_adjoints,
        const int* d_tape_positions,
        const int* d_variable_indices,
        BatchOutputs* d_outputs,
        int num_scenarios,
        int max_tape_size_per_scenario,
        int max_vars_per_scenario)
    {
        if (num_scenarios <= 0) return;
        
        int block_size = 256;
        int grid_size = (num_scenarios + block_size - 1) / block_size;
        
        batch_aad_reverse_kernel<<<grid_size, block_size>>>(
            d_tape, d_values, d_adjoints, d_tape_positions, d_variable_indices, d_outputs,
            num_scenarios, max_tape_size_per_scenario, max_vars_per_scenario);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in batch_aad_reverse: %s\n", cudaGetErrorString(error));
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
