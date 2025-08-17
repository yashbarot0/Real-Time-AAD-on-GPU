// ===== blackscholes_aad_kernels.cu =====
// GPU Black-Scholes kernels with AAD support
// Implements step-by-step Black-Scholes construction using proven CPU method

#include "AADTypes.h"
#include "device_functions.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

// Main forward pass kernel - implements step-by-step Black-Scholes with AAD
__global__ void batch_blackscholes_forward_kernel(
    const BatchInputs* inputs,
    GPUTapeEntry* tape,
    double* values,
    int* tape_positions,
    BatchOutputs* outputs,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario)
{
    int scenario_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (scenario_id >= num_scenarios) return;
    
    // Calculate memory offsets for this scenario
    int tape_offset = scenario_id * max_tape_size_per_scenario;
    int var_offset = scenario_id * max_vars_per_scenario;
    
    // Local pointers for this scenario
    GPUTapeEntry* local_tape = &tape[tape_offset];
    double* local_values = &values[var_offset];
    
    // Local counters (thread-local, no atomics needed within scenario)
    int local_tape_pos = 0;
    int local_var_idx = 0;
    
    // Get input parameters for this scenario
    double S_val = inputs->spot_prices[scenario_id];
    double K_val = inputs->strike_prices[scenario_id];
    double T_val = inputs->times_to_expiry[scenario_id];
    double r_val = inputs->risk_free_rates[scenario_id];
    double sigma_val = inputs->volatilities[scenario_id];
    
    // Handle edge cases first
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
    
    // Record input variables in AAD tape
    int S_idx = record_constant(S_val, local_values, &local_var_idx);
    int K_idx = record_constant(K_val, local_values, &local_var_idx);
    int T_idx = record_constant(T_val, local_values, &local_var_idx);
    int r_idx = record_constant(r_val, local_values, &local_var_idx);
    int sigma_idx = record_constant(sigma_val, local_values, &local_var_idx);
    
    // Step-by-step Black-Scholes construction using AAD
    // Step 1: Calculate sigma * sqrt(T)
    int T_sqrt_idx = aad_sqrt(T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int sigma_sqrt_T_idx = aad_mul(sigma_idx, T_sqrt_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 2: Calculate ln(S/K)
    int S_over_K_idx = aad_div(S_idx, K_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int ln_S_over_K_idx = aad_log(S_over_K_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 3: Calculate (r + 0.5*sigma^2)*T
    int sigma_squared_idx = aad_mul(sigma_idx, sigma_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int half_sigma_squared_idx = aad_mul_const(sigma_squared_idx, 0.5, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int r_plus_half_sigma2_idx = aad_add(r_idx, half_sigma_squared_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int drift_term_idx = aad_mul(r_plus_half_sigma2_idx, T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 4: Calculate d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
    int d1_numerator_idx = aad_add(ln_S_over_K_idx, drift_term_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int d1_idx = aad_div(d1_numerator_idx, sigma_sqrt_T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 5: Calculate d2 = d1 - sigma*sqrt(T)
    int d2_idx = aad_sub(d1_idx, sigma_sqrt_T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 6: Calculate N(d1) and N(d2)
    int N_d1_idx = aad_norm_cdf(d1_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int N_d2_idx = aad_norm_cdf(d2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 7: Calculate exp(-r*T)
    int neg_r_idx = aad_neg(r_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int neg_rT_idx = aad_mul(neg_r_idx, T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int discount_idx = aad_exp(neg_rT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 8: Calculate S*N(d1)
    int S_N_d1_idx = aad_mul(S_idx, N_d1_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 9: Calculate K*exp(-r*T)*N(d2)
    int K_discount_idx = aad_mul(K_idx, discount_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int K_discount_N_d2_idx = aad_mul(K_discount_idx, N_d2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 10: Calculate final call price = S*N(d1) - K*exp(-r*T)*N(d2)
    int call_price_idx = aad_sub(S_N_d1_idx, K_discount_N_d2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Store the forward pass result
    outputs->option_prices[scenario_id] = local_values[call_price_idx];
    
    // Store tape position for reverse pass
    tape_positions[scenario_id] = local_tape_pos;
    
    // Initialize Greeks to zero (will be computed by reverse pass)
    outputs->deltas[scenario_id] = 0.0;
    outputs->vegas[scenario_id] = 0.0;
    outputs->gammas[scenario_id] = 0.0;
    outputs->thetas[scenario_id] = 0.0;
    outputs->rhos[scenario_id] = 0.0;
}

// Reverse pass kernel for computing Greeks via AAD
__global__ void batch_aad_reverse_kernel(
    const BatchInputs* inputs,
    GPUTapeEntry* tape,
    double* values,
    int* tape_positions,
    BatchOutputs* outputs,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario)
{
    int scenario_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (scenario_id >= num_scenarios) return;
    
    int tape_size = tape_positions[scenario_id];
    if (tape_size == 0) return; // No tape to process
    
    // Calculate memory offsets for this scenario
    int tape_offset = scenario_id * max_tape_size_per_scenario;
    int var_offset = scenario_id * max_vars_per_scenario;
    
    // Local pointers for this scenario
    GPUTapeEntry* local_tape = &tape[tape_offset];
    double* local_values = &values[var_offset];
    
    // Allocate adjoint array (local to this thread)
    extern __shared__ double shared_adjoints[];
    double* adjoints = &shared_adjoints[threadIdx.x * max_vars_per_scenario];
    
    // Initialize adjoints to zero
    for (int i = 0; i < max_vars_per_scenario; i++) {
        adjoints[i] = 0.0;
    }
    
    // Set adjoint of output variable (call price) to 1.0
    // The call price is the last variable created
    int output_var_idx = tape_size > 0 ? local_tape[tape_size - 1].result_idx : 0;
    adjoints[output_var_idx] = 1.0;
    
    // Reverse pass: traverse tape backwards
    for (int i = tape_size - 1; i >= 0; i--) {
        GPUTapeEntry& entry = local_tape[i];
        double result_adjoint = adjoints[entry.result_idx];
        
        if (result_adjoint == 0.0) continue; // Skip if no gradient flows through
        
        // Propagate gradients based on operation type
        switch (entry.op_type) {
            case AADOpType::ADD:
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                adjoints[entry.input2_idx] += result_adjoint * entry.partial2;
                break;
                
            case AADOpType::SUB:
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                adjoints[entry.input2_idx] += result_adjoint * entry.partial2;
                break;
                
            case AADOpType::MUL:
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                adjoints[entry.input2_idx] += result_adjoint * entry.partial2;
                break;
                
            case AADOpType::DIV:
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                adjoints[entry.input2_idx] += result_adjoint * entry.partial2;
                break;
                
            case AADOpType::LOG:
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                break;
                
            case AADOpType::EXP:
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                break;
                
            case AADOpType::SQRT:
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                break;
                
            case AADOpType::NORM_CDF:
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                break;
                
            case AADOpType::NEG:
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                break;
                
            default:
                // Handle other operations as needed
                break;
        }
    }
    
    // Extract Greeks from adjoints of input variables
    // Input variables are at indices 0-4: S, K, T, r, sigma
    outputs->deltas[scenario_id] = adjoints[0];  // ∂Price/∂S
    outputs->rhos[scenario_id] = adjoints[3];    // ∂Price/∂r
    outputs->vegas[scenario_id] = adjoints[4];   // ∂Price/∂σ
    
    // For Theta, we need ∂Price/∂T but with negative sign (time decay)
    outputs->thetas[scenario_id] = -adjoints[2]; // -∂Price/∂T
    
    // Gamma requires second-order derivatives (not implemented in first-order AAD)
    outputs->gammas[scenario_id] = 0.0;
}

// Enhanced Black-Scholes kernel with comprehensive numerical stability
__global__ void batch_blackscholes_forward_stable_kernel(
    const BatchInputs* inputs,
    GPUTapeEntry* tape,
    double* values,
    int* tape_positions,
    BatchOutputs* outputs,
    int* error_flags,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario)
{
    int scenario_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (scenario_id >= num_scenarios) return;
    
    // Get input parameters
    double S_val = inputs->spot_prices[scenario_id];
    double K_val = inputs->strike_prices[scenario_id];
    double T_val = inputs->times_to_expiry[scenario_id];
    double r_val = inputs->risk_free_rates[scenario_id];
    double sigma_val = inputs->volatilities[scenario_id];
    
    // Use enhanced Black-Scholes calculation with numerical stability
    double price, delta, vega, gamma, theta, rho;
    device_black_scholes_call(S_val, K_val, T_val, r_val, sigma_val,
                             &price, &delta, &vega, &gamma, &theta, &rho);
    
    // Store results
    outputs->option_prices[scenario_id] = price;
    outputs->deltas[scenario_id] = delta;
    outputs->vegas[scenario_id] = vega;
    outputs->gammas[scenario_id] = gamma;
    outputs->thetas[scenario_id] = theta;
    outputs->rhos[scenario_id] = rho;
    
    // Store tape position (0 for now since we're using analytical calculation)
    tape_positions[scenario_id] = 0;
    if (error_flags) error_flags[scenario_id] = 0;
}

// C interface for launching the forward pass kernel
extern "C" {
    void launch_batch_blackscholes_forward(
        const BatchInputs* d_inputs,
        GPUTapeEntry* d_tape,
        double* d_values,
        int* d_tape_positions,
        BatchOutputs* d_outputs,
        int num_scenarios,
        int max_tape_size_per_scenario,
        int max_vars_per_scenario)
    {
        if (num_scenarios <= 0) return;
        
        // Calculate optimal block size
        int block_size = 256;
        int grid_size = (num_scenarios + block_size - 1) / block_size;
        
        // Launch the forward pass kernel
        batch_blackscholes_forward_kernel<<<grid_size, block_size>>>(
            d_inputs, d_tape, d_values, d_tape_positions, d_outputs,
            num_scenarios, max_tape_size_per_scenario, max_vars_per_scenario);
        
        // Check for kernel launch errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in batch_blackscholes_forward: %s\n", 
                   cudaGetErrorString(error));
        }
        
        // Synchronize to ensure completion
        cudaDeviceSynchronize();
    }
    
    void launch_batch_blackscholes_forward_stable(
        const BatchInputs* d_inputs,
        GPUTapeEntry* d_tape,
        double* d_values,
        int* d_tape_positions,
        BatchOutputs* d_outputs,
        int* d_error_flags,
        int num_scenarios,
        int max_tape_size_per_scenario,
        int max_vars_per_scenario)
    {
        if (num_scenarios <= 0) return;
        
        int block_size = 256;
        int grid_size = (num_scenarios + block_size - 1) / block_size;
        
        batch_blackscholes_forward_stable_kernel<<<grid_size, block_size>>>(
            d_inputs, d_tape, d_values, d_tape_positions, d_outputs, d_error_flags,
            num_scenarios, max_tape_size_per_scenario, max_vars_per_scenario);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in batch_blackscholes_forward_stable: %s\n", 
                   cudaGetErrorString(error));
        }
        
        cudaDeviceSynchronize();
    }
    
    void launch_batch_aad_reverse(
        const BatchInputs* d_inputs,
        GPUTapeEntry* d_tape,
        double* d_values,
        int* d_tape_positions,
        BatchOutputs* d_outputs,
        int num_scenarios,
        int max_tape_size_per_scenario,
        int max_vars_per_scenario)
    {
        if (num_scenarios <= 0) return;
        
        int block_size = 256;
        int grid_size = (num_scenarios + block_size - 1) / block_size;
        
        // Calculate shared memory size for adjoints
        size_t shared_mem_size = block_size * max_vars_per_scenario * sizeof(double);
        
        batch_aad_reverse_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_inputs, d_tape, d_values, d_tape_positions, d_outputs,
            num_scenarios, max_tape_size_per_scenario, max_vars_per_scenario);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in batch_aad_reverse: %s\n", 
                   cudaGetErrorString(error));
        }
        
        cudaDeviceSynchronize();
    }
}