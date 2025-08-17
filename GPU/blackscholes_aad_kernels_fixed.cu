// ===== blackscholes_aad_kernels.cu =====
// GPU Black-Scholes kernels with AAD support
// Implements step-by-step Black-Scholes construction using proven CPU method

#include "AADTypes.h"
#include "device_functions.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

// Local, non-atomic recorders for per-thread AAD tape building
__device__ inline int record_constant_local(double value, double* values, int* next_var_idx) {
    int idx = (*next_var_idx)++;
    values[idx] = value;
    return idx;
}

__device__ inline int record_unary_op_local(
    AADOpType op_type, int input_idx, double result_val, double partial,
    GPUTapeEntry* tape, double* values, int* tape_pos, int* next_var_idx,
    int max_tape_size)
{
    int result_idx = (*next_var_idx)++;
    values[result_idx] = result_val;
    int t = (*tape_pos)++;
    if (t < max_tape_size) {
        tape[t] = GPUTapeEntry(result_idx, op_type, input_idx, -1, 0.0, partial, 0.0);
    }
    return result_idx;
}

__device__ inline int record_binary_op_local(
    AADOpType op_type, int input1_idx, int input2_idx,
    double result_val, double partial1, double partial2,
    GPUTapeEntry* tape, double* values, int* tape_pos, int* next_var_idx,
    int max_tape_size)
{
    int result_idx = (*next_var_idx)++;
    values[result_idx] = result_val;
    int t = (*tape_pos)++;
    if (t < max_tape_size) {
        tape[t] = GPUTapeEntry(result_idx, op_type, input1_idx, input2_idx, 0.0, partial1, partial2);
    }
    return result_idx;
}

// AAD ops (local)
__device__ inline int aad_add_local(int a_idx, int b_idx, double* values,
    GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size)
{
    double a = values[a_idx], b = values[b_idx];
    return record_binary_op_local(AADOpType::ADD, a_idx, b_idx, a + b, 1.0, 1.0,
        tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_sub_local(int a_idx, int b_idx, double* values,
    GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size)
{
    double a = values[a_idx], b = values[b_idx];
    return record_binary_op_local(AADOpType::SUB, a_idx, b_idx, a - b, 1.0, -1.0,
        tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_mul_local(int a_idx, int b_idx, double* values,
    GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size)
{
    double a = values[a_idx], b = values[b_idx];
    return record_binary_op_local(AADOpType::MUL, a_idx, b_idx, a * b, b, a,
        tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_div_local(int a_idx, int b_idx, double* values,
    GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size)
{
    double a = values[a_idx], b = values[b_idx];
    double res = safe_divide(a, b);
    double p1 = safe_divide(1.0, b);
    double p2 = safe_divide(-a, b * b);
    return record_binary_op_local(AADOpType::DIV, a_idx, b_idx, res, p1, p2,
        tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_log_local(int x_idx, double* values,
    GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size)
{
    double x = values[x_idx];
    double res = safe_log(x);
    double p = (x > 1e-15) ? 1.0 / x : 1.0 / 1e-15;
    return record_unary_op_local(AADOpType::LOG, x_idx, res, p,
        tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_exp_local(int x_idx, double* values,
    GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size)
{
    double x = values[x_idx];
    double res = safe_exp(x);
    return record_unary_op_local(AADOpType::EXP, x_idx, res, res,
        tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_sqrt_local(int x_idx, double* values,
    GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size)
{
    double x = values[x_idx];
    double res = safe_sqrt(x);
    double p = (res > 1e-15) ? 0.5 / res : 0.0;
    return record_unary_op_local(AADOpType::SQRT, x_idx, res, p,
        tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_norm_cdf_local(int x_idx, double* values,
    GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size)
{
    double x = values[x_idx];
    double res = device_norm_cdf(x);
    double p = device_norm_pdf(x);
    return record_unary_op_local(AADOpType::NORM_CDF, x_idx, res, p,
        tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_mul_const_local(int x_idx, double c, double* values,
    GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size)
{
    double x = values[x_idx];
    double res = x * c;
    return record_unary_op_local(AADOpType::MUL, x_idx, res, c,
        tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_neg_local(int x_idx, double* values,
    GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size)
{
    double x = values[x_idx];
    double res = -x;
    return record_unary_op_local(AADOpType::NEG, x_idx, res, -1.0,
        tape, values, tape_pos, next_var_idx, max_tape_size);
}

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
    
    // Record input variables in AAD tape (local recorders)
    int S_idx    = record_constant_local(S_val, local_values, &local_var_idx);
    int K_idx    = record_constant_local(K_val, local_values, &local_var_idx);
    int T_idx    = record_constant_local(T_val, local_values, &local_var_idx);
    int r_idx    = record_constant_local(r_val, local_values, &local_var_idx);
    int sigma_idx= record_constant_local(sigma_val, local_values, &local_var_idx);
    
    // Step-by-step Black-Scholes construction using AAD (local ops)
    int T_sqrt_idx          = aad_sqrt_local(T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int sigma_sqrt_T_idx    = aad_mul_local(sigma_idx, T_sqrt_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int S_over_K_idx        = aad_div_local(S_idx, K_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int ln_S_over_K_idx     = aad_log_local(S_over_K_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int sigma_squared_idx   = aad_mul_local(sigma_idx, sigma_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int half_sigma2_idx     = aad_mul_const_local(sigma_squared_idx, 0.5, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int r_plus_half_sigma2  = aad_add_local(r_idx, half_sigma2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int drift_term_idx      = aad_mul_local(r_plus_half_sigma2, T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int d1_num_idx          = aad_add_local(ln_S_over_K_idx, drift_term_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int d1_idx              = aad_div_local(d1_num_idx, sigma_sqrt_T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int d2_idx              = aad_sub_local(d1_idx, sigma_sqrt_T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int N_d1_idx            = aad_norm_cdf_local(d1_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int N_d2_idx            = aad_norm_cdf_local(d2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int neg_r_idx           = aad_neg_local(r_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int neg_rT_idx          = aad_mul_local(neg_r_idx, T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int discount_idx        = aad_exp_local(neg_rT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int S_Nd1_idx           = aad_mul_local(S_idx, N_d1_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int K_disc_idx          = aad_mul_local(K_idx, discount_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int K_disc_Nd2_idx      = aad_mul_local(K_disc_idx, N_d2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int call_price_idx      = aad_sub_local(S_Nd1_idx, K_disc_Nd2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    outputs->option_prices[scenario_id] = local_values[call_price_idx];
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
    
    // Use per-scenario values buffer as adjoint storage during reverse pass
    double* adjoints = local_values;
    
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
        switch (static_cast<int>(entry.op_type)) {
            case static_cast<int>(AADOpType::ADD):
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                adjoints[entry.input2_idx] += result_adjoint * entry.partial2;
                break;
                
            case static_cast<int>(AADOpType::SUB):
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                adjoints[entry.input2_idx] += result_adjoint * entry.partial2;
                break;
                
            case static_cast<int>(AADOpType::MUL):
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                adjoints[entry.input2_idx] += result_adjoint * entry.partial2;
                break;
                
            case static_cast<int>(AADOpType::DIV):
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                adjoints[entry.input2_idx] += result_adjoint * entry.partial2;
                break;
                
            case static_cast<int>(AADOpType::LOG):
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                break;
                
            case static_cast<int>(AADOpType::EXP):
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                break;
                
            case static_cast<int>(AADOpType::SQRT):
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                break;
                
            case static_cast<int>(AADOpType::NORM_CDF):
                adjoints[entry.input1_idx] += result_adjoint * entry.partial1;
                break;
                
            case static_cast<int>(AADOpType::NEG):
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

    // New: uniquely named reverse pass launcher for the fixed AAD kernels
    void launch_bs_aad_reverse_fixed(
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
        
        batch_aad_reverse_kernel<<<grid_size, block_size>>>(
            d_inputs, d_tape, d_values, d_tape_positions, d_outputs,
            num_scenarios, max_tape_size_per_scenario, max_vars_per_scenario);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in launch_bs_aad_reverse_fixed: %s\n", cudaGetErrorString(error));
        }
        
        cudaDeviceSynchronize();
    }
}