// ===== blackscholes_aad_kernels.cu =====
// GPU Black-Scholes kernels with AAD support
// Implements step-by-step Black-Scholes construction using proven CPU method

#include "AADTypes.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Safe mathematical operations for numerical stability
__device__ inline double safe_log(double x) {
    const double min_val = 1e-15;
    return (x > min_val) ? log(x) : log(min_val);
}

__device__ inline double safe_exp(double x) {
    const double max_val = 700.0;  // Prevent overflow
    const double min_val = -700.0; // Prevent underflow
    x = fmax(fmin(x, max_val), min_val);
    return exp(x);
}

__device__ inline double safe_sqrt(double x) {
    return sqrt(fmax(x, 0.0));
}

__device__ inline double safe_divide(double numerator, double denominator) {
    const double min_denom = 1e-15;
    return (fabs(denominator) > min_denom) ? numerator / denominator : 0.0;
}

// Enhanced normal CDF with numerical stability
__device__ double device_norm_cdf(double x) {
    // Handle extreme values
    if (x < -8.0) return 0.0;
    if (x > 8.0) return 1.0;
    
    // Use the relationship: Φ(x) = 0.5 * (1 + erf(x/√2))
    const double sqrt2 = 1.4142135623730951; // √2
    return 0.5 * (1.0 + erf(x / sqrt2));
}

// Normal PDF for derivative calculations
__device__ double device_norm_pdf(double x) {
    const double inv_sqrt_2pi = 0.3989422804014327; // 1/√(2π)
    return inv_sqrt_2pi * safe_exp(-0.5 * x * x);
}

// Thread-local AAD tape recording functions
__device__ int record_constant(double value, double* values, int* next_var_idx) {
    int idx = atomicAdd(next_var_idx, 1);
    values[idx] = value;
    return idx;
}

__device__ int record_unary_op(
    AADOpType op_type, int input_idx, double result_val, double partial,
    GPUTapeEntry* tape, double* values, int* tape_pos, int* next_var_idx,
    int max_tape_size) {
    
    // Get next variable index
    int result_idx = atomicAdd(next_var_idx, 1);
    values[result_idx] = result_val;
    
    // Record tape entry
    int tape_idx = atomicAdd(tape_pos, 1);
    if (tape_idx < max_tape_size) {
        tape[tape_idx] = GPUTapeEntry(result_idx, op_type, input_idx, -1, 0.0, partial, 0.0);
    }
    
    return result_idx;
}

__device__ int record_binary_op(
    AADOpType op_type, int input1_idx, int input2_idx, 
    double result_val, double partial1, double partial2,
    GPUTapeEntry* tape, double* values, int* tape_pos, int* next_var_idx,
    int max_tape_size) {
    
    // Get next variable index
    int result_idx = atomicAdd(next_var_idx, 1);
    values[result_idx] = result_val;
    
    // Record tape entry
    int tape_idx = atomicAdd(tape_pos, 1);
    if (tape_idx < max_tape_size) {
        tape[tape_idx] = GPUTapeEntry(result_idx, op_type, input1_idx, input2_idx, 
                                     0.0, partial1, partial2);
    }
    
    return result_idx;
}

// AAD-aware mathematical operations
__device__ int aad_add(int a_idx, int b_idx, double* values, 
                      GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double a_val = values[a_idx];
    double b_val = values[b_idx];
    double result = a_val + b_val;
    return record_binary_op(AADOpType::ADD, a_idx, b_idx, result, 1.0, 1.0,
                           tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ int aad_sub(int a_idx, int b_idx, double* values,
                      GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double a_val = values[a_idx];
    double b_val = values[b_idx];
    double result = a_val - b_val;
    return record_binary_op(AADOpType::SUB, a_idx, b_idx, result, 1.0, -1.0,
                           tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ int aad_mul(int a_idx, int b_idx, double* values,
                      GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double a_val = values[a_idx];
    double b_val = values[b_idx];
    double result = a_val * b_val;
    return record_binary_op(AADOpType::MUL, a_idx, b_idx, result, b_val, a_val,
                           tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ int aad_div(int a_idx, int b_idx, double* values,
                      GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double a_val = values[a_idx];
    double b_val = values[b_idx];
    double result = safe_divide(a_val, b_val);
    double partial1 = safe_divide(1.0, b_val);
    double partial2 = safe_divide(-a_val, b_val * b_val);
    return record_binary_op(AADOpType::DIV, a_idx, b_idx, result, partial1, partial2,
                           tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ int aad_log(int x_idx, double* values,
                      GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = safe_log(x_val);
    double partial = (x_val > 1e-15) ? 1.0 / x_val : 1.0 / 1e-15;
    return record_unary_op(AADOpType::LOG, x_idx, result, partial,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ int aad_exp(int x_idx, double* values,
                      GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = safe_exp(x_val);
    return record_unary_op(AADOpType::EXP, x_idx, result, result,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ int aad_sqrt(int x_idx, double* values,
                       GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = safe_sqrt(x_val);
    double partial = (result > 1e-15) ? 0.5 / result : 0.0;
    return record_unary_op(AADOpType::SQRT, x_idx, result, partial,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ int aad_norm_cdf(int x_idx, double* values,
                           GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = device_norm_cdf(x_val);
    double partial = device_norm_pdf(x_val); // Derivative of CDF is PDF
    return record_unary_op(AADOpType::NORM_CDF, x_idx, result, partial,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

// Multiply by constant (no tape recording needed)
__device__ int aad_mul_const(int x_idx, double constant, double* values,
                            GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = x_val * constant;
    return record_unary_op(AADOpType::MUL, x_idx, result, constant,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

// Add constant (no tape recording needed for constant part)
__device__ int aad_add_const(int x_idx, double constant, double* values,
                            GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = x_val + constant;
    return record_unary_op(AADOpType::ADD, x_idx, result, 1.0,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

// Negate operation
__device__ int aad_neg(int x_idx, double* values,
                      GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = -x_val;
    return record_unary_op(AADOpType::NEG, x_idx, result, -1.0,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}// Main f
orward pass kernel - implements step-by-step Black-Scholes with AAD
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
    
    // Step-by-step Black-Scholes construction with AAD recording
    // Following the proven CPU implementation pattern
    
    // Create input variables
    int S_idx = local_var_idx++;
    local_values[S_idx] = S_val;
    
    int K_idx = local_var_idx++;
    local_values[K_idx] = K_val;
    
    int T_idx = local_var_idx++;
    local_values[T_idx] = T_val;
    
    int r_idx = local_var_idx++;
    local_values[r_idx] = r_val;
    
    int sigma_idx = local_var_idx++;
    local_values[sigma_idx] = sigma_val;
    
    // Step 1: sqrtT = sqrt(T)
    int sqrtT_idx = aad_sqrt(T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 2: S/K
    int S_over_K_idx = aad_div(S_idx, K_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 3: log(S/K)
    int log_S_over_K_idx = aad_log(S_over_K_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 4: sigma^2
    int sigma_squared_idx = aad_mul(sigma_idx, sigma_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 5: 0.5 * sigma^2
    int half_sigma_squared_idx = aad_mul_const(sigma_squared_idx, 0.5, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 6: r + 0.5 * sigma^2
    int r_plus_half_sigma_squared_idx = aad_add(r_idx, half_sigma_squared_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 7: (r + 0.5 * sigma^2) * T
    int drift_term_idx = aad_mul(r_plus_half_sigma_squared_idx, T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 8: log(S/K) + (r + 0.5 * sigma^2) * T
    int numerator_idx = aad_add(log_S_over_K_idx, drift_term_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 9: sigma * sqrt(T)
    int sigma_sqrtT_idx = aad_mul(sigma_idx, sqrtT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 10: d1 = numerator / (sigma * sqrt(T))
    int d1_idx = aad_div(numerator_idx, sigma_sqrtT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 11: d2 = d1 - sigma * sqrt(T)
    int d2_idx = aad_sub(d1_idx, sigma_sqrtT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 12: N(d1) = norm_cdf(d1)
    int N_d1_idx = aad_norm_cdf(d1_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 13: N(d2) = norm_cdf(d2)
    int N_d2_idx = aad_norm_cdf(d2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 14: S * N(d1)
    int term1_idx = aad_mul(S_idx, N_d1_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 15: -r * T
    int neg_rT_idx = aad_mul(r_idx, T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    neg_rT_idx = aad_neg(neg_rT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 16: exp(-r * T)
    int discount_idx = aad_exp(neg_rT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 17: K * exp(-r * T)
    int discounted_K_idx = aad_mul(K_idx, discount_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 18: K * exp(-r * T) * N(d2)
    int term2_idx = aad_mul(discounted_K_idx, N_d2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Step 19: Final call price = S * N(d1) - K * exp(-r * T) * N(d2)
    int call_price_idx = aad_sub(term1_idx, term2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Store the final price
    outputs->option_prices[scenario_id] = local_values[call_price_idx];
    
    // Store tape position for this scenario
    tape_positions[scenario_id] = local_tape_pos;
    
    // Store variable indices for reverse pass (we'll need these for Greeks computation)
    // For now, we'll compute analytical Greeks as a fallback
    // The reverse pass kernel will compute AAD Greeks
    
    // Compute analytical Greeks for validation
    double d1_val = local_values[d1_idx];
    double d2_val = local_values[d2_idx];
    double N_d1_val = local_values[N_d1_idx];
    double N_d2_val = local_values[N_d2_idx];
    double sqrt_T_val = local_values[sqrtT_idx];
    double discount_val = local_values[discount_idx];
    
    // Calculate PDF for Greeks
    double phi_d1 = device_norm_pdf(d1_val);
    
    // Analytical Greeks (for validation - AAD will compute these in reverse pass)
    outputs->deltas[scenario_id] = N_d1_val;
    outputs->vegas[scenario_id] = S_val * phi_d1 * sqrt_T_val;
    outputs->gammas[scenario_id] = safe_divide(phi_d1, S_val * sigma_val * sqrt_T_val);
    outputs->thetas[scenario_id] = -safe_divide(S_val * phi_d1 * sigma_val, 2.0 * sqrt_T_val) - r_val * K_val * discount_val * N_d2_val;
    outputs->rhos[scenario_id] = K_val * T_val * discount_val * N_d2_val;
}// C
 interface for launching the forward pass kernel
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
}//
 Reverse pass kernel for AAD gradient computation
__global__ void batch_aad_reverse_kernel(
    const GPUTapeEntry* tape,
    double* values,
    double* adjoints,
    const int* tape_sizes,
    const int* variable_indices,
    BatchOutputs* outputs,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario)
{
    int scenario_id = blockIdx.x;
    int tape_idx = threadIdx.x;
    
    if (scenario_id >= num_scenarios) return;
    
    int tape_size = tape_sizes[scenario_id];
    if (tape_idx >= tape_size) return;
    
    // Calculate memory offsets for this scenario
    int tape_offset = scenario_id * max_tape_size_per_scenario;
    int var_offset = scenario_id * max_vars_per_scenario;
    
    // Local pointers for this scenario
    const GPUTapeEntry* local_tape = &tape[tape_offset];
    double* local_adjoints = &adjoints[var_offset];
    
    // Shared memory for local adjoint accumulation to reduce atomic operations
    extern __shared__ double shared_adjoints[];
    
    // Initialize shared memory
    if (tape_idx < max_vars_per_scenario) {
        shared_adjoints[tape_idx] = 0.0;
    }
    __syncthreads();
    
    // Process tape in reverse order for backpropagation
    int reverse_idx = tape_size - 1 - tape_idx;
    
    if (reverse_idx >= 0) {
        const GPUTapeEntry& entry = local_tape[reverse_idx];
        
        // Get the adjoint of the result variable
        double result_adj = local_adjoints[entry.result_idx];
        
        // Skip if no adjoint to propagate
        if (result_adj != 0.0) {
            // Propagate to first input
            if (entry.input1_idx >= 0) {
                double contribution = result_adj * entry.partial1;
                if (contribution != 0.0) {
                    atomicAdd(&local_adjoints[entry.input1_idx], contribution);
                }
            }
            
            // Propagate to second input
            if (entry.input2_idx >= 0) {
                double contribution = result_adj * entry.partial2;
                if (contribution != 0.0) {
                    atomicAdd(&local_adjoints[entry.input2_idx], contribution);
                }
            }
        }
    }
    
    __syncthreads();
    
    // Extract Greeks from adjoints (only thread 0 does this)
    if (tape_idx == 0) {
        // Variable indices: S=0, K=1, T=2, r=3, sigma=4
        outputs->deltas[scenario_id] = local_adjoints[0];   // dP/dS
        outputs->vegas[scenario_id] = local_adjoints[4];    // dP/dsigma
        outputs->thetas[scenario_id] = local_adjoints[2];   // dP/dT
        outputs->rhos[scenario_id] = local_adjoints[3];     // dP/dr
        
        // Gamma requires second-order derivatives, which we'll compute separately
        // For now, keep the analytical gamma from the forward pass
    }
}

// Cooperative reverse pass kernel with better memory access patterns
__global__ void batch_aad_reverse_cooperative_kernel(
    const GPUTapeEntry* tape,
    double* values,
    double* adjoints,
    const int* tape_sizes,
    BatchOutputs* outputs,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario)
{
    int scenario_id = blockIdx.x;
    
    if (scenario_id >= num_scenarios) return;
    
    int tape_size = tape_sizes[scenario_id];
    
    // Calculate memory offsets for this scenario
    int tape_offset = scenario_id * max_tape_size_per_scenario;
    int var_offset = scenario_id * max_vars_per_scenario;
    
    // Local pointers for this scenario
    const GPUTapeEntry* local_tape = &tape[tape_offset];
    double* local_adjoints = &adjoints[var_offset];
    
    // Initialize the adjoint of the final result (option price) to 1.0
    if (threadIdx.x == 0) {
        // The final result is typically the last variable created
        // We need to identify which variable index corresponds to the option price
        // For now, we'll set it based on the tape structure
        if (tape_size > 0) {
            const GPUTapeEntry& last_entry = local_tape[tape_size - 1];
            local_adjoints[last_entry.result_idx] = 1.0;
        }
    }
    __syncthreads();
    
    // Process tape entries in reverse order
    // Each thread processes multiple entries to ensure all are covered
    int entries_per_thread = (tape_size + blockDim.x - 1) / blockDim.x;
    
    for (int i = 0; i < entries_per_thread; i++) {
        int tape_idx = threadIdx.x * entries_per_thread + i;
        int reverse_idx = tape_size - 1 - tape_idx;
        
        if (reverse_idx >= 0 && reverse_idx < tape_size) {
            const GPUTapeEntry& entry = local_tape[reverse_idx];
            
            // Get the adjoint of the result variable
            double result_adj = local_adjoints[entry.result_idx];
            
            // Skip if no adjoint to propagate
            if (result_adj != 0.0) {
                // Propagate to first input
                if (entry.input1_idx >= 0) {
                    double contribution = result_adj * entry.partial1;
                    if (contribution != 0.0) {
                        atomicAdd(&local_adjoints[entry.input1_idx], contribution);
                    }
                }
                
                // Propagate to second input
                if (entry.input2_idx >= 0) {
                    double contribution = result_adj * entry.partial2;
                    if (contribution != 0.0) {
                        atomicAdd(&local_adjoints[entry.input2_idx], contribution);
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Extract Greeks from adjoints (only thread 0 does this)
    if (threadIdx.x == 0) {
        // Variable indices: S=0, K=1, T=2, r=3, sigma=4
        outputs->deltas[scenario_id] = local_adjoints[0];   // dP/dS
        outputs->vegas[scenario_id] = local_adjoints[4];    // dP/dsigma  
        outputs->thetas[scenario_id] = local_adjoints[2];   // dP/dT
        outputs->rhos[scenario_id] = local_adjoints[3];     // dP/dr
    }
}

// Optimized reverse pass with memory bandwidth optimization
__global__ void batch_aad_reverse_optimized_kernel(
    const GPUTapeEntry* tape,
    double* values,
    double* adjoints,
    const int* tape_sizes,
    BatchOutputs* outputs,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario)
{
    int scenario_id = blockIdx.x;
    
    if (scenario_id >= num_scenarios) return;
    
    int tape_size = tape_sizes[scenario_id];
    
    // Calculate memory offsets for this scenario
    int tape_offset = scenario_id * max_tape_size_per_scenario;
    int var_offset = scenario_id * max_vars_per_scenario;
    
    // Local pointers for this scenario
    const GPUTapeEntry* local_tape = &tape[tape_offset];
    double* local_adjoints = &adjoints[var_offset];
    
    // Use shared memory for better memory access patterns
    extern __shared__ double shared_data[];
    double* shared_adjoints = shared_data;
    
    // Initialize shared memory with current adjoints
    for (int i = threadIdx.x; i < max_vars_per_scenario; i += blockDim.x) {
        shared_adjoints[i] = (i < max_vars_per_scenario) ? local_adjoints[i] : 0.0;
    }
    __syncthreads();
    
    // Initialize the final result adjoint
    if (threadIdx.x == 0 && tape_size > 0) {
        const GPUTapeEntry& last_entry = local_tape[tape_size - 1];
        shared_adjoints[last_entry.result_idx] = 1.0;
    }
    __syncthreads();
    
    // Process tape in reverse order with coalesced memory access
    for (int reverse_idx = tape_size - 1; reverse_idx >= 0; reverse_idx--) {
        if (threadIdx.x == 0) {
            const GPUTapeEntry& entry = local_tape[reverse_idx];
            
            double result_adj = shared_adjoints[entry.result_idx];
            
            if (result_adj != 0.0) {
                // Propagate to inputs
                if (entry.input1_idx >= 0) {
                    double contribution = result_adj * entry.partial1;
                    shared_adjoints[entry.input1_idx] += contribution;
                }
                
                if (entry.input2_idx >= 0) {
                    double contribution = result_adj * entry.partial2;
                    shared_adjoints[entry.input2_idx] += contribution;
                }
            }
        }
        __syncthreads();
    }
    
    // Copy results back to global memory
    for (int i = threadIdx.x; i < max_vars_per_scenario; i += blockDim.x) {
        if (i < max_vars_per_scenario) {
            local_adjoints[i] = shared_adjoints[i];
        }
    }
    __syncthreads();
    
    // Extract Greeks (only thread 0)
    if (threadIdx.x == 0) {
        outputs->deltas[scenario_id] = shared_adjoints[0];   // dP/dS
        outputs->vegas[scenario_id] = shared_adjoints[4];    // dP/dsigma
        outputs->thetas[scenario_id] = shared_adjoints[2];   // dP/dT
        outputs->rhos[scenario_id] = shared_adjoints[3];     // dP/dr
    }
}/
/ C interface for launching reverse pass kernels
extern "C" {
    void launch_batch_aad_reverse(
        const GPUTapeEntry* d_tape,
        double* d_values,
        double* d_adjoints,
        const int* d_tape_sizes,
        const int* d_variable_indices,
        BatchOutputs* d_outputs,
        int num_scenarios,
        int max_tape_size_per_scenario,
        int max_vars_per_scenario)
    {
        if (num_scenarios <= 0) return;
        
        int block_size = min(max_tape_size_per_scenario, 1024);
        int grid_size = num_scenarios;
        
        // Shared memory for local adjoint accumulation
        size_t shared_mem_size = max_vars_per_scenario * sizeof(double);
        
        batch_aad_reverse_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_tape, d_values, d_adjoints, d_tape_sizes, d_variable_indices,
            d_outputs, num_scenarios, max_tape_size_per_scenario, max_vars_per_scenario);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in batch_aad_reverse: %s\n", 
                   cudaGetErrorString(error));
        }
        
        cudaDeviceSynchronize();
    }
    
    void launch_batch_aad_reverse_cooperative(
        const GPUTapeEntry* d_tape,
        double* d_values,
        double* d_adjoints,
        const int* d_tape_sizes,
        BatchOutputs* d_outputs,
        int num_scenarios,
        int max_tape_size_per_scenario,
        int max_vars_per_scenario)
    {
        if (num_scenarios <= 0) return;
        
        int block_size = 256;  // Optimal for most GPUs
        int grid_size = num_scenarios;
        
        batch_aad_reverse_cooperative_kernel<<<grid_size, block_size>>>(
            d_tape, d_values, d_adjoints, d_tape_sizes, d_outputs,
            num_scenarios, max_tape_size_per_scenario, max_vars_per_scenario);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in batch_aad_reverse_cooperative: %s\n", 
                   cudaGetErrorString(error));
        }
        
        cudaDeviceSynchronize();
    }
    
    void launch_batch_aad_reverse_optimized(
        const GPUTapeEntry* d_tape,
        double* d_values,
        double* d_adjoints,
        const int* d_tape_sizes,
        BatchOutputs* d_outputs,
        int num_scenarios,
        int max_tape_size_per_scenario,
        int max_vars_per_scenario)
    {
        if (num_scenarios <= 0) return;
        
        int block_size = 256;
        int grid_size = num_scenarios;
        
        // Shared memory for better memory access patterns
        size_t shared_mem_size = max_vars_per_scenario * sizeof(double);
        
        batch_aad_reverse_optimized_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_tape, d_values, d_adjoints, d_tape_sizes, d_outputs,
            num_scenarios, max_tape_size_per_scenario, max_vars_per_scenario);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Kernel launch error in batch_aad_reverse_optimized: %s\n", 
                   cudaGetErrorString(error));
        }
        
        cudaDeviceSynchronize();
    }
}/
/ Enhanced Black-Scholes kernel with comprehensive numerical stability
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
    
    // Bounds checking for array access
    if (!check_array_bounds(scenario_id, num_scenarios)) return;
    
    // Calculate memory offsets for this scenario
    int tape_offset = scenario_id * max_tape_size_per_scenario;
    int var_offset = scenario_id * max_vars_per_scenario;
    
    // Local pointers for this scenario with bounds checking
    GPUTapeEntry* local_tape = &tape[tape_offset];
    double* local_values = &values[var_offset];
    
    // Local counters
    int local_tape_pos = 0;
    int local_var_idx = 0;
    
    // Get input parameters with safe array access
    double S_val = safe_array_read(inputs->spot_prices, scenario_id, num_scenarios, 100.0);
    double K_val = safe_array_read(inputs->strike_prices, scenario_id, num_scenarios, 100.0);
    double T_val = safe_array_read(inputs->times_to_expiry, scenario_id, num_scenarios, 0.25);
    double r_val = safe_array_read(inputs->risk_free_rates, scenario_id, num_scenarios, 0.05);
    double sigma_val = safe_array_read(inputs->volatilities, scenario_id, num_scenarios, 0.2);
    
    // Validate and sanitize parameters
    bool params_valid = validate_and_sanitize_parameters(&S_val, &K_val, &T_val, &r_val, &sigma_val);
    
    // Initialize error flags
    int local_error_flags = params_valid ? 0 : 0x01;
    
    // Handle edge cases with enhanced stability
    double price, delta, vega, gamma, theta, rho;
    
    // Check for edge cases first
    bool is_edge_case = (T_val <= 1e-8) || (sigma_val <= 1e-8) || (sigma_val >= 10.0) || 
                       (S_val / K_val < 0.01) || (S_val / K_val > 100.0);
    
    if (is_edge_case) {
        handle_black_scholes_edge_cases(S_val, K_val, T_val, r_val, sigma_val,
                                       &price, &delta, &vega, &gamma, &theta, &rho);
        
        // Apply graceful degradation
        apply_graceful_degradation(&price, &delta, &vega, &gamma, &theta, &rho);
        
        // Store results with bounds checking
        safe_array_write(outputs->option_prices, scenario_id, num_scenarios, price);
        safe_array_write(outputs->deltas, scenario_id, num_scenarios, delta);
        safe_array_write(outputs->vegas, scenario_id, num_scenarios, vega);
        safe_array_write(outputs->gammas, scenario_id, num_scenarios, gamma);
        safe_array_write(outputs->thetas, scenario_id, num_scenarios, theta);
        safe_array_write(outputs->rhos, scenario_id, num_scenarios, rho);
        
        tape_positions[scenario_id] = 0;
        if (error_flags) error_flags[scenario_id] = local_error_flags | 0x02; // Edge case flag
        return;
    }
    
    // Normal Black-Scholes computation with enhanced numerical stability
    // Create input variables with bounds checking
    if (local_var_idx >= max_vars_per_scenario) {
        local_error_flags |= 0x04; // Variable overflow
        if (error_flags) error_flags[scenario_id] = local_error_flags;
        return;
    }
    
    int S_idx = local_var_idx++;
    local_values[S_idx] = S_val;
    
    int K_idx = local_var_idx++;
    local_values[K_idx] = K_val;
    
    int T_idx = local_var_idx++;
    local_values[T_idx] = T_val;
    
    int r_idx = local_var_idx++;
    local_values[r_idx] = r_val;
    
    int sigma_idx = local_var_idx++;
    local_values[sigma_idx] = sigma_val;
    
    // Step-by-step Black-Scholes with enhanced error checking
    int sqrtT_idx = aad_sqrt(T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    if (local_tape_pos >= max_tape_size_per_scenario) {
        local_error_flags |= 0x08; // Tape overflow
        if (error_flags) error_flags[scenario_id] = local_error_flags;
        return;
    }
    
    int S_over_K_idx = aad_div(S_idx, K_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int log_S_over_K_idx = aad_log(S_over_K_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int sigma_squared_idx = aad_mul(sigma_idx, sigma_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int half_sigma_squared_idx = aad_mul_const(sigma_squared_idx, 0.5, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int r_plus_half_sigma_squared_idx = aad_add(r_idx, half_sigma_squared_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int drift_term_idx = aad_mul(r_plus_half_sigma_squared_idx, T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int numerator_idx = aad_add(log_S_over_K_idx, drift_term_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int sigma_sqrtT_idx = aad_mul(sigma_idx, sqrtT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int d1_idx = aad_div(numerator_idx, sigma_sqrtT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int d2_idx = aad_sub(d1_idx, sigma_sqrtT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int N_d1_idx = aad_norm_cdf(d1_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int N_d2_idx = aad_norm_cdf(d2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int term1_idx = aad_mul(S_idx, N_d1_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int neg_rT_idx = aad_mul(r_idx, T_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    neg_rT_idx = aad_neg(neg_rT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int discount_idx = aad_exp(neg_rT_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int discounted_K_idx = aad_mul(K_idx, discount_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int term2_idx = aad_mul(discounted_K_idx, N_d2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    int call_price_idx = aad_sub(term1_idx, term2_idx, local_values, local_tape, &local_tape_pos, &local_var_idx, max_tape_size_per_scenario);
    
    // Extract results with numerical validation
    price = local_values[call_price_idx];
    
    // Compute analytical Greeks for validation and fallback
    double d1_val = local_values[d1_idx];
    double d2_val = local_values[d2_idx];
    double N_d1_val = local_values[N_d1_idx];
    double N_d2_val = local_values[N_d2_idx];
    double sqrt_T_val = local_values[sqrtT_idx];
    double discount_val = local_values[discount_idx];
    
    double phi_d1 = safe_norm_pdf(d1_val);
    
    delta = N_d1_val;
    vega = S_val * phi_d1 * sqrt_T_val;
    gamma = safe_divide_enhanced(phi_d1, S_val * sigma_val * sqrt_T_val);
    theta = -safe_divide_enhanced(S_val * phi_d1 * sigma_val, 2.0 * sqrt_T_val) - r_val * K_val * discount_val * N_d2_val;
    rho = K_val * T_val * discount_val * N_d2_val;
    
    // Apply graceful degradation
    apply_graceful_degradation(&price, &delta, &vega, &gamma, &theta, &rho);
    
    // Detect numerical errors
    local_error_flags |= detect_numerical_errors(S_val, K_val, T_val, r_val, sigma_val,
                                                 price, delta, vega, gamma, theta, rho);
    
    // Store results with bounds checking
    safe_array_write(outputs->option_prices, scenario_id, num_scenarios, price);
    safe_array_write(outputs->deltas, scenario_id, num_scenarios, delta);
    safe_array_write(outputs->vegas, scenario_id, num_scenarios, vega);
    safe_array_write(outputs->gammas, scenario_id, num_scenarios, gamma);
    safe_array_write(outputs->thetas, scenario_id, num_scenarios, theta);
    safe_array_write(outputs->rhos, scenario_id, num_scenarios, rho);
    
    // Store tape position and error flags
    tape_positions[scenario_id] = local_tape_pos;
    if (error_flags) error_flags[scenario_id] = local_error_flags;
}

// C interface for the stable kernel
extern "C" {
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
}