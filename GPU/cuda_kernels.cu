
// ===== cuda_kernels.cu =====
#include "AADTypes.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ double device_erf(double x) {
    // Abramowitz and Stegun approximation
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
    
    int sign = 1;
    if (x < 0) sign = -1;
    x = fabs(x);
    
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    
    return sign * y;
}

__device__ double device_norm_cdf(double x) {
    return 0.5 * (1.0 + device_erf(x / sqrt(2.0)));
}

__global__ void propagate_kernel(
    const GPUTapeEntry* tape,
    double* values,
    double* adjoints,
    int tape_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= tape_size) return;
    
    int reverse_idx = tape_size - 1 - idx;
    const GPUTapeEntry& entry = tape[reverse_idx];
    
    double result_adj = adjoints[entry.result_idx];
    
    if (entry.input1_idx >= 0) {
        atomicAdd(&adjoints[entry.input1_idx], result_adj * entry.partial1);
    }
    
    if (entry.input2_idx >= 0) {
        atomicAdd(&adjoints[entry.input2_idx], result_adj * entry.partial2);
    }
}

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
    
    double sqrtT = sqrt(T);
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    
    double N_d1 = device_norm_cdf(d1);
    double N_d2 = device_norm_cdf(d2);
    
    double price = S * N_d1 - K * exp(-r * T) * N_d2;
    prices[idx] = price;
    
    // Greeks
    double phi_d1 = exp(-0.5 * d1 * d1) / sqrt(2.0 * M_PI);
    deltas[idx] = N_d1;
    vegas[idx] = S * phi_d1 * sqrtT;
    gammas[idx] = phi_d1 / (S * sigma * sqrtT);
}

extern "C" {
    void launch_propagate_kernel(
        const GPUTapeEntry* d_tape,
        double* d_values,
        double* d_adjoints,
        int tape_size)
    {
        int block_size = 256;
        int grid_size = (tape_size + block_size - 1) / block_size;
        
        propagate_kernel<<<grid_size, block_size>>>(
            d_tape, d_values, d_adjoints, tape_size);
        
        cudaDeviceSynchronize();
    }
    
    void launch_batch_blackscholes(
        double* d_S, double* d_K, double* d_T, double* d_r, double* d_sigma,
        double* d_prices, double* d_deltas, double* d_vegas, double* d_gammas,
        int num_scenarios)
    {
        int block_size = 256;
        int grid_size = (num_scenarios + block_size - 1) / block_size;
        
        batch_blackscholes_kernel<<<grid_size, block_size>>>(
            d_S, d_K, d_T, d_r, d_sigma,
            d_prices, d_deltas, d_vegas, d_gammas,
            num_scenarios);
        
        cudaDeviceSynchronize();
    }
}
