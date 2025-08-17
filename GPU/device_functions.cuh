#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "AADTypes.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// AAD tape recording functions
__device__ inline int record_constant(double value, double* values, int* next_var_idx) {
    int idx = atomicAdd(next_var_idx, 1);
    values[idx] = value;
    return idx;
}

__device__ inline int record_unary_op(
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

__device__ inline int record_binary_op(
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

// AAD arithmetic operations
__device__ inline int aad_add(int a_idx, int b_idx, double* values,
                             GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double a_val = values[a_idx];
    double b_val = values[b_idx];
    double result = a_val + b_val;
    return record_binary_op(AADOpType::ADD, a_idx, b_idx, result, 1.0, 1.0,
                           tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_sub(int a_idx, int b_idx, double* values,
                             GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double a_val = values[a_idx];
    double b_val = values[b_idx];
    double result = a_val - b_val;
    return record_binary_op(AADOpType::SUB, a_idx, b_idx, result, 1.0, -1.0,
                           tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_mul(int a_idx, int b_idx, double* values,
                             GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double a_val = values[a_idx];
    double b_val = values[b_idx];
    double result = a_val * b_val;
    return record_binary_op(AADOpType::MUL, a_idx, b_idx, result, b_val, a_val,
                           tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_div(int a_idx, int b_idx, double* values,
                             GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double a_val = values[a_idx];
    double b_val = values[b_idx];
    double result = safe_divide(a_val, b_val);
    double partial1 = safe_divide(1.0, b_val);
    double partial2 = safe_divide(-a_val, b_val * b_val);
    return record_binary_op(AADOpType::DIV, a_idx, b_idx, result, partial1, partial2,
                           tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_log(int x_idx, double* values,
                             GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = safe_log(x_val);
    double partial = (x_val > 1e-15) ? 1.0 / x_val : 1.0 / 1e-15;
    return record_unary_op(AADOpType::LOG, x_idx, result, partial,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_exp(int x_idx, double* values,
                             GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = safe_exp(x_val);
    return record_unary_op(AADOpType::EXP, x_idx, result, result,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_sqrt(int x_idx, double* values,
                              GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = safe_sqrt(x_val);
    double partial = (result > 1e-15) ? 0.5 / result : 0.0;
    return record_unary_op(AADOpType::SQRT, x_idx, result, partial,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_norm_cdf(int x_idx, double* values,
                                  GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = device_norm_cdf(x_val);
    double partial = device_norm_pdf(x_val); // Derivative of CDF is PDF
    return record_unary_op(AADOpType::NORM_CDF, x_idx, result, partial,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_mul_const(int x_idx, double constant, double* values,
                                   GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = x_val * constant;
    return record_unary_op(AADOpType::MUL, x_idx, result, constant,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

__device__ inline int aad_neg(int x_idx, double* values,
                             GPUTapeEntry* tape, int* tape_pos, int* next_var_idx, int max_tape_size) {
    double x_val = values[x_idx];
    double result = -x_val;
    return record_unary_op(AADOpType::NEG, x_idx, result, -1.0,
                          tape, values, tape_pos, next_var_idx, max_tape_size);
}

// Safe mathematical operations for numerical stability
__device__ inline double safe_log(double x) {
    const double min_val = 1e-15;
    return (x > min_val) ? log(x) : log(min_val);
}

__device__ inline double safe_exp(double x) {
    const double max_exp = 700.0;  // Avoid overflow
    const double min_exp = -700.0; // Avoid underflow
    return exp(fmax(min_exp, fmin(max_exp, x)));
}

__device__ inline double safe_sqrt(double x) {
    return sqrt(fmax(0.0, x));
}

__device__ inline double safe_divide(double numerator, double denominator) {
    const double epsilon = 1e-15;
    return (fabs(denominator) > epsilon) ? numerator / denominator : 0.0;
}

// Mathematical functions
__device__ inline double device_erf(double x) {
    // Use CUDA's built-in erf function
    return erf(x);
}

__device__ inline double device_erf_approx(double x) {
    // Abramowitz and Stegun approximation for erf
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    int sign = (x >= 0) ? 1 : -1;
    x = fabs(x);

    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * safe_exp(-x * x);

    return sign * y;
}

__device__ inline double device_norm_cdf(double x) {
    return 0.5 * (1.0 + device_erf(x / sqrt(2.0)));
}

__device__ inline double device_norm_pdf(double x) {
    const double inv_sqrt_2pi = 0.3989422804014327; // 1/sqrt(2*pi)
    return inv_sqrt_2pi * safe_exp(-0.5 * x * x);
}

// Derivative functions
__device__ inline double device_log_derivative(double x) {
    return safe_divide(1.0, x);
}

__device__ inline double device_exp_derivative(double x, double exp_val) {
    return exp_val; // d/dx exp(x) = exp(x)
}

__device__ inline double device_sqrt_derivative(double x, double sqrt_val) {
    return safe_divide(0.5, sqrt_val); // d/dx sqrt(x) = 1/(2*sqrt(x))
}

__device__ inline double device_erf_derivative(double x) {
    return (2.0 / sqrt(M_PI)) * safe_exp(-x * x);
}

// Black-Scholes helper functions
__device__ inline double device_d1(double S, double K, double T, double r, double sigma) {
    return safe_divide(safe_log(S / K) + (r + 0.5 * sigma * sigma) * T, sigma * safe_sqrt(T));
}

__device__ inline double device_d2(double d1, double sigma, double T) {
    return d1 - sigma * safe_sqrt(T);
}

// Black-Scholes calculation function
__device__ inline void device_black_scholes_call(
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
