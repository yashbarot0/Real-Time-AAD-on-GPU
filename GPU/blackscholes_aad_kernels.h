// ===== blackscholes_aad_kernels.h =====
// Header for GPU Black-Scholes AAD kernels

#pragma once

#include "AADTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward pass kernel functions
void launch_batch_blackscholes_forward(
    const BatchInputs* d_inputs,
    GPUTapeEntry* d_tape,
    double* d_values,
    int* d_tape_positions,
    BatchOutputs* d_outputs,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario);

void launch_batch_blackscholes_forward_stable(
    const BatchInputs* d_inputs,
    GPUTapeEntry* d_tape,
    double* d_values,
    int* d_tape_positions,
    BatchOutputs* d_outputs,
    int* d_error_flags,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario);

// Reverse pass kernel functions
void launch_batch_aad_reverse(
    const GPUTapeEntry* d_tape,
    double* d_values,
    double* d_adjoints,
    const int* d_tape_positions,
    const int* d_variable_indices,
    BatchOutputs* d_outputs,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario);

void launch_batch_aad_reverse_cooperative(
    const GPUTapeEntry* d_tape,
    double* d_values,
    double* d_adjoints,
    const int* d_tape_sizes,
    BatchOutputs* d_outputs,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario);

void launch_batch_aad_reverse_optimized(
    const GPUTapeEntry* d_tape,
    double* d_values,
    double* d_adjoints,
    const int* d_tape_sizes,
    BatchOutputs* d_outputs,
    int num_scenarios,
    int max_tape_size_per_scenario,
    int max_vars_per_scenario);

// Numerical stability functions (device functions, included for completeness)
// These are defined in numerical_stability.cu and used internally by kernels

#ifdef __cplusplus
}
#endif