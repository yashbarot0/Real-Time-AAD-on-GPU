// ===== GPUAADTape.cpp =====
#include "GPUAADTape.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

// CUDA kernel declarations
extern "C" void launch_propagate_kernel(
    const GPUTapeEntry* d_tape,
    double* d_values,
    double* d_adjoints,
    int tape_size);

extern "C" void launch_batch_blackscholes(
    double* d_S, double* d_K, double* d_T, double* d_r, double* d_sigma,
    double* d_prices, double* d_deltas, double* d_vegas, double* d_gammas,
    int num_scenarios);

GPUAADTape::GPUAADTape(int max_vars, int max_ops)
    : max_vars_(max_vars), max_tape_size_(max_ops), num_vars_(0), 
      tape_size_(0), gpu_allocated_(false), d_values_(nullptr), 
      d_adjoints_(nullptr), d_tape_(nullptr)
{
    values_.reserve(max_vars_);
    adjoints_.reserve(max_vars_);
    tape_.reserve(max_tape_size_);
    allocate_gpu_memory();
}

GPUAADTape::~GPUAADTape() {
    free_gpu_memory();
}

void GPUAADTape::allocate_gpu_memory() {
    if (gpu_allocated_) return;
    
    cudaError_t err;
    
    err = cudaMalloc(&d_values_, max_vars_ * sizeof(double));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for values");
    }
    
    err = cudaMalloc(&d_adjoints_, max_vars_ * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_values_);
        throw std::runtime_error("Failed to allocate GPU memory for adjoints");
    }
    
    err = cudaMalloc(&d_tape_, max_tape_size_ * sizeof(GPUTapeEntry));
    if (err != cudaSuccess) {
        cudaFree(d_values_);
        cudaFree(d_adjoints_);
        throw std::runtime_error("Failed to allocate GPU memory for tape");
    }
    
    gpu_allocated_ = true;
}

void GPUAADTape::free_gpu_memory() {
    if (!gpu_allocated_) return;
    
    cudaFree(d_values_);
    cudaFree(d_adjoints_);
    cudaFree(d_tape_);
    
    gpu_allocated_ = false;
}

int GPUAADTape::create_variable(double value) {
    if (num_vars_ >= max_vars_) {
        throw std::runtime_error("Maximum number of variables exceeded");
    }
    
    int idx = num_vars_++;
    values_.push_back(value);
    adjoints_.push_back(0.0);
    
    return idx;
}

void GPUAADTape::set_adjoint(int var_idx, double adj) {
    if (var_idx >= 0 && var_idx < num_vars_) {
        adjoints_[var_idx] = adj;
    }
}

void GPUAADTape::clear_adjoints() {
    std::fill(adjoints_.begin(), adjoints_.end(), 0.0);
}

int GPUAADTape::record_binary_op(AADOpType op_type, int input1, int input2, 
                                 double result_val, double partial1, double partial2) {
    if (tape_size_ >= max_tape_size_) {
        throw std::runtime_error("Maximum tape size exceeded");
    }
    
    int result_idx = create_variable(result_val);
    
    GPUTapeEntry entry;
    entry.result_idx = result_idx;
    entry.op_type = static_cast<int>(op_type);
    entry.input1_idx = input1;
    entry.input2_idx = input2;
    entry.constant = 0.0;
    entry.partial1 = partial1;
    entry.partial2 = partial2;
    
    tape_.push_back(entry);
    tape_size_++;
    
    return result_idx;
}

int GPUAADTape::record_unary_op(AADOpType op_type, int input, double result_val, double partial) {
    if (tape_size_ >= max_tape_size_) {
        throw std::runtime_error("Maximum tape size exceeded");
    }
    
    int result_idx = create_variable(result_val);
    
    GPUTapeEntry entry;
    entry.result_idx = result_idx;
    entry.op_type = static_cast<int>(op_type);
    entry.input1_idx = input;
    entry.input2_idx = -1;
    entry.constant = 0.0;
    entry.partial1 = partial;
    entry.partial2 = 0.0;
    
    tape_.push_back(entry);
    tape_size_++;
    
    return result_idx;
}

int GPUAADTape::record_constant(double value) {
    return create_variable(value);
}

void GPUAADTape::copy_to_gpu() {
    if (!gpu_allocated_) return;
    
    cudaMemcpy(d_values_, values_.data(), num_vars_ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjoints_, adjoints_.data(), num_vars_ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tape_, tape_.data(), tape_size_ * sizeof(GPUTapeEntry), cudaMemcpyHostToDevice);
}

void GPUAADTape::copy_from_gpu() {
    if (!gpu_allocated_) return;
    
    cudaMemcpy(adjoints_.data(), d_adjoints_, num_vars_ * sizeof(double), cudaMemcpyDeviceToHost);
}

void GPUAADTape::propagate_gpu() {
    if (!gpu_allocated_ || tape_size_ == 0) return;
    
    copy_to_gpu();
    launch_propagate_kernel(d_tape_, d_values_, d_adjoints_, tape_size_);
    copy_from_gpu();
}

double GPUAADTape::get_value(int idx) const {
    if (idx >= 0 && idx < num_vars_) {
        return values_[idx];
    }
    return 0.0;
}

double GPUAADTape::get_adjoint(int idx) const {
    if (idx >= 0 && idx < num_vars_) {
        return adjoints_[idx];
    }
    return 0.0;
}

void GPUAADTape::clear_tape() {
    tape_.clear();
    values_.clear();
    adjoints_.clear();
    num_vars_ = 0;
    tape_size_ = 0;
}
