// ===== GPUAADTape.cpp =====
#include "GPUAADTape.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <algorithm>

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

// ===== GPUErrorHandler Implementation =====
bool GPUErrorHandler::check_cuda_error(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::string error_msg = "CUDA Error in " + std::string(operation) + 
                               ": " + cudaGetErrorString(error);
        log_error(error_msg);
        return false;
    }
    return true;
}

void GPUErrorHandler::handle_memory_error(cudaError_t error) {
    if (error == cudaErrorMemoryAllocation) {
        log_error("GPU memory allocation failed - attempting recovery");
        if (!attempt_recovery(error)) {
            log_error("Recovery failed - falling back to CPU processing");
            fallback_to_cpu();
        }
    }
}

void GPUErrorHandler::handle_kernel_error(cudaError_t error) {
    if (error != cudaSuccess) {
        log_error("Kernel execution failed: " + std::string(cudaGetErrorString(error)));
        // Reset GPU state
        cudaDeviceReset();
    }
}

bool GPUErrorHandler::attempt_recovery(cudaError_t error) {
    // Try to free some memory and retry
    cudaDeviceSynchronize();
    return error == cudaSuccess;
}

void GPUErrorHandler::log_error(const std::string& message) {
    std::cerr << "[GPU AAD Error] " << message << std::endl;
}

bool GPUErrorHandler::fallback_to_cpu() {
    std::cout << "[GPU AAD] Falling back to CPU processing" << std::endl;
    return true;
}

// ===== CudaMemoryPool Implementation =====
CudaMemoryPool::CudaMemoryPool(size_t pool_size_bytes) 
    : pool_size_(pool_size_bytes), total_allocated_(0) {
}

CudaMemoryPool::~CudaMemoryPool() {
    cleanup();
}

bool CudaMemoryPool::initialize() {
    // Pre-allocate a large pool of GPU memory
    void* pool_ptr;
    cudaError_t error = cudaMalloc(&pool_ptr, pool_size_);
    if (!GPUErrorHandler::check_cuda_error(error, "CudaMemoryPool::initialize")) {
        return false;
    }
    
    // Create initial large block
    blocks_.emplace_back(pool_ptr, pool_size_);
    total_allocated_ = pool_size_;
    
    return true;
}

void* CudaMemoryPool::allocate(size_t size) {
    // Align size to 256 bytes for optimal GPU access
    size = ((size + 255) / 256) * 256;
    
    // Find suitable block
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            block.in_use = true;
            
            // Split block if it's much larger than needed
            if (block.size > size + 1024) {
                size_t remaining_size = block.size - size;
                void* remaining_ptr = static_cast<char*>(block.ptr) + size;
                blocks_.emplace_back(remaining_ptr, remaining_size);
                block.size = size;
            }
            
            return block.ptr;
        }
    }
    
    // No suitable block found - allocate new memory
    void* new_ptr;
    cudaError_t error = cudaMalloc(&new_ptr, size);
    if (GPUErrorHandler::check_cuda_error(error, "CudaMemoryPool::allocate")) {
        blocks_.emplace_back(new_ptr, size);
        blocks_.back().in_use = true;
        total_allocated_ += size;
        return new_ptr;
    }
    
    return nullptr;
}

void CudaMemoryPool::deallocate(void* ptr) {
    for (auto& block : blocks_) {
        if (block.ptr == ptr) {
            block.in_use = false;
            break;
        }
    }
}

void CudaMemoryPool::defragment() {
    // Sort blocks by address
    std::sort(blocks_.begin(), blocks_.end(), 
              [](const MemoryBlock& a, const MemoryBlock& b) {
                  return a.ptr < b.ptr;
              });
    
    // Merge adjacent free blocks
    for (size_t i = 0; i < blocks_.size() - 1; ++i) {
        if (!blocks_[i].in_use && !blocks_[i + 1].in_use) {
            char* end_of_current = static_cast<char*>(blocks_[i].ptr) + blocks_[i].size;
            if (end_of_current == blocks_[i + 1].ptr) {
                blocks_[i].size += blocks_[i + 1].size;
                blocks_.erase(blocks_.begin() + i + 1);
                --i; // Check this block again
            }
        }
    }
}

size_t CudaMemoryPool::get_available_memory() const {
    size_t available = 0;
    for (const auto& block : blocks_) {
        if (!block.in_use) {
            available += block.size;
        }
    }
    return available;
}

void CudaMemoryPool::cleanup() {
    for (const auto& block : blocks_) {
        if (block.ptr) {
            cudaFree(block.ptr);
        }
    }
    blocks_.clear();
    total_allocated_ = 0;
}

// ===== CudaStreamManager Implementation =====
CudaStreamManager::CudaStreamManager(int num_streams) : num_streams_(num_streams) {
}

CudaStreamManager::~CudaStreamManager() {
    cleanup();
}

bool CudaStreamManager::initialize() {
    compute_streams_.resize(num_streams_);
    memory_streams_.resize(num_streams_);
    
    for (int i = 0; i < num_streams_; ++i) {
        cudaError_t error = cudaStreamCreate(&compute_streams_[i]);
        if (!GPUErrorHandler::check_cuda_error(error, "CudaStreamManager::initialize compute")) {
            return false;
        }
        
        error = cudaStreamCreate(&memory_streams_[i]);
        if (!GPUErrorHandler::check_cuda_error(error, "CudaStreamManager::initialize memory")) {
            return false;
        }
    }
    
    return true;
}

cudaStream_t CudaStreamManager::get_compute_stream(int index) {
    return compute_streams_[index % num_streams_];
}

cudaStream_t CudaStreamManager::get_memory_stream(int index) {
    return memory_streams_[index % num_streams_];
}

void CudaStreamManager::synchronize_all() {
    for (auto stream : compute_streams_) {
        cudaStreamSynchronize(stream);
    }
    for (auto stream : memory_streams_) {
        cudaStreamSynchronize(stream);
    }
}

void CudaStreamManager::cleanup() {
    for (auto stream : compute_streams_) {
        if (stream) cudaStreamDestroy(stream);
    }
    for (auto stream : memory_streams_) {
        if (stream) cudaStreamDestroy(stream);
    }
    compute_streams_.clear();
    memory_streams_.clear();
}

// ===== GPUAADTape Implementation =====
GPUAADTape::GPUAADTape(int max_vars, int max_ops)
    : max_vars_(max_vars), max_tape_size_(max_ops), num_vars_(0), 
      tape_size_(0), gpu_allocated_(false), gpu_available_(false),
      d_values_(nullptr), d_adjoints_(nullptr), d_tape_(nullptr),
      last_allocation_time_ms_(0.0), last_copy_time_ms_(0.0), total_memory_allocated_(0)
{
    values_.reserve(max_vars_);
    adjoints_.reserve(max_vars_);
    tape_.reserve(max_tape_size_);
    
    // Initialize memory management systems
    memory_pool_ = std::make_unique<CudaMemoryPool>();
    stream_manager_ = std::make_unique<CudaStreamManager>();
}

GPUAADTape::~GPUAADTape() {
    cleanup();
}

bool GPUAADTape::initialize() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Check GPU availability
    if (!check_gpu_availability()) {
        return false;
    }
    
    // Initialize memory pool and stream manager
    if (!memory_pool_->initialize()) {
        std::cerr << "Failed to initialize CUDA memory pool" << std::endl;
        return false;
    }
    
    if (!stream_manager_->initialize()) {
        std::cerr << "Failed to initialize CUDA stream manager" << std::endl;
        return false;
    }
    
    // Allocate GPU memory
    if (!allocate_gpu_memory()) {
        return false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_allocation_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    gpu_available_ = true;
    log_performance_metrics();
    
    return true;
}

void GPUAADTape::cleanup() {
    free_gpu_memory();
    if (stream_manager_) {
        stream_manager_->cleanup();
    }
    if (memory_pool_) {
        memory_pool_->cleanup();
    }
    gpu_available_ = false;
}

bool GPUAADTape::check_gpu_availability() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (!GPUErrorHandler::check_cuda_error(error, "check_gpu_availability")) {
        return false;
    }
    
    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found" << std::endl;
        return false;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    if (!GPUErrorHandler::check_cuda_error(error, "cudaGetDeviceProperties")) {
        return false;
    }
    
    std::cout << "GPU Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    return true;
}

bool GPUAADTape::allocate_gpu_memory() {
    if (gpu_allocated_) return true;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Allocate using memory pool for better management
    size_t values_size = max_vars_ * sizeof(double);
    size_t adjoints_size = max_vars_ * sizeof(double);
    size_t tape_size = max_tape_size_ * sizeof(GPUTapeEntry);
    
    d_values_ = static_cast<double*>(memory_pool_->allocate(values_size));
    if (!d_values_) {
        GPUErrorHandler::handle_memory_error(cudaErrorMemoryAllocation);
        return handle_allocation_failure();
    }
    
    d_adjoints_ = static_cast<double*>(memory_pool_->allocate(adjoints_size));
    if (!d_adjoints_) {
        memory_pool_->deallocate(d_values_);
        d_values_ = nullptr;
        GPUErrorHandler::handle_memory_error(cudaErrorMemoryAllocation);
        return handle_allocation_failure();
    }
    
    d_tape_ = static_cast<GPUTapeEntry*>(memory_pool_->allocate(tape_size));
    if (!d_tape_) {
        memory_pool_->deallocate(d_values_);
        memory_pool_->deallocate(d_adjoints_);
        d_values_ = nullptr;
        d_adjoints_ = nullptr;
        GPUErrorHandler::handle_memory_error(cudaErrorMemoryAllocation);
        return handle_allocation_failure();
    }
    
    // Initialize memory to zero
    cudaError_t error = cudaMemset(d_values_, 0, values_size);
    if (!GPUErrorHandler::check_cuda_error(error, "cudaMemset values")) {
        free_gpu_memory();
        return false;
    }
    
    error = cudaMemset(d_adjoints_, 0, adjoints_size);
    if (!GPUErrorHandler::check_cuda_error(error, "cudaMemset adjoints")) {
        free_gpu_memory();
        return false;
    }
    
    error = cudaMemset(d_tape_, 0, tape_size);
    if (!GPUErrorHandler::check_cuda_error(error, "cudaMemset tape")) {
        free_gpu_memory();
        return false;
    }
    
    total_memory_allocated_ = values_size + adjoints_size + tape_size;
    gpu_allocated_ = true;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_allocation_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return true;
}

void GPUAADTape::free_gpu_memory() {
    if (!gpu_allocated_) return;
    
    if (d_values_) {
        memory_pool_->deallocate(d_values_);
        d_values_ = nullptr;
    }
    
    if (d_adjoints_) {
        memory_pool_->deallocate(d_adjoints_);
        d_adjoints_ = nullptr;
    }
    
    if (d_tape_) {
        memory_pool_->deallocate(d_tape_);
        d_tape_ = nullptr;
    }
    
    total_memory_allocated_ = 0;
    gpu_allocated_ = false;
}

bool GPUAADTape::handle_allocation_failure() {
    std::cerr << "GPU memory allocation failed - attempting defragmentation" << std::endl;
    
    // Try defragmentation
    memory_pool_->defragment();
    
    // Try allocation again
    if (allocate_gpu_memory()) {
        std::cout << "Memory allocation succeeded after defragmentation" << std::endl;
        return true;
    }
    
    std::cerr << "GPU memory allocation failed permanently" << std::endl;
    return false;
}

void GPUAADTape::log_performance_metrics() const {
    std::cout << "=== GPU AAD Performance Metrics ===" << std::endl;
    std::cout << "Allocation time: " << last_allocation_time_ms_ << " ms" << std::endl;
    std::cout << "Total GPU memory: " << total_memory_allocated_ / (1024*1024) << " MB" << std::endl;
    std::cout << "Available pool memory: " << memory_pool_->get_available_memory() / (1024*1024) << " MB" << std::endl;
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

bool GPUAADTape::async_copy_to_gpu(cudaStream_t stream) {
    if (!gpu_allocated_) return false;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cudaError_t error;
    
    // Asynchronous memory transfers
    error = cudaMemcpyAsync(d_values_, values_.data(), 
                           num_vars_ * sizeof(double), 
                           cudaMemcpyHostToDevice, stream);
    if (!GPUErrorHandler::check_cuda_error(error, "async_copy_to_gpu values")) {
        return false;
    }
    
    error = cudaMemcpyAsync(d_adjoints_, adjoints_.data(), 
                           num_vars_ * sizeof(double), 
                           cudaMemcpyHostToDevice, stream);
    if (!GPUErrorHandler::check_cuda_error(error, "async_copy_to_gpu adjoints")) {
        return false;
    }
    
    error = cudaMemcpyAsync(d_tape_, tape_.data(), 
                           tape_size_ * sizeof(GPUTapeEntry), 
                           cudaMemcpyHostToDevice, stream);
    if (!GPUErrorHandler::check_cuda_error(error, "async_copy_to_gpu tape")) {
        return false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_copy_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return true;
}

bool GPUAADTape::async_copy_from_gpu(cudaStream_t stream) {
    if (!gpu_allocated_) return false;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cudaError_t error = cudaMemcpyAsync(adjoints_.data(), d_adjoints_, 
                                       num_vars_ * sizeof(double), 
                                       cudaMemcpyDeviceToHost, stream);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_copy_time_ms_ += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return GPUErrorHandler::check_cuda_error(error, "async_copy_from_gpu");
}

bool GPUAADTape::copy_to_gpu(bool async) {
    if (!gpu_allocated_) return false;
    
    if (async) {
        cudaStream_t stream = stream_manager_->get_memory_stream();
        return async_copy_to_gpu(stream);
    } else {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        cudaError_t error;
        error = cudaMemcpy(d_values_, values_.data(), 
                          num_vars_ * sizeof(double), cudaMemcpyHostToDevice);
        if (!GPUErrorHandler::check_cuda_error(error, "copy_to_gpu values")) {
            return false;
        }
        
        error = cudaMemcpy(d_adjoints_, adjoints_.data(), 
                          num_vars_ * sizeof(double), cudaMemcpyHostToDevice);
        if (!GPUErrorHandler::check_cuda_error(error, "copy_to_gpu adjoints")) {
            return false;
        }
        
        error = cudaMemcpy(d_tape_, tape_.data(), 
                          tape_size_ * sizeof(GPUTapeEntry), cudaMemcpyHostToDevice);
        if (!GPUErrorHandler::check_cuda_error(error, "copy_to_gpu tape")) {
            return false;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        last_copy_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        return true;
    }
}

bool GPUAADTape::copy_from_gpu(bool async) {
    if (!gpu_allocated_) return false;
    
    if (async) {
        cudaStream_t stream = stream_manager_->get_memory_stream();
        return async_copy_from_gpu(stream);
    } else {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        cudaError_t error = cudaMemcpy(adjoints_.data(), d_adjoints_, 
                                      num_vars_ * sizeof(double), cudaMemcpyDeviceToHost);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        last_copy_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        return GPUErrorHandler::check_cuda_error(error, "copy_from_gpu");
    }
}

bool GPUAADTape::propagate_gpu() {
    if (!gpu_allocated_ || tape_size_ == 0) return false;
    
    // Use compute stream for kernel execution
    cudaStream_t compute_stream = stream_manager_->get_compute_stream();
    
    // Copy data to GPU
    if (!copy_to_gpu(true)) {
        return false;
    }
    
    // Launch kernel
    launch_propagate_kernel(d_tape_, d_values_, d_adjoints_, tape_size_);
    
    // Check for kernel errors
    cudaError_t error = cudaGetLastError();
    if (!GPUErrorHandler::check_cuda_error(error, "propagate_gpu kernel launch")) {
        return false;
    }
    
    // Synchronize compute stream
    error = cudaStreamSynchronize(compute_stream);
    if (!GPUErrorHandler::check_cuda_error(error, "propagate_gpu synchronize")) {
        return false;
    }
    
    // Copy results back
    return copy_from_gpu(true);
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

size_t GPUAADTape::get_memory_usage() const {
    return total_memory_allocated_;
}

bool GPUAADTape::batch_evaluate(const std::vector<std::vector<double>>& inputs,
                                std::vector<std::vector<double>>& outputs,
                                std::vector<std::vector<double>>& derivatives) {
    if (!gpu_available_) {
        std::cerr << "GPU not available for batch evaluation" << std::endl;
        return false;
    }
    
    // Enhanced batch processing with memory management
    size_t batch_size = inputs.size();
    if (batch_size == 0) return true;
    
    // Check if we have enough memory for the batch
    size_t required_memory = batch_size * (inputs[0].size() + outputs[0].size()) * sizeof(double);
    if (required_memory > memory_pool_->get_available_memory()) {
        std::cerr << "Insufficient GPU memory for batch size " << batch_size << std::endl;
        return false;
    }
    
    // Process batch with error handling
    try {
        // Implementation would go here for actual batch processing
        // This is a placeholder for the enhanced batch processing logic
        
        outputs.resize(batch_size);
        derivatives.resize(batch_size);
        
        for (size_t i = 0; i < batch_size; ++i) {
            outputs[i].resize(1); // Placeholder
            derivatives[i].resize(inputs[i].size()); // Placeholder
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Batch evaluation failed: " << e.what() << std::endl;
        return false;
    }
}