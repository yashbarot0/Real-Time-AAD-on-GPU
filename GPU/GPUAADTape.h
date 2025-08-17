// ===== GPUAADTape.h =====
#pragma once

#include "AADTypes.h"
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <unordered_map>
#include <queue>

// Forward declarations
class CudaMemoryPool;
class CudaStreamManager;

// Error handling utility
class GPUErrorHandler {
public:
    static bool check_cuda_error(cudaError_t error, const char* operation);
    static void handle_memory_error(cudaError_t error);
    static void handle_kernel_error(cudaError_t error);
    static bool attempt_recovery(cudaError_t error);
    
private:
    static void log_error(const std::string& message);
    static bool fallback_to_cpu();
};

// Memory pool for efficient GPU allocation
class CudaMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        
        MemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(false) {}
    };
    
    std::vector<MemoryBlock> blocks_;
    size_t total_allocated_;
    size_t pool_size_;
    
public:
    CudaMemoryPool(size_t pool_size_bytes = 1024 * 1024 * 1024); // 1GB default
    ~CudaMemoryPool();
    
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void defragment();
    size_t get_available_memory() const;
    bool initialize();
    void cleanup();
};

// CUDA stream management for asynchronous operations
class CudaStreamManager {
private:
    std::vector<cudaStream_t> compute_streams_;
    std::vector<cudaStream_t> memory_streams_;
    int num_streams_;
    
public:
    CudaStreamManager(int num_streams = 4);
    ~CudaStreamManager();
    
    cudaStream_t get_compute_stream(int index = 0);
    cudaStream_t get_memory_stream(int index = 0);
    void synchronize_all();
    bool initialize();
    void cleanup();
};

class GPUAADTape {
private:
    std::vector<double> values_;
    std::vector<double> adjoints_;
    std::vector<GPUTapeEntry> tape_;
    
    // GPU memory pointers
    double* d_values_;
    double* d_adjoints_;
    GPUTapeEntry* d_tape_;
    
    // Memory management
    std::unique_ptr<CudaMemoryPool> memory_pool_;
    std::unique_ptr<CudaStreamManager> stream_manager_;
    
    // State tracking
    int num_vars_;
    int tape_size_;
    int max_vars_;
    int max_tape_size_;
    bool gpu_allocated_;
    bool gpu_available_;
    
    // Performance tracking
    mutable double last_allocation_time_ms_;
    mutable double last_copy_time_ms_;
    mutable size_t total_memory_allocated_;
    
    // Memory management methods
    bool allocate_gpu_memory();
    void free_gpu_memory();
    bool check_gpu_availability();
    
    // Asynchronous operations
    bool async_copy_to_gpu(cudaStream_t stream = 0);
    bool async_copy_from_gpu(cudaStream_t stream = 0);
    
    // Error handling
    bool handle_allocation_failure();
    void log_performance_metrics() const;

public:
    GPUAADTape(int max_vars = 100000, int max_ops = 1000000);
    ~GPUAADTape();
    
    // Initialization and cleanup
    bool initialize();
    void cleanup();
    bool is_gpu_available() const { return gpu_available_; }
    
    // Variable management
    int create_variable(double value);
    void set_adjoint(int var_idx, double adj);
    void clear_adjoints();
    
    // Operation recording
    int record_binary_op(AADOpType op_type, int input1, int input2, 
                        double result_val, double partial1, double partial2);
    int record_unary_op(AADOpType op_type, int input, double result_val, double partial);
    int record_constant(double value);
    
    // GPU operations with error handling
    bool copy_to_gpu(bool async = false);
    bool copy_from_gpu(bool async = false);
    bool propagate_gpu();
    
    // Accessors
    double get_value(int idx) const;
    double get_adjoint(int idx) const;
    void clear_tape();
    
    // Memory and performance monitoring
    size_t get_memory_usage() const;
    double get_last_allocation_time() const { return last_allocation_time_ms_; }
    double get_last_copy_time() const { return last_copy_time_ms_; }
    
    // Batch operations with enhanced memory management
    bool batch_evaluate(const std::vector<std::vector<double>>& inputs,
                       std::vector<std::vector<double>>& outputs,
                       std::vector<std::vector<double>>& derivatives);
    
    // Stream management access
    CudaStreamManager* get_stream_manager() const { return stream_manager_.get(); }
    CudaMemoryPool* get_memory_pool() const { return memory_pool_.get(); }
};