// ===== GPUAADTape.h =====
#pragma once

#include "AADTypes.h"
#include <vector>
#include <memory>

class GPUAADTape {
private:
    std::vector<double> values_;
    std::vector<double> adjoints_;
    std::vector<GPUTapeEntry> tape_;
    
    // GPU memory
    double* d_values_;
    double* d_adjoints_;
    GPUTapeEntry* d_tape_;
    
    int num_vars_;
    int tape_size_;
    int max_vars_;
    int max_tape_size_;
    bool gpu_allocated_;
    
    void allocate_gpu_memory();
    void free_gpu_memory();

public:
    GPUAADTape(int max_vars = 100000, int max_ops = 1000000);
    ~GPUAADTape();
    
    // Variable management
    int create_variable(double value);
    void set_adjoint(int var_idx, double adj);
    void clear_adjoints();
    
    // Operation recording
    int record_binary_op(AADOpType op_type, int input1, int input2, 
                        double result_val, double partial1, double partial2);
    int record_unary_op(AADOpType op_type, int input, double result_val, double partial);
    int record_constant(double value);
    
    // GPU operations
    void copy_to_gpu();
    void copy_from_gpu();
    void propagate_gpu();
    
    // Accessors
    double get_value(int idx) const;
    double get_adjoint(int idx) const;
    void clear_tape();
    
    // Batch operations
    void batch_evaluate(const std::vector<std::vector<double>>& inputs,
                       std::vector<std::vector<double>>& outputs,
                       std::vector<std::vector<double>>& derivatives);
};