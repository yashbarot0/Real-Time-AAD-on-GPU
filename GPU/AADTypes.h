// ===== AADTypes.h =====
#pragma once

enum class AADOpType {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    LOG = 4,
    EXP = 5,
    SQRT = 6,
    ERF = 7,
    NORM_CDF = 8,
    NEG = 9,
    POW = 10,
    ABS = 11,
    MAX = 12,
    MIN = 13,
    SAFE_LOG = 14,
    SAFE_SQRT = 15,
    SAFE_DIV = 16
};

struct GPUTapeEntry {
    int result_idx;
    int op_type;
    int input1_idx;
    int input2_idx;
    double constant;
    double partial1;
    double partial2;
    
    // Constructor for easier initialization
    __host__ __device__ GPUTapeEntry() 
        : result_idx(-1), op_type(0), input1_idx(-1), input2_idx(-1), 
          constant(0.0), partial1(0.0), partial2(0.0) {}
    
    __host__ __device__ GPUTapeEntry(int res_idx, AADOpType op, int in1_idx, int in2_idx, 
                                    double const_val, double p1, double p2)
        : result_idx(res_idx), op_type(static_cast<int>(op)), input1_idx(in1_idx), 
          input2_idx(in2_idx), constant(const_val), partial1(p1), partial2(p2) {}
};

// Performance metrics structure
struct GPUPerformanceMetrics {
    double allocation_time_ms;
    double copy_time_ms;
    double kernel_time_ms;
    size_t memory_usage_bytes;
    size_t peak_memory_bytes;
    int scenarios_processed;
    double throughput_scenarios_per_sec;
    
    GPUPerformanceMetrics() 
        : allocation_time_ms(0.0), copy_time_ms(0.0), kernel_time_ms(0.0),
          memory_usage_bytes(0), peak_memory_bytes(0), scenarios_processed(0),
          throughput_scenarios_per_sec(0.0) {}
};

// Batch processing structures
struct BatchInputs {
    double* spot_prices;      // [num_scenarios]
    double* strike_prices;    // [num_scenarios]
    double* times_to_expiry;  // [num_scenarios]
    double* risk_free_rates;  // [num_scenarios]
    double* volatilities;     // [num_scenarios]
    int num_scenarios;
    
    BatchInputs() : spot_prices(nullptr), strike_prices(nullptr), 
                   times_to_expiry(nullptr), risk_free_rates(nullptr),
                   volatilities(nullptr), num_scenarios(0) {}
};

struct BatchOutputs {
    double* option_prices;    // [num_scenarios]
    double* deltas;          // [num_scenarios]
    double* vegas;           // [num_scenarios]
    double* gammas;          // [num_scenarios]
    double* thetas;          // [num_scenarios]
    double* rhos;            // [num_scenarios]
    
    GPUPerformanceMetrics metrics;
    
    BatchOutputs() : option_prices(nullptr), deltas(nullptr), vegas(nullptr),
                    gammas(nullptr), thetas(nullptr), rhos(nullptr) {}
};



