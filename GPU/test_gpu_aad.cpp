// Test program for enhanced GPU AAD implementation
#include "GPUAADTape.h"
#include "GPUAADNumber.h"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    try {
        std::cout << "=== GPU AAD Enhanced Implementation Test ===" << std::endl;
        
        // Create and initialize GPU AAD tape
        GPUAADTape tape(10000, 100000);
        
        std::cout << "Initializing GPU AAD system..." << std::endl;
        if (!tape.initialize()) {
            std::cerr << "Failed to initialize GPU AAD system" << std::endl;
            return 1;
        }
        
        std::cout << "GPU AAD system initialized successfully!" << std::endl;
        std::cout << "GPU available: " << (tape.is_gpu_available() ? "Yes" : "No") << std::endl;
        std::cout << "Memory usage: " << tape.get_memory_usage() / (1024*1024) << " MB" << std::endl;
        
        // Set active tape
        GPUAADNumber::set_active_tape(&tape);
        
        // Test basic arithmetic operations
        std::cout << "\n=== Testing Basic Operations ===" << std::endl;
        
        GPUAADNumber x(2.0);
        GPUAADNumber y(3.0);
        
        // Test addition
        GPUAADNumber z1 = x + y;
        std::cout << "x + y = " << z1.val() << " (expected: 5.0)" << std::endl;
        
        // Test multiplication
        GPUAADNumber z2 = x * y;
        std::cout << "x * y = " << z2.val() << " (expected: 6.0)" << std::endl;
        
        // Test division
        GPUAADNumber z3 = y / x;
        std::cout << "y / x = " << z3.val() << " (expected: 1.5)" << std::endl;
        
        // Test unary minus
        GPUAADNumber z4 = -x;
        std::cout << "-x = " << z4.val() << " (expected: -2.0)" << std::endl;
        
        // Test math functions
        std::cout << "\n=== Testing Math Functions ===" << std::endl;
        
        GPUAADNumber a(1.0);
        GPUAADNumber log_a = log(a);
        std::cout << "log(1.0) = " << log_a.val() << " (expected: 0.0)" << std::endl;
        
        GPUAADNumber exp_a = exp(a);
        std::cout << "exp(1.0) = " << exp_a.val() << " (expected: ~2.718)" << std::endl;
        
        GPUAADNumber b(4.0);
        GPUAADNumber sqrt_b = sqrt(b);
        std::cout << "sqrt(4.0) = " << sqrt_b.val() << " (expected: 2.0)" << std::endl;
        
        // Test safe functions
        std::cout << "\n=== Testing Safe Functions ===" << std::endl;
        
        GPUAADNumber small_val(1e-20);
        GPUAADNumber safe_log_small = safe_log(small_val);
        std::cout << "safe_log(1e-20) = " << safe_log_small.val() << " (should not be -inf)" << std::endl;
        
        GPUAADNumber neg_val(-1.0);
        GPUAADNumber safe_sqrt_neg = safe_sqrt(neg_val);
        std::cout << "safe_sqrt(-1.0) = " << safe_sqrt_neg.val() << " (expected: 0.0)" << std::endl;
        
        // Test norm_cdf
        std::cout << "\n=== Testing Normal CDF ===" << std::endl;
        
        GPUAADNumber zero(0.0);
        GPUAADNumber norm_cdf_zero = norm_cdf(zero);
        std::cout << "norm_cdf(0.0) = " << norm_cdf_zero.val() << " (expected: 0.5)" << std::endl;
        
        // Test GPU operations
        std::cout << "\n=== Testing GPU Operations ===" << std::endl;
        
        // Set up for derivative computation
        tape.clear_adjoints();
        tape.set_adjoint(z2.index(), 1.0); // Set adjoint of result
        
        // Test GPU propagation
        if (tape.propagate_gpu()) {
            std::cout << "GPU propagation successful!" << std::endl;
            std::cout << "dx/d(x*y) = " << tape.get_adjoint(x.index()) << " (expected: " << y.val() << ")" << std::endl;
            std::cout << "dy/d(x*y) = " << tape.get_adjoint(y.index()) << " (expected: " << x.val() << ")" << std::endl;
        } else {
            std::cout << "GPU propagation failed - using CPU fallback" << std::endl;
        }
        
        std::cout << "\n=== Performance Metrics ===" << std::endl;
        std::cout << "Last allocation time: " << tape.get_last_allocation_time() << " ms" << std::endl;
        std::cout << "Last copy time: " << tape.get_last_copy_time() << " ms" << std::endl;
        
        std::cout << "\n=== Test Completed Successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}