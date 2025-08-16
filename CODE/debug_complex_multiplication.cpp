#include <iostream>
#include <iomanip>
#include "AADNumber.h"

void debug_detailed(const AADNumber& num, const std::string& name) {
    std::cout << name << ": val=" << num.val() << ", adj=" << num.adj() 
              << ", deps=" << num.node->dependencies.size() 
              << ", ptr=" << num.node.get() << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Testing Complex Multiplication ===" << std::endl;
    
    // Test 1: S * (simple expression)
    std::cout << "\n--- Test 1: S * (x + 1) ---" << std::endl;
    AADNumber S1(100.0);
    AADNumber x1(2.0);
    AADNumber expr1 = x1 + 1.0;  // expr1 = x1 + 1
    AADNumber result1 = S1 * expr1;  // result1 = S1 * (x1 + 1)
    
    std::cout << "Before propagation:" << std::endl;
    debug_detailed(S1, "S1");
    debug_detailed(x1, "x1");
    debug_detailed(expr1, "expr1");
    debug_detailed(result1, "result1");
    
    result1.setAdj(1.0);
    result1.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "S1.adj() = " << S1.adj() << " (should be 3.0)" << std::endl;
    std::cout << "x1.adj() = " << x1.adj() << " (should be 100.0)" << std::endl;
    
    // Test 2: S * log(x)
    std::cout << "\n--- Test 2: S * log(x) ---" << std::endl;
    AADNumber S2(100.0);
    AADNumber x2(2.0);
    AADNumber log_x = log(x2);
    AADNumber result2 = S2 * log_x;
    
    std::cout << "Before propagation:" << std::endl;
    debug_detailed(S2, "S2");
    debug_detailed(x2, "x2");
    debug_detailed(log_x, "log_x");
    debug_detailed(result2, "result2");
    
    result2.setAdj(1.0);
    result2.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "S2.adj() = " << S2.adj() << " (should be log(2) = " << std::log(2.0) << ")" << std::endl;
    std::cout << "x2.adj() = " << x2.adj() << " (should be 100/2 = 50.0)" << std::endl;
    
    // Test 3: S * norm_cdf(x) - the problematic case
    std::cout << "\n--- Test 3: S * norm_cdf(x) ---" << std::endl;
    AADNumber S3(100.0);
    AADNumber x3(0.5);
    AADNumber ncdf_x = norm_cdf(x3);
    AADNumber result3 = S3 * ncdf_x;
    
    std::cout << "Before propagation:" << std::endl;
    debug_detailed(S3, "S3");
    debug_detailed(x3, "x3");
    debug_detailed(ncdf_x, "ncdf_x");
    debug_detailed(result3, "result3");
    
    result3.setAdj(1.0);
    result3.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "S3.adj() = " << S3.adj() << std::endl;
    std::cout << "x3.adj() = " << x3.adj() << std::endl;
    
    // Let's also check the dependencies more carefully
    std::cout << "\nDetailed dependency check for result3:" << std::endl;
    if (result3.node->dependencies.size() >= 2) {
        std::cout << "result3.dep[0]: partial=" << result3.node->dependencies[0].second 
                  << ", points to " << result3.node->dependencies[0].first << std::endl;
        std::cout << "result3.dep[1]: partial=" << result3.node->dependencies[1].second 
                  << ", points to " << result3.node->dependencies[1].first << std::endl;
        std::cout << "S3.node.get() = " << S3.node.get() << std::endl;
        std::cout << "ncdf_x.node.get() = " << ncdf_x.node.get() << std::endl;
    }
    
    return 0;
}