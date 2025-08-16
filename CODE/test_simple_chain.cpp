#include <iostream>
#include <iomanip>
#include "AADNumber.h"

void debug_node_info(const AADNumber& num, const std::string& name) {
    std::cout << name << ": val=" << num.val() << ", adj=" << num.adj() 
              << ", deps=" << num.node->dependencies.size() << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Testing Simple Chain vs norm_cdf Chain ===" << std::endl;
    
    // Test 1: S * log(d1) instead of S * norm_cdf(d1)
    std::cout << "\n--- Test 1: S * log(d1) ---" << std::endl;
    AADNumber S1(100.0);
    AADNumber d1_val(0.5);  // Use positive value for log
    AADNumber result1 = S1 * log(d1_val);
    
    debug_node_info(S1, "S1");
    debug_node_info(d1_val, "d1_val");
    debug_node_info(result1, "result1");
    
    result1.setAdj(1.0);
    result1.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "S1.adj() = " << S1.adj() << " (should be log(0.5) = " << std::log(0.5) << ")" << std::endl;
    std::cout << "d1_val.adj() = " << d1_val.adj() << " (should be 100/0.5 = " << (100.0/0.5) << ")" << std::endl;
    
    // Test 2: S * norm_cdf(d1) - same as before
    std::cout << "\n--- Test 2: S * norm_cdf(d1) ---" << std::endl;
    AADNumber S2(100.0);
    AADNumber d1_val2(0.5);
    AADNumber result2 = S2 * norm_cdf(d1_val2);
    
    debug_node_info(S2, "S2");
    debug_node_info(d1_val2, "d1_val2");
    debug_node_info(result2, "result2");
    
    result2.setAdj(1.0);
    result2.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "S2.adj() = " << S2.adj() << std::endl;
    std::cout << "d1_val2.adj() = " << d1_val2.adj() << std::endl;
    
    // Test 3: Just norm_cdf(d1) by itself
    std::cout << "\n--- Test 3: Just norm_cdf(d1) ---" << std::endl;
    AADNumber d1_val3(0.5);
    AADNumber result3 = norm_cdf(d1_val3);
    
    debug_node_info(d1_val3, "d1_val3");
    debug_node_info(result3, "result3");
    
    result3.setAdj(1.0);
    result3.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "d1_val3.adj() = " << d1_val3.adj() << " (should be ~0.352)" << std::endl;
    
    return 0;
}