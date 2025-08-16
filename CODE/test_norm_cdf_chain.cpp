#include <iostream>
#include <iomanip>
#include "AADNumber.h"

void debug_node_info(const AADNumber& num, const std::string& name) {
    std::cout << name << ": val=" << num.val() << ", adj=" << num.adj() 
              << ", deps=" << num.node->dependencies.size() << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Testing norm_cdf Chain ===" << std::endl;
    
    // Test 1: Simple norm_cdf
    std::cout << "\n--- Test 1: Simple norm_cdf(x) ---" << std::endl;
    AADNumber x(0.5);
    AADNumber n = norm_cdf(x);
    
    debug_node_info(x, "x");
    debug_node_info(n, "n");
    
    n.setAdj(1.0);
    n.propagate();
    
    std::cout << "After propagation:" << std::endl;
    debug_node_info(x, "x");
    std::cout << "x.adj() = " << x.adj() << " (should be ~0.352)" << std::endl;
    
    // Test 2: Chain with multiplication
    std::cout << "\n--- Test 2: S * norm_cdf(d1) ---" << std::endl;
    AADNumber S(100.0);
    AADNumber d1(-0.31);  // Use the d1 value from our previous test
    AADNumber result = S * norm_cdf(d1);
    
    std::cout << "Before propagation:" << std::endl;
    debug_node_info(S, "S");
    debug_node_info(d1, "d1");
    debug_node_info(result, "result");
    
    result.setAdj(1.0);
    result.propagate();
    
    std::cout << "After propagation:" << std::endl;
    debug_node_info(S, "S");
    debug_node_info(d1, "d1");
    std::cout << "S.adj() = " << S.adj() << std::endl;
    std::cout << "d1.adj() = " << d1.adj() << std::endl;
    
    // Test 3: Full first term of Black-Scholes
    std::cout << "\n--- Test 3: Full first term ---" << std::endl;
    AADNumber S2(100.0);
    AADNumber K2(105.0);
    AADNumber T2(0.25);
    AADNumber r2(0.05);
    AADNumber sigma2(0.2);
    
    // Build d1 step by step
    AADNumber sqrtT = sqrt(T2);
    AADNumber d1_full = (log(S2 / K2) + (r2 + sigma2 * sigma2 * 0.5) * T2) / (sigma2 * sqrtT);
    AADNumber term1 = S2 * norm_cdf(d1_full);
    
    std::cout << "d1_full = " << d1_full.val() << std::endl;
    std::cout << "term1 = " << term1.val() << std::endl;
    
    debug_node_info(term1, "term1");
    
    term1.setAdj(1.0);
    term1.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "S2.adj() = " << S2.adj() << std::endl;
    std::cout << "sigma2.adj() = " << sigma2.adj() << std::endl;
    
    return 0;
}