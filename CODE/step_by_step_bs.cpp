#include <iostream>
#include <iomanip>
#include "AADNumber.h"

void debug_node_info(const AADNumber& num, const std::string& name) {
    std::cout << name << ": val=" << num.val() << ", adj=" << num.adj() 
              << ", deps=" << num.node->dependencies.size() << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Step-by-Step Black-Scholes Construction ===" << std::endl;
    
    AADNumber S(100.0);
    AADNumber K(105.0);
    AADNumber T(0.25);
    AADNumber r(0.05);
    AADNumber sigma(0.2);
    
    std::cout << "\nInput variables:" << std::endl;
    debug_node_info(S, "S");
    debug_node_info(K, "K");
    debug_node_info(T, "T");
    debug_node_info(r, "r");
    debug_node_info(sigma, "sigma");
    
    // Step 1: sqrt(T)
    AADNumber sqrtT = sqrt(T);
    std::cout << "\nAfter sqrt(T):" << std::endl;
    debug_node_info(sqrtT, "sqrtT");
    
    // Step 2: S/K
    AADNumber ratio = S / K;
    std::cout << "\nAfter S/K:" << std::endl;
    debug_node_info(ratio, "ratio");
    
    // Step 3: log(S/K)
    AADNumber log_ratio = log(ratio);
    std::cout << "\nAfter log(S/K):" << std::endl;
    debug_node_info(log_ratio, "log_ratio");
    
    // Step 4: sigma^2
    AADNumber sigma_sq = sigma * sigma;
    std::cout << "\nAfter sigma^2:" << std::endl;
    debug_node_info(sigma_sq, "sigma_sq");
    
    // Step 5: 0.5 * sigma^2
    AADNumber half_sigma_sq = sigma_sq * 0.5;
    std::cout << "\nAfter 0.5 * sigma^2:" << std::endl;
    debug_node_info(half_sigma_sq, "half_sigma_sq");
    
    // Step 6: r + 0.5 * sigma^2
    AADNumber drift = r + half_sigma_sq;
    std::cout << "\nAfter r + 0.5*sigma^2:" << std::endl;
    debug_node_info(drift, "drift");
    
    // Step 7: (r + 0.5 * sigma^2) * T
    AADNumber drift_T = drift * T;
    std::cout << "\nAfter drift * T:" << std::endl;
    debug_node_info(drift_T, "drift_T");
    
    // Step 8: log(S/K) + (r + 0.5 * sigma^2) * T
    AADNumber numerator = log_ratio + drift_T;
    std::cout << "\nAfter numerator:" << std::endl;
    debug_node_info(numerator, "numerator");
    
    // Step 9: sigma * sqrt(T)
    AADNumber denominator = sigma * sqrtT;
    std::cout << "\nAfter denominator:" << std::endl;
    debug_node_info(denominator, "denominator");
    
    // Step 10: d1
    AADNumber d1 = numerator / denominator;
    std::cout << "\nAfter d1:" << std::endl;
    debug_node_info(d1, "d1");
    
    // Now test propagation from d1
    std::cout << "\n=== Testing propagation from d1 ===" << std::endl;
    d1.setAdj(1.0);
    d1.propagate();
    
    std::cout << "After d1 propagation:" << std::endl;
    debug_node_info(S, "S");
    debug_node_info(K, "K");
    debug_node_info(T, "T");
    debug_node_info(r, "r");
    debug_node_info(sigma, "sigma");
    
    return 0;
}