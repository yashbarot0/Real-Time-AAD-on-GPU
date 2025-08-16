#include <iostream>
#include <iomanip>
#include "AADNumber.h"

// The exact Black-Scholes function from the real-time code
AADNumber BlackScholesCall(
    const AADNumber& S, const AADNumber& K,
    const AADNumber& T, const AADNumber& r,
    const AADNumber& sigma)
{
    AADNumber sqrtT = sqrt(T);
    AADNumber d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    AADNumber d2 = d1 - sigma * sqrtT;
    AADNumber call = S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2);
    return call;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Testing Full Black-Scholes Step by Step ===" << std::endl;
    
    // Create fresh variables
    AADNumber S(100.0);
    AADNumber K(105.0);
    AADNumber T(0.25);
    AADNumber r(0.05);
    AADNumber sigma(0.2);
    
    std::cout << "Inputs: S=" << S.val() << ", K=" << K.val() << ", T=" << T.val() 
              << ", r=" << r.val() << ", sigma=" << sigma.val() << std::endl;
    
    // Test each component separately first
    std::cout << "\n--- Testing Components ---" << std::endl;
    
    // Test 1: sqrt(T)
    AADNumber sqrtT = sqrt(T);
    std::cout << "sqrt(T) = " << sqrtT.val() << std::endl;
    sqrtT.setAdj(1.0);
    sqrtT.propagate();
    std::cout << "d(sqrt(T))/dT = " << T.adj() << " (should be " << (0.5/sqrtT.val()) << ")" << std::endl;
    
    // Reset for next test
    T = AADNumber(0.25);  // Create fresh variable
    
    // Test 2: log(S/K)
    AADNumber ratio = S / K;
    AADNumber log_ratio = log(ratio);
    std::cout << "log(S/K) = " << log_ratio.val() << std::endl;
    log_ratio.setAdj(1.0);
    log_ratio.propagate();
    std::cout << "d(log(S/K))/dS = " << S.adj() << " (should be " << (1.0/S.val()) << ")" << std::endl;
    
    // Reset for full test
    S = AADNumber(100.0);
    K = AADNumber(105.0);
    T = AADNumber(0.25);
    r = AADNumber(0.05);
    sigma = AADNumber(0.2);
    
    std::cout << "\n--- Testing Full Black-Scholes ---" << std::endl;
    
    AADNumber price = BlackScholesCall(S, K, T, r, sigma);
    std::cout << "Black-Scholes price = " << price.val() << std::endl;
    
    // Check if price has dependencies
    std::cout << "Price node dependencies: " << price.node->dependencies.size() << std::endl;
    
    price.setAdj(1.0);
    std::cout << "Set price.adj() = 1.0" << std::endl;
    
    price.propagate();
    std::cout << "Propagation completed" << std::endl;
    
    std::cout << "\nGreeks:" << std::endl;
    std::cout << "Delta (dP/dS) = " << S.adj() << std::endl;
    std::cout << "Vega (dP/dsigma) = " << sigma.adj() << std::endl;
    std::cout << "Rho (dP/dr) = " << r.adj() << std::endl;
    std::cout << "Theta (dP/dT) = " << T.adj() << std::endl;
    
    // Check individual node states
    std::cout << "\nDebugging info:" << std::endl;
    std::cout << "S.adj() = " << S.adj() << ", dependencies = " << S.node->dependencies.size() << std::endl;
    std::cout << "sigma.adj() = " << sigma.adj() << ", dependencies = " << sigma.node->dependencies.size() << std::endl;
    std::cout << "price.adj() = " << price.adj() << ", dependencies = " << price.node->dependencies.size() << std::endl;
    
    return 0;
}