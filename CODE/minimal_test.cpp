#include <iostream>
#include <iomanip>
#include "AADNumber.h"

// Test the exact Black-Scholes function from the real-time code
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
    
    // Use the exact same parameters as the real-time test
    AADNumber S(100.0);
    AADNumber K(105.0);
    AADNumber T(0.25);
    AADNumber r(0.05);
    AADNumber sigma(0.2);
    
    std::cout << "=== Testing Exact Black-Scholes from Real-Time Code ===" << std::endl;
    std::cout << "S = " << S.val() << std::endl;
    std::cout << "K = " << K.val() << std::endl;
    std::cout << "T = " << T.val() << std::endl;
    std::cout << "r = " << r.val() << std::endl;
    std::cout << "sigma = " << sigma.val() << std::endl;
    
    AADNumber price = BlackScholesCall(S, K, T, r, sigma);
    std::cout << "Price = " << price.val() << std::endl;
    
    // Set adjoint and propagate
    price.setAdj(1.0);
    std::cout << "Set price.adj() = 1.0" << std::endl;
    
    price.propagate();
    std::cout << "Propagation completed" << std::endl;
    
    std::cout << "\nGreeks:" << std::endl;
    std::cout << "Delta (dP/dS) = " << S.adj() << std::endl;
    std::cout << "Vega (dP/dsigma) = " << sigma.adj() << std::endl;
    std::cout << "Rho (dP/dr) = " << r.adj() << std::endl;
    std::cout << "Theta (dP/dT) = " << T.adj() << std::endl;
    
    // Let's also test if the individual variables have any dependencies
    std::cout << "\nDebugging info:" << std::endl;
    std::cout << "S dependencies: " << S.node->dependencies.size() << std::endl;
    std::cout << "price dependencies: " << price.node->dependencies.size() << std::endl;
    std::cout << "price.adj() = " << price.adj() << std::endl;
    
    return 0;
}