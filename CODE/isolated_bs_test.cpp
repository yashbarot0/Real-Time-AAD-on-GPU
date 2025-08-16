#include <iostream>
#include <iomanip>
#include "AADNumber.h"

// Black-Scholes function
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
    
    std::cout << "=== Isolated Black-Scholes Test ===" << std::endl;
    
    // Create completely fresh variables
    AADNumber S(100.0);
    AADNumber K(105.0);
    AADNumber T(0.25);
    AADNumber r(0.05);
    AADNumber sigma(0.2);
    
    std::cout << "Computing Black-Scholes..." << std::endl;
    AADNumber price = BlackScholesCall(S, K, T, r, sigma);
    std::cout << "Price = " << price.val() << std::endl;
    
    std::cout << "Setting adjoint..." << std::endl;
    price.setAdj(1.0);
    
    std::cout << "Propagating..." << std::endl;
    price.propagate();
    
    std::cout << "Results:" << std::endl;
    std::cout << "Delta = " << S.adj() << std::endl;
    std::cout << "Vega = " << sigma.adj() << std::endl;
    std::cout << "Rho = " << r.adj() << std::endl;
    std::cout << "Theta = " << T.adj() << std::endl;
    
    // Let's also manually check if we can trace the path from price to S
    std::cout << "\nManual tracing:" << std::endl;
    std::cout << "Price has " << price.node->dependencies.size() << " dependencies" << std::endl;
    
    if (price.node->dependencies.size() > 0) {
        auto& first_dep = price.node->dependencies[0];
        std::cout << "First dependency: partial=" << first_dep.second << ", adj=" << first_dep.first->adj << std::endl;
        std::cout << "First dependency has " << first_dep.first->dependencies.size() << " dependencies" << std::endl;
    }
    
    return 0;
}