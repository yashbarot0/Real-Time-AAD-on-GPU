#include <iostream>
#include <iomanip>
#include "AADNumber.h"

// Simplified Black-Scholes for debugging
AADNumber SimpleBlackScholes(const AADNumber& S, const AADNumber& sigma) {
    // Very simple version: just S * sigma (should give dP/dS = sigma, dP/dsigma = S)
    return S * sigma;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    AADNumber S(100.0);
    AADNumber sigma(0.2);
    
    std::cout << "=== Simple Test: P = S * sigma ===" << std::endl;
    std::cout << "S = " << S.val() << std::endl;
    std::cout << "sigma = " << sigma.val() << std::endl;
    
    AADNumber price = SimpleBlackScholes(S, sigma);
    std::cout << "Price = " << price.val() << std::endl;
    
    // DON'T reset adjoints - let's see what happens
    price.setAdj(1.0);
    price.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "dP/dS = " << S.adj() << " (should be " << sigma.val() << ")" << std::endl;
    std::cout << "dP/dsigma = " << sigma.adj() << " (should be " << S.val() << ")" << std::endl;
    
    return 0;
}