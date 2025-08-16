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
    
    std::cout << "=== Testing Black-Scholes (Fixed Version) ===" << std::endl;
    
    // Create variables ONCE and don't recreate them
    AADNumber S(100.0);
    AADNumber K(105.0);
    AADNumber T(0.25);
    AADNumber r(0.05);
    AADNumber sigma(0.2);
    
    std::cout << "Inputs: S=" << S.val() << ", K=" << K.val() << ", T=" << T.val() 
              << ", r=" << r.val() << ", sigma=" << sigma.val() << std::endl;
    
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
    
    // Analytical values for comparison
    double S_val = S.val(), K_val = K.val(), T_val = T.val(), r_val = r.val(), sigma_val = sigma.val();
    double sqrt_T = std::sqrt(T_val);
    double d1 = (std::log(S_val/K_val) + (r_val + 0.5*sigma_val*sigma_val)*T_val) / (sigma_val * sqrt_T);
    
    double N_d1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
    double n_d1 = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * d1 * d1);
    
    double analytical_delta = N_d1;
    double analytical_vega = S_val * n_d1 * sqrt_T;
    
    std::cout << "\nAnalytical comparison:" << std::endl;
    std::cout << "Analytical Delta = " << analytical_delta << std::endl;
    std::cout << "Analytical Vega = " << analytical_vega << std::endl;
    
    std::cout << "\nAAD vs Analytical:" << std::endl;
    std::cout << "Delta error = " << std::abs(S.adj() - analytical_delta) << std::endl;
    std::cout << "Vega error = " << std::abs(sigma.adj() - analytical_vega) << std::endl;
    
    return 0;
}