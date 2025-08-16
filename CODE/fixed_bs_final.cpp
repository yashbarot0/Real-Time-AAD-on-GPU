#include <iostream>
#include <iomanip>
#include "AADNumber.h"

AADNumber BlackScholesCall_Fixed(
    const AADNumber& S, const AADNumber& K,
    const AADNumber& T, const AADNumber& r,
    const AADNumber& sigma)
{
    // Build d1 and d2
    AADNumber sqrtT = sqrt(T);
    AADNumber d1 = (log(S / K) + (r + sigma * sigma * 0.5) * T) / (sigma * sqrtT);
    AADNumber d2 = d1 - sigma * sqrtT;
    
    // Build the two terms separately
    AADNumber N_d1 = norm_cdf(d1);
    AADNumber N_d2 = norm_cdf(d2);
    
    AADNumber term1 = S * N_d1;
    AADNumber discount = exp(-r * T);
    AADNumber term2 = K * discount * N_d2;
    
    // Final result
    AADNumber call = term1 - term2;
    return call;
}

void debug_node_info(const AADNumber& num, const std::string& name) {
    std::cout << name << ": val=" << num.val() << ", adj=" << num.adj() 
              << ", deps=" << num.node->dependencies.size() << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Fixed Black-Scholes Test ===" << std::endl;
    
    AADNumber S(100.0);
    AADNumber K(105.0);
    AADNumber T(0.25);
    AADNumber r(0.05);
    AADNumber sigma(0.2);
    
    std::cout << "Computing Black-Scholes with fixed implementation..." << std::endl;
    AADNumber price = BlackScholesCall_Fixed(S, K, T, r, sigma);
    std::cout << "Price = " << price.val() << std::endl;
    
    std::cout << "\nBefore propagation:" << std::endl;
    debug_node_info(S, "S");
    debug_node_info(sigma, "sigma");
    debug_node_info(price, "price");
    
    price.setAdj(1.0);
    price.propagate();
    
    std::cout << "\nAfter propagation:" << std::endl;
    std::cout << "Delta (dP/dS) = " << S.adj() << std::endl;
    std::cout << "Vega (dP/dsigma) = " << sigma.adj() << std::endl;
    std::cout << "Rho (dP/dr) = " << r.adj() << std::endl;
    std::cout << "Theta (dP/dT) = " << T.adj() << std::endl;
    
    // Analytical comparison
    double S_val = 100.0, K_val = 105.0, T_val = 0.25, r_val = 0.05, sigma_val = 0.2;
    double sqrt_T = std::sqrt(T_val);
    double d1 = (std::log(S_val/K_val) + (r_val + 0.5*sigma_val*sigma_val)*T_val) / (sigma_val * sqrt_T);
    
    double N_d1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
    double n_d1 = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * d1 * d1);
    
    double analytical_delta = N_d1;
    double analytical_vega = S_val * n_d1 * sqrt_T;
    
    std::cout << "\nAnalytical comparison:" << std::endl;
    std::cout << "Analytical Delta = " << analytical_delta << std::endl;
    std::cout << "Analytical Vega = " << analytical_vega << std::endl;
    
    std::cout << "\nErrors:" << std::endl;
    std::cout << "Delta error = " << std::abs(S.adj() - analytical_delta) << std::endl;
    std::cout << "Vega error = " << std::abs(sigma.adj() - analytical_vega) << std::endl;
    
    return 0;
}