#include <iostream>
#include <iomanip>
#include "AADNumber.h"

// Black-Scholes Call Option Formula with detailed debugging
AADNumber BlackScholesCall_Debug(
    const AADNumber& S, const AADNumber& K,
    const AADNumber& T, const AADNumber& r,
    const AADNumber& sigma)
{
    std::cout << "=== Black-Scholes Computation Debug ===" << std::endl;
    std::cout << "Inputs:" << std::endl;
    std::cout << "S = " << S.val() << std::endl;
    std::cout << "K = " << K.val() << std::endl;
    std::cout << "T = " << T.val() << std::endl;
    std::cout << "r = " << r.val() << std::endl;
    std::cout << "sigma = " << sigma.val() << std::endl;
    
    AADNumber sqrtT = sqrt(T);
    std::cout << "sqrt(T) = " << sqrtT.val() << std::endl;
    
    AADNumber log_ratio = log(S / K);
    std::cout << "log(S/K) = " << log_ratio.val() << std::endl;
    
    AADNumber sigma_squared = sigma * sigma;
    std::cout << "sigma^2 = " << sigma_squared.val() << std::endl;
    
    AADNumber drift_term = (r + sigma_squared * 0.5) * T;
    std::cout << "drift term = " << drift_term.val() << std::endl;
    
    AADNumber numerator = log_ratio + drift_term;
    std::cout << "numerator = " << numerator.val() << std::endl;
    
    AADNumber denominator = sigma * sqrtT;
    std::cout << "denominator = " << denominator.val() << std::endl;
    
    AADNumber d1 = numerator / denominator;
    std::cout << "d1 = " << d1.val() << std::endl;
    
    AADNumber d2 = d1 - sigma * sqrtT;
    std::cout << "d2 = " << d2.val() << std::endl;
    
    AADNumber N_d1 = norm_cdf(d1);
    std::cout << "N(d1) = " << N_d1.val() << std::endl;
    
    AADNumber N_d2 = norm_cdf(d2);
    std::cout << "N(d2) = " << N_d2.val() << std::endl;
    
    AADNumber discount = exp(-r * T);
    std::cout << "exp(-rT) = " << discount.val() << std::endl;
    
    AADNumber term1 = S * N_d1;
    std::cout << "S * N(d1) = " << term1.val() << std::endl;
    
    AADNumber term2 = K * discount * N_d2;
    std::cout << "K * exp(-rT) * N(d2) = " << term2.val() << std::endl;
    
    AADNumber call = term1 - term2;
    std::cout << "Call price = " << call.val() << std::endl;
    
    return call;
}

void test_full_black_scholes() {
    std::cout << std::fixed << std::setprecision(6);
    
    // Standard Black-Scholes parameters
    AADNumber S(100.0);   // At-the-money
    AADNumber K(100.0);   // Strike
    AADNumber T(0.25);    // 3 months
    AADNumber r(0.05);    // 5% risk-free rate
    AADNumber sigma(0.2); // 20% volatility
    
    AADNumber price = BlackScholesCall_Debug(S, K, T, r, sigma);
    
    std::cout << "\n=== Computing Greeks ===" << std::endl;
    
    // Reset only the visited flags, not the adjoints
    price.node->reset_visited();
    
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
    
    // Analytical Black-Scholes Greeks for comparison
    double S_val = S.val(), K_val = K.val(), T_val = T.val(), r_val = r.val(), sigma_val = sigma.val();
    double sqrt_T = std::sqrt(T_val);
    double d1 = (std::log(S_val/K_val) + (r_val + 0.5*sigma_val*sigma_val)*T_val) / (sigma_val * sqrt_T);
    double d2 = d1 - sigma_val * sqrt_T;
    
    double N_d1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
    double n_d1 = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * d1 * d1); // PDF
    
    double analytical_delta = N_d1;
    double analytical_vega = S_val * n_d1 * sqrt_T;
    
    std::cout << "\nAnalytical comparison:" << std::endl;
    std::cout << "Analytical Delta = " << analytical_delta << std::endl;
    std::cout << "Analytical Vega = " << analytical_vega << std::endl;
}

int main() {
    test_full_black_scholes();
    return 0;
}