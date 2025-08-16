#include <iostream>
#include <iomanip>
#include "AADNumber.h"

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Minimal Working Black-Scholes ===" << std::endl;
    
    // Use the exact same pattern that worked in debug_exp_issue.cpp Test 5
    AADNumber S(100.0);
    AADNumber K(100.0);
    AADNumber T(0.25);
    AADNumber r(0.05);
    AADNumber sigma(0.2);
    
    std::cout << "Building Black-Scholes step by step..." << std::endl;
    
    // Step 1: Build d1
    AADNumber sqrtT = sqrt(T);
    AADNumber log_SK = log(S / K);
    AADNumber sigma_sq = sigma * sigma;
    AADNumber drift = r + sigma_sq * 0.5;
    AADNumber drift_T = drift * T;
    AADNumber numerator = log_SK + drift_T;
    AADNumber denominator = sigma * sqrtT;
    AADNumber d1 = numerator / denominator;
    
    std::cout << "d1 = " << d1.val() << std::endl;
    
    // Step 2: Build d2
    AADNumber sigma_sqrtT = sigma * sqrtT;
    AADNumber d2 = d1 - sigma_sqrtT;
    
    std::cout << "d2 = " << d2.val() << std::endl;
    
    // Step 3: Build N(d1) and N(d2)
    AADNumber N_d1 = norm_cdf(d1);
    AADNumber N_d2 = norm_cdf(d2);
    
    std::cout << "N(d1) = " << N_d1.val() << std::endl;
    std::cout << "N(d2) = " << N_d2.val() << std::endl;
    
    // Step 4: Build first term
    AADNumber term1 = S * N_d1;
    std::cout << "term1 = " << term1.val() << std::endl;
    
    // Step 5: Build discount factor (using working pattern)
    AADNumber product_rT = r * T;
    AADNumber neg_product_rT = -product_rT;
    AADNumber discount = exp(neg_product_rT);
    
    std::cout << "discount = " << discount.val() << std::endl;
    
    // Step 6: Build second term
    AADNumber K_discount = K * discount;
    AADNumber term2 = K_discount * N_d2;
    
    std::cout << "term2 = " << term2.val() << std::endl;
    
    // Step 7: Final price
    AADNumber price = term1 - term2;
    
    std::cout << "price = " << price.val() << std::endl;
    
    // Step 8: Compute Greeks
    std::cout << "\nComputing Greeks..." << std::endl;
    
    price.setAdj(1.0);
    price.propagate();
    
    std::cout << "Delta = " << S.adj() << std::endl;
    std::cout << "Vega = " << sigma.adj() << std::endl;
    std::cout << "Rho = " << r.adj() << std::endl;
    std::cout << "Theta = " << T.adj() << std::endl;
    
    return 0;
}