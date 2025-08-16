#include <iostream>
#include <iomanip>
#include "AADNumber.h"

void test_individual_derivatives() {
    std::cout << std::fixed << std::setprecision(6);
    
    // Test each Greek individually
    std::cout << "=== Testing Individual Greeks ===" << std::endl;
    
    // Test 1: Delta (dP/dS) - we know this works
    std::cout << "\n--- Test 1: Delta (dP/dS) ---" << std::endl;
    AADNumber S1(100.0);
    AADNumber K1(100.0);
    AADNumber simple_payoff = S1 - K1;  // Simple payoff
    
    simple_payoff.setAdj(1.0);
    simple_payoff.propagate();
    std::cout << "dPayoff/dS = " << S1.adj() << " (should be 1.0)" << std::endl;
    
    // Test 2: Vega (dP/dsigma) - test with sigma in a simple expression
    std::cout << "\n--- Test 2: Vega (dP/dsigma) ---" << std::endl;
    AADNumber sigma2(0.2);
    AADNumber sigma_expr = sigma2 * sigma2;  // sigma^2
    
    sigma_expr.setAdj(1.0);
    sigma_expr.propagate();
    std::cout << "d(sigma^2)/dsigma = " << sigma2.adj() << " (should be 0.4)" << std::endl;
    
    // Test 3: Rho (dP/dr) - test with r in exp(-r*T)
    std::cout << "\n--- Test 3: Rho (dP/dr) ---" << std::endl;
    AADNumber r3(0.05);
    AADNumber T3(0.25);
    AADNumber discount = exp(-r3 * T3);
    
    discount.setAdj(1.0);
    discount.propagate();
    std::cout << "d(exp(-r*T))/dr = " << r3.adj() << " (should be " << (-T3.val() * discount.val()) << ")" << std::endl;
    std::cout << "d(exp(-r*T))/dT = " << T3.adj() << " (should be " << (-r3.val() * discount.val()) << ")" << std::endl;
    
    // Test 4: Full d1 expression
    std::cout << "\n--- Test 4: Full d1 expression ---" << std::endl;
    AADNumber S4(100.0);
    AADNumber K4(100.0);
    AADNumber T4(0.25);
    AADNumber r4(0.05);
    AADNumber sigma4(0.2);
    
    AADNumber sqrtT = sqrt(T4);
    AADNumber d1 = (log(S4 / K4) + (r4 + sigma4 * sigma4 * 0.5) * T4) / (sigma4 * sqrtT);
    
    std::cout << "d1 = " << d1.val() << std::endl;
    
    d1.setAdj(1.0);
    d1.propagate();
    
    std::cout << "dd1/dS = " << S4.adj() << std::endl;
    std::cout << "dd1/dsigma = " << sigma4.adj() << std::endl;
    std::cout << "dd1/dr = " << r4.adj() << std::endl;
    std::cout << "dd1/dT = " << T4.adj() << std::endl;
    
    // Test 5: norm_cdf(d1) with respect to all variables
    std::cout << "\n--- Test 5: norm_cdf(d1) derivatives ---" << std::endl;
    AADNumber S5(100.0);
    AADNumber K5(100.0);
    AADNumber T5(0.25);
    AADNumber r5(0.05);
    AADNumber sigma5(0.2);
    
    AADNumber sqrtT5 = sqrt(T5);
    AADNumber d1_5 = (log(S5 / K5) + (r5 + sigma5 * sigma5 * 0.5) * T5) / (sigma5 * sqrtT5);
    AADNumber N_d1 = norm_cdf(d1_5);
    
    std::cout << "N(d1) = " << N_d1.val() << std::endl;
    
    N_d1.setAdj(1.0);
    N_d1.propagate();
    
    std::cout << "dN(d1)/dS = " << S5.adj() << std::endl;
    std::cout << "dN(d1)/dsigma = " << sigma5.adj() << std::endl;
    std::cout << "dN(d1)/dr = " << r5.adj() << std::endl;
    std::cout << "dN(d1)/dT = " << T5.adj() << std::endl;
}

int main() {
    test_individual_derivatives();
    return 0;
}