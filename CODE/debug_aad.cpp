#include <iostream>
#include <iomanip>
#include "AADNumber.h"

void test_simple_function() {
    std::cout << "=== Testing Simple Function: f(x,y) = x * y ===" << std::endl;
    
    AADNumber x(3.0);
    AADNumber y(4.0);
    
    std::cout << "Before computation:" << std::endl;
    std::cout << "x.val() = " << x.val() << ", x.adj() = " << x.adj() << std::endl;
    std::cout << "y.val() = " << y.val() << ", y.adj() = " << y.adj() << std::endl;
    
    AADNumber z = x * y;
    std::cout << "z = x * y = " << z.val() << std::endl;
    
    // Set adjoint of output
    z.setAdj(1.0);
    std::cout << "Set z.adj() = 1.0" << std::endl;
    
    // Propagate
    z.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "x.adj() = " << x.adj() << " (should be " << y.val() << ")" << std::endl;
    std::cout << "y.adj() = " << y.adj() << " (should be " << x.val() << ")" << std::endl;
    std::cout << std::endl;
}

void test_black_scholes_simple() {
    std::cout << "=== Testing Simple Black-Scholes Components ===" << std::endl;
    
    AADNumber S(100.0);  // Spot price
    AADNumber K(105.0);  // Strike price
    AADNumber T(0.25);   // Time to expiry
    AADNumber r(0.05);   // Risk-free rate
    AADNumber sigma(0.2); // Volatility
    
    std::cout << "Input values:" << std::endl;
    std::cout << "S = " << S.val() << std::endl;
    std::cout << "K = " << K.val() << std::endl;
    std::cout << "T = " << T.val() << std::endl;
    std::cout << "r = " << r.val() << std::endl;
    std::cout << "sigma = " << sigma.val() << std::endl;
    
    // Test individual components
    AADNumber ratio = S / K;
    std::cout << "S/K = " << ratio.val() << std::endl;
    
    AADNumber log_ratio = log(ratio);
    std::cout << "log(S/K) = " << log_ratio.val() << std::endl;
    
    // Test a simple payoff: max(S-K, 0) approximated as S-K for ITM
    AADNumber payoff = S - K;
    std::cout << "Simple payoff (S-K) = " << payoff.val() << std::endl;
    
    // Test derivatives
    payoff.setAdj(1.0);
    payoff.propagate();
    
    std::cout << "After propagation of payoff:" << std::endl;
    std::cout << "dPayoff/dS = " << S.adj() << " (should be 1.0)" << std::endl;
    std::cout << "dPayoff/dK = " << K.adj() << " (should be -1.0)" << std::endl;
    std::cout << std::endl;
}

void test_norm_cdf() {
    std::cout << "=== Testing norm_cdf Function ===" << std::endl;
    
    AADNumber x(0.5);
    std::cout << "x = " << x.val() << std::endl;
    
    AADNumber result = norm_cdf(x);
    std::cout << "norm_cdf(x) = " << result.val() << std::endl;
    
    result.setAdj(1.0);
    result.propagate();
    
    std::cout << "d(norm_cdf)/dx = " << x.adj() << std::endl;
    
    // Analytical derivative of norm_cdf is the PDF: (1/sqrt(2π)) * exp(-x²/2)
    double analytical = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x.val() * x.val());
    std::cout << "Analytical derivative = " << analytical << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    test_simple_function();
    test_black_scholes_simple();
    test_norm_cdf();
    
    return 0;
}