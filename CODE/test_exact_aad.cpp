#include <iostream>
#include <iomanip>
#include "AADNumber.h"

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Testing Exact AADNumber Implementation ===" << std::endl;
    
    // Test 1: Simple multiplication (should work)
    std::cout << "\n--- Test 1: Simple Multiplication ---" << std::endl;
    AADNumber x(3.0);
    AADNumber y(4.0);
    AADNumber z = x * y;
    
    std::cout << "z = x * y = " << z.val() << std::endl;
    z.setAdj(1.0);
    z.propagate();
    
    std::cout << "x.adj() = " << x.adj() << " (should be 4)" << std::endl;
    std::cout << "y.adj() = " << y.adj() << " (should be 3)" << std::endl;
    
    // Test 2: Chain of operations
    std::cout << "\n--- Test 2: Chain of Operations ---" << std::endl;
    AADNumber a(2.0);
    AADNumber b(3.0);
    AADNumber c = a * b;  // c = 6
    AADNumber d = c + a;  // d = 8
    
    std::cout << "d = (a * b) + a = " << d.val() << std::endl;
    d.setAdj(1.0);
    d.propagate();
    
    std::cout << "a.adj() = " << a.adj() << " (should be 4: 3 from multiplication + 1 from addition)" << std::endl;
    std::cout << "b.adj() = " << b.adj() << " (should be 2)" << std::endl;
    
    // Test 3: Test with log function
    std::cout << "\n--- Test 3: Logarithm Function ---" << std::endl;
    AADNumber u(2.0);
    AADNumber v = log(u);
    
    std::cout << "v = log(u) = " << v.val() << std::endl;
    v.setAdj(1.0);
    v.propagate();
    
    std::cout << "u.adj() = " << u.adj() << " (should be 0.5 = 1/u)" << std::endl;
    
    // Test 4: Test with norm_cdf
    std::cout << "\n--- Test 4: norm_cdf Function ---" << std::endl;
    AADNumber w(0.5);
    AADNumber n = norm_cdf(w);
    
    std::cout << "n = norm_cdf(w) = " << n.val() << std::endl;
    n.setAdj(1.0);
    n.propagate();
    
    std::cout << "w.adj() = " << w.adj() << " (should be ~0.352)" << std::endl;
    
    // Test 5: Simple Black-Scholes component
    std::cout << "\n--- Test 5: Simple BS Component ---" << std::endl;
    AADNumber S(100.0);
    AADNumber K(105.0);
    AADNumber ratio = S / K;
    AADNumber log_ratio = log(ratio);
    
    std::cout << "log_ratio = log(S/K) = " << log_ratio.val() << std::endl;
    log_ratio.setAdj(1.0);
    log_ratio.propagate();
    
    std::cout << "S.adj() = " << S.adj() << " (should be 1/(S) = " << (1.0/S.val()) << ")" << std::endl;
    std::cout << "K.adj() = " << K.adj() << " (should be -1/K = " << (-1.0/K.val()) << ")" << std::endl;
    
    return 0;
}