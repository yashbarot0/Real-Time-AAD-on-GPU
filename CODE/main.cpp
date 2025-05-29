// test_main.cpp
#include <iostream>
#include <chrono>
#include "AADNumber.h"

// Black-Scholes Call Option Formula with AADNumbers
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
    constexpr int NUM_RUNS = 1000000;
    auto start = std::chrono::high_resolution_clock::now();

    double total = 0.0;

    for (int i = 0; i < NUM_RUNS; ++i) {
        AADNumber S(100.0);
        AADNumber K(100.0);
        AADNumber T(1.0);
        AADNumber r(0.05);
        AADNumber sigma(0.2);

        AADNumber price = BlackScholesCall(S, K, T, r, sigma);
        price.setAdj(1.0);
        price.propagate();

        total += price.val();  // prevent compiler optimization
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Total simulated price sum: " << total << "\n";
    std::cout << "Total time: " << elapsed.count() << " seconds\n";
    std::cout << "Avg time per evaluation: " << (elapsed.count() / NUM_RUNS * 1e6) << " µs\n";

    return 0;
}
