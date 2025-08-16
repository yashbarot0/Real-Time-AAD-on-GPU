#include <iostream>
#include <iomanip>
#include "AADNumber.h"

struct MarketData {
    std::string symbol;
    double spot_price;
    double strike_price;
    double time_to_expiry;
    double risk_free_rate;
    double volatility;
    
    MarketData(const std::string& sym, double spot, double strike, double T, double r, double vol)
        : symbol(sym), spot_price(spot), strike_price(strike), time_to_expiry(T), 
          risk_free_rate(r), volatility(vol) {}
};

struct OptionGreeks {
    double price;
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
    
    OptionGreeks() : price(0), delta(0), gamma(0), theta(0), vega(0), rho(0) {}
};

OptionGreeks computeGreeks(const MarketData& data) {
    OptionGreeks result;
    
    // Create AAD variables
    AADNumber S(data.spot_price);
    AADNumber K(data.strike_price);
    AADNumber T(data.time_to_expiry);
    AADNumber r(data.risk_free_rate);
    AADNumber sigma(data.volatility);
    
    // Compute option price using the working Black-Scholes
    AADNumber sqrtT = sqrt(T);
    AADNumber d1 = (log(S / K) + (r + sigma * sigma * 0.5) * T) / (sigma * sqrtT);
    AADNumber d2 = d1 - sigma * sqrtT;
    
    AADNumber N_d1 = norm_cdf(d1);
    AADNumber N_d2 = norm_cdf(d2);
    
    AADNumber term1 = S * N_d1;
    
    // Build exp(-r*T) step by step
    AADNumber product_rT = r * T;
    AADNumber neg_product_rT = -product_rT;
    AADNumber discount = exp(neg_product_rT);
    
    AADNumber term2 = K * discount * N_d2;
    
    AADNumber price = term1 - term2;
    result.price = price.val();
    
    // Compute Greeks via AAD
    price.setAdj(1.0);
    price.propagate();
    
    result.delta = S.adj();
    result.vega = sigma.adj();
    result.rho = r.adj();
    result.theta = T.adj();
    
    return result;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Testing Greeks Calculation ===" << std::endl;
    
    // Test with standard Black-Scholes parameters
    MarketData data("TEST", 100.0, 100.0, 0.25, 0.05, 0.2);
    
    std::cout << "Input parameters:" << std::endl;
    std::cout << "S = " << data.spot_price << std::endl;
    std::cout << "K = " << data.strike_price << std::endl;
    std::cout << "T = " << data.time_to_expiry << std::endl;
    std::cout << "r = " << data.risk_free_rate << std::endl;
    std::cout << "σ = " << data.volatility << std::endl;
    
    OptionGreeks greeks = computeGreeks(data);
    
    std::cout << "\nAAD Results:" << std::endl;
    std::cout << "Price = " << greeks.price << std::endl;
    std::cout << "Delta = " << greeks.delta << std::endl;
    std::cout << "Vega = " << greeks.vega << std::endl;
    std::cout << "Rho = " << greeks.rho << std::endl;
    std::cout << "Theta = " << greeks.theta << std::endl;
    
    // Analytical Black-Scholes for comparison
    double S = 100.0, K = 100.0, T = 0.25, r = 0.05, sigma = 0.2;
    double sqrt_T = std::sqrt(T);
    double d1 = (std::log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;
    
    double N_d1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
    double N_d2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));
    double n_d1 = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * d1 * d1);
    
    double analytical_price = S * N_d1 - K * std::exp(-r * T) * N_d2;
    double analytical_delta = N_d1;
    double analytical_vega = S * n_d1 * sqrt_T;
    double analytical_rho = K * T * std::exp(-r * T) * N_d2;
    double analytical_theta = -(S * n_d1 * sigma) / (2 * sqrt_T) - r * K * std::exp(-r * T) * N_d2;
    
    std::cout << "\nAnalytical Results:" << std::endl;
    std::cout << "Price = " << analytical_price << std::endl;
    std::cout << "Delta = " << analytical_delta << std::endl;
    std::cout << "Vega = " << analytical_vega << std::endl;
    std::cout << "Rho = " << analytical_rho << std::endl;
    std::cout << "Theta = " << analytical_theta << std::endl;
    
    std::cout << "\nErrors:" << std::endl;
    std::cout << "Price error = " << std::abs(greeks.price - analytical_price) << std::endl;
    std::cout << "Delta error = " << std::abs(greeks.delta - analytical_delta) << std::endl;
    std::cout << "Vega error = " << std::abs(greeks.vega - analytical_vega) << std::endl;
    std::cout << "Rho error = " << std::abs(greeks.rho - analytical_rho) << std::endl;
    std::cout << "Theta error = " << std::abs(greeks.theta - analytical_theta) << std::endl;
    
    return 0;
}