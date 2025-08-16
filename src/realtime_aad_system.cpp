#include "data_interface.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

// Include your AAD headers (adjust paths as needed)
#ifdef USE_GPU
#include "../GPU/GPUAADNumber.h"
#include "../GPU/GPUAADTape.h"
using AADNumber = GPUAADNumber;
using AADTape = GPUAADTape;
#else
#include "../CODE/AADNumber.h"
using AADTape = void; // CPU version doesn't need explicit tape
#endif

class RealTimeAADSystem {
private:
    RealTimeDataInterface data_interface;
    bool running = false;
    
    // Black-Scholes with AAD
    AADNumber BlackScholesCall(const AADNumber& S, const AADNumber& K,
                              const AADNumber& T, const AADNumber& r,
                              const AADNumber& sigma) {
        AADNumber sqrtT = sqrt(T);
        AADNumber d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        AADNumber d2 = d1 - sigma * sqrtT;
        AADNumber call = S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2);
        return call;
    }
    
    OptionGreeks computeGreeks(const MarketData& data) {
        OptionGreeks result;
        
#ifdef USE_GPU
        GPUAADTape tape;
        GPUAADNumber::set_active_tape(&tape);
#endif
        
        // Create AAD variables
        AADNumber S(data.spot_price);
        AADNumber K(data.strike_price);
        AADNumber T(data.time_to_expiry);
        AADNumber r(data.risk_free_rate);
        AADNumber sigma(data.volatility);
        
        // Compute option price
        AADNumber price = BlackScholesCall(S, K, T, r, sigma);
        result.price = price.val();
        
        // Compute Greeks via AAD
#ifdef USE_GPU
        tape.set_adjoint(price.index(), 1.0);
        tape.propagate_gpu();
        
        result.delta = S.adj();
        result.vega = sigma.adj();
        result.rho = r.adj();
        result.theta = T.adj();
#else
        price.setAdj(1.0);
        price.propagate();
        
        result.delta = S.adj();
        result.vega = sigma.adj();
        result.rho = r.adj();
        result.theta = T.adj();
#endif
        
        return result;
    }
    
    void processMarketData(const std::vector<MarketData>& market_data) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "Processing " << market_data.size() << " market data points" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<OptionGreeks> results;
        results.reserve(market_data.size());
        
        for (const auto& data : market_data) {
            auto greeks = computeGreeks(data);
            results.push_back(greeks);
            
            // Print results
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Symbol: " << std::setw(6) << data.symbol 
                      << " | Spot: $" << std::setw(8) << data.spot_price
                      << " | Strike: $" << std::setw(8) << data.strike_price << std::endl;
            
            std::cout << "  Price: $" << std::setw(8) << greeks.price
                      << " | Delta: " << std::setw(8) << greeks.delta
                      << " | Vega: " << std::setw(8) << greeks.vega
                      << " | Theta: " << std::setw(8) << greeks.theta
                      << " | Rho: " << std::setw(8) << greeks.rho << std::endl;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "Computation completed in " << duration.count() << " µs" << std::endl;
        std::cout << "Average time per option: " << (duration.count() / market_data.size()) << " µs" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }
    
public:
    bool initialize() {
        std::cout << "Initializing Real-Time AAD System..." << std::endl;
        
#ifdef USE_GPU
        std::cout << "Using GPU AAD implementation" << std::endl;
#else
        std::cout << "Using CPU AAD implementation" << std::endl;
#endif
        
        // Initialize data interface
        if (!data_interface.initialize_python_bridge()) {
            std::cerr << "Failed to initialize data bridge" << std::endl;
            return false;
        }
        
        // Set up callback for incoming data
        data_interface.set_data_callback([this](const std::vector<MarketData>& data) {
            processMarketData(data);
        });
        
        return true;
    }
    
    void start(const std::vector<std::string>& symbols = {"AAPL", "MSFT", "GOOGL", "TSLA"}) {
        if (!initialize()) {
            std::cerr << "Failed to initialize system" << std::endl;
            return;
        }
        
        std::cout << "Starting real-time processing for symbols: ";
        for (const auto& symbol : symbols) {
            std::cout << symbol << " ";
        }
        std::cout << std::endl;
        
        running = true;
        data_interface.start_data_stream(symbols);
        
        std::cout << "System running. Press Ctrl+C to stop..." << std::endl;
        
        // Keep running until stopped
        while (running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    void stop() {
        running = false;
        data_interface.stop_data_stream();
        std::cout << "System stopped." << std::endl;
    }
};

int main() {
    RealTimeAADSystem system;
    
    try {
        // Start the real-time system
        system.start({"AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"});
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}