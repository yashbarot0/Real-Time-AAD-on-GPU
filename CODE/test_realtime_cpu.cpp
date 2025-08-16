#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <thread>
#include <random>
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

class RealTimeCPUAAD {
private:
    // Black-Scholes Call Option with AAD
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
        
        // Create AAD variables
        AADNumber S(data.spot_price);
        AADNumber K(data.strike_price);
        AADNumber T(data.time_to_expiry);
        AADNumber r(data.risk_free_rate);
        AADNumber sigma(data.volatility);
        
        // Compute option price using step-by-step construction
        AADNumber sqrtT = sqrt(T);
        AADNumber d1 = (log(S / K) + (r + sigma * sigma * 0.5) * T) / (sigma * sqrtT);
        AADNumber d2 = d1 - sigma * sqrtT;
        
        AADNumber N_d1 = norm_cdf(d1);
        AADNumber N_d2 = norm_cdf(d2);
        
        AADNumber term1 = S * N_d1;
        
        // Build exp(-r*T) step by step (exact pattern from working test)
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
    
    std::vector<MarketData> generateSyntheticData(const std::vector<std::string>& symbols) {
        std::vector<MarketData> data;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_dist(80.0, 120.0);
        std::uniform_real_distribution<> vol_dist(0.15, 0.35);
        
        for (const auto& symbol : symbols) {
            double spot = price_dist(gen);
            double strike = spot * 1.05; // 5% OTM
            double T = 0.25; // 3 months
            double r = 0.05; // 5% risk-free rate
            double vol = vol_dist(gen);
            
            data.emplace_back(symbol, spot, strike, T, r, vol);
        }
        
        return data;
    }
    
    void printResults(const std::vector<MarketData>& market_data, 
                     const std::vector<OptionGreeks>& results,
                     double computation_time_us) {
        
        std::cout << "\n" << std::string(100, '=') << std::endl;
        std::cout << "REAL-TIME AAD OPTION PRICING RESULTS" << std::endl;
        std::cout << std::string(100, '=') << std::endl;
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::setw(8) << "Symbol" 
                  << std::setw(10) << "Spot" 
                  << std::setw(10) << "Strike"
                  << std::setw(10) << "Vol"
                  << std::setw(12) << "Price"
                  << std::setw(10) << "Delta"
                  << std::setw(10) << "Vega"
                  << std::setw(10) << "Theta"
                  << std::setw(10) << "Rho" << std::endl;
        
        std::cout << std::string(100, '-') << std::endl;
        
        for (size_t i = 0; i < market_data.size(); ++i) {
            const auto& data = market_data[i];
            const auto& greeks = results[i];
            
            std::cout << std::setw(8) << data.symbol
                      << std::setw(10) << data.spot_price
                      << std::setw(10) << data.strike_price
                      << std::setw(10) << data.volatility
                      << std::setw(12) << greeks.price
                      << std::setw(10) << greeks.delta
                      << std::setw(10) << greeks.vega
                      << std::setw(10) << greeks.theta
                      << std::setw(10) << greeks.rho << std::endl;
        }
        
        std::cout << std::string(100, '=') << std::endl;
        std::cout << "Performance Metrics:" << std::endl;
        std::cout << "  Total computation time: " << computation_time_us << " µs" << std::endl;
        std::cout << "  Average time per option: " << (computation_time_us / market_data.size()) << " µs" << std::endl;
        std::cout << "  Options per second: " << (1e6 / (computation_time_us / market_data.size())) << std::endl;
        std::cout << std::string(100, '=') << std::endl;
    }
    
public:
    void runSingleBatch() {
        std::cout << "=== Single Batch Test ===" << std::endl;
        
        std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"};
        auto market_data = generateSyntheticData(symbols);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<OptionGreeks> results;
        results.reserve(market_data.size());
        
        for (const auto& data : market_data) {
            results.push_back(computeGreeks(data));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        printResults(market_data, results, duration.count());
    }
    
    void runRealTimeSimulation(int duration_seconds = 30) {
        std::cout << "\n=== Real-Time Simulation (" << duration_seconds << " seconds) ===" << std::endl;
        
        std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX"};
        
        auto start_time = std::chrono::steady_clock::now();
        int batch_count = 0;
        double total_computation_time = 0.0;
        int total_options = 0;
        
        while (true) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            
            if (elapsed.count() >= duration_seconds) {
                break;
            }
            
            // Generate new market data (simulating real-time feed)
            auto market_data = generateSyntheticData(symbols);
            
            // Process the batch
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            std::vector<OptionGreeks> results;
            results.reserve(market_data.size());
            
            for (const auto& data : market_data) {
                results.push_back(computeGreeks(data));
            }
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start);
            
            batch_count++;
            total_computation_time += batch_duration.count();
            total_options += market_data.size();
            
            // Print periodic updates
            if (batch_count % 10 == 0) {
                std::cout << "Batch " << batch_count 
                          << " | Options: " << market_data.size()
                          << " | Time: " << batch_duration.count() << " µs"
                          << " | Avg/option: " << (batch_duration.count() / market_data.size()) << " µs"
                          << std::endl;
            }
            
            // Simulate real-time delay (e.g., market data updates every 1 second)
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
        // Final statistics
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "REAL-TIME SIMULATION SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Total batches processed: " << batch_count << std::endl;
        std::cout << "Total options processed: " << total_options << std::endl;
        std::cout << "Total computation time: " << total_computation_time << " µs" << std::endl;
        std::cout << "Average time per option: " << (total_computation_time / total_options) << " µs" << std::endl;
        std::cout << "Average options per second: " << (total_options / duration_seconds) << std::endl;
        std::cout << "Computational throughput: " << (1e6 / (total_computation_time / total_options)) << " options/sec" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }
    
    void runBenchmark(int num_options = 10000) {
        std::cout << "\n=== Performance Benchmark (" << num_options << " options) ===" << std::endl;
        
        // Generate large dataset
        std::vector<MarketData> all_data;
        std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"};
        
        for (int i = 0; i < num_options; ++i) {
            std::string symbol = symbols[i % symbols.size()];
            double spot = 100.0 + (i % 50) - 25; // Vary prices
            double strike = spot * (1.0 + 0.1 * (i % 10 - 5) / 10.0); // Vary moneyness
            double T = 0.1 + 0.8 * (i % 10) / 10.0; // Vary time to expiry
            double r = 0.02 + 0.06 * (i % 5) / 5.0; // Vary interest rate
            double vol = 0.1 + 0.4 * (i % 8) / 8.0; // Vary volatility
            
            all_data.emplace_back(symbol, spot, strike, T, r, vol);
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<OptionGreeks> results;
        results.reserve(all_data.size());
        
        for (const auto& data : all_data) {
            results.push_back(computeGreeks(data));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Benchmark Results:" << std::endl;
        std::cout << "  Options processed: " << num_options << std::endl;
        std::cout << "  Total time: " << duration.count() << " µs (" << (duration.count() / 1000.0) << " ms)" << std::endl;
        std::cout << "  Average time per option: " << (duration.count() / num_options) << " µs" << std::endl;
        std::cout << "  Throughput: " << (1e6 / (duration.count() / num_options)) << " options/second" << std::endl;
        
        // Calculate some statistics
        double total_price = 0.0, total_delta = 0.0;
        for (const auto& result : results) {
            total_price += result.price;
            total_delta += result.delta;
        }
        
        std::cout << "  Average option price: $" << (total_price / num_options) << std::endl;
        std::cout << "  Average delta: " << (total_delta / num_options) << std::endl;
    }
};

int main() {
    std::cout << "Real-Time CPU AAD System Test" << std::endl;
    std::cout << "=============================" << std::endl;
    
    RealTimeCPUAAD system;
    
    try {
        // Test 1: Single batch
        system.runSingleBatch();
        
        // Test 2: Performance benchmark
        system.runBenchmark(1000);
        
        // Test 3: Real-time simulation (uncomment for longer test)
        std::cout << "\nStarting real-time simulation (10 seconds)..." << std::endl;
        system.runRealTimeSimulation(10);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}