#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>

struct MarketData {
    std::string symbol;
    double spot_price;
    double strike_price;
    double time_to_expiry;
    double risk_free_rate;
    double volatility;
    double dividend_yield;
    long long timestamp;
    
    MarketData() : spot_price(0), strike_price(0), time_to_expiry(0), 
                   risk_free_rate(0), volatility(0), dividend_yield(0), timestamp(0) {}
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

class RealTimeDataInterface {
public:
    using DataCallback = std::function<void(const std::vector<MarketData>&)>;
    using ResultCallback = std::function<void(const std::vector<OptionGreeks>&)>;
    
    RealTimeDataInterface();
    ~RealTimeDataInterface();
    
    // Data source management
    bool initialize_python_bridge();
    bool start_data_stream(const std::vector<std::string>& symbols);
    void stop_data_stream();
    
    // Callbacks for real-time processing
    void set_data_callback(DataCallback callback);
    void set_result_callback(ResultCallback callback);
    
    // Manual data input (for testing)
    void push_market_data(const std::vector<MarketData>& data);
    
    // Get latest data
    std::vector<MarketData> get_latest_data();
    bool has_new_data() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};