#!/usr/bin/env python3
"""
Real-time financial data fetcher for AAD system
Supports multiple data sources: Yahoo Finance, Alpha Vantage, etc.
"""

import yfinance as yf
import requests
import json
import time
import csv
import numpy as np
from datetime import datetime
import threading
import queue

class RealTimeDataFetcher:
    def __init__(self, symbols=['AAPL', 'MSFT', 'GOOGL'], update_interval=60):
        self.symbols = symbols
        self.update_interval = update_interval  # seconds
        self.data_queue = queue.Queue()
        self.running = False
        
    def fetch_yahoo_data(self, symbol):
        """Fetch real-time data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            # Get current price and basic info
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'symbol': symbol,
                    'price': float(latest['Close']),
                    'volume': int(latest['Volume']),
                    'timestamp': datetime.now().isoformat(),
                    'bid': info.get('bid', latest['Close']),
                    'ask': info.get('ask', latest['Close']),
                    'volatility': self.calculate_volatility(hist['Close'])
                }
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_volatility(self, prices, window=20):
        """Calculate rolling volatility (annualized)"""
        if len(prices) < window:
            return 0.2  # Default volatility
        
        returns = np.log(prices / prices.shift(1)).dropna()
        volatility = returns.rolling(window=window).std().iloc[-1]
        return float(volatility * np.sqrt(252))  # Annualized
    
    def fetch_alpha_vantage_data(self, symbol, api_key):
        """Fetch from Alpha Vantage (requires API key)"""
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': 07UXAH521O1BNY6V
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'symbol': symbol,
                    'price': float(quote['05. price']),
                    'change': float(quote['09. change']),
                    'change_percent': quote['10. change percent'].rstrip('%'),
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            print(f"Alpha Vantage error for {symbol}: {e}")
            return None
    
    def generate_synthetic_data(self, symbol, base_price=100.0):
        """Generate synthetic market data for testing"""
        # Simple GBM simulation
        dt = 1.0 / 252 / 24 / 60  # 1 minute in years
        drift = 0.05  # 5% annual drift
        volatility = 0.2  # 20% annual volatility
        
        random_shock = np.random.normal(0, 1)
        price_change = base_price * (drift * dt + volatility * np.sqrt(dt) * random_shock)
        new_price = base_price + price_change
        
        return {
            'symbol': symbol,
            'price': max(new_price, 0.01),  # Prevent negative prices
            'timestamp': datetime.now().isoformat(),
            'volatility': volatility,
            'risk_free_rate': 0.05
        }
    
    def start_streaming(self, use_synthetic=True):
        """Start real-time data streaming"""
        self.running = True
        
        def stream_worker():
            while self.running:
                batch_data = []
                
                for symbol in self.symbols:
                    if use_synthetic:
                        data = self.generate_synthetic_data(symbol)
                    else:
                        data = self.fetch_yahoo_data(symbol)
                    
                    if data:
                        batch_data.append(data)
                
                # Put batch in queue for processing
                if batch_data:
                    self.data_queue.put(batch_data)
                    print(f"[{datetime.now()}] Fetched data for {len(batch_data)} symbols")
                
                time.sleep(self.update_interval)
        
        # Start streaming in background thread
        self.stream_thread = threading.Thread(target=stream_worker)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        print(f"Started streaming data for {self.symbols}")
    
    def get_latest_batch(self):
        """Get latest batch of market data"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop_streaming(self):
        """Stop data streaming"""
        self.running = False
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join()
        print("Stopped data streaming")
    
    def save_to_csv(self, filename="market_data.csv"):
        """Save streaming data to CSV for later analysis"""
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'symbol', 'price', 'volatility', 'volume']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            while self.running:
                batch = self.get_latest_batch()
                if batch:
                    for data in batch:
                        writer.writerow(data)
                time.sleep(1)

# Example usage
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = RealTimeDataFetcher(['AAPL', 'MSFT', 'TSLA'], update_interval=30)
    
    # Start streaming (synthetic data for demo)
    fetcher.start_streaming(use_synthetic=True)
    
    try:
        # Simulate processing for 5 minutes
        for i in range(10):
            batch = fetcher.get_latest_batch()
            if batch:
                print(f"\nBatch {i+1}:")
                for data in batch:
                    print(f"  {data['symbol']}: ${data['price']:.2f} (vol: {data.get('volatility', 0):.3f})")
            time.sleep(30)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        fetcher.stop_streaming()