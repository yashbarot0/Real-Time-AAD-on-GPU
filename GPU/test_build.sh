#!/bin/bash

echo "Building with fixed AAD function linking..."

cd build

# Clean and rebuild
make clean
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful! Testing AAD implementation..."
    
    echo ""
    echo "=== Testing Black-Scholes with TRUE AAD ==="
    ./test_blackscholes_aad
    
else
    echo "❌ Build failed!"
    exit 1
fi
