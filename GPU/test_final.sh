#!/bin/bash

echo "🔨 Building with properly ordered functions..."

cd build

# Clean and rebuild
make clean
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful! Now testing TRUE AAD implementation..."
    
    echo ""
    echo "=== Testing Black-Scholes with AAD tape recording ==="
    ./test_blackscholes_aad
    
    echo ""
    echo "=== Testing Main GPU AAD Program ==="
    ./GPU_AAD
    
else
    echo "❌ Build failed!"
    exit 1
fi
