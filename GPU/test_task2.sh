#!/bin/bash

echo "Building GPU AAD with updated Task 2 implementation..."

cd build

# Clean and rebuild
make clean
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "Build successful! Testing the implementation..."
    
    echo ""
    echo "=== Testing GPU AAD ==="
    ./test_gpu_aad
    
    echo ""
    echo "=== Testing Black-Scholes AAD with TRUE AAD recording ==="
    ./test_blackscholes_aad
    
    echo ""
    echo "=== Running Main Program ==="
    ./GPU_AAD
    
else
    echo "Build failed! Check errors above."
    exit 1
fi
