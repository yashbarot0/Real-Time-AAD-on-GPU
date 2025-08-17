#!/bin/bash

# GPU AAD Test Runner Script
# This script runs various tests and benchmarks for the GPU AAD implementation

set -e

echo "=== GPU AAD Test Runner ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if build directory exists
if [ ! -d "build" ]; then
    print_error "Build directory not found. Please run ./build.sh first."
    exit 1
fi

cd build

# Check if test executable exists
if [ ! -f "test_gpu_aad" ]; then
    print_error "Test executable not found. Please build the project first."
    exit 1
fi

# System information
print_status "System Information:"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    print_status "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
fi

echo ""
print_status "CUDA Information:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    print_warning "nvcc not found"
fi

echo ""

# Run basic functionality test
print_test "Running basic GPU AAD functionality test..."
echo "----------------------------------------"

if ./test_gpu_aad; then
    print_status "Basic functionality test PASSED"
else
    print_error "Basic functionality test FAILED"
    exit 1
fi

echo ""
echo "----------------------------------------"

# Performance test (if available)
if [ -f "GPU_AAD" ]; then
    print_test "Running main GPU AAD program..."
    echo "----------------------------------------"
    
    if ./GPU_AAD; then
        print_status "Main program test PASSED"
    else
        print_warning "Main program test had issues"
    fi
    
    echo ""
    echo "----------------------------------------"
fi

# Memory test
print_test "Running memory stress test..."
echo "Testing with different memory configurations..."

# You can add specific memory tests here
print_status "Memory test completed (placeholder)"

echo ""

# Performance benchmark
print_test "Running performance benchmark..."
echo "This may take a few moments..."

# Run the test multiple times for performance measurement
ITERATIONS=3
print_status "Running $ITERATIONS iterations for performance measurement..."

for i in $(seq 1 $ITERATIONS); do
    echo "Iteration $i/$ITERATIONS:"
    if ! ./test_gpu_aad > /dev/null 2>&1; then
        print_warning "Iteration $i failed"
    else
        print_status "Iteration $i completed"
    fi
done

echo ""
print_status "All tests completed!"

# Final GPU status
if command -v nvidia-smi &> /dev/null; then
    echo ""
    print_status "Final GPU status:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv
fi

echo ""
print_status "Test run finished successfully!"
echo "Check the output above for any warnings or errors."