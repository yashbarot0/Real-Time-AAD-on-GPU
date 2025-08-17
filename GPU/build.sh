#!/bin/bash

# GPU AAD Build Script
# This script automates the build process for the GPU AAD implementation

set -e  # Exit on any error

echo "=== GPU AAD Build Script ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_status "CUDA found: version $CUDA_VERSION"
else
    print_error "CUDA not found. Please install CUDA toolkit first."
    echo "Visit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
    print_status "CMake found: version $CMAKE_VERSION"
    
    # Check if version is >= 3.18
    CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d. -f1)
    CMAKE_MINOR=$(echo $CMAKE_VERSION | cut -d. -f2)
    if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 18 ]); then
        print_error "CMake version 3.18+ required, found $CMAKE_VERSION"
        exit 1
    fi
else
    print_error "CMake not found. Please install CMake 3.18+"
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    print_status "Checking GPU..."
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits | while read line; do
        print_status "GPU: $line"
    done
else
    print_warning "nvidia-smi not found. GPU may not be available."
fi

# Set CUDA environment variables if not set
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        print_status "Set CUDA_HOME to $CUDA_HOME"
    fi
fi

if [ -n "$CUDA_HOME" ]; then
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

# Create build directory
print_status "Creating build directory..."
mkdir -p build
cd build

# Clean previous build if requested
if [ "$1" = "clean" ]; then
    print_status "Cleaning previous build..."
    rm -rf *
fi

# Configure with CMake
print_status "Configuring with CMake..."
if ! cmake ..; then
    print_error "CMake configuration failed!"
    exit 1
fi

# Build the project
print_status "Building project..."
NPROC=$(nproc 2>/dev/null || echo 4)
print_status "Using $NPROC parallel jobs"

if ! make -j$NPROC; then
    print_error "Build failed!"
    print_status "Trying with verbose output..."
    make VERBOSE=1
    exit 1
fi

print_status "Build completed successfully!"

# Check if executables were created
if [ -f "test_gpu_aad" ]; then
    print_status "Test executable created: test_gpu_aad"
else
    print_warning "Test executable not found"
fi

if [ -f "GPU_AAD" ]; then
    print_status "Main executable created: GPU_AAD"
else
    print_warning "Main executable not found"
fi

print_status "Build script completed!"
echo ""
echo "To run the test:"
echo "  cd build && ./test_gpu_aad"
echo ""
echo "To run the main program:"
echo "  cd build && ./GPU_AAD"