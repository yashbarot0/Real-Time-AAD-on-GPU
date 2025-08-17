# GPU AAD Implementation - Server Setup Guide

## Prerequisites

### 1. CUDA Toolkit Installation
```bash
# Check if CUDA is already installed
nvcc --version
nvidia-smi

# If not installed, install CUDA toolkit (Ubuntu/Debian)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Or use conda (recommended for easier management)
conda install -c nvidia cuda-toolkit
```

### 2. CMake (version 3.18+)
```bash
# Check CMake version
cmake --version

# Install if needed
sudo apt-get install cmake

# Or install newer version
wget https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0-linux-x86_64.sh
chmod +x cmake-3.25.0-linux-x86_64.sh
sudo ./cmake-3.25.0-linux-x86_64.sh --prefix=/usr/local --skip-license
```

### 3. GCC/G++ Compiler
```bash
# Install build essentials
sudo apt-get install build-essential
```

## Build Instructions

### Option 1: Quick Build Script
```bash
# Make the build script executable and run it
chmod +x build.sh
./build.sh
```

### Option 2: Manual Build
```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)

# Or build with verbose output for debugging
make VERBOSE=1
```

## Running the Code

### 1. Test the GPU AAD Implementation
```bash
# Run the test program
./build/test_gpu_aad

# Expected output should show:
# - GPU initialization success
# - Basic arithmetic operations
# - Math function tests
# - GPU propagation results
# - Performance metrics
```

### 2. Run the Main Application
```bash
# Run the main GPU AAD program
./build/GPU_AAD
```

## Troubleshooting

### CUDA Not Found
```bash
# Set CUDA paths manually
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Then rebuild
cd build && make clean && cmake .. && make -j$(nproc)
```

### GPU Architecture Issues
If you get architecture-related errors, update the CMakeLists.txt:
```cmake
# Find your GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv

# Then update CMAKE_CUDA_ARCHITECTURES in CMakeLists.txt
# For example, for RTX 3080 (Ampere), use "86"
# For RTX 2080 (Turing), use "75"
```

### Memory Issues
```bash
# Check GPU memory
nvidia-smi

# If you get out-of-memory errors, reduce batch sizes in the code
# or use the CPU fallback mode
```

### Compilation Errors
```bash
# Clean build and try again
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
make VERBOSE=1
```

## Performance Monitoring

The test program will output performance metrics including:
- GPU memory allocation time
- Data transfer times
- Kernel execution times
- Memory usage statistics

## Server-Specific Considerations

### SSH with X11 Forwarding (if needed for debugging)
```bash
ssh -X username@server
```

### Running in Background
```bash
# Use screen or tmux for long-running processes
screen -S gpu_aad
./build/test_gpu_aad
# Ctrl+A, D to detach

# Reattach later
screen -r gpu_aad
```

### Resource Monitoring
```bash
# Monitor GPU usage while running
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

## Expected Test Output

When successful, you should see output like:
```
=== GPU AAD Enhanced Implementation Test ===
Initializing GPU AAD system...
GPU Device: NVIDIA GeForce RTX 3080
Compute Capability: 8.6
Global Memory: 10240 MB
GPU AAD system initialized successfully!
GPU available: Yes
Memory usage: 95 MB

=== Testing Basic Operations ===
x + y = 5 (expected: 5.0)
x * y = 6 (expected: 6.0)
y / x = 1.5 (expected: 1.5)
-x = -2 (expected: -2.0)

=== Testing Math Functions ===
log(1.0) = 0 (expected: 0.0)
exp(1.0) = 2.71828 (expected: ~2.718)
sqrt(4.0) = 2 (expected: 2.0)

=== Test Completed Successfully ===
```