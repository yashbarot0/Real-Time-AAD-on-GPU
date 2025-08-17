# GPU AAD Build Instructions

## Build Commands for CUDA Server

```bash
# Navigate to GPU directory
cd /path/to/Real-Time-AAD-on-GPU/GPU

# Clean previous builds (if any)
rm -rf build

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)

# Optional: Build specific targets
make GPU_AAD                    # Main executable
make test_gpu_aad              # GPU AAD tests
make test_blackscholes_aad     # Black-Scholes specific tests
```

## Troubleshooting

### If you get CUDA toolkit errors:
```bash
# Set CUDA toolkit path if needed
export CUDA_ROOT=/usr/local/cuda
export PATH=$CUDA_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH

# Then retry cmake
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT ..
```

### If you get architecture errors:
Edit CMakeLists.txt and adjust the CUDA architectures line:
```cmake
# For older GPUs (Tesla K80, etc.)
set(CMAKE_CUDA_ARCHITECTURES "35;50;60")

# For newer GPUs (V100, A100, etc.)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

# For mixed environments
set(CMAKE_CUDA_ARCHITECTURES "35;50;60;70;75;80;86")
```

## Testing

```bash
# Run basic GPU test
./test_gpu_aad

# Run Black-Scholes AAD test
./test_blackscholes_aad

# Run main program
./GPU_AAD
```

## Files Modified to Fix Linking Issue

1. `device_functions.cuh` - NEW: Shared device functions header
2. `cuda_kernels.cu` - UPDATED: Uses shared header
3. `blackscholes_aad_kernels_fixed.cu` - UPDATED: Uses shared header  
4. `CMakeLists.txt` - FIXED: Removed syntax error
