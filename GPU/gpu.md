# GPU AAD Implementation - Build Instructions

## Prerequisites

1. **NVIDIA GPU** with CUDA Compute Capability 6.0+ (GeForce GTX 10 series or newer)
2. **CUDA Toolkit 11.0+** (Download from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads))
3. **CMake 3.18+**
4. **C++ Compiler** (GCC 7+, Visual Studio 2019+, or Clang 8+)

## File Structure

Create the following directory structure:

```
gpu_aad/
├── CMakeLists.txt
├── main.cpp
├── AADTypes.h
├── GPUAADTape.h
├── GPUAADTape.cpp
├── GPUAADNumber.h
├── GPUAADNumber.cpp
└── cuda_kernels.cu
```

## Building

### Linux/macOS
```bash
mkdir gpu_aad
cd gpu_aad

# Copy all the code files from the artifact above

mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Windows (Visual Studio)
```cmd
mkdir gpu_aad
cd gpu_aad

REM Copy all the code files from the artifact above

mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

## Running

```bash
# Linux/macOS
./GPU_AAD

# Windows
Release\GPU_AAD.exe
```

## Expected Output

```
=== Single Evaluation Test ===
Price: 10.4506
Delta (dP/dS): 0.636831
Vega (dP/dsigma): 37.1641
Rho (dP/dr): 53.2325
Theta (dP/dT): -4.31508

=== GPU AAD Benchmark ===
Total runs: 10000
Average price: 10.4506
Average delta: 0.636831
Total time: 0.285 seconds
Avg time per evaluation: 28.5 µ