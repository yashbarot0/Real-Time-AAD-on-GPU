Here's the full `CMakeLists.txt` for your root directory, now added to your project. Next, here’s a **`Dockerfile`** and some **example code stubs** to get you started.

---

### 🐳 `Dockerfile` (for CUDA + CMake + Catch2)

```Dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 python3-pip \
    libgtest-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Catch2
RUN git clone https://github.com/catchorg/Catch2.git /Catch2 && \
    cd /Catch2 && cmake -Bbuild -H. && cmake --build build --target install

# Set workdir
WORKDIR /app

# Copy project
COPY . .

# Build
RUN cmake -S . -B build && cmake --build build

# Default command
CMD [\"./build/tests/test_runner\"]
```

---

### 🧪 Example Test (Catch2)

**`tests/test_tape.cpp`**

```cpp
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "aad/tape.h"

TEST_CASE("Basic AAD tape functionality", "[aad]") {
    Tape tape;
    tape.record(1.0, 0.5);
    REQUIRE(tape.size() == 1);
}
```

---

### 💻 CUDA Kernel Example

**`src/aad/forward_kernel.cu`**

```cpp
#include <cuda_runtime.h>
#include "aad/forward_kernel.h"

__global__ void forward_pass_kernel(double* input, double* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        output[idx] = input[idx] * input[idx];  // dummy forward op
    }
}

void launch_forward_pass(double* input, double* output, int N) {
    double *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(double));
    cudaMalloc(&d_output, N * sizeof(double));
    cudaMemcpy(d_input, input, N * sizeof(double), cudaMemcpyHostToDevice);

    forward_pass_kernel<<<(N + 255) / 256, 256>>>(d_input, d_output, N);
    cudaMemcpy(output, d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
```

---

Would you like me to generate:

1. The full subdirectory `CMakeLists.txt` (for `src/` and `tests/`)?
2. Python script for plotting?
3. `black_scholes_model.h/.cu` stubs?

Let me know how deep you'd like to go right now.
