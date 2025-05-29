

# Real-Time AAD on GPU: Implementation Guide

This guide walks through building a real-time Adjoint Algorithmic Differentiation (AAD) system in C++/CUDA for financial risk (e.g. option sensitivities). We cover the theory and practical steps needed to leverage an NVIDIA RTX 2080 GPU (Turing, compute capability 7.5) and optional Google Colab GPU to compute derivatives (Greeks) efficiently. The following sections explain the concepts, required tools, design patterns, optimizations, and testing needed for a robust GPU-based AAD prototype.

## 1. Overview of AAD and Its Relevance in Finance

**Algorithmic Differentiation (AD)** is a set of techniques to compute exact derivatives (sensitivities) of a function defined by a computer program. It works by applying the chain rule at the level of code operations, yielding derivatives accurate to machine precision (unlike finite differences which introduce truncation error). There are two main modes: *forward (tangent) mode* and *reverse (adjoint) mode*. Reverse-mode AD (often called AAD) computes gradients of many inputs with respect to a small number of outputs by recording intermediate operations during a forward pass (the “tape”) and then propagating sensitivities backwards.

In computational **finance**, AAD is especially powerful for risk management. For example, computing *Greeks* (sensitivities of a portfolio’s value to market factors) traditionally requires repeating pricing computations hundreds of times (bumping each input and revaluing). AAD can compute thousands of such sensitivities in roughly one pass through the code. As noted by Capriotti & Giles, AAD “has revolutionized the way risk is computed in the financial industry,” especially for Monte Carlo and PDE-based pricing. In practice, AAD can yield 10–1000× speedups over naive finite-difference Greeks (depending on complexity) by leveraging the fact that the cost of a reverse sweep is only a small multiple of the forward valuation.

*Key point:* In AAD, one executes the original pricing code (the *primal* or forward pass) while recording enough information (the *tape*) to replay the operations. Then a reverse pass computes gradients of a single output (e.g. a final price or P\&L) with respect to all inputs. This is ideal for finance where e.g. one P\&L has derivatives with respect to hundreds of market risk factors. The computational cost is roughly proportional to one extra forward evaluation (plus overhead), instead of hundreds as with finite differences.

## 2. Software and Hardware Requirements

* **GPU Hardware:** NVIDIA RTX 2080 (Turing GPU, Compute Capability 7.5) or similar. This architecture supports CUDA starting from version 10.0. For best performance, use the latest CUDA (e.g. CUDA 12.x) and drivers. The GTX/Turing GPUs have 8–12 GB memory; ensure enough memory for your tape and data.
* **Host System:** A 64-bit CPU system (Linux or Windows). At least 16 GB RAM is recommended, though real-time tasks may fit in less if carefully managed. A Linux environment (Ubuntu, CentOS) is common in HPC; Windows with NVidia Nsight support is also feasible.
* **CUDA Toolkit:** Install a CUDA Toolkit compatible with your GPU. RTX 2080 is fully supported by CUDA 10 and later. As of 2025 the latest is CUDA 12.9 (May 2025). Use NVCC (the NVIDIA compiler) and a C++ compiler (GCC 9+ on Linux, or MSVC 2019/2022 on Windows) that is compatible with your CUDA version.
* **C++ Standard:** Use modern C++ (C++14 or later) for cleaner code and safe concurrency primitives (`std::atomic`, etc.). The CUDA compiler supports many C++11/C++14 features on device code.
* **GPU Libraries:** Leverage NVIDIA libraries where helpful:

  * **cuBLAS** (dense linear algebra) for matrix operations or linear solves.
  * **cuRAND** for generating random numbers (e.g. simulating Monte Carlo paths on GPU).
  * **Thrust** (part of CUDA) for high-level parallel primitives (sort, scan, transform). Thrust works well for device vectors and transformations with functors.
  * **cuSPARSE**, **cuSOLVER** if sparse matrices or solvers are needed (e.g. for calibration).
  * **NVIDIA Nsight Tools:** For profiling and debugging, install Nsight Systems and Nsight Compute. (Note: the old nvprof/Visual Profiler are deprecated, but still available on older systems.)
* **Other Tools:** Standard build tools (CMake is recommended for managing compilation) and version control (Git). For unit testing, C++ frameworks like Google Test or Catch2 can help structure tests.
* **Optional (Google Colab):** If using Colab, note that the GPUs there are typically T4 or P100, with up to \~16 GB. Code targeting RTX 2080 should run on Colab, but check that the CUDA version matches (Colab often has CUDA 11.x). Data upload may need network considerations. Colab can be useful for prototyping but an on-prem HPC cluster is likely faster and more reliable for continuous runs.

## 3. Code Structure and Modular Design

A clean modular design is key. We suggest organizing the code into components such as:

* **AD Tape Module:** A *tape* data structure records each elementary operation during the forward pass. For example, define a `struct TapeEntry { OpType op; int lhs, rhs, out; double localDeriv; };` where `OpType` enumerates add/mul/div/etc. Each GPU thread (or block) logs operations in its segment of the tape. Use a global `std::atomic<int>` (or `atomicAdd`) to assign slots thread-safely. The tape can be allocated in CUDA-managed (`cudaMallocManaged`) or pinned host memory for easy access from host/device. For example:

  ```cpp
  // Simplified tape entry
  struct TapeEntry { 
      char op;       // '+','*','sin', etc.
      int lhs, rhs;  // indices of input variables
      int out;       // index of output variable
      double dval;   // local derivative (∂out/∂lhs etc.)
  };
  __device__ __managed__ TapeEntry *tape;         // global tape buffer
  __device__ std::atomic<int> tapePos{0};        // position index
  ```

  (Indices like `lhs, rhs, out` can be offsets into an array of AD values.) Each operation’s operator overload will do something like:

  ```cpp
  __device__ ADVal operator+(const ADVal &a, const ADVal &b) {
      ADVal r; r.val = a.val + b.val;
      int idx = tapePos.fetch_add(1);
      tape[idx] = {'+', a.index, b.index, r.index, 1.0 /* d (a+b)/d a */};
      return r;
  }
  ```

  This records the operation on the tape (here storing derivative w\.r.t. each input). In practice you’d store both derivatives or record symmetric entries. The tape module also includes logic to reset/clear the tape between iterations. The tape design follows Gremse *et al.*’s pattern: two large arrays (values and local derivatives) preallocated and filled atomically.

* **Primal (Forward) Kernel(s):** These CUDA kernels compute the target quantity (e.g. option price or portfolio value) for each data point or Monte Carlo path. They use *AD-enabled* data types so that overloaded arithmetic automatically logs to the tape. For instance, define a device class `ADVal` that holds `val` and an `adjointIndex`, with overloaded operators as above. A sample kernel might be:

  ```cpp
  __global__ void computePrice(ADVal *S, ADVal *option) {
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      ADVal s = S[tid];
      // Example payoff: option = exp(-r*T) * max(s-K, 0)
      ADVal payoff = max(s - ADVal(K), ADVal(0.0));
      ADVal price = exp(ADVal(-r*T)) * payoff;
      option[tid] = price;
      // The operators here automatically record add/mul on `tape`.
  }
  ```

  Each thread processes one scenario or path. All intermediate arithmetic logs its operation to the tape (using the global `tapePos` index) so that after the forward pass, the tape contains a computational graph (reverse AD DAG) of all operations.

* **Adjoint (Backward) Sweep:** After the forward pass finishes, launch a reverse (adjoint) kernel to propagate gradients from outputs back to inputs. One approach is to have one thread (or a small group) walk backwards through the tape entries in reverse order, accumulating gradients. For example:

  ```cpp
  __global__ void computeAdjoint(double *grads /*size=numVars*/, int tapeSize) {
      int idx = blockIdx.x*blockDim.x + threadIdx.x;
      if (idx == 0) {  // assume single-thread gradient pass for simplicity
          grads[outputIndex] = 1.0;  // d(output)/d(output) = 1
          for(int k = tapeSize-1; k >= 0; --k) {
              auto e = tape[k];
              double gout = grads[e.out];
              if (e.op == '+') {
                  // For addition: d(lhs)+d(rhs) += gout
                  atomicAdd(&grads[e.lhs], gout * e.dval /*==1.0*/);
                  atomicAdd(&grads[e.rhs], gout * e.dval /*==1.0*/);
              } 
              else if (e.op == '*') {
                  // For multiplication: use stored local derivatives
                  atomicAdd(&grads[e.lhs], gout * e.dval /*=b.val*/);
                  atomicAdd(&grads[e.rhs], gout * /*a.val or similar*/);
              }
              // handle other ops (sin, exp, /, etc.)
          }
      }
  }
  ```

  In practice, you can parallelize the backward sweep (e.g. each warp or block handles a segment of the tape) as long as you manage race conditions (often by atomic adds). The key is: once the tape is filled, the adjoint pass is conceptually a single pass through the tape from end to start, updating gradients. Gremse *et al.* note that the tape (with \~24 bytes/edge) stores the DAG and then “the adjoints are propagated backwards to compute the gradient”.

* **Memory Management:** Allocate large buffers in GPU memory (using `cudaMalloc` or unified memory) for: tape entries, AD value array, gradients, and input data. Pinned (page-locked) host memory can be used for fast host-to-device copies of real-time data. Use `cudaMemcpyAsync` with streams to overlap data transfers and computation (see below).

* **Host Orchestration:** On the host side, a loop should stream incoming data (market parameters or paths) into GPU memory, launch the forward kernel, then launch the backward kernel, and finally read out the computed sensitivities. Use double-buffering (see Sec. 5) to hide data transfer latency. The host code manages synchronization and error checking (e.g. using `cudaGetLastError()` after kernels as best practice).

*Example Code Snippet (Operator Overload)*:

```cpp
// Simplified AD value class
struct ADVal {
    double val;
    int idx;  // index of this variable in a global value array
    __device__ ADVal(double v=0.0) : val(v), idx(-1) {}
    __device__ ADVal operator+(const ADVal &o) const {
        ADVal r; r.val = val + o.val;
        int pos = tapePos.fetch_add(1);
        tape[pos] = {'+', idx, o.idx, r.idx, 1.0};  // store d(r)/d(x) = 1
        return r;
    }
    __device__ ADVal operator*(const ADVal &o) const {
        ADVal r; r.val = val * o.val;
        int pos = tapePos.fetch_add(1);
        tape[pos] = {'*', idx, o.idx, r.idx, o.val};  // d(x*y)/d(x) = y
        return r;
    }
    // ... similarly for -, /, sin, exp, etc.
};
```

This modular structure — separate tape, forward, and backward components — makes the code more maintainable. For large real-world models (e.g. LMM, Heston model), one can further subdivide the forward kernel into device functions and write corresponding adjoint functions for each, as recommended by Georges *et al.*.

## 4. GPU Optimization Strategies

Efficient GPU code requires attention to memory access patterns and parallel occupancy:

* **Memory Coalescing:** Arrange data so that consecutive threads access consecutive memory addresses. For example, if each thread works on one asset or path, store its data in arrays of structures or structure of arrays so that `threadIdx.x` reads contiguous elements. As the CUDA Best Practices Guide notes, “global memory loads and stores by threads of a warp are coalesced… into as few transactions as possible”. Uncoalesced (strided or random) access will severely limit bandwidth.

* **Use of Shared Memory:** For frequently reused data or reductions within a block, copy global memory into `__shared__` memory to avoid repeated global loads. This is especially useful for small lookup tables or block-level reductions. Note that shared memory has *banked* architecture: avoid bank conflicts by using padded indexing or by ensuring that warps access distinct banks. Aside from bank conflicts, shared memory access is very fast and can significantly accelerate inner-loop computations.

* **Occupancy Tuning:** Choose block and grid dimensions to maximize warp occupancy without exceeding register/shared limits. Occupancy is the ratio of active warps to the hardware limit; while “higher occupancy does not always equate to higher performance,” low occupancy (<30%) can prevent hiding latency. Use `nvcc`’s `-maxrregcount` or `__launch_bounds__` to adjust registers per thread if needed. Use NVIDIA’s Occupancy Calculator or Nsight Compute to check occupancy and resource usage. Aim for at least \~50–70% occupancy if possible, but focus first on algorithmic efficiency.

* **Minimize Divergence:** Write kernels to minimize branch divergence within warps. For example, if an option payoff has an `if (S>K)` condition, consider computing both paths and using predication or masking. Divergent branches cause serialization. Similarly, avoid very divergent code paths in the backward sweep; if needed, handle special cases (e.g. payoff = 0) outside the main loop or with separate kernels.

* **Memory Footprint:** Keep data in fast memory and registers when possible. For the tape, preallocate sufficiently large buffers (`cudaMallocManaged` often pre-reserves virtual space with demand paging) so you don’t need to reallocate each iteration. Free buffers only at shutdown. If the tape gets very large (many GB), one can consider “checkpointing”: recompute parts of the forward pass instead of storing all intermediate values (a trade-off between compute and memory).

* **Atomic Operations:** The tape implementation uses atomic increments (`atomicAdd`) to append operations. While slower than plain writes, their contention is usually low if each thread only writes a few entries. Gremse *et al.* successfully use `std::atomic` to serialize tape writes. To reduce contention, you can give each thread (or warp) a precomputed block of the tape so it can write without atomics (e.g. by doing a prefix-sum of operation counts first).

* **Library Functions:** Use Thrust for parallel operations (e.g. initializing data, reductions). Many AD implementations (including the one by Gremse *et al.*) use custom device vectors that work with Thrust’s algorithms. For example, you might call `thrust::transform` on an array of `AdjointDouble` values to apply a vector operation. But be mindful that standard library functions usually do not log to tape; you must intercept math calls (like `sin`, `exp`) by providing overloaded device implementations.

## 5. Real-Time Data Handling Strategy

For real-time or streaming data, you must continually feed market data (or simulated data) into the GPU and process it with minimal delay:

* **Data Streams:** Read or generate data in batches. If using real historical or live data, use a producer-consumer model: one thread (or process) ingests raw market ticks or time-series data and fills a host buffer, while another thread pushes that data to the GPU. For synthetic or simulation use-cases, you can generate random paths directly on the GPU (e.g. with cuRAND), eliminating host-GPU transfer. In either case, aim to process batches small enough for low latency (milliseconds) but large enough to keep the GPU busy.

* **Asynchronous Copies:** Use CUDA streams and `cudaMemcpyAsync` to overlap host-to-device transfer and GPU computation. For example:

  ```cpp
  // Allocate page-locked (pinned) host memory for faster DMA
  cudaHostAlloc((void**)&h_data, bytes, cudaHostAllocDefault);
  cudaMalloc((void**)&d_data, bytes);
  // Stream 0: copy batch N to GPU, stream 1: run kernel on previous batch
  cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream0);
  computeForward<<<grid,block,0,stream1>>>(d_data, ...);
  ```

  As CUDA docs explain, `cudaMemcpyAsync` is non-blocking and requires pinned memory. By alternating streams (double-buffering), you can hide latency: while one chunk is being transferred, the previous chunk is being processed. This is essential for *real-time* throughput.

* **Batching vs. Per-Event:** Decide on a trade-off between latency and GPU efficiency. Small batches give low latency but underutilize the GPU; large batches increase throughput but add delay. Profiling will help choose a good batch size. In finance, it’s common to accumulate, say, a few dozen or hundred paths before launching kernels.

* **Synthetic vs Historical Data:** For prototyping, you can generate synthetic market paths (e.g. Geometric Brownian Motion) on the fly using cuRAND in a GPU kernel. Alternatively, use historical data sets (e.g. time-series of asset prices) in CSV/HDF5 files and stream them. For streaming, tools like Python Kafka, RAPIDS cuDF, or plain file I/O can simulate real-time feeds; these feed into the C++ code via sockets or shared memory. The key is that the CUDA side only sees well-defined buffers arriving periodically.

* **GPU Buffers Management:** Keep persistent GPU buffers in a ring (queue) if using multiple asynchronous streams. For example, allocate two device buffers; while kernel `i` runs on buffer A, the host can copy new data into buffer B. Rotate buffers each iteration.

*Example (Pseudo-code) for Overlap:*

```cpp
cudaStream_t streamH, streamK;
cudaStreamCreate(&streamH);
cudaStreamCreate(&streamK);
while (running) {
    // Copy to GPU on streamH (asynchronous)
    cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, streamH);
    // Run forward kernel on streamK
    forwardKernel<<<grid, block, 0, streamK>>>(d_input, d_output);
    // After forward, optionally launch backward on same or new stream
    backwardKernel<<<..., streamK>>>( /*use same stream to enforce order*/ );
    // Meanwhile, CPU can prepare next h_input
}
```

By using streams and pinned memory, data transfer and compute occur in parallel, which is critical for real-time performance.

## 6. Benchmarking and Performance Evaluation

To assess performance, measure both **throughput** and **latency**. Key metrics include:

* **Throughput:** number of price calculations or scenarios processed per second. E.g., “million option evaluations per second”.
* **Latency:** time to complete one batch or one update from data arrival to derivative output.
* **Kernel Time:** time spent in forward vs backward kernels. Measure via `cudaEventRecord` around kernels or use Nsight.

Use the following approaches:

* **CUDA Events:** Insert `cudaEventCreate`, `cudaEventRecord` before and after kernels to time GPU execution. This gives precise kernel runtimes.

* **Profiling Tools:** NVIDIA recommends using **Nsight Systems** (for overall tracing) and **Nsight Compute** (kernel-level metrics). These tools report metrics like memory bandwidth, achieved occupancy, instruction utilization, and highlight bottlenecks. For example, Nsight Compute can show *Memory Load Efficiency*, helping you confirm if accesses are coalesced.

* **Performance Metrics:** Track achieved GFLOPS or memory GB/s compared to peak, check occupancy, and identify if code is memory-bound or compute-bound. For a mixed workload (exp, log, etc.), expect being limited by memory and special-function units.

* **Baseline Comparison:** Always compare to a reference: e.g. a CPU implementation (single-threaded or multi-threaded) and/or a finite-difference scheme. According to Giles & Glasserman and others, AAD on GPU can be orders-of-magnitude faster than naive methods. For instance, one study reported that GPU processing “achieves the highest performance, asymptotically” compared to CPU/vectorized code. Record the speedup over CPU (e.g. “5× faster than OpenMP CPU code, 50× faster than serial”) to quantify benefit.

* **Unit Workloads:** Test on simple problems first (e.g. computing derivatives of a Black–Scholes price) to validate correctness and profile. Then scale up to realistic workloads (Monte Carlo with many paths, or a full portfolio).

**Example:** If one forward+backward run on the GPU takes 5 ms for 1e5 paths, that’s 20 batch/s latency (50 ms/batch). If each batch has 1000 paths, that’s 100k/s throughput. You would compare this with how long a single CPU core takes to do the same (e.g., 500 ms).

For detailed profiling: set up a short run with Nsight Compute to capture memory metrics. Also use NVIDIA’s **roofline** model if applicable, to see if you are near theoretical limits.

## 7. Testing, Debugging, and Validation

Robust testing ensures correctness of the AAD implementation:

* **Unit Tests:** For basic arithmetic operations (`+`, `*`, `sin`, etc.), write tests that compare the AD gradient to known results. For example, test a small function \$f(x,y)=x\sin(y)\$: compute \$\partial f/\partial x\$ and \$\partial f/\partial y\$ via AAD and compare to analytic values. Use a C++ unit-test framework (Google Test, Catch2) to automate these checks.

* **Gradient Checking:** Implement a *finite-difference check* on a subset of variables. For a given input, slightly perturb one input and see how the output changes (finite difference). Compare \$(f(x+ε)-f(x-ε))/(2ε)\$ to the AAD-computed gradient; they should match to within small numerical error. This is a standard check for AD correctness.

* **Consistency with CPU:** Run the same forward computation on CPU (with double precision) and compare outputs to GPU results. Any discrepancy may indicate a bug in logic or precision issues.

* **CUDA Debugging:** Use `cuda-memcheck` (or compute sanitizers) to catch illegal memory accesses or race conditions in kernels. Also use Nsight’s `printf` debugging: you can instrument device code with `printf(...)` (though this serializes and slows kernels, it can help spot bad values).

* **Numerical Accuracy:** Finance models may be sensitive to floating-point. Test with both single and double precision. The tape gradients will accumulate many operations; make sure no catastrophic cancellation occurs. If needed, use double precision or Kahan summation for critical accumulations. Check stability by perturbing inputs at different scales.

* **Debugging Workflow:** If a CUDA kernel is incorrect, use CUDA-GDB or Nsight Compute’s debugging mode to inspect thread-local values. Check that each tape entry was recorded correctly (e.g. print out the first few tape entries on failure).

* **Visualization:** Optionally, use Python (Matplotlib) to plot results: e.g., plot the computed gradients vs. finite-difference approximations for a range of input values. Or plot kernel execution timelines from Nsight to visually inspect overlaps.

The key is *confidence*: each part (forward, tape, backward) should be validated on simple cases before assembling the full pipeline.

## 8. Deployment and Real-World Considerations

Once the prototype works, consider steps to harden and deploy it:

* **Build and Packaging:** Use CMake to manage the build system. Ensure all dependencies (CUDA, libraries) are correctly detected. For reproducibility, consider Docker or Singularity containers with the exact CUDA and driver environment.

* **Performance in Production:** In a real trading environment, integrate the AAD module into the risk system. This may involve interfacing with data sources (e.g. market data feeds), which might use protocols like FIX or custom APIs. Write a lightweight C++ API or shared library that other systems can call (passing input data and retrieving sensitivities).

* **Scalability:** If data volumes grow, think about multi-GPU or distributed computing. For instance, you could split Monte Carlo scenarios across multiple GPUs or nodes using MPI. NVIDIA’s *GPUDirect* and NVLink can help with inter-GPU communication if needed. If real-time latency is critical, use CPU cores for orchestration but let each GPU run independently on separate tasks.

* **Monitoring and Logging:** Add detailed logs (with timestamps) of throughput and latency. In production, integrate with monitoring tools (Prometheus, Grafana, etc.) to watch performance. Log any GPU errors or memory issues. Monitor GPU utilization (using `nvidia-smi` or NVML) to ensure the card is properly used.

* **Error Handling:** Check the return of every CUDA API call and kernel launch (`cudaGetLastError()`), and handle errors gracefully. For example, if the data feed lags, have a mechanism to skip or queue inputs without crashing.

* **Precision/Security:** Decide on numerical precision: if double precision is too slow, mixed precision might be possible (compute payoffs in FP32, adjoints in FP64, etc.), but validate carefully. Also ensure no sensitive data (market data) is mishandled if the code runs on shared infrastructure.

* **Real-Time Operation:** For strict latency requirements, tune batch size and avoid any file I/O on the critical path. You might offload logging or reporting to asynchronous threads so as not to stall the GPU computation.

In summary, a real-world prototype means not just fast code, but a maintainable, tested, and well-integrated solution. With careful design as outlined above — modular code, optimized GPU usage, streaming input, and rigorous testing — a GPU-accelerated AAD system can turn hours of batch risk calculations into sub-second real-time updates.







Great. I’ll prepare a modular project and file structure for your AAD-on-GPU implementation using CUDA/C++, targeting the Black-Scholes model with Catch2 tests, CMake and Docker setup, and Python-based visualization.

I'll let you know as soon as it's ready for you to start setting up your GitHub repository.


# Project Structure Overview

An HPC CUDA/C++ project should separate concerns into distinct folders (code, headers, tests, docs, etc.). A common layout is: `src/` for implementation, `include/` for public headers, `tests/` for unit tests, plus directories for build scripts, docs, and deployment. This modular organization makes the codebase easy to navigate and maintain. For example, most Unix-style projects use `src/` for source files and `include/` for headers, while modern CMake-based projects also use directories like `scripts/` or `docker/` for tools and containers. Below is a suggested Git repo layout:

* `src/`: **Core implementation code (C++/CUDA)**. This contains all source files. Example files:

  * `tape.cu` – Implements the AAD *tape* data structure that records intermediate values during the forward pass (so the adjoint can be computed in reverse).
  * `tape.h` – Declarations for the tape interface (push/pop operations, memory management).
  * `forward_kernel.cu` – CUDA kernel(s) performing the forward evaluation of the model (e.g. Monte Carlo simulation or vectorized Black–Scholes pricing) while logging operations to the tape.
  * `forward_kernel.h` – Host-side wrappers or declarations to launch forward CUDA kernels.
  * `reverse_sweep.cu` – CUDA kernel(s) for the reverse-mode sweep that backpropagates sensitivities using values from the tape.
  * `reverse_sweep.h` – Declarations for invoking the reverse sweep.
  * `black_scholes_model.cu` – Model-specific code implementing the Black–Scholes formulas (e.g. payoff and drift/diffusion updates). Used by the forward pass.
  * `black_scholes_model.h` – Header for Black–Scholes routines (option payoff, analytical prices, etc.).
  * `cuda_utils.cu` – CUDA utility functions (error-checking wrappers, device query, simple device-side math, etc.).
  * `cuda_utils.h` – Declarations for CUDA helpers (e.g. `checkCudaError`, safe kernel launch, etc.).
  * `memory_manager.cpp` / `.h` – GPU memory management utilities (RAII wrappers or functions for allocating/freeing/copying unified or pinned memory).
  * (Other utilities as needed, such as RNG wrappers, device math.)

* `include/`: **Public headers**. Any headers that define the library’s public interface (for example, `AADTape.h`, `BlackScholesModel.h`, or utility headers) go here. This keeps interface files separate from implementation. Downstream code can include headers from this directory.

* `tests/`: **Catch2 unit tests**. Organize test code here. Each test source is typically paired with the functionality it tests, e.g.:

  * `test_tape.cpp` – Tests for the tape (e.g., pushing/popping values and adjoints works correctly).
  * `test_forward.cpp` – Tests that the forward CUDA kernels produce correct Black–Scholes prices.
  * `test_reverse.cpp` – Tests that the reverse sweep produces correct gradients/Greeks.
  * Additional tests (e.g. `test_black_scholes.cpp` for model-specific logic).
    The `tests/` folder will have its own `CMakeLists.txt` enabling testing and linking against Catch2 and the core library.

* `cmake/`: **CMake modules and helper scripts (optional)**. Put any custom CMake modules or scripts here (for example, a `FindCatch2.cmake` if not using FetchContent, or other reusable CMake scripts). This is not always needed, but is useful if the build requires non-standard find routines.

* `scripts/` (or `python/`): **Python utilities and visualization**. For example, include:

  * `plot_results.py` – A Python script that reads output data (e.g. CSV or JSON of simulated prices/gradients) and generates plots using Matplotlib or Plotly.
  * `run_experiments.py` – Scripts to run simulations or parameter sweeps and collect results.
  * Any helper scripts (data converters, performance profiling tools, etc.).

* `docker/`: **Container files**. Put Docker-related configuration here:

  * `Dockerfile` – Builds an environment with NVIDIA CUDA (e.g. using an `nvidia/cuda` base image), installs dependencies (CMake, compilers, Catch2, Python libraries, etc.), and compiles the project.
  * `docker-compose.yml` (optional) – If multiple services or multi-stage builds are needed (for example, a builder image and a runner image, or to easily launch an interactive container). Use NVIDIA Docker / container toolkit to allow GPU access in the container.

* **Top-level files** (in project root):

  * `CMakeLists.txt` – The root CMake script that sets the project name (with `LANGUAGES CXX CUDA`), the C++/CUDA standards, and calls `add_subdirectory(src)` and `add_subdirectory(tests)`.
  * `README.md`, `LICENSE`, etc. – Standard docs.
  * `.gitignore` – Ignore build artifacts, binaries, etc.
  * (Optional) `CHANGELOG.md`, `docs/` for detailed documentation.

This layout follows common C/C++ conventions and cleanly separates each concern. For example, NVIDIA’s CUDA/CMake examples similarly place all CUDA/C++ sources in one library via `add_library`.

## CMake Build System

Use a “Modern CMake” approach with out-of-source builds. The **root `CMakeLists.txt`** should:

* Set a minimum CMake version and `project()` with `LANGUAGES CXX CUDA` to enable CUDA support.
* Configure global options (e.g. `set(CMAKE_CXX_STANDARD 17)`, CUDA architecture flags like `CMAKE_CUDA_ARCHITECTURES`).
* Use `find_package(CUDA REQUIRED)` or rely on CMake’s builtin CUDA support.
* Call `add_subdirectory(src)` and `add_subdirectory(tests)`.

In **`src/CMakeLists.txt`**, build the main library or executable:

* E.g. `add_library(aad_core STATIC tape.cu forward_kernel.cu reverse_sweep.cu black_scholes_model.cu cuda_utils.cu)`. Link any necessary CUDA libraries (e.g. curand, cufft if used).
* Use `target_compile_features(aad_core PUBLIC cxx_std_17)` to enforce C++17 (or C++14) compatibility.
* Enable separable compilation for device functions if needed: `set_target_properties(aad_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)`.
* If building an executable (e.g. `add_executable(run_simulation main.cpp)`), link it with the `aad_core` library.

In **`tests/CMakeLists.txt`**, integrate Catch2 and define test targets:

* Call `enable_testing()`.
* Use `FetchContent_Declare()` or `add_subdirectory()` to get Catch2 (for example, using `FetchContent` is common in Modern CMake).
* For each test source, e.g. `test_tape.cpp`, do `add_executable(test_tape test_tape.cpp)`, then `target_link_libraries(test_tape PRIVATE aad_core Catch2::Catch2)`.
* Optionally use `include(Catch)` to simplify, and `add_test(NAME tape_test COMMAND test_tape)` to register with CTest.

This keeps build logic modular: the root CMake orchestrates subdirectories, while each subdirectory’s CMakeLists focuses on its own targets. NVIDIA recommends explicitly listing CUDA sources in `add_library`/`add_executable` as shown above.

## Code Modularity and Best Practices

* **Group related code into modules**. For example, place all AAD-related code (tape, forward/reverse sweep kernels) in one module, Black–Scholes model code in another, and CUDA helpers in a utilities module. This “modular” approach (each module with its own sources and headers) makes it easy to find and test code. Each module can live in its own subdirectory under `src/` if the codebase grows large (e.g. `src/aad/`, `src/model/`, `src/utils/`).
* **Separate interface from implementation**. Put public class/function declarations in headers in `include/`, and definitions in `src/`. For example, `AADTape.h` in `include/` with its implementation in `tape.cu`. This aligns with standard conventions and allows users (or other modules) to include only the headers they need.
* **Keep files small and focused**. Each source file should implement a specific functionality (e.g. one CUDA kernel or one class). This improves readability and aids testing. If a file grows too large, split it into multiple files by functionality.
* **Use namespaces or classes** to avoid name collisions (e.g. an `aad::` namespace or a `BlackScholes` class). This is a good C++ practice in general.
* **Leverage CUDA best practices**. For example, minimize data transfers between host and device, use pinned or unified memory for performance, and wrap CUDA calls with error-checking (put in `cuda_utils.cu`). Group related device kernels in the same file or namespace.
* **Documentation and inline comments**. Even though not a folder, keep the code well-documented. Brief comments on non-obvious CUDA kernel parameters or AAD steps will help future maintainers.
* **Version control hygiene**: Keep the Git history clean by committing logically (see below) and use `.gitignore` to exclude build artifacts (`build/`, `*.o`, `*.exe`, etc.).

By following these practices, the codebase will be cleanly organized, easy to extend, and simpler to debug. As one blogger notes, splitting a project into directories by concern (headers, sources, tests, scripts) is a widely recommended convention, and grouping code into modules makes isolated testing straightforward.

## Visualization and Analysis Scripts

Place any data analysis or plotting scripts (in Python) in the `scripts/` directory. For example, after running the GPU executable to generate results (e.g. CSV of option prices or Greek sensitivities), a Python script like `scripts/plot_results.py` can load the data and produce charts. Keep these scripts lightweight and document their usage (e.g. read from `README`). Use Matplotlib or Plotly as needed. These scripts are *optional* helpers, so they need not be part of the CMake build; they can assume the user has Python installed.

## Docker/Container Configuration

Containerization ensures reproducibility. In `docker/`, include at least a `Dockerfile` that:

* Uses an NVIDIA CUDA base image (e.g. `nvidia/cuda:12.0-devel-ubuntu20.04`).
* Installs required tools: `build-essential`, `cmake`, any math libraries, Python with Matplotlib/Plotly, etc.
* Copies the project into the image and builds it (using CMake with an out-of-source `build/` directory).
* Sets the entrypoint or CMD as needed (or leave it to run interactively).

Optionally, a `docker-compose.yml` can simplify running the container with GPU access and volume mounts. For example, mount a host `data/` directory to the container so that results produced by the GPU program can be read by the host. Use the NVIDIA Docker runtime or the modern `--gpus` flag. Including instructions in `README.md` on how to build and run the Docker image (and how to use `docker-compose up`) will help users replicate the environment.

## Branch and Commit Guidelines

Use clear, descriptive Git workflows for collaboration and history. For example:

* **Branch naming**: Adopt a prefix strategy. Use prefixes like `feature/`, `bugfix/`, `hotfix/`, or `docs/` to indicate purpose. Keep names short, lowercase, and hyphen-separated (kebab-case). For instance, a new feature branch might be `feature/aad-black-scholes`, and a bug fix branch could be `bugfix/memory-leak`. This makes branches self-explanatory.
* **Commit messages**: Write messages in the imperative mood (e.g. “Add reverse sweep kernel”, not “Added…”), so that messages read like commands. You can optionally follow a [Conventional Commits](https://www.conventionalcommits.org/) style: prefix with a type such as `feat:`, `fix:`, `docs:`, e.g. `feat(backend): implement CUDA memory manager`. Keep the subject line under \~50 characters, and use the body of the commit to explain *why* the change was made if it’s not obvious. Examples:

  * `feature/aad-backprop` (branch), then commits like `feat(aad): implement tape push/pop operations`.
  * `bugfix/memcopy-bounds`, commit `fix(cuda): correct bounds in device memory copy`.

Following these conventions improves collaboration and project clarity. It also aids automated tools (e.g. generating changelogs or associating commits with issues).
