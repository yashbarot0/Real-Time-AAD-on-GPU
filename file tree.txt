aad-gpu-black-scholes/
│
├── CMakeLists.txt                  # Root CMake configuration
├── README.md                       # Project description and setup guide
├── .gitignore                      # Ignore build, logs, binaries
├── LICENSE                         # (Optional) Open source license
│
├── include/                        # Public headers
│   ├── aad/
│   │   ├── tape.h                  # AAD tape class declarations
│   │   ├── forward_kernel.h        # Forward CUDA kernel interface
│   │   └── reverse_sweep.h         # Reverse sweep kernel interface
│   ├── model/
│   │   └── black_scholes_model.h   # Black-Scholes specific logic
│   └── utils/
│       └── cuda_utils.h            # CUDA helpers and error handling
│
├── src/                            # Source and CUDA files
│   ├── aad/
│   │   ├── tape.cu                 # Tape structure and logic
│   │   ├── forward_kernel.cu       # Forward pass CUDA kernel
│   │   └── reverse_sweep.cu        # Reverse pass CUDA kernel
│   ├── model/
│   │   └── black_scholes_model.cu  # Black-Scholes implementation
│   └── utils/
│       └── cuda_utils.cu           # CUDA utilities and memory mgmt
│
├── tests/                          # Catch2 unit tests
│   ├── CMakeLists.txt              # Adds tests to CMake
│   ├── test_tape.cpp               # Test AAD tape structure
│   ├── test_forward.cpp            # Test forward kernel
│   └── test_reverse.cpp            # Test reverse sweep
│
├── scripts/                        # Python visualization & tools
│   ├── plot_results.py             # Matplotlib/Plotly plots of output
│   └── run_simulation.py           # Example runner / automation script
│
├── cmake/                          # Extra CMake modules (optional)
│   └── FindCatch2.cmake            # Find Catch2 if not using FetchContent
│
├── docker/                         # Docker configuration
│   ├── Dockerfile                  # Docker image for GPU build + run
│   └── docker-compose.yml          # (Optional) GPU runtime config
│
└── data/                           # (Optional) simulation input/output
    ├── results.csv                 # Output from simulation (read by plot)
    └── config.json                 # (Optional) input configuration
