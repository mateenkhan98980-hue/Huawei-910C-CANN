# Ascend CANN 8.0 Runtime & Operator Wrapper Libraries

Production-ready **user-space** wrapper layer that attaches to official Huawei CANN 8.0 libraries and `libascend_hal.so` (from the Ascend **driver** package — firmware/`.ko` are **not** built from this repo).

**Architecture & concept:** see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│   User Application (test_gemm.cpp)                   │
├──────────────────────────────────────────────────────┤
│   Public API Layer                                   │
│  ├─ ascendcl_wrapper.h (Runtime)                     │
│  ├─ aclnn_wrapper.h (Operators)                      │
│  ├─ acl_compiler.h (Graph)                           │
│  └─ gemm_kernel.h (GEMM Kernel)                      │
├──────────────────────────────────────────────────────┤
│   Implementation Layer (src/)                        │
│  ├─ libascendcl.so (Runtime)                         │
│  ├─ libaclnn_ops.so (Operators)                      │
│  └─ libacl_op_compiler.so (Compiler)                 │
├──────────────────────────────────────────────────────┤
│   Official CANN 8.0 Libraries                        │
│  ├─ libacl.so                                        │
│  ├─ libaclnn.so                                      │
│  └─ libascend_hal.so (Hardware Driver)               │
├──────────────────────────────────────────────────────┤
│   Huawei Ascend 910C Hardware                        │
│  ├─ 32 AI Cores                                      │
│  ├─ Cube Units (256 TFLOPS peak)                     │
│  └─ 1.2 TB/s HBM Bandwidth                           │
└──────────────────────────────────────────────────────┘
```

## File Structure

```
.
├── include/                      # Header files
│   ├── ascendcl_wrapper.h       # Runtime management (Device, Stream, Memory, Context)
│   ├── aclnn_wrapper.h          # Operator interface (MatMul, BiasAdd, ReLU)
│   ├── acl_compiler.h           # Graph compilation and optimization
│   └── gemm_kernel.h            # GEMM kernel interface
├── src/                          # Implementation files
│   ├── libascendcl.cpp          # Runtime wrapper implementation
│   ├── acl_runtime.cpp          # Runtime initialization
│   ├── acl_memory.cpp           # Memory management
│   ├── acl_device.cpp           # Device management
│   ├── acl_stream.cpp           # Stream management
│   ├── libaclnn_ops.cpp         # Operator implementations
│   ├── gemm_kernel.cpp          # GEMM kernel implementation
│   ├── libacl_op_compiler.cpp   # Compiler implementation
│   └── acl_graph.cpp            # Graph implementation
├── tests/
│   ├── test_gemm.cpp            # GEMM + graph tests (requires NPU)
│   └── test_cuda_compat.cpp     # CUDA-compat smoke test
├── docs/
│   └── ARCHITECTURE.md          # Concept: layers, driver boundary, validation
├── CMakeLists.txt               # CMake build configuration
├── build.sh                     # Build script
└── README.md                    # This file
```

## Building

### Prerequisites

1. **Ascend Toolkit 8.0** installed at `/usr/local/Ascend/ascend-toolkit/latest` or `/usr/local/Ascend`
2. **Ascend Driver** installed at `/usr/local/Ascend/driver`
3. **CMake 3.10+**
4. **GCC 7.0+** or **Clang 5.0+**
5. **Build tools**: `make`, `git`

### Quick Build

```bash
# Set environment variables (if not in default locations)
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_DRIVER=/usr/local/Ascend/driver

# Build
chmod +x build.sh
./build.sh
```

### Manual Build

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_HOME=/usr/local/Ascend \
    -DASCEND_DRIVER=/usr/local/Ascend/driver

make -j$(nproc)
make install
```

## Running Tests

```bash
cd build

# Run individual test
./test_gemm

# Or run via CMake
ctest -V
```

## Key Features

### Production stack (this repo)

| Component | Status |
|-----------|--------|
| Full CMake build (3 libs + optional CUDA compat + tests) | ✓ |
| GEMM numerical tests vs CPU reference | ✓ |
| CUDA-compat kernel registry (no silent placeholders) | ✓ |
| Single-NPU HCCL collectives (`world_size=1`) | ✓ |
| Multi-process HCCL | Requires `ASCEND_HCCL_ROOT_INFO_PATH` |

### ✓ Runtime quality
- Thread-safe singleton patterns
- Comprehensive error handling with CHECK_ACL macros
- Memory safety checks (null pointer validation)
- Resource cleanup on exception
- Detailed logging for debugging

### ✓ Performance Optimized
- Direct CANN library calls (zero overhead wrapper)
- Async DMA and stream management
- Tiling strategy for Cube Core utilization
- Peak 256 TFLOPS on Ascend 910

### ✓ Standard Compliance
- C++17 standard
- Modern RAII patterns
- Exception-safe code
- STL containers for memory safety

### ✓ Hardware Support
- Ascend 910 / 910C chips
- 32 AI Cores per chip
- 256 TFLOPS theoretical peak
- 1.2 TB/s HBM bandwidth

## API Overview

### Runtime Management

```cpp
// Initialize CANN runtime
ascendcl::initializeRuntime();

// Create device and set active
ascendcl::Device device(0);
device.setActive();

// Create stream for async operations
auto stream = ascendcl::createStream();

// Allocate device memory
auto memory = ascendcl::allocateMemory(size_in_bytes, huge_page=true);
memory->copyFromHost(host_ptr, size);
memory->copyToHost(host_ptr, size);

// Finalize runtime
ascendcl::finalizeRuntime();
```

### GEMM Kernel

```cpp
// Create kernel for M x N x K GEMM
auto gemm = std::make_unique<kernel::GEMMKernel>(M, N, K);

// Execute: D = A * B + C
aclError ret = gemm->execute(a_data, b_data, c_data, d_data);

// Get performance metrics
const auto& metrics = gemm->getMetrics();
std::cout << "Achieved: " << metrics.achieved_tflops << " TFLOPS" << std::endl;
```

## Troubleshooting

### Build Errors

**Error: `CANN headers not found`**
- Verify ASCEND_HOME environment variable
- Check CANN toolkit installation: `ls $ASCEND_HOME/ascend-toolkit/latest/include/acl/acl.h`

**Error: `libascend_hal.so not found`**
- Verify ASCEND_DRIVER environment variable
- Check driver installation: `ls $ASCEND_DRIVER/lib64/libascend_hal.so`

**Error: `CMake configuration failed`**
- Clean build: `rm -rf build && mkdir build`
- Verify CMake version: `cmake --version` (need 3.10+)

### Runtime Errors

**Error: `Failed to initialize AscendCL context`**
- Verify device is detected: `npu-smi info` (Huawei tool)
- Check driver is loaded: `lsmod | grep ascend`

**Error: `Device not found`**
- Ensure at least one Ascend device is attached
- Check device ID is valid: `npu-smi info -l`

## Performance Tuning

### Memory Allocation
- Use huge pages for better performance: `allocateMemory(size, true)`
- Align data to 32-byte boundaries for DMA efficiency
- Pre-allocate memory pools to reduce allocation overhead

### Kernel Execution
- Overlap DMA transfers with computation using async streams
- Tile matrices to fit in Unified Buffer (32 KB per core)
- Use matrix sizes that are multiples of 16 (Cube Core native size)

### Example Optimization
```cpp
// Good: 512x512x512 (multiples of 16)
kernel::GEMMKernel gemm(512, 512, 512);

// Better: 1024x1024x1024 (better utilization)
kernel::GEMMKernel gemm(1024, 1024, 1024);
```

## License

Production-ready implementation for Huawei Ascend CANN 8.0.

## Support

For issues or questions:
1. Check Huawei CANN documentation: https://www.hiascend.com/
2. Review error messages in stdout/stderr (detailed logging)
3. Verify hardware is functional with Huawei tools
4. Check CMakeLists.txt CANN library paths
