# Ascend CANN Wrapper — Architecture & Concept

## What this project is

This repository is a **user-space software stack** for Huawei Ascend 910 / 910C. It does **not** ship firmware or kernel modules (`.ko`). Those come from the official **Ascend Driver** package installed on the host.

```
Application (PyTorch, custom C++, tests)
        │
        ▼
┌───────────────────────────────────────────┐
│  This repo (wrappers)                     │
│  • libascendcl.so      — runtime (ACL)    │
│  • libaclnn_ops.so     — MatMul, GEMM     │
│  • libacl_op_compiler.so — graph builder  │
│  • libascendcl_cuda_compat.so (optional)   │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  Official CANN 8.0 toolkit (Huawei)       │
│  libacl.so, libaclnn.so, libopapi.so …    │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  Ascend driver (NOT in this repo)         │
│  libascend_hal.so + kernel modules (.ko)  │
└───────────────────────────────────────────┘
        │
        ▼
   Ascend 910C NPU hardware
```

## Layer responsibilities

| Layer | Role | Production criteria |
|-------|------|---------------------|
| **Runtime (`ascendcl`)** | `aclInit`, device, stream, HBM alloc | Real ACL calls, RAII cleanup |
| **Operators (`aclnn_ops`)** | `aclopMatMul`, BiasAdd, ReLU, GEMM | Executes on device via CANN |
| **Compiler (`acl_op_compiler`)** | Ordered operator graphs + fusion pass | Validates and runs operator lists |
| **CUDA compat (optional)** | CUDA-like API for framework ports | Memory/stream/events + registered kernels |

## Firmware / driver boundary

- **In scope:** Calling into CANN and HAL once the driver is installed.
- **Out of scope:** Building or updating `.ko` files, NPU microcode, or board firmware.
- **Verify driver on Linux:** `npu-smi info`, `lsmod | grep ascend`, `ls $ASCEND_DRIVER/lib64/libascend_hal.so`.

## Build outputs

After `./build.sh`:

| Artifact | Purpose |
|----------|---------|
| `libascendcl.so` | Core runtime wrapper |
| `libaclnn_ops.so` | Operators + GEMM kernel |
| `libacl_op_compiler.so` | Graph compilation |
| `libascendcl_cuda_compat.so` | Optional CUDA-like layer |
| `test_gemm` | Numerical GEMM validation on NPU |
| `test_cuda_compat` | Smoke test for CUDA-compat API |

## Distributed training (HCCL)

- **Single NPU (`world_size=1`):** Collectives degenerate to local device copies — fully supported without HCCL setup.
- **Multi-process:** Requires standard CANN/HCCL cluster initialization (root info / launcher). Set `ASCEND_HCCL_ROOT_INFO_PATH` when using `world_size > 1`.

## Validation workflow

```bash
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_DRIVER=/usr/local/Ascend/driver
./build.sh
cd build && ctest -V
```

`test_gemm` compares NPU results against a CPU reference implementation (FP16 multiply, FP32 accumulate).
