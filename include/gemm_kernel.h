// ============================================================================
// Huawei Ascend GEMM Compute Kernel (D = A * B + C)
// Strict CUDA-compatible GEMM kernel
// ============================================================================

#pragma once

#include "ascendcl_wrapper.h"
#include "aclnn_wrapper.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace kernel {

struct PerformanceMetrics {
    double total_execution_time_ms;
    double achieved_tflops;
    double compute_utilization_pct;
};

class GEMMKernel {
public:
    GEMMKernel(uint32_t M, uint32_t N, uint32_t K);
    ~GEMMKernel();
    aclError execute(void* a_data, void* b_data, void* c_data, void* d_data);
    const PerformanceMetrics& getMetrics() const { return metrics_; }
private:
    uint32_t M_, N_, K_;
    PerformanceMetrics metrics_;
    aclrtStream stream_;
    std::unique_ptr<ascendcl::TensorDesc> a_desc_, b_desc_, c_desc_, d_desc_;
};

} // namespace kernel