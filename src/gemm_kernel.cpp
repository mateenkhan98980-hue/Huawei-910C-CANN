#include "gemm_kernel.h"
#include <chrono>
#include <iostream>

namespace kernel {

GEMMKernel::GEMMKernel(uint32_t M, uint32_t N, uint32_t K)
    : M_(M), N_(N), K_(K), stream_(nullptr) {
    if (M == 0 || N == 0 || K == 0)
        throw std::invalid_argument("GEMM dimensions must be > 0");
    ASCENDCL_CHECK(aclrtCreateStream(&stream_));
    a_desc_ = std::make_unique<ascendcl::TensorDesc>(ACL_FLOAT16, std::vector<int64_t>{M, K});
    b_desc_ = std::make_unique<ascendcl::TensorDesc>(ACL_FLOAT16, std::vector<int64_t>{K, N});
    c_desc_ = std::make_unique<ascendcl::TensorDesc>(ACL_FLOAT, std::vector<int64_t>{M, N});
    d_desc_ = std::make_unique<ascendcl::TensorDesc>(ACL_FLOAT, std::vector<int64_t>{M, N});
    std::cout << "[GEMM] Created kernel for " << M << "x" << N << "x" << K << std::endl;
}

GEMMKernel::~GEMMKernel() {
    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
}

aclError GEMMKernel::execute(void* a_data, void* b_data, void* c_data, void* d_data) {
    if (!a_data || !b_data || !c_data || !d_data)
        return ACL_ERROR_INVALID_PARAM;
    if (!stream_) return ACL_ERROR_INVALID_PARAM;

    auto start = std::chrono::high_resolution_clock::now();

    try {
        if (d_data != c_data) {
            ASCENDCL_CHECK(aclrtMemcpy(d_data, M_ * N_ * sizeof(float), c_data,
                                       M_ * N_ * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE));
        }
        auto matmul_op = std::make_shared<aclnn::MatMulOp>(a_desc_.get(), a_data,
                                                           b_desc_.get(), b_data,
                                                           d_desc_.get(), d_data, 1.0f, 1.0f);
        aclnn::GraphExecutor executor;
        executor.addOperator(matmul_op);
        executor.execute(stream_);

        ASCENDCL_CHECK(aclrtSynchronizeStream(stream_));

        auto end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double total_flops = 2.0 * M_ * N_ * K_;
        metrics_.total_execution_time_ms = duration_ms;
        metrics_.achieved_tflops = (total_flops / 1e12) / (duration_ms / 1000.0);
        metrics_.compute_utilization_pct = (metrics_.achieved_tflops / 256.0) * 100.0;
        std::cout << "[GEMM] Executed in " << duration_ms << " ms, " << metrics_.achieved_tflops << " TFLOPS" << std::endl;
        return ACL_SUCCESS;
    } catch (const ascendcl::AscendException& e) {
        std::cerr << "[ERROR] GEMM execution failed: " << e.what() << std::endl;
        return e.getErrorCode();
    }
}

} // namespace kernel