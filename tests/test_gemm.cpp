// ============================================================================
// GEMM Kernel Test Suite - CANN 8.0
// Validates numerical correctness against CPU reference implementation
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <memory>

#include "ascendcl_wrapper.h"
#include "aclnn_wrapper.h"
#include "acl_compiler.h"
#include "gemm_kernel.h"
#include "ascend_util.h"

namespace {

float randomFloat() {
    return static_cast<float>((rand() % 1000) - 500) / 100.0f;
}

bool compareMatrices(
    const std::vector<float>& computed,
    const std::vector<float>& reference,
    float tolerance = 5e-2f) {
    if (computed.size() != reference.size()) {
        std::cerr << "Size mismatch: computed=" << computed.size()
                  << " reference=" << reference.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < computed.size(); ++i) {
        const float ref_val = reference[i];
        const float diff = std::abs(computed[i] - ref_val);
        const float allowed = std::max(tolerance, tolerance * std::abs(ref_val));
        if (diff > allowed) {
            std::cerr << "Mismatch at index " << i << ": computed=" << computed[i]
                      << " reference=" << ref_val << " diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

bool testBasicGEMM() {
    std::cout << "\n========== TEST: Basic GEMM (16x16x16) ==========" << std::endl;

    const uint32_t M = 16;
    const uint32_t N = 16;
    const uint32_t K = 16;
    const size_t a_size = static_cast<size_t>(M) * K * sizeof(uint16_t);
    const size_t b_size = static_cast<size_t>(K) * N * sizeof(uint16_t);
    const size_t c_size = static_cast<size_t>(M) * N * sizeof(float);

    auto a_mem = ascendcl::allocateMemory(a_size, true);
    auto b_mem = ascendcl::allocateMemory(b_size, true);
    auto c_mem = ascendcl::allocateMemory(c_size, true);
    auto d_mem = ascendcl::allocateMemory(c_size, true);

    std::vector<uint16_t> a_fp16(static_cast<size_t>(M) * K);
    std::vector<uint16_t> b_fp16(static_cast<size_t>(K) * N);
    std::vector<float> c_host(static_cast<size_t>(M) * N);

    for (size_t i = 0; i < a_fp16.size(); ++i) {
        a_fp16[i] = ascend::floatToFp16(randomFloat());
    }
    for (size_t i = 0; i < b_fp16.size(); ++i) {
        b_fp16[i] = ascend::floatToFp16(randomFloat());
    }
    for (size_t i = 0; i < c_host.size(); ++i) {
        c_host[i] = randomFloat();
    }

    std::vector<float> expected;
    ascend::cpuGemmFp16AccFp32(a_fp16, b_fp16, c_host, expected, M, N, K);

    a_mem->copyFromHost(a_fp16.data(), a_size);
    b_mem->copyFromHost(b_fp16.data(), b_size);
    c_mem->copyFromHost(c_host.data(), c_size);

    auto gemm = std::make_unique<kernel::GEMMKernel>(M, N, K);
    const aclError ret = gemm->execute(
        a_mem->getData(),
        b_mem->getData(),
        c_mem->getData(),
        d_mem->getData());

    if (ret != ACL_SUCCESS) {
        std::cout << "✗ FAILED (error code: " << ret << ")" << std::endl;
        return false;
    }

    std::vector<float> result(static_cast<size_t>(M) * N);
    d_mem->copyToHost(result.data(), c_size);

    if (!compareMatrices(result, expected)) {
        std::cout << "✗ FAILED (numerical mismatch)" << std::endl;
        return false;
    }

    const auto& metrics = gemm->getMetrics();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Execution time: " << metrics.total_execution_time_ms << " ms" << std::endl;
    std::cout << "  Achieved TFLOPS: " << metrics.achieved_tflops << std::endl;
    std::cout << "✓ PASSED" << std::endl;
    return true;
}

bool testLargeGEMM() {
    std::cout << "\n========== TEST: Large GEMM (512x512x512) ==========" << std::endl;

    const uint32_t M = 512;
    const uint32_t N = 512;
    const uint32_t K = 512;
    const size_t a_size = static_cast<size_t>(M) * K * sizeof(uint16_t);
    const size_t b_size = static_cast<size_t>(K) * N * sizeof(uint16_t);
    const size_t c_size = static_cast<size_t>(M) * N * sizeof(float);

    auto a_mem = ascendcl::allocateMemory(a_size, true);
    auto b_mem = ascendcl::allocateMemory(b_size, true);
    auto c_mem = ascendcl::allocateMemory(c_size, true);
    auto d_mem = ascendcl::allocateMemory(c_size, true);

    std::vector<uint16_t> a_fp16(static_cast<size_t>(M) * K, ascend::floatToFp16(0.01f));
    std::vector<uint16_t> b_fp16(static_cast<size_t>(K) * N, ascend::floatToFp16(0.01f));
    std::vector<float> c_host(static_cast<size_t>(M) * N, 1.0f);

    std::vector<float> expected;
    ascend::cpuGemmFp16AccFp32(a_fp16, b_fp16, c_host, expected, M, N, K);

    a_mem->copyFromHost(a_fp16.data(), a_size);
    b_mem->copyFromHost(b_fp16.data(), b_size);
    c_mem->copyFromHost(c_host.data(), c_size);

    auto gemm = std::make_unique<kernel::GEMMKernel>(M, N, K);
    const aclError ret = gemm->execute(
        a_mem->getData(),
        b_mem->getData(),
        c_mem->getData(),
        d_mem->getData());

    if (ret != ACL_SUCCESS) {
        std::cout << "✗ FAILED" << std::endl;
        return false;
    }

    std::vector<float> result(static_cast<size_t>(M) * N);
    d_mem->copyToHost(result.data(), c_size);

    if (!compareMatrices(result, expected, 0.2f)) {
        std::cout << "✗ FAILED (numerical mismatch)" << std::endl;
        return false;
    }

    const auto& metrics = gemm->getMetrics();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Achieved TFLOPS: " << metrics.achieved_tflops << std::endl;
    std::cout << "✓ PASSED" << std::endl;
    return true;
}

bool testGraphCompilation() {
    std::cout << "\n========== TEST: Graph Compilation ==========" << std::endl;

    auto builder = std::make_unique<acl_compiler::GraphBuilder>();
    auto graph = builder->createGraph("test_graph");
    if (!graph) {
        std::cout << "✗ FAILED (graph creation)" << std::endl;
        return false;
    }

    builder->compileAll();
    std::cout << "✓ PASSED" << std::endl;
    return true;
}

} // namespace

int main() {
    std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   ASCEND CANN 8.0 Wrapper Test Suite       ║" << std::endl;
    std::cout << "║   Target: Ascend 910C + libascend_hal      ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    int failures = 0;

    try {
        ascendcl::initializeRuntime();
        ascendcl::Device device(0);
        device.setActive();

        if (ascendcl::Device::getDeviceCount() == 0) {
            std::cerr << "[FATAL] No Ascend devices detected. Check driver (.ko) and npu-smi." << std::endl;
            return 1;
        }

        if (!testBasicGEMM()) {
            ++failures;
        }
        if (!testLargeGEMM()) {
            ++failures;
        }
        if (!testGraphCompilation()) {
            ++failures;
        }

        ascendcl::finalizeRuntime();
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] Test suite failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n========== TEST SUMMARY ==========" << std::endl;
    if (failures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    }

    std::cout << failures << " TEST(S) FAILED" << std::endl;
    return 1;
}
