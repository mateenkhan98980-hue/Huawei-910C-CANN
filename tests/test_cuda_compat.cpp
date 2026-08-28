// ============================================================================
// CUDA-compatible wrapper smoke tests
// ============================================================================

#include "ascendcl_cuda_compat.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "\n========== CUDA COMPAT WRAPPER TESTS ==========" << std::endl;

    try {
        ascendcl::RuntimeContext::getInstance().initialize(0);

        if (ascendcl::Device::getDeviceCount() == 0) {
            std::cerr << "[FATAL] No Ascend devices detected." << std::endl;
            return 1;
        }

        ascendcl::Device device(0);
        device.setActive();

        auto stream = std::make_unique<ascendcl::Stream>();
        auto memory = std::make_unique<ascendcl::DeviceMemory>(4096, true);

        std::vector<float> host_data(1024, 3.14f);
        ascendcl::MemcpyHtoD(memory->getData(), host_data.data(), host_data.size() * sizeof(float), stream.get());

        std::vector<float> read_back(1024, 0.0f);
        ascendcl::MemcpyDtoH(read_back.data(), memory->getData(), read_back.size() * sizeof(float), stream.get());
        stream->synchronize();

        ascendcl::Event start;
        ascendcl::Event end;
        start.record(stream.get());
        end.record(stream.get());
        end.synchronize();
        const float elapsed = ascendcl::Event::elapsedTime(start, end);

        const auto ops = ascendcl::OperatorRegistry::getInstance().listOps();
        std::cout << "Registered operators: " << ops.size() << std::endl;
        for (const auto& op : ops) {
            std::cout << "  - " << op << std::endl;
        }

        std::vector<void*> kernel_args;
        ascendcl::KernelLauncher::launchKernel(
            stream.get(), 1, 1, 0, "ascend::noop", kernel_args);

        ascendcl::DistributedContext::getInstance().initDistributed(0, 1);
        ascendcl::DistributedContext::getInstance().allReduce(
            memory->getData(), memory->getData(), 256, 0);
        ascendcl::DistributedContext::getInstance().finalize();

        std::cout << "Event elapsed (ms): " << elapsed << std::endl;
        std::cout << ascendcl::RuntimeContext::getInstance().getStatistics() << std::endl;
        std::cout << "✓ CUDA compat smoke test passed" << std::endl;

        ascendcl::RuntimeContext::getInstance().finalize();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << std::endl;
        return 1;
    }
}
