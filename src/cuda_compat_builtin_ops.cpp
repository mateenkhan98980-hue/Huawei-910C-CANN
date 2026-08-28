// ============================================================================
// Built-in CUDA-compat kernels and operators (registered at library load)
// ============================================================================

#include "ascendcl_cuda_compat.h"
#include <cstring>

namespace ascendcl {
namespace {

aclError deviceMemcpyKernel(const std::vector<void*>& args, aclrtStream stream) {
    if (args.size() < 3) {
        return ACL_ERROR_INVALID_PARAM;
    }

    void* dst = args[0];
    const void* src = args[1];
    size_t size = 0;
    std::memcpy(&size, &args[2], sizeof(size));

    if (!dst || !src || size == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }

    return aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
}

aclError noopKernel(const std::vector<void*>& args, aclrtStream stream) {
    (void)args;
    (void)stream;
    return ACL_SUCCESS;
}

struct BuiltinRegistration {
    BuiltinRegistration() {
        auto& registry = OperatorRegistry::getInstance();
        registry.registerSimpleKernel("ascend::noop", noopKernel);
        registry.registerSimpleKernel("ascend::d2d_memcpy", deviceMemcpyKernel);
        registry.registerOp(
            "ascend::noop_op",
            [](const std::vector<void*>&,
               const std::vector<void*>&,
               const std::vector<aclTensorDesc*>&,
               const std::vector<aclTensorDesc*>&,
               aclrtStream stream) -> aclError {
                (void)stream;
                return ACL_SUCCESS;
            });
    }
};

const BuiltinRegistration g_builtin_registration;

} // namespace
} // namespace ascendcl
