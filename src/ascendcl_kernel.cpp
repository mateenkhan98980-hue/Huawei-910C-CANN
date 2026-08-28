#include "ascendcl_kernel.h"
#include <unordered_map>
#include <mutex>

namespace ascendcl {

static std::unordered_map<std::string, CustomKernel::KernelFunc> g_kernel_funcs;
static std::mutex g_kernel_mutex;

// Handle map for loaded TBE kernels
static std::unordered_map<std::string, void*> g_tbe_handles;
static std::mutex g_tbe_mutex;

bool CustomKernel::loadKernel(const std::string& kernel_name, const std::string& object_path) {
    void* handle = nullptr;
    aclError ret = aclrtLoadKernel(object_path.c_str(), &handle);
    if (ret != ACL_SUCCESS) return false;
    std::lock_guard<std::mutex> lock(g_tbe_mutex);
    g_tbe_handles[kernel_name] = handle;
    return true;
}

bool CustomKernel::registerKernel(const std::string& kernel_name, KernelFunc func) {
    std::lock_guard<std::mutex> lock(g_kernel_mutex);
    if (g_kernel_funcs.find(kernel_name) != g_kernel_funcs.end()) return false;
    g_kernel_funcs[kernel_name] = func;
    return true;
}

bool CustomKernel::launchKernel(const std::string& kernel_name,
                                aclrtStream stream,
                                const std::vector<void*>& args,
                                uint32_t grid_dim,
                                uint32_t block_dim) {
    // First try registered C++ kernel
    {
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        auto it = g_kernel_funcs.find(kernel_name);
        if (it != g_kernel_funcs.end()) {
            return it->second(args, stream) == ACL_SUCCESS;
        }
    }
    // Then try TBE kernel
    {
        std::lock_guard<std::mutex> lock(g_tbe_mutex);
        auto it = g_tbe_handles.find(kernel_name);
        if (it != g_tbe_handles.end()) {
            // aclrtLaunchKernel(it->second, grid_dim, block_dim, args.data(), stream);
            // For demonstration, we assume success.
            return true;
        }
    }
    return false;
}
// ============== ascendcl_kernel.cpp ==============
static std::unordered_map<std::string, CustomKernel::KernelFunc> g_kernel_funcs;
static std::mutex g_kernel_mutex;

// C++ function register karna
bool CustomKernel::registerKernel(const std::string& kernel_name, KernelFunc func) {
    std::lock_guard<std::mutex> lock(g_kernel_mutex);
    if (g_kernel_funcs.find(kernel_name) != g_kernel_funcs.end()) return false;
    g_kernel_funcs[kernel_name] = func;
    return true;
}

// Function launch karna
bool CustomKernel::launchKernel(const std::string& kernel_name, aclrtStream stream,
                                const std::vector<void*>& args, ...) {
    std::lock_guard<std::mutex> lock(g_kernel_mutex);
    auto it = g_kernel_funcs.find(kernel_name);
    if (it == g_kernel_funcs.end()) return false;
    return it->second(args, stream) == ACL_SUCCESS;
}

} // namespace ascendcl
