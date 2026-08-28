#pragma once

#include "ascendcl_cuda_compat.h"
#include <string>
#include <vector>
#include <functional>

namespace ascendcl {

class CustomKernel {
public:
    // Load a pre‑compiled TBE kernel from .o file
    static bool loadKernel(const std::string& kernel_name, const std::string& object_path);
    
    // Launch a loaded kernel with arguments
    static bool launchKernel(const std::string& kernel_name,
                             aclrtStream stream,
                             const std::vector<void*>& args,
                             uint32_t grid_dim = 1,
                             uint32_t block_dim = 1);
    
    // Register a user‑defined C++ function as a kernel (like CUDA __global__)
    using KernelFunc = std::function<aclError(const std::vector<void*>&, aclrtStream)>;
    static bool registerKernel(const std::string& kernel_name, KernelFunc func);
};

} // namespace ascendcl