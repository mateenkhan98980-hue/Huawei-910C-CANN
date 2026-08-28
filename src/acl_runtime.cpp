// ============================================================================
// AscendCL Runtime Support - CANN 8.0
// ============================================================================

#include "ascendcl_wrapper.h"
#include <iostream>

namespace ascendcl {

// Global runtime initialization flag
bool g_runtime_initialized = false;

void initializeRuntime() {
    if (g_runtime_initialized) return;
    
    try {
        Context::getInstance().init();
        g_runtime_initialized = true;
        std::cout << "[Runtime] AscendCL runtime initialized" << std::endl;
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to initialize runtime: " << e.what() << std::endl;
        throw;
    }
}

void finalizeRuntime() {
    if (!g_runtime_initialized) return;
    
    try {
        Context::getInstance().finalize();
        g_runtime_initialized = false;
        std::cout << "[Runtime] AscendCL runtime finalized" << std::endl;
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to finalize runtime" << std::endl;
    }
}

} // namespace ascendcl