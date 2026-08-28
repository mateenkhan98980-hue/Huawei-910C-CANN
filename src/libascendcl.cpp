// ============================================================================
// AscendCL Runtime Wrapper Implementation - CANN 8.0
// Attaches directly to libascend_hal.so and official CANN runtime
// ============================================================================

#include "ascendcl_wrapper.h"
#include <iostream>
#include <cstring>

namespace ascendcl {

// ============================================================================
// CONTEXT SINGLETON
// ============================================================================

Context& Context::getInstance() {
    static Context instance;
    return instance;
}

void Context::init() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) return;
    
    try {
        CHECK_ACL(aclInit(nullptr));
        initialized_ = true;
        std::cout << "[AscendCL] Context initialized with CANN 8.0" << std::endl;
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to initialize AscendCL context: " 
                  << e.what() << std::endl;
        throw;
    }
}

void Context::finalize() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) return;
    
    try {
        CHECK_ACL(aclFinalize());
        initialized_ = false;
        std::cout << "[AscendCL] Context finalized" << std::endl;
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to finalize AscendCL context" << std::endl;
    }
}

// ============================================================================
// DEVICE MANAGEMENT
// ============================================================================

Device::Device(int device_id)
    : device_id_(device_id), is_active_(false) {
    
    try {
        CHECK_ACL(aclrtSetDevice(device_id));
        is_active_ = true;
        std::cout << "[Device] Set device " << device_id << " as active" << std::endl;
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to set device " << device_id << std::endl;
        throw;
    }
}

Device::~Device() {
    if (is_active_) {
        try {
            aclrtResetDevice(device_id_);
        } catch (...) {
            std::cerr << "[ERROR] Failed to reset device " << device_id_ << std::endl;
        }
    }
}

void Device::setActive() {
    CHECK_ACL(aclrtSetDevice(device_id_));
}

int Device::getDeviceCount() {
    uint32_t count = 0;
    try {
        CHECK_ACL(aclrtGetDeviceCount(&count));
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to get device count" << std::endl;
        return 0;
    }
    return static_cast<int>(count);
}

// ============================================================================
// STREAM MANAGEMENT
// ============================================================================

Stream::Stream() : stream_(nullptr) {
    try {
        CHECK_ACL(aclrtCreateStream(&stream_));
        std::cout << "[Stream] Created stream" << std::endl;
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to create stream" << std::endl;
        throw;
    }
}

Stream::~Stream() {
    if (stream_) {
        try {
            aclrtDestroyStream(stream_);
        } catch (...) {
            std::cerr << "[ERROR] Failed to destroy stream" << std::endl;
        }
    }
}

void Stream::synchronize() {
    try {
        CHECK_ACL(aclrtSynchronizeStream(stream_));
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to synchronize stream" << std::endl;
        throw;
    }
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

Memory::Memory(size_t size, bool huge_page)
    : data_(nullptr), size_(size) {
    
    try {
        aclrtMemMallocPolicy policy = huge_page ? 
            ACL_MEM_MALLOC_HUGE_FIRST : ACL_MEM_MALLOC_NORMAL_ONLY;
        CHECK_ACL(aclrtMalloc(&data_, size, policy));
        std::cout << "[Memory] Allocated " << size << " bytes on device" << std::endl;
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to allocate device memory" << std::endl;
        throw;
    }
}

Memory::~Memory() {
    if (data_) {
        try {
            aclrtFree(data_);
        } catch (...) {
            std::cerr << "[ERROR] Failed to free device memory" << std::endl;
        }
    }
}

void Memory::copyToHost(void* host_ptr, size_t size) const {
    if (!host_ptr) {
        throw AscendException(ACL_ERROR_INVALID_PARAM, "Host pointer is null");
    }
    if (size > size_) {
        throw AscendException(ACL_ERROR_INVALID_PARAM, 
            "Copy size exceeds allocated memory");
    }
    
    try {
        CHECK_ACL(aclrtMemcpy(host_ptr, size, data_, size, 
                             ACL_MEMCPY_DEVICE_TO_HOST));
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to copy memory to host" << std::endl;
        throw;
    }
}

void Memory::copyFromHost(const void* host_ptr, size_t size) {
    if (!host_ptr) {
        throw AscendException(ACL_ERROR_INVALID_PARAM, "Host pointer is null");
    }
    if (size > size_) {
        throw AscendException(ACL_ERROR_INVALID_PARAM, 
            "Copy size exceeds allocated memory");
    }
    
    try {
        CHECK_ACL(aclrtMemcpy(data_, size, host_ptr, size, 
                             ACL_MEMCPY_HOST_TO_DEVICE));
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to copy memory from host" << std::endl;
        throw;
    }
}

void Memory::copyToDevice(void* device_ptr, size_t size) const {
    if (!device_ptr) {
        throw AscendException(ACL_ERROR_INVALID_PARAM, "Device pointer is null");
    }
    if (size > size_) {
        throw AscendException(ACL_ERROR_INVALID_PARAM, 
            "Copy size exceeds allocated memory");
    }
    
    try {
        CHECK_ACL(aclrtMemcpy(device_ptr, size, data_, size, 
                             ACL_MEMCPY_DEVICE_TO_DEVICE));
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to copy memory device-to-device" << std::endl;
        throw;
    }
}

// ============================================================================
// TENSOR DESCRIPTOR
// ============================================================================

TensorDesc::TensorDesc(aclDataType dtype, const std::vector<int64_t>& shape)
    : desc_(nullptr), dtype_(dtype), shape_(shape) {
    
    try {
        if (shape.empty()) {
            throw AscendException(ACL_ERROR_INVALID_PARAM, "Shape cannot be empty");
        }
        
        std::vector<int64_t> dims(shape);
        CHECK_ACL(aclCreateTensorDesc(dtype, dims.size(), dims.data(), 
                                     ACL_FORMAT_ND, &desc_));
        std::cout << "[TensorDesc] Created tensor descriptor" << std::endl;
    } catch (const AscendException& e) {
        std::cerr << "[ERROR] Failed to create tensor descriptor" << std::endl;
        throw;
    }
}

TensorDesc::~TensorDesc() {
    if (desc_) {
        try {
            aclDestroyTensorDesc(desc_);
        } catch (...) {
            std::cerr << "[ERROR] Failed to destroy tensor descriptor" << std::endl;
        }
    }
}

} // namespace ascendcl