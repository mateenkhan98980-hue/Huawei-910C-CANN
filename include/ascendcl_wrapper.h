// ============================================================================
// AscendCL Runtime Wrapper - CANN 8.0 Compatible
// Strict, production-ready wrapper
// ============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <iostream>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "acl/acl_base.h"

namespace ascendcl {

// ============================================================================
// ERROR HANDLING
// ============================================================================
class AscendException : public std::runtime_error {
public:
    explicit AscendException(aclError err, const std::string& msg = "")
        : std::runtime_error(msg), error_code_(err) {}
    aclError getErrorCode() const { return error_code_; }
private:
    aclError error_code_;
};

#define CHECK_ACL(expr) \
    do { \
        aclError ret = (expr); \
        if (ret != ACL_SUCCESS) { \
            throw AscendException(ret, "ACL error at " __FILE__ ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// ============================================================================
// DEVICE MANAGEMENT
// ============================================================================
class Device {
public:
    Device(int device_id = 0);
    ~Device();
    void setActive();
    int getId() const { return device_id_; }
    static int getDeviceCount();
private:
    int device_id_;
    bool is_active_;
};

// ============================================================================
// STREAM MANAGEMENT
// ============================================================================
class Stream {
public:
    Stream();
    ~Stream();
    void synchronize();
    aclrtStream getHandle() const { return stream_; }
private:
    aclrtStream stream_;
};

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================
class Memory {
public:
    Memory(size_t size, bool huge_page = true);
    ~Memory();
    void* getData() const { return data_; }
    size_t getSize() const { return size_; }
    void copyToHost(void* host_ptr, size_t size) const;
    void copyFromHost(const void* host_ptr, size_t size);
    void copyToDevice(void* device_ptr, size_t size) const;
private:
    void* data_;
    size_t size_;
};

// ============================================================================
// TENSOR DESCRIPTOR
// ============================================================================
class TensorDesc {
public:
    TensorDesc(aclDataType dtype, const std::vector<int64_t>& shape);
    ~TensorDesc();
    aclTensorDesc* getHandle() const { return desc_; }
    aclDataType getDataType() const { return dtype_; }
    std::vector<int64_t> getShape() const { return shape_; }
private:
    aclTensorDesc* desc_;
    aclDataType dtype_;
    std::vector<int64_t> shape_;
};

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================
class Context {
public:
    static Context& getInstance();
    void init();
    void finalize();
    bool isInitialized() const { return initialized_; }
private:
    Context() : initialized_(false) {}
    ~Context() { if (initialized_) finalize(); }
    bool initialized_;
    mutable std::mutex mutex_;
};

// ============================================================================
// RUNTIME MANAGEMENT
// ============================================================================
void initializeRuntime();
void finalizeRuntime();
std::shared_ptr<Memory> allocateMemory(size_t size, bool huge_page = true);
std::shared_ptr<Stream> createStream();

} // namespace ascendcl