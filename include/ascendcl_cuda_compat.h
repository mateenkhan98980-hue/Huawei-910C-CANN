// ============================================================================
// CUDA-Compatible Wrapper Layer for Huawei Ascend 910C
// Strict CUDA API surface - all functions return proper error codes
// ============================================================================

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>
#include <map>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "acl/acl_op.h"
#include "acl/acl_nn.h"
#include "hccl/hccl.h"

namespace ascendcl {

// ============================================================================
// ERROR HANDLING - strict
// ============================================================================
class AscendException : public std::runtime_error {
public:
    explicit AscendException(aclError err, const std::string& msg = "")
        : std::runtime_error(msg), error_code_(err) {
        std::cerr << "[AscendCL Error] Code: " << static_cast<int>(err) 
                  << " Message: " << msg << std::endl;
    }
    aclError getErrorCode() const { return error_code_; }
private:
    aclError error_code_;
};

#define ASCENDCL_CHECK(expr) \
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
    explicit Device(int device_id = 0);
    ~Device();
    void setActive();
    int getId() const { return device_id_; }
    static int getDeviceCount();
    static std::string getDeviceProperties(int device_id);
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
    bool isReady() const;
private:
    aclrtStream stream_;
};

// ============================================================================
// EVENT MANAGEMENT
// ============================================================================
class Event {
public:
    Event();
    ~Event();
    void record(Stream* stream);
    void synchronize();
    aclrtEvent getHandle() const { return event_; }
    static float elapsedTime(Event& start, Event& end);
private:
    aclrtEvent event_;
};

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================
class DeviceMemory {
public:
    explicit DeviceMemory(size_t size, bool huge_page = true);
    ~DeviceMemory();
    void* getData() const { return data_; }
    size_t getSize() const { return size_; }
    size_t getAlignedSize() const { return aligned_size_; }
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    DeviceMemory(DeviceMemory&& other) noexcept;
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;
private:
    void* data_;
    size_t size_;
    size_t aligned_size_;
    static size_t alignSize(size_t size);
};

// ============================================================================
// MEMORY OPERATIONS
// ============================================================================
void MemcpyHtoD(void* dst, const void* src, size_t size, Stream* stream = nullptr);
void MemcpyDtoH(void* dst, const void* src, size_t size, Stream* stream = nullptr);
void MemcpyDtoD(void* dst, const void* src, size_t size, Stream* stream = nullptr);
void Memcpy(void* dst, const void* src, size_t size, const std::string& copy_type = "D2H");
void Memset(void* dst, int value, size_t size);

// ============================================================================
// ALLOCATION API
// ============================================================================
std::shared_ptr<DeviceMemory> DeviceMalloc(size_t size, bool huge_page = true);
void* MallocHost(size_t size);
void FreeHost(void* ptr);

// ============================================================================
// OPERATOR REGISTRY
// ============================================================================
using OperatorFunc = std::function<aclError(
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<aclTensorDesc*>& input_descs,
    const std::vector<aclTensorDesc*>& output_descs,
    aclrtStream stream)>;

class OperatorRegistry {
public:
    static OperatorRegistry& getInstance();
    bool registerOp(const std::string& op_name, const OperatorFunc& func);
    bool hasOp(const std::string& op_name) const;
    aclError executeOp(const std::string& op_name,
                       const std::vector<void*>& inputs,
                       const std::vector<void*>& outputs,
                       const std::vector<aclTensorDesc*>& input_descs,
                       const std::vector<aclTensorDesc*>& output_descs,
                       aclrtStream stream);
    using SimpleKernelFunc = std::function<aclError(const std::vector<void*>&, aclrtStream)>;
    bool registerSimpleKernel(const std::string& kernel_name, const SimpleKernelFunc& func);
    bool hasSimpleKernel(const std::string& kernel_name) const;
    aclError executeSimpleKernel(const std::string& kernel_name,
                                 const std::vector<void*>& args,
                                 aclrtStream stream);
    std::vector<std::string> listOps() const;
private:
    OperatorRegistry() = default;
    std::map<std::string, OperatorFunc> operators_;
    std::map<std::string, SimpleKernelFunc> simple_kernels_;
    mutable std::mutex registry_mutex_;
};

// ============================================================================
// KERNEL LAUNCHER
// ============================================================================
class KernelLauncher {
public:
    static void launchKernel(
        Stream* stream,
        uint32_t grid_dim,
        uint32_t block_dim,
        uint32_t shared_mem,
        const std::string& kernel_name,
        const std::vector<void*>& args);
};

// ============================================================================
// TENSOR DESCRIPTOR
// ============================================================================
class TensorDescriptor {
public:
    TensorDescriptor(aclDataType dtype, const std::vector<int64_t>& shape);
    ~TensorDescriptor();
    aclTensorDesc* getHandle() const { return desc_; }
    aclDataType getDataType() const { return dtype_; }
    std::vector<int64_t> getShape() const { return shape_; }
private:
    aclTensorDesc* desc_;
    aclDataType dtype_;
    std::vector<int64_t> shape_;
};

// ============================================================================
// DISTRIBUTED CONTEXT (HCCL)
// ============================================================================
class DistributedContext {
public:
    static DistributedContext& getInstance();
    void initDistributed(int rank, int world_size,
                        const std::string& master_addr = "127.0.0.1",
                        int master_port = 29500);
    bool isInitialized() const { return initialized_; }
    int getRank() const { return rank_; }
    int getWorldSize() const { return world_size_; }
    void allReduce(void* send_buf, void* recv_buf, size_t size, int root = 0);
    void broadcast(void* buf, size_t size, int root);
    void allGather(void* send_buf, void* recv_buf, size_t size);
    void barrier();
    void finalize();
private:
    DistributedContext() : initialized_(false), rank_(-1), world_size_(0), hccl_comm_(nullptr) {}
    void initHCCL();
    bool initialized_;
    int rank_;
    int world_size_;
    HcclComm hccl_comm_;
    std::string master_addr_;
    int master_port_;
};

// ============================================================================
// RUNTIME CONTEXT
// ============================================================================
class RuntimeContext {
public:
    static RuntimeContext& getInstance();
    void initialize(int device_id = 0);
    void finalize();
    bool isInitialized() const { return initialized_; }
    int getCurrentDevice() const { return current_device_; }
    std::string getStatistics() const;
private:
    RuntimeContext() : initialized_(false), current_device_(-1) {}
    bool initialized_;
    int current_device_;
    std::mutex context_mutex_;
};

// ============================================================================
// PYTORCH HELPERS
// ============================================================================
namespace pytorch {
void registerCustomOp(const std::string& op_name, const OperatorFunc& func);
std::shared_ptr<DeviceMemory> tensorToDevice(const void* tensor_data, size_t size);
void* deviceToTensor(const std::shared_ptr<DeviceMemory>& device_mem);
std::string createAtenTensor(const std::shared_ptr<DeviceMemory>& data,
                            const std::vector<int64_t>& shape,
                            const std::string& dtype);
} // namespace pytorch

// ============================================================================
// GLOBAL INITIALIZATION
// ============================================================================
void initializeAscend(int device_id = 0, bool enable_distributed = false);
void finalizeAscend();

} // namespace ascendcl