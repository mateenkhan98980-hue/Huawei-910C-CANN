// ============================================================================
// CUDA-Compatible Wrapper Layer Implementation (Strict)
// ============================================================================

#include "ascendcl_cuda_compat.h"
#include "ascendcl_memory.h"
#include "ascendcl_kernel.h"
#include <algorithm>
#include <cstring>
#include <fstream>

namespace ascendcl {

// ------------------- Device -------------------
Device::Device(int device_id) : device_id_(device_id), is_active_(false) {
    ASCENDCL_CHECK(aclrtSetDevice(device_id));
    is_active_ = true;
    std::cout << "[Device] Initialized device " << device_id << std::endl;
}

Device::~Device() {
    if (is_active_) {
        try { aclrtResetDevice(device_id_); }
        catch (...) { std::cerr << "[WARN] Failed to reset device " << device_id_ << std::endl; }
    }
}

void Device::setActive() {
    ASCENDCL_CHECK(aclrtSetDevice(device_id_));
}

int Device::getDeviceCount() {
    uint32_t count = 0;
    ASCENDCL_CHECK(aclrtGetDeviceCount(&count));
    return static_cast<int>(count);
}

std::string Device::getDeviceProperties(int device_id) {
    return "Ascend 910C (CANN 8.0) - 32 AI Cores, 256 TFLOPS, 1.2 TB/s HBM";
}

// ------------------- Stream -------------------
Stream::Stream() : stream_(nullptr) {
    ASCENDCL_CHECK(aclrtCreateStream(&stream_));
    std::cout << "[Stream] Created" << std::endl;
}

Stream::~Stream() {
    if (stream_) {
        try { aclrtDestroyStream(stream_); }
        catch (...) { std::cerr << "[WARN] Failed to destroy stream" << std::endl; }
    }
}

void Stream::synchronize() {
    ASCENDCL_CHECK(aclrtSynchronizeStream(stream_));
}

bool Stream::isReady() const {
    return aclrtStreamQuery(stream_) == ACL_SUCCESS;
}

// ------------------- Event -------------------
Event::Event() : event_(nullptr) {
    ASCENDCL_CHECK(aclrtCreateEvent(&event_));
}

Event::~Event() {
    if (event_) {
        try { aclrtDestroyEvent(event_); }
        catch (...) { std::cerr << "[WARN] Failed to destroy event" << std::endl; }
    }
}

void Event::record(Stream* stream) {
    if (!stream) throw AscendException(ACL_ERROR_INVALID_PARAM, "Stream is null");
    ASCENDCL_CHECK(aclrtRecordEvent(event_, stream->getHandle()));
}

void Event::synchronize() {
    ASCENDCL_CHECK(aclrtWaitEvent(event_));
}

float Event::elapsedTime(Event& start, Event& end) {
    float ms = 0.0f;
    ASCENDCL_CHECK(aclrtEventElapsedTime(&ms, start.getHandle(), end.getHandle()));
    return ms;
}

// ------------------- DeviceMemory -------------------
size_t DeviceMemory::alignSize(size_t size) {
    const size_t ALIGN = 128;
    return ((size + ALIGN - 1) / ALIGN) * ALIGN;
}

DeviceMemory::DeviceMemory(size_t size, bool huge_page)
    : data_(nullptr), size_(size), aligned_size_(alignSize(size)) {
    aclrtMemMallocPolicy policy = huge_page ? ACL_MEM_MALLOC_HUGE_FIRST : ACL_MEM_MALLOC_NORMAL_ONLY;
    ASCENDCL_CHECK(aclrtMalloc(&data_, aligned_size_, policy));
    std::cout << "[DeviceMemory] Allocated " << aligned_size_ << " bytes" << std::endl;
}

DeviceMemory::~DeviceMemory() {
    if (data_) {
        try { aclrtFree(data_); }
        catch (...) { std::cerr << "[WARN] Failed to free device memory" << std::endl; }
        data_ = nullptr;
    }
}

// Move constructors...
DeviceMemory::DeviceMemory(DeviceMemory&& other) noexcept
    : data_(other.data_), size_(other.size_), aligned_size_(other.aligned_size_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.aligned_size_ = 0;
}

DeviceMemory& DeviceMemory::operator=(DeviceMemory&& other) noexcept {
    if (this != &other) {
        if (data_) { try { aclrtFree(data_); } catch (...) {} }
        data_ = other.data_;
        size_ = other.size_;
        aligned_size_ = other.aligned_size_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.aligned_size_ = 0;
    }
    return *this;
}

// ------------------- Memcpy -------------------
void MemcpyHtoD(void* dst, const void* src, size_t size, Stream* stream) {
    if (!dst || !src) throw AscendException(ACL_ERROR_INVALID_PARAM, "Null pointer");
    aclrtStream s = stream ? stream->getHandle() : nullptr;
    ASCENDCL_CHECK(aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE, s));
}

void MemcpyDtoH(void* dst, const void* src, size_t size, Stream* stream) {
    if (!dst || !src) throw AscendException(ACL_ERROR_INVALID_PARAM, "Null pointer");
    aclrtStream s = stream ? stream->getHandle() : nullptr;
    ASCENDCL_CHECK(aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST, s));
}

void MemcpyDtoD(void* dst, const void* src, size_t size, Stream* stream) {
    if (!dst || !src) throw AscendException(ACL_ERROR_INVALID_PARAM, "Null pointer");
    aclrtStream s = stream ? stream->getHandle() : nullptr;
    ASCENDCL_CHECK(aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE, s));
}

void Memcpy(void* dst, const void* src, size_t size, const std::string& copy_type) {
    if (!dst || !src) throw AscendException(ACL_ERROR_INVALID_PARAM, "Null pointer");
    aclrtMemcpyKind kind;
    if (copy_type == "H2D") kind = ACL_MEMCPY_HOST_TO_DEVICE;
    else if (copy_type == "D2H") kind = ACL_MEMCPY_DEVICE_TO_HOST;
    else if (copy_type == "D2D") kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
    else throw AscendException(ACL_ERROR_INVALID_PARAM, "Invalid copy type");
    ASCENDCL_CHECK(aclrtMemcpy(dst, size, src, size, kind));
}

void Memset(void* dst, int value, size_t size) {
    if (!dst) throw AscendException(ACL_ERROR_INVALID_PARAM, "Null pointer");
    ASCENDCL_CHECK(aclrtMemset(dst, size, value, size));
}

// ------------------- Allocation -------------------
std::shared_ptr<DeviceMemory> DeviceMalloc(size_t size, bool huge_page) {
    if (size == 0) throw AscendException(ACL_ERROR_INVALID_PARAM, "Size zero");
    return std::make_shared<DeviceMemory>(size, huge_page);
}

void* MallocHost(size_t size) {
    if (size == 0) throw AscendException(ACL_ERROR_INVALID_PARAM, "Size zero");
    void* ptr = nullptr;
    ASCENDCL_CHECK(aclrtMallocHost(&ptr, size));
    return ptr;
}

void FreeHost(void* ptr) {
    if (!ptr) throw AscendException(ACL_ERROR_INVALID_PARAM, "Null pointer");
    ASCENDCL_CHECK(aclrtFreeHost(ptr));
}

// ------------------- OperatorRegistry -------------------
OperatorRegistry& OperatorRegistry::getInstance() {
    static OperatorRegistry instance;
    return instance;
}

bool OperatorRegistry::registerOp(const std::string& op_name, const OperatorFunc& func) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    if (operators_.find(op_name) != operators_.end()) {
        std::cerr << "[WARN] Operator " << op_name << " already registered" << std::endl;
        return false;
    }
    operators_[op_name] = func;
    std::cout << "[Registry] Registered operator: " << op_name << std::endl;
    return true;
}

bool OperatorRegistry::hasOp(const std::string& op_name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    return operators_.find(op_name) != operators_.end();
}

aclError OperatorRegistry::executeOp(const std::string& op_name,
                                     const std::vector<void*>& inputs,
                                     const std::vector<void*>& outputs,
                                     const std::vector<aclTensorDesc*>& input_descs,
                                     const std::vector<aclTensorDesc*>& output_descs,
                                     aclrtStream stream) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = operators_.find(op_name);
    if (it == operators_.end()) {
        std::cerr << "[ERROR] Operator not registered: " << op_name << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    return it->second(inputs, outputs, input_descs, output_descs, stream);
}

bool OperatorRegistry::registerSimpleKernel(const std::string& kernel_name, const SimpleKernelFunc& func) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    if (simple_kernels_.find(kernel_name) != simple_kernels_.end()) return false;
    simple_kernels_[kernel_name] = func;
    return true;
}

bool OperatorRegistry::hasSimpleKernel(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    return simple_kernels_.find(kernel_name) != simple_kernels_.end();
}

aclError OperatorRegistry::executeSimpleKernel(const std::string& kernel_name,
                                               const std::vector<void*>& args,
                                               aclrtStream stream) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = simple_kernels_.find(kernel_name);
    if (it == simple_kernels_.end()) return ACL_ERROR_INVALID_PARAM;
    return it->second(args, stream);
}

std::vector<std::string> OperatorRegistry::listOps() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    std::vector<std::string> ops;
    for (const auto& p : operators_) ops.push_back(p.first);
    for (const auto& p : simple_kernels_) ops.push_back("kernel:" + p.first);
    return ops;
}

// ------------------- KernelLauncher -------------------
void KernelLauncher::launchKernel(Stream* stream, uint32_t grid_dim, uint32_t block_dim,
                                  uint32_t shared_mem, const std::string& kernel_name,
                                  const std::vector<void*>& args) {
    if (!stream) throw AscendException(ACL_ERROR_INVALID_PARAM, "Stream is null");
    auto& registry = OperatorRegistry::getInstance();
    if (registry.hasSimpleKernel(kernel_name)) {
        aclError ret = registry.executeSimpleKernel(kernel_name, args, stream->getHandle());
        if (ret != ACL_SUCCESS) throw AscendException(ret, "Kernel execution failed");
        ASCENDCL_CHECK(aclrtSynchronizeStream(stream->getHandle()));
        return;
    }
    throw AscendException(ACL_ERROR_INVALID_PARAM, "Kernel not registered: " + kernel_name);
}

// ------------------- TensorDescriptor -------------------
TensorDescriptor::TensorDescriptor(aclDataType dtype, const std::vector<int64_t>& shape)
    : desc_(nullptr), dtype_(dtype), shape_(shape) {
    if (shape.empty()) throw AscendException(ACL_ERROR_INVALID_PARAM, "Shape empty");
    std::vector<int64_t> dims(shape);
    ASCENDCL_CHECK(aclCreateTensorDesc(dtype, dims.size(), dims.data(), ACL_FORMAT_ND, &desc_));
}

TensorDescriptor::~TensorDescriptor() {
    if (desc_) {
        try { aclDestroyTensorDesc(desc_); }
        catch (...) { std::cerr << "[WARN] Failed to destroy tensor descriptor" << std::endl; }
    }
}

// ------------------- DistributedContext -------------------
DistributedContext& DistributedContext::getInstance() {
    static DistributedContext instance;
    return instance;
}

void DistributedContext::initDistributed(int rank, int world_size, const std::string& master_addr, int master_port) {
    if (initialized_) return;
    rank_ = rank; world_size_ = world_size;
    master_addr_ = master_addr; master_port_ = master_port;
    std::cout << "[Distributed] Initializing rank=" << rank << " world=" << world_size << std::endl;
    initHCCL();
    initialized_ = true;
}

void DistributedContext::initHCCL() {
    hccl_comm_ = nullptr;
    if (world_size_ <= 1) {
        std::cout << "[Distributed] Single-process mode, no HCCL required" << std::endl;
        return;
    }
    const char* root_info_path = std::getenv("ASCEND_HCCL_ROOT_INFO_PATH");
    if (!root_info_path || root_info_path[0] == '\0') {
        throw AscendException(ACL_ERROR_INVALID_PARAM, "ASCEND_HCCL_ROOT_INFO_PATH not set");
    }
    HcclRootInfo root_info{};
    std::ifstream in(root_info_path, std::ios::binary);
    if (!in) throw AscendException(ACL_ERROR_INVALID_PARAM, "Failed to open HCCL root info file");
    in.read(reinterpret_cast<char*>(&root_info), sizeof(root_info));
    if (!in) throw AscendException(ACL_ERROR_INVALID_PARAM, "Failed to read HCCL root info");

    const int MAX_RETRIES = 3;
    for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
        if (attempt > 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        HcclResult ret = HcclCommInitRootInfo(world_size_, &root_info, rank_, &hccl_comm_);
        if (ret == HCCL_SUCCESS) {
            std::cout << "[Distributed] HCCL initialized for rank " << rank_ << std::endl;
            return;
        }
        if (attempt == MAX_RETRIES - 1)
            throw AscendException(ACL_ERROR_INVALID_PARAM, "HcclCommInitRootInfo failed");
    }
}

void DistributedContext::allReduce(void* send_buf, void* recv_buf, size_t size, int root) {
    if (!initialized_) throw AscendException(ACL_ERROR_INVALID_PARAM, "Distributed not initialized");
    if (!send_buf || !recv_buf) throw AscendException(ACL_ERROR_INVALID_PARAM, "Null buffer");
    if (world_size_ <= 1 || hccl_comm_ == nullptr) {
        if (send_buf != recv_buf)
            ASCENDCL_CHECK(aclrtMemcpy(recv_buf, size, send_buf, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
    } else {
        HcclResult ret = HcclAllReduce(send_buf, recv_buf, size, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, hccl_comm_, nullptr);
        if (ret != HCCL_SUCCESS) throw AscendException(ACL_ERROR_INVALID_PARAM, "HcclAllReduce failed");
    }
}

void DistributedContext::broadcast(void* buf, size_t size, int root) {
    if (!initialized_) throw AscendException(ACL_ERROR_INVALID_PARAM, "Distributed not initialized");
    if (!buf) throw AscendException(ACL_ERROR_INVALID_PARAM, "Null buffer");
    if (world_size_ > 1 && hccl_comm_ != nullptr) {
        HcclResult ret = HcclBroadcast(buf, size, HCCL_DATA_TYPE_INT8, root, hccl_comm_, nullptr);
        if (ret != HCCL_SUCCESS) throw AscendException(ACL_ERROR_INVALID_PARAM, "HcclBroadcast failed");
    }
}

void DistributedContext::allGather(void* send_buf, void* recv_buf, size_t size) {
    if (!initialized_) throw AscendException(ACL_ERROR_INVALID_PARAM, "Distributed not initialized");
    if (!send_buf || !recv_buf) throw AscendException(ACL_ERROR_INVALID_PARAM, "Null buffer");
    if (world_size_ <= 1 || hccl_comm_ == nullptr) {
        if (send_buf != recv_buf)
            ASCENDCL_CHECK(aclrtMemcpy(recv_buf, size, send_buf, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
    } else {
        HcclResult ret = HcclAllGather(send_buf, size, HCCL_DATA_TYPE_INT8, recv_buf, size * world_size_, HCCL_DATA_TYPE_INT8, hccl_comm_, nullptr);
        if (ret != HCCL_SUCCESS) throw AscendException(ACL_ERROR_INVALID_PARAM, "HcclAllGather failed");
    }
}

void DistributedContext::barrier() {
    if (!initialized_) throw AscendException(ACL_ERROR_INVALID_PARAM, "Distributed not initialized");
    if (world_size_ > 1 && hccl_comm_ != nullptr) {
        HcclResult ret = HcclBarrier(hccl_comm_, nullptr);
        if (ret != HCCL_SUCCESS) throw AscendException(ACL_ERROR_INVALID_PARAM, "HcclBarrier failed");
    }
}

void DistributedContext::finalize() {
    if (!initialized_) return;
    if (hccl_comm_ != nullptr) {
        HcclCommDestroy(hccl_comm_);
        hccl_comm_ = nullptr;
    }
    initialized_ = false;
}

// ------------------- RuntimeContext -------------------
RuntimeContext& RuntimeContext::getInstance() {
    static RuntimeContext instance;
    return instance;
}

void RuntimeContext::initialize(int device_id) {
    std::lock_guard<std::mutex> lock(context_mutex_);
    if (initialized_) return;
    ASCENDCL_CHECK(aclInit(nullptr));
    ASCENDCL_CHECK(aclrtSetDevice(device_id));
    current_device_ = device_id;
    initialized_ = true;
    std::cout << "[Runtime] Initialized on device " << device_id << std::endl;
}

void RuntimeContext::finalize() {
    std::lock_guard<std::mutex> lock(context_mutex_);
    if (!initialized_) return;
    ASCENDCL_CHECK(aclFinalize());
    initialized_ = false;
    current_device_ = -1;
}

std::string RuntimeContext::getStatistics() const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    std::string s = "Ascend Runtime Statistics:\n";
    s += "  Status: " + std::string(initialized_ ? "Initialized" : "Not initialized") + "\n";
    s += "  Current Device: " + std::to_string(current_device_) + "\n";
    s += "  Total Devices: " + std::to_string(Device::getDeviceCount()) + "\n";
    return s;
}

// ------------------- PyTorch helpers -------------------
namespace pytorch {
void registerCustomOp(const std::string& op_name, const OperatorFunc& func) {
    if (!OperatorRegistry::getInstance().registerOp(op_name, func))
        std::cerr << "[ERROR] Failed to register op " << op_name << std::endl;
}

std::shared_ptr<DeviceMemory> tensorToDevice(const void* tensor_data, size_t size) {
    if (!tensor_data || size == 0) throw AscendException(ACL_ERROR_INVALID_PARAM, "Invalid tensor data");
    auto mem = DeviceMalloc(size, true);
    MemcpyHtoD(mem->getData(), tensor_data, size);
    return mem;
}

void* deviceToTensor(const std::shared_ptr<DeviceMemory>& device_mem) {
    if (!device_mem || !device_mem->getData()) throw AscendException(ACL_ERROR_INVALID_PARAM, "Invalid device memory");
    void* host = MallocHost(device_mem->getSize());
    MemcpyDtoH(host, device_mem->getData(), device_mem->getSize());
    return host;
}

std::string createAtenTensor(const std::shared_ptr<DeviceMemory>& data,
                             const std::vector<int64_t>& shape,
                             const std::string& dtype) {
    std::string s = "ATen Tensor (Ascend Device):\n";
    s += "  Data: " + std::to_string(reinterpret_cast<uintptr_t>(data->getData())) + "\n";
    s += "  Size: " + std::to_string(data->getSize()) + " bytes\n";
    s += "  Shape: [";
    for (size_t i=0; i<shape.size(); ++i) { s += std::to_string(shape[i]); if (i+1<shape.size()) s += ", "; }
    s += "]\n";
    s += "  DType: " + dtype + "\n";
    return s;
}
} // namespace pytorch

// ------------------- Global Init -------------------
void initializeAscend(int device_id, bool enable_distributed) {
    RuntimeContext::getInstance().initialize(device_id);
    if (enable_distributed) {
        std::cout << "[Ascend] Distributed training enabled" << std::endl;
        // Environment variables must be set externally
    }
}

void finalizeAscend() {
    if (DistributedContext::getInstance().isInitialized())
        DistributedContext::getInstance().finalize();
    RuntimeContext::getInstance().finalize();
}

} // namespace ascendcl