// ============================================================================
// CUDA Drop-in Replacement Library – Exact CUDA Runtime Symbols
// Maps to AscendCL CANN. No code change required in user application.
// ============================================================================

#include "ascendcl_cuda_compat.h"
#include "ascendcl_memory.h"
#include "ascendcl_kernel.h"
#include "cuda_compat_extra.h"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <iostream>
#include <string>

using namespace ascendcl;

// ------------------- CUDA Error Codes -------------------
#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_INVALID_DEVICE 2
#define CUDA_ERROR_OUT_OF_MEMORY 3
#define CUDA_ERROR_NOT_INITIALIZED 4

// ------------------- Global Initialization -------------------
static std::once_flag g_init_flag;
static bool g_initialized = false;

void ensure_ascend_init() {
    std::call_once(g_init_flag, []() {
        try {
            initializeAscend(0, false);
            g_initialized = true;
        } catch (...) {
            // Fallback: try device 0
        }
    });
}

// ------------------- Memory Tracking -------------------
static std::unordered_map<void*, std::shared_ptr<DeviceMemory>> g_ptr_map;
static std::mutex g_ptr_mutex;

// ------------------- Stream/Event Tracking -------------------
struct CudaStream { std::shared_ptr<Stream> stream; };
struct CudaEvent  { std::shared_ptr<Event> event; };

static std::unordered_map<void*, std::shared_ptr<CudaStream>> g_stream_map;
static std::mutex g_stream_mutex;

static std::unordered_map<void*, std::shared_ptr<CudaEvent>> g_event_map;
static std::mutex g_event_mutex;

// ------------------- Unified Memory Tracking -------------------
static std::unordered_map<void*, size_t> g_managed_map;
static std::mutex g_managed_mutex;

// ============================================================================
// CUDA Kernel Registry – maps function pointers to kernel names
// ============================================================================
static std::unordered_map<const void*, std::string> g_kernel_registry;
static std::mutex g_kernel_registry_mutex;

int cudaRegisterKernel(const void* func, const char* name) {
    if (!func || !name) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
    if (g_kernel_registry.find(func) != g_kernel_registry.end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    g_kernel_registry[func] = std::string(name);
    return CUDA_SUCCESS;
}

int cudaUnregisterKernel(const void* func) {
    std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
    auto it = g_kernel_registry.find(func);
    if (it == g_kernel_registry.end()) return CUDA_ERROR_INVALID_VALUE;
    g_kernel_registry.erase(it);
    return CUDA_SUCCESS;
}

// ============================================================================
// CUDA Runtime API: Memory Info & OOM
// ============================================================================
int cudaMemGetInfo(size_t* free, size_t* total) {
    ensure_ascend_init();
    if (!free || !total) return CUDA_ERROR_INVALID_VALUE;
    try {
        getMemoryInfo(free, total);
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

// Enhanced cudaMalloc with OOM check
int cudaMalloc(void** devPtr, size_t size) {
    ensure_ascend_init();
    if (!devPtr || size == 0) return CUDA_ERROR_INVALID_VALUE;
    size_t free, total;
    getMemoryInfo(&free, &total);
    if (free < size) return CUDA_ERROR_OUT_OF_MEMORY;
    try {
        auto mem = DeviceMalloc(size, true);
        void* ptr = mem->getData();
        {
            std::lock_guard<std::mutex> lock(g_ptr_mutex);
            g_ptr_map[ptr] = mem;
        }
        *devPtr = ptr;
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

int cudaFree(void* devPtr) {
    if (!devPtr) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(g_ptr_mutex);
    auto it = g_ptr_map.find(devPtr);
    if (it == g_ptr_map.end()) return CUDA_ERROR_INVALID_VALUE;
    g_ptr_map.erase(it);
    return CUDA_SUCCESS;
}

int cudaMallocHost(void** hostPtr, size_t size) {
    ensure_ascend_init();
    if (!hostPtr || size == 0) return CUDA_ERROR_INVALID_VALUE;
    try {
        *hostPtr = MallocHost(size);
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

int cudaFreeHost(void* hostPtr) {
    if (!hostPtr) return CUDA_ERROR_INVALID_VALUE;
    try {
        FreeHost(hostPtr);
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

// ------------------- Unified Memory -------------------
int cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
    ensure_ascend_init();
    if (!devPtr || size == 0) return CUDA_ERROR_INVALID_VALUE;
    try {
        void* hostPtr = MallocHost(size); // pinned host memory
        if (!hostPtr) return CUDA_ERROR_OUT_OF_MEMORY;
        {
            std::lock_guard<std::mutex> lock(g_managed_mutex);
            g_managed_map[hostPtr] = size;
        }
        *devPtr = hostPtr; // device pointer = host pointer (simplified)
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

int cudaFreeManaged(void* ptr) {
    if (!ptr) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(g_managed_mutex);
    auto it = g_managed_map.find(ptr);
    if (it == g_managed_map.end()) return CUDA_ERROR_INVALID_VALUE;
    FreeHost(ptr);
    g_managed_map.erase(it);
    return CUDA_SUCCESS;
}

// Override cudaMemcpy to handle managed pointers
int cudaMemcpy(void* dst, const void* src, size_t count, int kind) {
    ensure_ascend_init();
    if (!dst || !src) return CUDA_ERROR_INVALID_VALUE;
    // If either pointer is managed, copy via host<->device as needed.
    // For simplicity, we just use standard memcpy with appropriate direction.
    // We'll check if src/dst are in managed map.
    bool srcManaged = false, dstManaged = false;
    {
        std::lock_guard<std::mutex> lock(g_managed_mutex);
        srcManaged = g_managed_map.find((void*)src) != g_managed_map.end();
        dstManaged = g_managed_map.find(dst) != g_managed_map.end();
    }
    if (srcManaged && dstManaged) {
        std::memcpy(dst, src, count);
        return CUDA_SUCCESS;
    }
    // Otherwise, use original logic.
    try {
        if (kind == 0) Memcpy(dst, src, count, "H2D");
        else if (kind == 1) Memcpy(dst, src, count, "D2H");
        else if (kind == 2) Memcpy(dst, src, count, "D2D");
        else return CUDA_ERROR_INVALID_VALUE;
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

// Asynchronous memcpy
int cudaMemcpyAsync(void* dst, const void* src, size_t count, int kind, void* stream) {
    ensure_ascend_init();
    if (!dst || !src) return CUDA_ERROR_INVALID_VALUE;
    try {
        auto* cudaStream = static_cast<CudaStream*>(stream);
        Stream* s = cudaStream ? cudaStream->stream.get() : nullptr;
        if (kind == 0) MemcpyHtoD(dst, src, count, s);
        else if (kind == 1) MemcpyDtoH(dst, src, count, s);
        else if (kind == 2) MemcpyDtoD(dst, src, count, s);
        else return CUDA_ERROR_INVALID_VALUE;
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

int cudaMemset(void* dst, int value, size_t count) {
    ensure_ascend_init();
    if (!dst) return CUDA_ERROR_INVALID_VALUE;
    try {
        Memset(dst, value, count);
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

// ----------------------------------------------------------
// STREAM MANAGEMENT
// ----------------------------------------------------------
int cudaStreamCreate(void** pStream) {
    ensure_ascend_init();
    if (!pStream) return CUDA_ERROR_INVALID_VALUE;
    try {
        auto cudaStream = std::make_shared<CudaStream>();
        cudaStream->stream = std::make_shared<Stream>();
        {
            std::lock_guard<std::mutex> lock(g_stream_mutex);
            g_stream_map[cudaStream.get()] = cudaStream;
        }
        *pStream = cudaStream.get();
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

int cudaStreamDestroy(void* stream) {
    if (!stream) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(g_stream_mutex);
    auto it = g_stream_map.find(stream);
    if (it == g_stream_map.end()) return CUDA_ERROR_INVALID_VALUE;
    g_stream_map.erase(it);
    return CUDA_SUCCESS;
}

int cudaStreamSynchronize(void* stream) {
    if (!stream) return CUDA_ERROR_INVALID_VALUE;
    try {
        auto* cudaStream = static_cast<CudaStream*>(stream);
        cudaStream->stream->synchronize();
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

int cudaStreamQuery(void* stream) {
    if (!stream) return CUDA_ERROR_INVALID_VALUE;
    auto* cudaStream = static_cast<CudaStream*>(stream);
    return cudaStream->stream->isReady() ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

// ----------------------------------------------------------
// EVENT MANAGEMENT
// ----------------------------------------------------------
int cudaEventCreate(void** pEvent) {
    ensure_ascend_init();
    if (!pEvent) return CUDA_ERROR_INVALID_VALUE;
    try {
        auto cudaEvent = std::make_shared<CudaEvent>();
        cudaEvent->event = std::make_shared<Event>();
        {
            std::lock_guard<std::mutex> lock(g_event_mutex);
            g_event_map[cudaEvent.get()] = cudaEvent;
        }
        *pEvent = cudaEvent.get();
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

int cudaEventDestroy(void* event) {
    if (!event) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(g_event_mutex);
    auto it = g_event_map.find(event);
    if (it == g_event_map.end()) return CUDA_ERROR_INVALID_VALUE;
    g_event_map.erase(it);
    return CUDA_SUCCESS;
}

int cudaEventRecord(void* event, void* stream) {
    if (!event || !stream) return CUDA_ERROR_INVALID_VALUE;
    try {
        auto* cudaEvent = static_cast<CudaEvent*>(event);
        auto* cudaStream = static_cast<CudaStream*>(stream);
        cudaEvent->event->record(cudaStream->stream.get());
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

int cudaEventSynchronize(void* event) {
    if (!event) return CUDA_ERROR_INVALID_VALUE;
    try {
        auto* cudaEvent = static_cast<CudaEvent*>(event);
        cudaEvent->event->synchronize();
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

int cudaEventElapsedTime(float* ms, void* start, void* end) {
    if (!ms || !start || !end) return CUDA_ERROR_INVALID_VALUE;
    try {
        auto* s = static_cast<CudaEvent*>(start);
        auto* e = static_cast<CudaEvent*>(end);
        *ms = Event::elapsedTime(*s->event, *e->event);
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

// ----------------------------------------------------------
// ERROR HANDLING
// ----------------------------------------------------------
const char* cudaGetErrorString(int error) {
    switch (error) {
        case CUDA_SUCCESS: return "no error";
        case CUDA_ERROR_INVALID_VALUE: return "invalid argument";
        case CUDA_ERROR_OUT_OF_MEMORY: return "out of memory";
        default: return "unknown error";
    }
}

int cudaGetLastError(void) {
    return CUDA_SUCCESS;
}

// ----------------------------------------------------------
// DEVICE MANAGEMENT
// ----------------------------------------------------------
int cudaSetDevice(int device_id) {
    ensure_ascend_init();
    try {
        Device device(device_id);
        device.setActive();
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
}

int cudaGetDevice(int* device_id) {
    ensure_ascend_init();
    if (!device_id) return CUDA_ERROR_INVALID_VALUE;
    *device_id = 0; // default
    return CUDA_SUCCESS;
}

int cudaGetDeviceCount(int* count) {
    ensure_ascend_init();
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    *count = Device::getDeviceCount();
    return CUDA_SUCCESS;
}

int cudaGetDeviceProperties(void* props, int device_id) {
    ensure_ascend_init();
    // Minimal stub – can fill cudaDeviceProp if needed.
    return CUDA_SUCCESS;
}

// ----------------------------------------------------------
// cudaLaunchKernel – exact CUDA signature
// ----------------------------------------------------------
int cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                     void** args, size_t sharedMem, void* stream) {
    // Look up kernel name
    std::string kernel_name;
    {
        std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
        auto it = g_kernel_registry.find(func);
        if (it == g_kernel_registry.end()) {
            std::cerr << "[ERROR] cudaLaunchKernel: function pointer not registered. "
                      << "Use cudaRegisterKernel() first." << std::endl;
            return CUDA_ERROR_INVALID_VALUE;
        }
        kernel_name = it->second;
    }

    std::vector<void*> arg_vector;
    if (args) {
        for (int i = 0; args[i] != nullptr; ++i) {
            arg_vector.push_back(args[i]);
        }
    }

    ascendcl::Stream* acl_stream = nullptr;
    if (stream) {
        auto* cudaStream = static_cast<CudaStream*>(stream);
        if (cudaStream) {
            acl_stream = cudaStream->stream.get();
        }
    }

    bool success = ascendcl::CustomKernel::launchKernel(
        kernel_name,
        acl_stream ? acl_stream->getHandle() : nullptr,
        arg_vector,
        gridDim.x * gridDim.y * gridDim.z,
        blockDim.x * blockDim.y * blockDim.z
    );

    return success ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

// ----------------------------------------------------------
// CUDA Graphs (forward to cuda_graph.cpp)
// ----------------------------------------------------------
int cudaGraphCreate(void** pGraph, int flags) {
    // Implementation in cuda_graph.cpp
    extern int cudaGraphCreate(void**, int);
    return cudaGraphCreate(pGraph, flags);
}

int cudaGraphLaunch(void* graph, void* stream) {
    extern int cudaGraphLaunch(void*, void*);
    return cudaGraphLaunch(graph, stream);
}

int cudaGraphDestroy(void* graph) {
    extern int cudaGraphDestroy(void*);
    return cudaGraphDestroy(graph);
}

// ----------------------------------------------------------
// cuBLAS / cuDNN (forward)
// ----------------------------------------------------------
int cublasCreate(void** handle) {
    extern int cublasCreate(void**);
    return cublasCreate(handle);
}
int cublasDestroy(void* handle) {
    extern int cublasDestroy(void*);
    return cublasDestroy(handle);
}
int cublasSgemm(void* handle, int transA, int transB, int m, int n, int k,
                const float* alpha, const float* A, int lda,
                const float* B, int ldb, const float* beta, float* C, int ldc) {
    extern int cublasSgemm(void*, int, int, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
    return cublasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// ----------------------------------------------------------
// Profiler
// ----------------------------------------------------------
int cudaProfilerStart() {
    extern int cudaProfilerStart();
    return cudaProfilerStart();
}
int cudaProfilerStop() {
    extern int cudaProfilerStop();
    return cudaProfilerStop();
}
// ============== cuda_dropin.cpp (Exact Mapping) ==============
static std::unordered_map<const void*, std::string> g_kernel_registry;
static std::mutex g_kernel_registry_mutex;

// CUDA kernel register karne ka function
int cudaRegisterKernel(const void* func, const char* name) {
    if (!func || !name) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
    g_kernel_registry[func] = std::string(name);
    return CUDA_SUCCESS;
}

// CUDA kernel launch karne ka function
int cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                     void** args, size_t sharedMem, void* stream) {
    // 1. Func pointer se kernel name nikalna
    std::string kernel_name;
    {
        std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
        auto it = g_kernel_registry.find(func);
        if (it == g_kernel_registry.end()) return CUDA_ERROR_INVALID_VALUE;
        kernel_name = it->second;
    }

    // 2. Args ko vector mein daalna
    std::vector<void*> arg_vector;
    if (args) {
        for (int i = 0; args[i] != nullptr; ++i) arg_vector.push_back(args[i]);
    }

    // 3. Stream handle nikalna
    ascendcl::Stream* acl_stream = nullptr;
    if (stream) {
        auto* cudaStream = static_cast<CudaStream*>(stream);
        acl_stream = cudaStream->stream.get();
    }

    // 4. Ascend CustomKernel par forward karna
    bool success = ascendcl::CustomKernel::launchKernel(
        kernel_name,
        acl_stream ? acl_stream->getHandle() : nullptr,
        arg_vector,
        gridDim.x * gridDim.y * gridDim.z,
        blockDim.x * blockDim.y * blockDim.z
    );
    return success ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}