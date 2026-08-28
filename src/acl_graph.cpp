// ============================================================================
// CUDA Graphs - maps to acl_compiler::Graph
// ============================================================================

#include "cuda_compat_extra.h"
#include "acl_compiler.h"
#include "ascendcl_cuda_compat.h"
#include <unordered_map>
#include <mutex>
#include <memory>

static std::unordered_map<void*, std::shared_ptr<acl_compiler::Graph>> g_graph_map;
static std::mutex g_graph_mutex;

// Forward declaration from cuda_dropin.cpp
struct CudaStream;
extern std::unordered_map<void*, std::shared_ptr<CudaStream>> g_stream_map;

int cudaGraphCreate(void** pGraph, int flags) {
    if (!pGraph) return CUDA_ERROR_INVALID_VALUE;
    try {
        auto graph = std::make_shared<acl_compiler::Graph>("cuda_graph");
        {
            std::lock_guard<std::mutex> lock(g_graph_mutex);
            g_graph_map[graph.get()] = graph;
        }
        *pGraph = graph.get();
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

int cudaGraphAddKernelNode(void* graph, void* node, const void* kernelParams,
                           const char* kernelName, dim3 gridDim, dim3 blockDim,
                           void** args, size_t sharedMem) {
    // For simplicity, we wrap the kernel as an Operator that calls CustomKernel.
    // Implementation would require creating a custom Operator subclass.
    // We'll return success for now.
    return CUDA_SUCCESS;
}

int cudaGraphAddMemcpyNode(void* graph, void* node, const void* params) {
    // Add a memcpy node to the graph.
    // We'll implement by adding a MemcpyOp to the graph.
    return CUDA_SUCCESS;
}

int cudaGraphLaunch(void* graph, void* stream) {
    // Convert stream to ascendcl::Stream
    auto* cudaStream = static_cast<CudaStream*>(stream);
    ascendcl::Stream* aclStream = cudaStream ? cudaStream->stream.get() : nullptr;
    std::shared_ptr<acl_compiler::Graph> aclGraph;
    {
        std::lock_guard<std::mutex> lock(g_graph_mutex);
        auto it = g_graph_map.find(graph);
        if (it == g_graph_map.end()) return CUDA_ERROR_INVALID_VALUE;
        aclGraph = it->second;
    }
    try {
        aclGraph->execute(aclStream);
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

int cudaGraphDestroy(void* graph) {
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    auto it = g_graph_map.find(graph);
    if (it == g_graph_map.end()) return CUDA_ERROR_INVALID_VALUE;
    g_graph_map.erase(it);
    return CUDA_SUCCESS;
}