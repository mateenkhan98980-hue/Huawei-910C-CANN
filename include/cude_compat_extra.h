// ============================================================================
// CUDA Extra APIs - Graphs, Unified Memory, cuBLAS, profiler
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

// ------------------- CUDA Graphs -------------------
int cudaGraphCreate(void** pGraph, int flags);
int cudaGraphAddKernelNode(void* graph, void* node, const void* kernelParams,
                           const char* kernelName, dim3 gridDim, dim3 blockDim,
                           void** args, size_t sharedMem);
int cudaGraphAddMemcpyNode(void* graph, void* node, const void* params);
int cudaGraphLaunch(void* graph, void* stream);
int cudaGraphDestroy(void* graph);

// ------------------- Unified Memory -------------------
int cudaMallocManaged(void** devPtr, size_t size, unsigned int flags = 0);
int cudaFreeManaged(void* ptr);

// ------------------- cuBLAS (stub) -------------------
int cublasCreate(void** handle);
int cublasDestroy(void* handle);
int cublasSgemm(void* handle, int transA, int transB,
                int m, int n, int k,
                const float* alpha,
                const float* A, int lda,
                const float* B, int ldb,
                const float* beta,
                float* C, int ldc);
int cublasHgemm(void* handle, int transA, int transB,
                int m, int n, int k,
                const void* alpha,
                const void* A, int lda,
                const void* B, int ldb,
                const void* beta,
                void* C, int ldc);

// ------------------- cuDNN (stub) -------------------
int cudnnCreate(void** handle);
int cudnnDestroy(void* handle);
int cudnnConvolutionForward(void* handle, void* convDesc,
                            const void* alpha, const void* xDesc, const void* x,
                            const void* wDesc, const void* w,
                            const void* beta, void* yDesc, void* y);

// ------------------- Profiler -------------------
int cudaProfilerStart();
int cudaProfilerStop();
void cudaProfilerRecordEvent(const char* name);

} // extern "C"