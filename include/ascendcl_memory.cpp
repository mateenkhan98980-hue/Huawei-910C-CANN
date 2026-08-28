#include "ascendcl_memory.h"
#include <numeric>
#include <chrono>

namespace ascendcl {

void getMemoryInfo(size_t* free, size_t* total) {
    aclrtMemInfo memInfo;
    ASCENDCL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &memInfo));
    if (free) *free = memInfo.free;
    if (total) *total = memInfo.total;
}

MemoryPool& MemoryPool::getInstance() {
    static MemoryPool instance;
    return instance;
}

std::shared_ptr<DeviceMemory> MemoryPool::allocate(size_t size, aclrtStream) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& block : blocks_) {
        if (block.free && block.size >= size) {
            block.free = false;
            return std::make_shared<DeviceMemory>(block.ptr, block.size, true);
        }
    }
    void* ptr = allocChunk(size);
    blocks_.push_back({ptr, size, false});
    return std::make_shared<DeviceMemory>(ptr, size, true);
}

void MemoryPool::defragment() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::sort(blocks_.begin(), blocks_.end(),
              [](const Block& a, const Block& b) { return a.ptr < b.ptr; });
    for (size_t i = 0; i + 1 < blocks_.size(); ) {
        if (blocks_[i].free && blocks_[i+1].free) {
            blocks_[i].size += blocks_[i+1].size;
            blocks_.erase(blocks_.begin() + i + 1);
        } else {
            ++i;
        }
    }
}

std::shared_ptr<DeviceMemory> DeviceMallocAsync(size_t size, aclrtStream) {
    return MemoryPool::getInstance().allocate(size);
}

double benchmarkBandwidth(size_t size, int iterations) {
    auto src = DeviceMalloc(size);
    auto dst = DeviceMalloc(size);
    auto stream = std::make_unique<Stream>();

    std::vector<char> host(size);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        MemcpyHtoD(dst->getData(), host.data(), size, stream.get());
        MemcpyDtoH(host.data(), dst->getData(), size, stream.get());
    }
    stream->synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double totalBytes = 2.0 * size * iterations;
    double seconds = std::chrono::duration<double>(end - start).count();
    return totalBytes / seconds / (1024*1024*1024); // GB/s
}

// allocChunk implementation
void* MemoryPool::allocChunk(size_t size) {
    void* ptr = nullptr;
    ASCENDCL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

} // namespace ascendcl