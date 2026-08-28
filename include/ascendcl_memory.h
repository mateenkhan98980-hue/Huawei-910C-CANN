#pragma once

#include "ascendcl_wrapper.h"
#include <memory>
#include <vector>
#include <mutex>

namespace ascendcl {

void getMemoryInfo(size_t* free, size_t* total);

class MemoryPool {
public:
    static MemoryPool& getInstance();
    std::shared_ptr<DeviceMemory> allocate(size_t size, aclrtStream stream = nullptr);
    void defragment();
    void printStats() const;
private:
    struct Block { void* ptr; size_t size; bool free; };
    std::vector<Block> blocks_;
    std::mutex mutex_;
    void* allocChunk(size_t size);
};

std::shared_ptr<DeviceMemory> DeviceMallocAsync(size_t size, aclrtStream stream = nullptr);
double benchmarkBandwidth(size_t size, int iterations = 10);

} // namespace ascendcl