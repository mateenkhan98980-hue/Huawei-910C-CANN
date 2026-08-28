// ============================================================================
// Profiler - simple CPU/GPU timing
// ============================================================================

#include "cuda_compat_extra.h"
#include "ascendcl_cuda_compat.h"
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <mutex>

static bool g_profiler_enabled = false;
static std::unordered_map<std::string, std::chrono::steady_clock::time_point> g_events;
static std::mutex g_profiler_mutex;

int cudaProfilerStart() {
    g_profiler_enabled = true;
    return 0;
}

int cudaProfilerStop() {
    g_profiler_enabled = false;
    return 0;
}

void cudaProfilerRecordEvent(const char* name) {
    if (!g_profiler_enabled) return;
    std::lock_guard<std::mutex> lock(g_profiler_mutex);
    auto now = std::chrono::steady_clock::now();
    g_events[name] = now;
    std::cout << "[Profiler] Event recorded: " << name << std::endl;
}