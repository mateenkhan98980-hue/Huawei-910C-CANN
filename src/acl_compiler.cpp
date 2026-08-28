#include "acl_compiler.h"
#include <iostream>
#include <unordered_map>

namespace acl_compiler {

class CompilerBackend {
public:
    static CompilerBackend& getInstance() {
        static CompilerBackend instance;
        return instance;
    }
    
    void registerOptimization(const std::string& name, 
                              std::shared_ptr<OptimizationPass> pass) {
        if (!pass) {
            throw std::invalid_argument("Optimization pass cannot be null");
        }
        optimizations_[name] = pass;
        std::cout << "[Compiler] Registered optimization: " << name << std::endl;
    }
    
    void applyOptimization(const std::string& name, Graph* graph) {
        if (!graph) {
            throw std::invalid_argument("Graph cannot be null");
        }
        auto it = optimizations_.find(name);
        if (it != optimizations_.end()) {
            it->second->apply(graph);
        }
    }
    
private:
    CompilerBackend() = default;
    std::unordered_map<std::string, std::shared_ptr<OptimizationPass>> optimizations_;
};

} // namespace acl_compiler