// ============================================================================
// ACL Operator Compiler - CANN 8.0
// Strict CUDA-compatible Graph compiler
// ============================================================================

#pragma once

#include "ascendcl_wrapper.h"
#include "aclnn_wrapper.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <stdexcept>

namespace acl_compiler {

// ============================================================================
// COMPUTATION GRAPH
// ============================================================================

class Graph {
public:
    Graph(const std::string& name);
    ~Graph();
    
    void addOperator(std::shared_ptr<aclnn::Operator> op);
    void compile();
    void execute(ascendcl::Stream* stream = nullptr);
    
    std::string getName() const { return name_; }
    size_t getOperatorCount() const { return operators_.size(); }
    
private:
    std::string name_;
    std::vector<std::shared_ptr<aclnn::Operator>> operators_;
    bool compiled_;
};

// ============================================================================
// GRAPH BUILDER
// ============================================================================

class GraphBuilder {
public:
    GraphBuilder();
    
    std::shared_ptr<Graph> createGraph(const std::string& name);
    void compileAll();
    void executeAll(ascendcl::Stream* stream = nullptr);
    
private:
    std::vector<std::shared_ptr<Graph>> graphs_;
};

// ============================================================================
// OPTIMIZATION PASS
// ============================================================================

class OptimizationPass {
public:
    virtual ~OptimizationPass() = default;
    virtual void apply(Graph* graph) = 0;
    virtual std::string getName() const = 0;
};

class OperatorFusionPass : public OptimizationPass {
public:
    void apply(Graph* graph) override;
    std::string getName() const override { return "OperatorFusion"; }
};

} // namespace acl_compiler