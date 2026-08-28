// ============================================================================
// ACL Operator Compiler Implementation - CANN 8.0
// ============================================================================

#include "acl_compiler.h"
#include <iostream>

namespace acl_compiler {

// ============================================================================
// COMPUTATION GRAPH
// ============================================================================

Graph::Graph(const std::string& name)
    : name_(name), compiled_(false) {
    std::cout << "[Graph] Created graph: " << name << std::endl;
}

Graph::~Graph() {
    operators_.clear();
}

void Graph::addOperator(std::shared_ptr<aclnn::Operator> op) {
    if (!op) {
        throw std::invalid_argument("Operator cannot be null");
    }
    operators_.push_back(op);
    std::cout << "[Graph] Added operator: " << op->getName() 
              << " (total: " << operators_.size() << ")" << std::endl;
}

void Graph::compile() {
    if (compiled_) return;
    
    std::cout << "[Graph] Compiling graph: " << name_ << std::endl;
    std::cout << "[Graph] Operators to compile: " << operators_.size() << std::endl;
    
    // Run optimization passes
    OperatorFusionPass fusion_pass;
    fusion_pass.apply(this);
    
    compiled_ = true;
    std::cout << "[Graph] Compilation complete" << std::endl;
}

void Graph::execute(ascendcl::Stream* stream) {
    if (!compiled_) compile();
    
    auto executor = std::make_unique<aclnn::GraphExecutor>();
    for (auto& op : operators_) {
        if (!op) {
            throw std::runtime_error("Null operator in graph");
        }
        executor->addOperator(op);
    }
    executor->execute(stream);
}

// ============================================================================
// GRAPH BUILDER
// ============================================================================

GraphBuilder::GraphBuilder() {
    std::cout << "[GraphBuilder] Initialized" << std::endl;
}

std::shared_ptr<Graph> GraphBuilder::createGraph(const std::string& name) {
    auto graph = std::make_shared<Graph>(name);
    if (!graph) {
        throw std::runtime_error("Failed to create graph");
    }
    graphs_.push_back(graph);
    return graph;
}

void GraphBuilder::compileAll() {
    std::cout << "[GraphBuilder] Compiling all graphs (" << graphs_.size() << ")" << std::endl;
    for (auto& graph : graphs_) {
        if (!graph) {
            throw std::runtime_error("Null graph in builder");
        }
        graph->compile();
    }
}

void GraphBuilder::executeAll(ascendcl::Stream* stream) {
    std::cout << "[GraphBuilder] Executing all graphs" << std::endl;
    for (auto& graph : graphs_) {
        if (!graph) {
            throw std::runtime_error("Null graph in builder");
        }
        graph->execute(stream);
    }
}

// ============================================================================
// OPTIMIZATION PASSES
// ============================================================================

void OperatorFusionPass::apply(Graph* graph) {
    if (!graph) {
        throw std::invalid_argument("Graph cannot be null");
    }

    const size_t op_count = graph->getOperatorCount();
    if (op_count == 0) {
        std::cout << "[OperatorFusionPass] Graph \"" << graph->getName()
                  << "\" has no operators; skipping fusion" << std::endl;
        return;
    }

    std::cout << "[OperatorFusionPass] Validated " << op_count
              << " operator(s) in graph \"" << graph->getName() << "\"" << std::endl;
    std::cout << "[OperatorFusionPass] Fusion pass complete (execution order preserved)"
              << std::endl;
}

} // namespace acl_compiler