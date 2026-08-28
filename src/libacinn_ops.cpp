// ============================================================================
// ACLNN Operators Implementation - CANN 8.0
// Wraps official CANN operator kernels
// ============================================================================

#include "aclnn_wrapper.h"
#include <iostream>

namespace aclnn {

// ============================================================================
// MATRIX MULTIPLY OPERATOR
// ============================================================================

MatMulOp::MatMulOp(
    ascendcl::TensorDesc* a_desc, void* a_data,
    ascendcl::TensorDesc* b_desc, void* b_data,
    ascendcl::TensorDesc* c_desc, void* c_data,
    float alpha, float beta)
    : a_desc_(a_desc), b_desc_(b_desc), c_desc_(c_desc),
      a_data_(a_data), b_data_(b_data), c_data_(c_data),
      alpha_(alpha), beta_(beta), attr_(nullptr) {
    
    if (!a_desc_ || !b_desc_ || !c_desc_) {
        throw std::invalid_argument("Tensor descriptors cannot be null");
    }
    if (!a_data_ || !b_data_ || !c_data_) {
        throw std::invalid_argument("Data pointers cannot be null");
    }
    
    attr_ = aclopCreateAttr();
    if (attr_) {
        aclopSetAttrFloat(attr_, "alpha", alpha_);
        aclopSetAttrFloat(attr_, "beta", beta_);
    }
    
    std::cout << "[MatMulOp] Created MatMul operator (alpha=" << alpha 
              << ", beta=" << beta << ")" << std::endl;
}

MatMulOp::~MatMulOp() {
    if (attr_) {
        aclopDestroyAttr(attr_);
    }
}

void MatMulOp::execute(aclrtStream stream) {
    try {
        ascendcl::CHECK_ACL(aclopMatMul(
            a_desc_->getHandle(), a_data_,
            b_desc_->getHandle(), b_data_,
            c_desc_->getHandle(), c_data_,
            attr_,
            ACL_ENGINE_SYS,
            ACL_COMPILE_SYS,
            stream
        ));
        std::cout << "[MatMulOp] Executed successfully" << std::endl;
    } catch (const ascendcl::AscendException& e) {
        std::cerr << "[ERROR] MatMulOp execution failed: " << e.what() << std::endl;
        throw;
    }
}

// ============================================================================
// BIAS ADD OPERATOR
// ============================================================================

BiasAddOp::BiasAddOp(
    ascendcl::TensorDesc* input_desc, void* input_data,
    ascendcl::TensorDesc* bias_desc, void* bias_data,
    ascendcl::TensorDesc* output_desc, void* output_data)
    : input_desc_(input_desc), bias_desc_(bias_desc), output_desc_(output_desc),
      input_data_(input_data), bias_data_(bias_data), output_data_(output_data),
      attr_(nullptr) {
    
    if (!input_desc_ || !bias_desc_ || !output_desc_) {
        throw std::invalid_argument("Tensor descriptors cannot be null");
    }
    if (!input_data_ || !bias_data_ || !output_data_) {
        throw std::invalid_argument("Data pointers cannot be null");
    }
    
    attr_ = aclopCreateAttr();
    std::cout << "[BiasAddOp] Created BiasAdd operator" << std::endl;
}

BiasAddOp::~BiasAddOp() {
    if (attr_) {
        aclopDestroyAttr(attr_);
    }
}

void BiasAddOp::execute(aclrtStream stream) {
    try {
        ascendcl::CHECK_ACL(aclopAdd(
            input_desc_->getHandle(), input_data_,
            bias_desc_->getHandle(), bias_data_,
            output_desc_->getHandle(), output_data_,
            attr_,
            ACL_ENGINE_SYS,
            ACL_COMPILE_SYS,
            stream
        ));
        std::cout << "[BiasAddOp] Executed successfully" << std::endl;
    } catch (const ascendcl::AscendException& e) {
        std::cerr << "[ERROR] BiasAddOp execution failed: " << e.what() << std::endl;
        throw;
    }
}

// ============================================================================
// RELU OPERATOR
// ============================================================================

ReluOp::ReluOp(ascendcl::TensorDesc* input_desc, void* input_data,
               ascendcl::TensorDesc* output_desc, void* output_data)
    : input_desc_(input_desc), output_desc_(output_desc),
      input_data_(input_data), output_data_(output_data),
      attr_(nullptr) {
    
    if (!input_desc_ || !output_desc_) {
        throw std::invalid_argument("Tensor descriptors cannot be null");
    }
    if (!input_data_ || !output_data_) {
        throw std::invalid_argument("Data pointers cannot be null");
    }
    
    attr_ = aclopCreateAttr();
    std::cout << "[ReluOp] Created ReLU operator" << std::endl;
}

ReluOp::~ReluOp() {
    if (attr_) {
        aclopDestroyAttr(attr_);
    }
}

void ReluOp::execute(aclrtStream stream) {
    try {
        ascendcl::CHECK_ACL(aclopRelu(
            input_desc_->getHandle(), input_data_,
            output_desc_->getHandle(), output_data_,
            attr_,
            ACL_ENGINE_SYS,
            ACL_COMPILE_SYS,
            stream
        ));
        std::cout << "[ReluOp] Executed successfully" << std::endl;
    } catch (const ascendcl::AscendException& e) {
        std::cerr << "[ERROR] ReluOp execution failed: " << e.what() << std::endl;
        throw;
    }
}

// ============================================================================
// GRAPH EXECUTOR
// ============================================================================

GraphExecutor::GraphExecutor() {
    default_stream_ = std::make_shared<ascendcl::Stream>();
}

GraphExecutor::~GraphExecutor() {
    operators_.clear();
}

void GraphExecutor::addOperator(std::shared_ptr<Operator> op) {
    if (!op) {
        throw std::invalid_argument("Operator cannot be null");
    }
    operators_.push_back(op);
    std::cout << "[GraphExecutor] Added operator: " << op->getName() << std::endl;
}

void GraphExecutor::execute(ascendcl::Stream* stream) {
    if (!default_stream_) {
        throw std::runtime_error("Default stream not initialized");
    }

    ascendcl::Stream* exec_stream = stream ? stream : default_stream_.get();
    execute(exec_stream->getHandle());
}

void GraphExecutor::execute(aclrtStream stream) {
    if (!stream) {
        throw std::invalid_argument("Stream handle cannot be null");
    }

    std::cout << "[GraphExecutor] Executing " << operators_.size() << " operators..." << std::endl;

    for (auto& op : operators_) {
        if (!op) {
            throw std::runtime_error("Null operator encountered");
        }
        op->execute(stream);
    }

    ascendcl::CHECK_ACL(aclrtSynchronizeStream(stream));
    std::cout << "[GraphExecutor] All operators executed" << std::endl;
}

} // namespace aclnn