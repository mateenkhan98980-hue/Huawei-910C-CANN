// ============================================================================
// ACLNN Operators Implementation (Full) - CANN 8.0
// Includes: MatMul, BiasAdd, ReLU, Conv2d, BatchNorm, MaxPool, Softmax,
// Add, Sub, Mul, Div, LeakyReLU, Sigmoid, Gelu, Dropout, LayerNorm, Attention.
// Uses ACLNN APIs for high performance.
// ============================================================================

#include "aclnn_wrapper.h"
#include "ascendcl_wrapper.h"
#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <acl/acl.h>
#include <acl/acl_nn.h>

namespace aclnn {

// ============================================================================
// Helper Functions for ACLNN Tensor Creation
// ============================================================================

// Convert ascendcl::TensorDesc + data to aclTensor*
static aclTensor* createAclTensor(ascendcl::TensorDesc* desc, void* data) {
    if (!desc || !data) return nullptr;
    aclTensor* tensor = nullptr;
    aclTensorDesc* innerDesc = desc->getHandle();
    aclError ret = aclCreateTensor(innerDesc, data, &tensor);
    if (ret != ACL_SUCCESS) {
        throw ascendcl::AscendException(ret, "Failed to create aclTensor");
    }
    return tensor;
}

// RAII wrapper for aclTensor
struct AclTensorDeleter {
    void operator()(aclTensor* t) const {
        if (t) aclDestroyTensor(t);
    }
};
using AclTensorPtr = std::unique_ptr<aclTensor, AclTensorDeleter>;

// Helper to set common attributes (for aclop-based ops)
static void setCommonAttr(aclopAttr* attr, float alpha, float beta) {
    if (attr) {
        aclopSetAttrFloat(attr, "alpha", alpha);
        aclopSetAttrFloat(attr, "beta", beta);
    }
}

// ============================================================================
// MATRIX MULTIPLY OPERATOR (MatMul)
// ============================================================================
class MatMulOp : public Operator {
public:
    MatMulOp(
        ascendcl::TensorDesc* a_desc, void* a_data,
        ascendcl::TensorDesc* b_desc, void* b_data,
        ascendcl::TensorDesc* c_desc, void* c_data,
        float alpha = 1.0f, float beta = 1.0f)
        : a_desc_(a_desc), b_desc_(b_desc), c_desc_(c_desc),
          a_data_(a_data), b_data_(b_data), c_data_(c_data),
          alpha_(alpha), beta_(beta) {
        if (!a_desc_ || !b_desc_ || !c_desc_)
            throw std::invalid_argument("Descriptors cannot be null");
        if (!a_data_ || !b_data_ || !c_data_)
            throw std::invalid_argument("Data pointers cannot be null");
        std::cout << "[MatMulOp] Created (alpha=" << alpha << ", beta=" << beta << ")" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr aTensor(createAclTensor(a_desc_, a_data_));
        AclTensorPtr bTensor(createAclTensor(b_desc_, b_data_));
        AclTensorPtr cTensor(createAclTensor(c_desc_, c_data_));

        size_t workspaceSize = 1024 * 1024; // 1 MB default
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnMatMul(
            aTensor.get(), bTensor.get(), cTensor.get(),
            alpha_, beta_, stream, workspace, workspaceSize);

        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnMatMul failed");
        }
        std::cout << "[MatMulOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "MatMul"; }

private:
    ascendcl::TensorDesc* a_desc_;
    ascendcl::TensorDesc* b_desc_;
    ascendcl::TensorDesc* c_desc_;
    void* a_data_;
    void* b_data_;
    void* c_data_;
    float alpha_, beta_;
};

// ============================================================================
// BIAS ADD OPERATOR (BiasAdd)
// ============================================================================
class BiasAddOp : public Operator {
public:
    BiasAddOp(
        ascendcl::TensorDesc* input_desc, void* input_data,
        ascendcl::TensorDesc* bias_desc, void* bias_data,
        ascendcl::TensorDesc* output_desc, void* output_data)
        : input_desc_(input_desc), bias_desc_(bias_desc), output_desc_(output_desc),
          input_data_(input_data), bias_data_(bias_data), output_data_(output_data) {
        if (!input_desc_ || !bias_desc_ || !output_desc_)
            throw std::invalid_argument("Descriptors cannot be null");
        if (!input_data_ || !bias_data_ || !output_data_)
            throw std::invalid_argument("Data pointers cannot be null");
        std::cout << "[BiasAddOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr inputTensor(createAclTensor(input_desc_, input_data_));
        AclTensorPtr biasTensor(createAclTensor(bias_desc_, bias_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnAdd(
            inputTensor.get(), biasTensor.get(), outTensor.get(),
            stream, workspace, workspaceSize);

        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnAdd (BiasAdd) failed");
        }
        std::cout << "[BiasAddOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "BiasAdd"; }

private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* bias_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* bias_data_;
    void* output_data_;
};

// ============================================================================
// RELU OPERATOR (ReLU)
// ============================================================================
class ReluOp : public Operator {
public:
    ReluOp(ascendcl::TensorDesc* input_desc, void* input_data,
           ascendcl::TensorDesc* output_desc, void* output_data)
        : input_desc_(input_desc), output_desc_(output_desc),
          input_data_(input_data), output_data_(output_data) {
        if (!input_desc_ || !output_desc_)
            throw std::invalid_argument("Descriptors cannot be null");
        if (!input_data_ || !output_data_)
            throw std::invalid_argument("Data pointers cannot be null");
        std::cout << "[ReluOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr inputTensor(createAclTensor(input_desc_, input_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnRelu(
            inputTensor.get(), outTensor.get(),
            stream, workspace, workspaceSize);

        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnRelu failed");
        }
        std::cout << "[ReluOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "ReLU"; }

private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* output_data_;
};

// ============================================================================
// CONV2D OPERATOR
// ============================================================================
class Conv2dOp : public Operator {
public:
    Conv2dOp(
        ascendcl::TensorDesc* input_desc, void* input_data,
        ascendcl::TensorDesc* weight_desc, void* weight_data,
        ascendcl::TensorDesc* output_desc, void* output_data,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& pads,
        int64_t dilations = 1, int64_t groups = 1)
        : input_desc_(input_desc), weight_desc_(weight_desc), output_desc_(output_desc),
          input_data_(input_data), weight_data_(weight_data), output_data_(output_data),
          strides_(strides), pads_(pads), dilations_(dilations), groups_(groups) {
        if (!input_desc_ || !weight_desc_ || !output_desc_)
            throw std::invalid_argument("Descriptors cannot be null");
        if (!input_data_ || !weight_data_ || !output_data_)
            throw std::invalid_argument("Data pointers cannot be null");
        std::cout << "[Conv2dOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr inputTensor(createAclTensor(input_desc_, input_data_));
        AclTensorPtr weightTensor(createAclTensor(weight_desc_, weight_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        aclnnConv2dDescriptor convDesc;
        aclnnCreateConv2dDescriptor(&convDesc);
        aclnnSetConv2dDescriptorStrides(convDesc, strides_.data(), strides_.size());
        aclnnSetConv2dDescriptorPads(convDesc, pads_.data(), pads_.size());
        aclnnSetConv2dDescriptorDilations(convDesc, dilations_);
        aclnnSetConv2dDescriptorGroups(convDesc, groups_);

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnConv2d(
            inputTensor.get(), weightTensor.get(), nullptr,
            outTensor.get(), convDesc, stream, workspace, workspaceSize);

        aclnnDestroyConv2dDescriptor(convDesc);
        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnConv2d failed");
        }
        std::cout << "[Conv2dOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "Conv2d"; }

private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* weight_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* weight_data_;
    void* output_data_;
    std::vector<int64_t> strides_;
    std::vector<int64_t> pads_;
    int64_t dilations_;
    int64_t groups_;
};

// ============================================================================
// BATCH NORM OPERATOR
// ============================================================================
class BatchNormOp : public Operator {
public:
    BatchNormOp(
        ascendcl::TensorDesc* input_desc, void* input_data,
        ascendcl::TensorDesc* scale_desc, void* scale_data,
        ascendcl::TensorDesc* offset_desc, void* offset_data,
        ascendcl::TensorDesc* mean_desc, void* mean_data,
        ascendcl::TensorDesc* variance_desc, void* variance_data,
        ascendcl::TensorDesc* output_desc, void* output_data,
        float epsilon = 1e-5f, float momentum = 0.9f)
        : input_desc_(input_desc), scale_desc_(scale_desc), offset_desc_(offset_desc),
          mean_desc_(mean_desc), variance_desc_(variance_desc), output_desc_(output_desc),
          input_data_(input_data), scale_data_(scale_data), offset_data_(offset_data),
          mean_data_(mean_data), variance_data_(variance_data), output_data_(output_data),
          epsilon_(epsilon), momentum_(momentum) {
        // Validation...
        std::cout << "[BatchNormOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr inputTensor(createAclTensor(input_desc_, input_data_));
        AclTensorPtr scaleTensor(createAclTensor(scale_desc_, scale_data_));
        AclTensorPtr offsetTensor(createAclTensor(offset_desc_, offset_data_));
        AclTensorPtr meanTensor(createAclTensor(mean_desc_, mean_data_));
        AclTensorPtr varTensor(createAclTensor(variance_desc_, variance_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        aclnnBatchNormDescriptor bnDesc;
        aclnnCreateBatchNormDescriptor(&bnDesc);
        aclnnSetBatchNormDescriptorEpsilon(bnDesc, epsilon_);
        aclnnSetBatchNormDescriptorMomentum(bnDesc, momentum_);

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnBatchNorm(
            inputTensor.get(), scaleTensor.get(), offsetTensor.get(),
            meanTensor.get(), varTensor.get(), outTensor.get(),
            bnDesc, stream, workspace, workspaceSize);

        aclnnDestroyBatchNormDescriptor(bnDesc);
        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnBatchNorm failed");
        }
        std::cout << "[BatchNormOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "BatchNorm"; }

private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* scale_desc_;
    ascendcl::TensorDesc* offset_desc_;
    ascendcl::TensorDesc* mean_desc_;
    ascendcl::TensorDesc* variance_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* scale_data_;
    void* offset_data_;
    void* mean_data_;
    void* variance_data_;
    void* output_data_;
    float epsilon_;
    float momentum_;
};

// ============================================================================
// MAX POOL OPERATOR
// ============================================================================
class MaxPoolOp : public Operator {
public:
    MaxPoolOp(
        ascendcl::TensorDesc* input_desc, void* input_data,
        ascendcl::TensorDesc* output_desc, void* output_data,
        const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& pads)
        : input_desc_(input_desc), output_desc_(output_desc),
          input_data_(input_data), output_data_(output_data),
          kernel_size_(kernel_size), strides_(strides), pads_(pads) {
        // Validation...
        std::cout << "[MaxPoolOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr inputTensor(createAclTensor(input_desc_, input_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        aclnnPoolingDescriptor poolDesc;
        aclnnCreatePoolingDescriptor(&poolDesc);
        aclnnSetPoolingDescriptorMode(poolDesc, ACLNN_POOLING_MAX);
        aclnnSetPoolingDescriptorKernelSize(poolDesc, kernel_size_.data(), kernel_size_.size());
        aclnnSetPoolingDescriptorStrides(poolDesc, strides_.data(), strides_.size());
        aclnnSetPoolingDescriptorPads(poolDesc, pads_.data(), pads_.size());

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnPooling(
            inputTensor.get(), outTensor.get(), poolDesc,
            stream, workspace, workspaceSize);

        aclnnDestroyPoolingDescriptor(poolDesc);
        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnMaxPool failed");
        }
        std::cout << "[MaxPoolOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "MaxPool"; }

private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* output_data_;
    std::vector<int64_t> kernel_size_;
    std::vector<int64_t> strides_;
    std::vector<int64_t> pads_;
};

// ============================================================================
// SOFTMAX OPERATOR
// ============================================================================
class SoftmaxOp : public Operator {
public:
    SoftmaxOp(ascendcl::TensorDesc* input_desc, void* input_data,
              ascendcl::TensorDesc* output_desc, void* output_data,
              int64_t axis = -1)
        : input_desc_(input_desc), output_desc_(output_desc),
          input_data_(input_data), output_data_(output_data), axis_(axis) {
        std::cout << "[SoftmaxOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr inputTensor(createAclTensor(input_desc_, input_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        aclnnSoftmaxDescriptor smDesc;
        aclnnCreateSoftmaxDescriptor(&smDesc);
        aclnnSetSoftmaxDescriptorAxis(smDesc, axis_);

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnSoftmax(
            inputTensor.get(), outTensor.get(), smDesc,
            stream, workspace, workspaceSize);

        aclnnDestroySoftmaxDescriptor(smDesc);
        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnSoftmax failed");
        }
        std::cout << "[SoftmaxOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "Softmax"; }

private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* output_data_;
    int64_t axis_;
};

// ============================================================================
// ADD OPERATOR
// ============================================================================
class AddOp : public Operator {
public:
    AddOp(ascendcl::TensorDesc* a_desc, void* a_data,
          ascendcl::TensorDesc* b_desc, void* b_data,
          ascendcl::TensorDesc* output_desc, void* output_data,
          float alpha = 1.0f, float beta = 1.0f)
        : a_desc_(a_desc), b_desc_(b_desc), output_desc_(output_desc),
          a_data_(a_data), b_data_(b_data), output_data_(output_data),
          alpha_(alpha), beta_(beta) {
        std::cout << "[AddOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr aTensor(createAclTensor(a_desc_, a_data_));
        AclTensorPtr bTensor(createAclTensor(b_desc_, b_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnAdd(
            aTensor.get(), bTensor.get(), outTensor.get(),
            stream, workspace, workspaceSize);

        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnAdd failed");
        }
        std::cout << "[AddOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "Add"; }

private:
    ascendcl::TensorDesc* a_desc_;
    ascendcl::TensorDesc* b_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* a_data_;
    void* b_data_;
    void* output_data_;
    float alpha_, beta_;
};

// ============================================================================
// SUB OPERATOR
// ============================================================================
class SubOp : public Operator {
public:
    SubOp(ascendcl::TensorDesc* a_desc, void* a_data,
          ascendcl::TensorDesc* b_desc, void* b_data,
          ascendcl::TensorDesc* output_desc, void* output_data,
          float alpha = 1.0f, float beta = 1.0f)
        : a_desc_(a_desc), b_desc_(b_desc), output_desc_(output_desc),
          a_data_(a_data), b_data_(b_data), output_data_(output_data),
          alpha_(alpha), beta_(beta) {
        std::cout << "[SubOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr aTensor(createAclTensor(a_desc_, a_data_)
        AclTensorPtr bTensor(createAclTensor(b_desc_, b_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnSub(
            aTensor.get(), bTensor.get(), outTensor.get(),
            stream, workspace, workspaceSize);

        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnSub failed");
        }
        std::cout << "[SubOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "Sub"; }

private:
    ascendcl::TensorDesc* a_desc_;
    ascendcl::TensorDesc* b_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* a_data_;
    void* b_data_;
    void* output_data_;
    float alpha_, beta_;
};

// ============================================================================
// MUL OPERATOR
// ============================================================================
class MulOp : public Operator {
public:
    MulOp(ascendcl::TensorDesc* a_desc, void* a_data,
          ascendcl::TensorDesc* b_desc, void* b_data,
          ascendcl::TensorDesc* output_desc, void* output_data,
          float alpha = 1.0f, float beta = 1.0f)
        : a_desc_(a_desc), b_desc_(b_desc), output_desc_(output_desc),
          a_data_(a_data), b_data_(b_data), output_data_(output_data),
          alpha_(alpha), beta_(beta) {
        std::cout << "[MulOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr aTensor(createAclTensor(a_desc_, a_data_));
        AclTensorPtr bTensor(createAclTensor(b_desc_, b_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnMul(
            aTensor.get(), bTensor.get(), outTensor.get(),
            stream, workspace, workspaceSize);

        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnMul failed");
        }
        std::cout << "[MulOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "Mul"; }

private:
    ascendcl::TensorDesc* a_desc_;
    ascendcl::TensorDesc* b_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* a_data_;
    void* b_data_;
    void* output_data_;
    float alpha_, beta_;
};

// ============================================================================
// DIV OPERATOR
// ============================================================================
class DivOp : public Operator {
public:
    DivOp(ascendcl::TensorDesc* a_desc, void* a_data,
          ascendcl::TensorDesc* b_desc, void* b_data,
          ascendcl::TensorDesc* output_desc, void* output_data,
          float alpha = 1.0f, float beta = 1.0f)
        : a_desc_(a_desc), b_desc_(b_desc), output_desc_(output_desc),
          a_data_(a_data), b_data_(b_data), output_data_(output_data),
          alpha_(alpha), beta_(beta) {
        std::cout << "[DivOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr aTensor(createAclTensor(a_desc_, a_data_));
        AclTensorPtr bTensor(createAclTensor(b_desc_, b_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnDiv(
            aTensor.get(), bTensor.get(), outTensor.get(),
            stream, workspace, workspaceSize);

        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnDiv failed");
        }
        std::cout << "[DivOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "Div"; }

private:
    ascendcl::TensorDesc* a_desc_;
    ascendcl::TensorDesc* b_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* a_data_;
    void* b_data_;
    void* output_data_;
    float alpha_, beta_;
};

// ============================================================================
// LEAKY RELU OPERATOR
// ============================================================================
class LeakyReluOp : public Operator {
public:
    LeakyReluOp(ascendcl::TensorDesc* input_desc, void* input_data,
                ascendcl::TensorDesc* output_desc, void* output_data,
                float negative_slope = 0.01f)
        : input_desc_(input_desc), output_desc_(output_desc),
          input_data_(input_data), output_data_(output_data),
          negative_slope_(negative_slope) {
        std::cout << "[LeakyReluOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr inputTensor(createAclTensor(input_desc_, input_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnLeakyRelu(
            inputTensor.get(), outTensor.get(), negative_slope_,
            stream, workspace, workspaceSize);

        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnLeakyRelu failed");
        }
        std::cout << "[LeakyReluOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "LeakyRelu"; }

private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* output_data_;
    float negative_slope_;
};

// ============================================================================
// SIGMOID OPERATOR
// ============================================================================
class SigmoidOp : public Operator {
public:
    SigmoidOp(ascendcl::TensorDesc* input_desc, void* input_data,
              ascendcl::TensorDesc* output_desc, void* output_data)
        : input_desc_(input_desc), output_desc_(output_desc),
          input_data_(input_data), output_data_(output_data) {
        std::cout << "[SigmoidOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr inputTensor(createAclTensor(input_desc_, input_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));

        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclnnStatus status = aclnnSigmoid(
            inputTensor.get(), outTensor.get(),
            stream, workspace, workspaceSize);

        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnSigmoid failed");
        }
        std::cout << "[SigmoidOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "Sigmoid"; }

private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* output_data_;
};

// ============================================================================
// GELU OPERATOR
// ============================================================================
class GeluOp : public Operator {
public:
    GeluOp(ascendcl::TensorDesc* input_desc, void* input_data,
           ascendcl::TensorDesc* output_desc, void* output_data)
        : input_desc_(input_desc), output_desc_(output_desc),
          input_data_(input_data), output_data_(output_data) {
        std::cout << "[GeluOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr inputTensor(createAclTensor(input_desc_, input_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));
        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnStatus status = aclnnGelu(inputTensor.get(), outTensor.get(), stream, workspace, workspaceSize);
        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnGelu failed");
        }
        std::cout << "[GeluOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "Gelu"; }

private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* output_data_;
};

// ============================================================================
// DROPOUT OPERATOR
// ============================================================================
class DropoutOp : public Operator {
public:
    DropoutOp(ascendcl::TensorDesc* input_desc, void* input_data,
              ascendcl::TensorDesc* output_desc, void* output_data,
              float keep_prob = 0.5f, float seed = 0.0f)
        : input_desc_(input_desc), output_desc_(output_desc),
          input_data_(input_data), output_data_(output_data),
          keep_prob_(keep_prob), seed_(seed) {
        std::cout << "[DropoutOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr inputTensor(createAclTensor(input_desc_, input_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));
        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnStatus status = aclnnDropout(inputTensor.get(), outTensor.get(), keep_prob_, seed_, stream, workspace, workspaceSize);
        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnDropout failed");
        }
        std::cout << "[DropoutOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "Dropout"; }

private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* output_data_;
    float keep_prob_;
    float seed_;
};

// ============================================================================
// LAYER NORM OPERATOR
// ============================================================================
class LayerNormOp : public Operator {
public:
    LayerNormOp(ascendcl::TensorDesc* input_desc, void* input_data,
                ascendcl::TensorDesc* gamma_desc, void* gamma_data,
                ascendcl::TensorDesc* beta_desc, void* beta_data,
                ascendcl::TensorDesc* output_desc, void* output_data,
                float epsilon = 1e-5f)
        : input_desc_(input_desc), gamma_desc_(gamma_desc), beta_desc_(beta_desc),
          output_desc_(output_desc), input_data_(input_data),
          gamma_data_(gamma_data), beta_data_(beta_data),
          output_data_(output_data), epsilon_(epsilon) {
        std::cout << "[LayerNormOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr inputTensor(createAclTensor(input_desc_, input_data_));
        AclTensorPtr gammaTensor(createAclTensor(gamma_desc_, gamma_data_));
        AclTensorPtr betaTensor(createAclTensor(beta_desc_, beta_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));
        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnStatus status = aclnnLayerNorm(inputTensor.get(), gammaTensor.get(), betaTensor.get(),
                                            outTensor.get(), epsilon_, stream, workspace, workspaceSize);
        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnLayerNorm failed");
        }
        std::cout << "[LayerNormOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "LayerNorm"; }

private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* gamma_desc_;
    ascendcl::TensorDesc* beta_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* gamma_data_;
    void* beta_data_;
    void* output_data_;
    float epsilon_;
};

// ============================================================================
// ATTENTION OPERATOR (simplified)
// ============================================================================
class AttentionOp : public Operator {
public:
    AttentionOp(ascendcl::TensorDesc* query_desc, void* query_data,
                ascendcl::TensorDesc* key_desc, void* key_data,
                ascendcl::TensorDesc* value_desc, void* value_data,
                ascendcl::TensorDesc* output_desc, void* output_data,
                float scale = 1.0f)
        : query_desc_(query_desc), key_desc_(key_desc), value_desc_(value_desc),
          output_desc_(output_desc), query_data_(query_data),
          key_data_(key_data), value_data_(value_data),
          output_data_(output_data), scale_(scale) {
        std::cout << "[AttentionOp] Created" << std::endl;
    }

    void execute(aclrtStream stream) override {
        AclTensorPtr qTensor(createAclTensor(query_desc_, query_data_));
        AclTensorPtr kTensor(createAclTensor(key_desc_, key_data_));
        AclTensorPtr vTensor(createAclTensor(value_desc_, value_data_));
        AclTensorPtr outTensor(createAclTensor(output_desc_, output_data_));
        size_t workspaceSize = 1024 * 1024;
        void* workspace = nullptr;
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnStatus status = aclnnAttention(qTensor.get(), kTensor.get(), vTensor.get(),
                                            outTensor.get(), scale_, stream, workspace, workspaceSize);
        aclrtFree(workspace);
        if (status != ACLNN_SUCCESS) {
            throw ascendcl::AscendException(ACL_ERROR_INVALID_PARAM, "aclnnAttention failed");
        }
        std::cout << "[AttentionOp] Executed (ACLNN)" << std::endl;
    }

    std::string getName() const override { return "Attention"; }

private:
    ascendcl::TensorDesc* query_desc_;
    ascendcl::TensorDesc* key_desc_;
    ascendcl::TensorDesc* value_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* query_data_;
    void* key_data_;
    void* value_data_;
    void* output_data_;
    float scale_;
};

// ============================================================================
// FACTORY FUNCTION: createOperator (with all operators)
// ============================================================================
std::shared_ptr<Operator> createOperator(
    const std::string& name,
    const std::vector<ascendcl::TensorDesc*>& descs,
    const std::vector<void*>& data,
    float alpha = 1.0f,
    float beta = 1.0f,
    const std::vector<int64_t>& extra_params = {})
{
    if (name == "MatMul") {
        if (descs.size() < 3 || data.size() < 3)
            throw std::invalid_argument("MatMul requires 3 descriptors and data");
        return std::make_shared<MatMulOp>(descs[0], data[0], descs[1], data[1], descs[2], data[2], alpha, beta);
    } else if (name == "BiasAdd") {
        if (descs.size() < 3 || data.size() < 3)
            throw std::invalid_argument("BiasAdd requires 3 descriptors and data");
        return std::make_shared<BiasAddOp>(descs[0], data[0], descs[1], data[1], descs[2], data[2]);
    } else if (name == "ReLU") {
        if (descs.size() < 2 || data.size() < 2)
            throw std::invalid_argument("ReLU requires 2 descriptors and data");
        return std::make_shared<ReluOp>(descs[0], data[0], descs[1], data[1]);
    } else if (name == "Conv2d") {
        if (descs.size() < 3 || data.size() < 3)
            throw std::invalid_argument("Conv2d requires 3 descriptors and data");
        std::vector<int64_t> strides = {1,1,1,1};
        std::vector<int64_t> pads = {0,0,0,0};
        int64_t dilations = 1, groups = 1;
        if (extra_params.size() >= 8) {
            strides = {extra_params[0], extra_params[1], extra_params[2], extra_params[3]};
            pads = {extra_params[4], extra_params[5], extra_params[6], extra_params[7]};
        }
        if (extra_params.size() >= 10) {
            dilations = extra_params[8];
            groups = extra_params[9];
        }
        return std::make_shared<Conv2dOp>(descs[0], data[0], descs[1], data[1], descs[2], data[2],
                                          strides, pads, dilations, groups);
    } else if (name == "BatchNorm") {
        if (descs.size() < 6 || data.size() < 6)
            throw std::invalid_argument("BatchNorm requires 6 descriptors and data");
        float epsilon = 1e-5f, momentum = 0.9f;
        if (extra_params.size() >= 2) {
            epsilon = static_cast<float>(extra_params[0]);
            momentum = static_cast<float>(extra_params[1]);
        }
        return std::make_shared<BatchNormOp>(
            descs[0], data[0], descs[1], data[1], descs[2], data[2],
            descs[3], data[3], descs[4], data[4], descs[5], data[5],
            epsilon, momentum);
    } else if (name == "MaxPool") {
        if (descs.size() < 2 || data.size() < 2)
            throw std::invalid_argument("MaxPool requires 2 descriptors and data");
        std::vector<int64_t> kernel_size = {2,2};
        std::vector<int64_t> strides = {2,2};
        std::vector<int64_t> pads = {0,0};
        if (extra_params.size() >= 6) {
            kernel_size = {extra_params[0], extra_params[1]};
            strides = {extra_params[2], extra_params[3]};
            pads = {extra_params[4], extra_params[5]};
        }
        return std::make_shared<MaxPoolOp>(descs[0], data[0], descs[1], data[1],
                                           kernel_size, strides, pads);
    } else if (name == "Softmax") {
        if (descs.size() < 2 || data.size() < 2)
            throw std::invalid_argument("Softmax requires 2 descriptors and data");
        int64_t axis = -1;
        if (!extra_params.empty()) axis = extra_params[0];
        return std::make_shared<SoftmaxOp>(descs[0], data[0], descs[1], data[1], axis);
    } else if (name == "Add") {
        if (descs.size() < 3 || data.size() < 3)
            throw std::invalid_argument("Add requires 3 descriptors and data");
        return std::make_shared<AddOp>(descs[0], data[0], descs[1], data[1], descs[2], data[2], alpha, beta);
    } else if (name == "Sub") {
        if (descs.size() < 3 || data.size() < 3)
            throw std::invalid_argument("Sub requires 3 descriptors and data");
        return std::make_shared<SubOp>(descs[0], data[0], descs[1], data[1], descs[2], data[2], alpha, beta);
    } else if (name == "Mul") {
        if (descs.size() < 3 || data.size() < 3)
            throw std::invalid_argument("Mul requires 3 descriptors and data");
        return std::make_shared<MulOp>(descs[0], data[0], descs[1], data[1], descs[2], data[2], alpha, beta);
    } else if (name == "Div") {
        if (descs.size() < 3 || data.size() < 3)
            throw std::invalid_argument("Div requires 3 descriptors and data");
        return std::make_shared<DivOp>(descs[0], data[0], descs[1], data[1], descs[2], data[2], alpha, beta);
    } else if (name == "LeakyRelu") {
        if (descs.size() < 2 || data.size() < 2)
            throw std::invalid_argument("LeakyRelu requires 2 descriptors and data");
        float negative_slope = 0.01f;
        if (!extra_params.empty()) negative_slope = static_cast<float>(extra_params[0]);
        return std::make_shared<LeakyReluOp>(descs[0], data[0], descs[1], data[1], negative_slope);
    } else if (name == "Sigmoid") {
        if (descs.size() < 2 || data.size() < 2)
            throw std::invalid_argument("Sigmoid requires 2 descriptors and data");
        return std::make_shared<SigmoidOp>(descs[0], data[0], descs[1], data[1]);
    } else if (name == "Gelu") {
        if (descs.size() < 2 || data.size() < 2)
            throw std::invalid_argument("Gelu requires 2 descriptors and data");
        return std::make_shared<GeluOp>(descs[0], data[0], descs[1], data[1]);
    } else if (name == "Dropout") {
        if (descs.size() < 2 || data.size() < 2)
            throw std::invalid_argument("Dropout requires 2 descriptors and data");
        float keep_prob = 0.5f;
        float seed = 0.0f;
        if (extra_params.size() >= 1) keep_prob = static_cast<float>(extra_params[0]);
        if (extra_params.size() >= 2) seed = static_cast<float>(extra_params[1]);
        return std::make_shared<DropoutOp>(descs[0], data[0], descs[1], data[1], keep_prob, seed);
    } else if (name == "LayerNorm") {
        if (descs.size() < 5 || data.size() < 5)
            throw std::invalid_argument("LayerNorm requires 5 descriptors and data");
        float epsilon = 1e-5f;
        if (!extra_params.empty()) epsilon = static_cast<float>(extra_params[0]);
        return std::make_shared<LayerNormOp>(descs[0], data[0], descs[1], data[1], descs[2], data[2],
                                             descs[3], data[3], descs[4], data[4], epsilon);
    } else if (name == "Attention") {
        if (descs.size() < 4 || data.size() < 4)
            throw std::invalid_argument("Attention requires 4 descriptors and data");
        float scale = 1.0f;
        if (!extra_params.empty()) scale = static_cast<float>(extra_params[0]);
        return std::make_shared<AttentionOp>(descs[0], data[0], descs[1], data[1], descs[2], data[2],
                                             descs[3], data[3], scale);
    } else {
        throw std::invalid_argument("Unknown operator name: " + name);
    }
}

// ============================================================================
// GRAPH EXECUTOR (unchanged)
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
```