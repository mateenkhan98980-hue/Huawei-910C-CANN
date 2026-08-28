// ============================================================================
// ACLNN Operator Wrapper - CANN 8.0
// Strict CUDA-compatible operator interface
// ============================================================================

#pragma once

#include "ascendcl_wrapper.h"
#include "acl/acl_op.h"
#include "acl/acl_nn.h"
#include <vector>
#include <memory>
#include <string>

namespace aclnn {

// ============================================================================
// OPERATOR INTERFACE
// ============================================================================

class Operator {
public:
    virtual ~Operator() = default;
    virtual void execute(aclrtStream stream) = 0;
    virtual std::string getName() const = 0;
};

// ============================================================================
// MATRIX MULTIPLY (GEMM) OPERATOR
// ============================================================================

class MatMulOp : public Operator {
public:
    MatMulOp(
        ascendcl::TensorDesc* a_desc, void* a_data,
        ascendcl::TensorDesc* b_desc, void* b_data,
        ascendcl::TensorDesc* c_desc, void* c_data,
        float alpha = 1.0f, float beta = 1.0f);
    
    ~MatMulOp() override;
    
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "MatMul"; }
    
private:
    ascendcl::TensorDesc* a_desc_;
    ascendcl::TensorDesc* b_desc_;
    ascendcl::TensorDesc* c_desc_;
    void* a_data_;
    void* b_data_;
    void* c_data_;
    float alpha_;
    float beta_;
    aclopAttr* attr_;
};

// ============================================================================
// BIAS ADD OPERATOR
// ============================================================================

class BiasAddOp : public Operator {
public:
    BiasAddOp(
        ascendcl::TensorDesc* input_desc, void* input_data,
        ascendcl::TensorDesc* bias_desc, void* bias_data,
        ascendcl::TensorDesc* output_desc, void* output_data);
    
    ~BiasAddOp() override;
    
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "BiasAdd"; }
    
private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* bias_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* bias_data_;
    void* output_data_;
    aclopAttr* attr_;
};

// ============================================================================
// ACTIVATION OPERATORS
// ============================================================================

class ReluOp : public Operator {
public:
    ReluOp(ascendcl::TensorDesc* input_desc, void* input_data,
           ascendcl::TensorDesc* output_desc, void* output_data);
    
    ~ReluOp() override;
    
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "ReLU"; }
    
private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* output_data_;
    aclopAttr* attr_;
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
        int64_t dilations = 1, int64_t groups = 1);
    ~Conv2dOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "Conv2d"; }
private:
    ascendcl::TensorDesc* input_desc_;
    ascendcl::TensorDesc* weight_desc_;
    ascendcl::TensorDesc* output_desc_;
    void* input_data_;
    void* weight_data_;
    void* output_data_;
    aclopAttr* attr_;
    std::vector<int64_t> strides_, pads_;
    int64_t dilations_, groups_;
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
        float epsilon = 1e-5f, float momentum = 0.9f);
    ~BatchNormOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "BatchNorm"; }
private:
    ascendcl::TensorDesc* input_desc_, *scale_desc_, *offset_desc_, *mean_desc_, *variance_desc_, *output_desc_;
    void* input_data_, *scale_data_, *offset_data_, *mean_data_, *variance_data_, *output_data_;
    float epsilon_, momentum_;
    aclopAttr* attr_;
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
        const std::vector<int64_t>& pads);
    ~MaxPoolOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "MaxPool"; }
private:
    ascendcl::TensorDesc* input_desc_, *output_desc_;
    void* input_data_, *output_data_;
    std::vector<int64_t> kernel_size_, strides_, pads_;
    aclopAttr* attr_;
};

// ============================================================================
// SOFTMAX OPERATOR
// ============================================================================
class SoftmaxOp : public Operator {
public:
    SoftmaxOp(ascendcl::TensorDesc* input_desc, void* input_data,
              ascendcl::TensorDesc* output_desc, void* output_data,
              int64_t axis = -1);
    ~SoftmaxOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "Softmax"; }
private:
    ascendcl::TensorDesc* input_desc_, *output_desc_;
    void* input_data_, *output_data_;
    int64_t axis_;
    aclopAttr* attr_;
};

// ============================================================================
// ELEMENTWISE OPERATORS (Add, Sub, Mul, Div)
// ============================================================================
class AddOp : public Operator {
public:
    AddOp(ascendcl::TensorDesc* a_desc, void* a_data,
          ascendcl::TensorDesc* b_desc, void* b_data,
          ascendcl::TensorDesc* output_desc, void* output_data,
          float alpha = 1.0f, float beta = 1.0f);
    ~AddOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "Add"; }
private:
    ascendcl::TensorDesc* a_desc_, *b_desc_, *output_desc_;
    void* a_data_, *b_data_, *output_data_;
    float alpha_, beta_;
    aclopAttr* attr_;
};

class SubOp : public Operator {
public:
    SubOp(ascendcl::TensorDesc* a_desc, void* a_data,
          ascendcl::TensorDesc* b_desc, void* b_data,
          ascendcl::TensorDesc* output_desc, void* output_data,
          float alpha = 1.0f, float beta = 1.0f);
    ~SubOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "Sub"; }
private:
    ascendcl::TensorDesc* a_desc_, *b_desc_, *output_desc_;
    void* a_data_, *b_data_, *output_data_;
    float alpha_, beta_;
    aclopAttr* attr_;
};

class MulOp : public Operator {
public:
    MulOp(ascendcl::TensorDesc* a_desc, void* a_data,
          ascendcl::TensorDesc* b_desc, void* b_data,
          ascendcl::TensorDesc* output_desc, void* output_data,
          float alpha = 1.0f, float beta = 1.0f);
    ~MulOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "Mul"; }
private:
    ascendcl::TensorDesc* a_desc_, *b_desc_, *output_desc_;
    void* a_data_, *b_data_, *output_data_;
    float alpha_, beta_;
    aclopAttr* attr_;
};

class DivOp : public Operator {
public:
    DivOp(ascendcl::TensorDesc* a_desc, void* a_data,
          ascendcl::TensorDesc* b_desc, void* b_data,
          ascendcl::TensorDesc* output_desc, void* output_data,
          float alpha = 1.0f, float beta = 1.0f);
    ~DivOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "Div"; }
private:
    ascendcl::TensorDesc* a_desc_, *b_desc_, *output_desc_;
    void* a_data_, *b_data_, *output_data_;
    float alpha_, beta_;
    aclopAttr* attr_;
};

// ============================================================================
// LEAKY RELU OPERATOR
// ============================================================================
class LeakyReluOp : public Operator {
public:
    LeakyReluOp(ascendcl::TensorDesc* input_desc, void* input_data,
                ascendcl::TensorDesc* output_desc, void* output_data,
                float negative_slope = 0.01f);
    ~LeakyReluOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "LeakyRelu"; }
private:
    ascendcl::TensorDesc* input_desc_, *output_desc_;
    void* input_data_, *output_data_;
    float negative_slope_;
    aclopAttr* attr_;
};

// ============================================================================
// SIGMOID OPERATOR
// ============================================================================
class SigmoidOp : public Operator {
public:
    SigmoidOp(ascendcl::TensorDesc* input_desc, void* input_data,
              ascendcl::TensorDesc* output_desc, void* output_data);
    ~SigmoidOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "Sigmoid"; }
private:
    ascendcl::TensorDesc* input_desc_, *output_desc_;
    void* input_data_, *output_data_;
    aclopAttr* attr_;
};

// ============================================================================
// GELU OPERATOR
// ============================================================================
class GeluOp : public Operator {
public:
    GeluOp(ascendcl::TensorDesc* input_desc, void* input_data,
           ascendcl::TensorDesc* output_desc, void* output_data);
    ~GeluOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "Gelu"; }
private:
    ascendcl::TensorDesc* input_desc_, *output_desc_;
    void* input_data_, *output_data_;
};

// ============================================================================
// DROPOUT OPERATOR
// ============================================================================
class DropoutOp : public Operator {
public:
    DropoutOp(ascendcl::TensorDesc* input_desc, void* input_data,
              ascendcl::TensorDesc* output_desc, void* output_data,
              float keep_prob = 0.5f, float seed = 0.0f);
    ~DropoutOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "Dropout"; }
private:
    ascendcl::TensorDesc* input_desc_, *output_desc_;
    void* input_data_, *output_data_;
    float keep_prob_, seed_;
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
                float epsilon = 1e-5f);
    ~LayerNormOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "LayerNorm"; }
private:
    ascendcl::TensorDesc* input_desc_, *gamma_desc_, *beta_desc_, *output_desc_;
    void* input_data_, *gamma_data_, *beta_data_, *output_data_;
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
                float scale = 1.0f);
    ~AttentionOp() override;
    void execute(aclrtStream stream) override;
    std::string getName() const override { return "Attention"; }
private:
    ascendcl::TensorDesc* query_desc_, *key_desc_, *value_desc_, *output_desc_;
    void* query_data_, *key_data_, *value_data_, *output_data_;
    float scale_;
};

// ============================================================================
// OPERATOR EXECUTOR (GRAPH BUILDER)
// ============================================================================

class GraphExecutor {
public:
    GraphExecutor();
    ~GraphExecutor();
    
    void addOperator(std::shared_ptr<Operator> op);
    void execute(ascendcl::Stream* stream = nullptr);
    void execute(aclrtStream stream);
    
private:
    std::vector<std::shared_ptr<Operator>> operators_;
    std::shared_ptr<ascendcl::Stream> default_stream_;
};

// ============================================================================
// FACTORY FUNCTION
// ============================================================================
std::shared_ptr<Operator> createOperator(
    const std::string& name,
    const std::vector<ascendcl::TensorDesc*>& descs,
    const std::vector<void*>& data,
    float alpha = 1.0f,
    float beta = 1.0f,
    const std::vector<int64_t>& extra_params = {});

} // namespace aclnn