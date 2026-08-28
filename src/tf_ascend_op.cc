#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "ascendcl_cuda_compat.h"

using namespace tensorflow;

class AscendMatMulOp : public OpKernel {
public:
    explicit AscendMatMulOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& a = context->input(0);
        const Tensor& b = context->input(1);

        // Convert to device memory
        auto a_mem = ascendcl::pytorch::tensorToDevice(a.data(), a.AllocatedBytes());
        auto b_mem = ascendcl::pytorch::tensorToDevice(b.data(), b.AllocatedBytes());

        // Allocate output
        TensorShape out_shape = a.shape();
        out_shape.set_dim(out_shape.dims()-1, b.shape().dim_size(1));
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        auto out_mem = ascendcl::DeviceMalloc(output->AllocatedBytes());

        // Run GEMM
        ascendcl::Stream stream;
        aclnn::MatMulOp matmul(/* descriptors */);
        // ... (simplified)
        // For brevity, we'll assume the GEMM kernel is used.
    }
};

REGISTER_OP("AscendMatMul")
    .Input("a: float")
    .Input("b: float")
    .Output("output: float");

REGISTER_KERNEL_BUILDER(Name("AscendMatMul").Device(DEVICE_GPU), AscendMatMulOp);