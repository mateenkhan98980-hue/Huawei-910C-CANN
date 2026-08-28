// ============================================================================
// torch_ascend.cpp – PyTorch Extension for Ascend NPU
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ascendcl_cuda_compat.h"
#include <torch/extension.h> // for ATen tensors

namespace py = pybind11;

// Convert PyTorch tensor to Ascend device memory
py::object tensor_to_ascend(py::object tensor_obj) {
    auto tensor = tensor_obj.cast<at::Tensor>();
    void* data = tensor.data_ptr();
    size_t size = tensor.nbytes();
    auto device_mem = ascendcl::pytorch::tensorToDevice(data, size);
    // Return a Python object that holds the shared_ptr
    return py::cast(device_mem);
}

// Convert Ascend memory back to PyTorch tensor
py::object device_to_tensor(py::object mem_obj, std::vector<int64_t> shape, std::string dtype) {
    auto mem = mem_obj.cast<std::shared_ptr<ascendcl::DeviceMemory>>();
    void* host_data = ascendcl::pytorch::deviceToTensor(mem);
    // Wrap host_data into a PyTorch tensor (must copy or manage memory)
    // For simplicity, we return a numpy array and convert to tensor.
    py::array_t<float> np_array(shape, host_data);
    return py::cast(np_array);
}

PYBIND11_MODULE(torch_ascend, m) {
    m.def("init", &ascendcl::initializeAscend, py::arg("device_id") = 0, py::arg("distributed") = false);
    m.def("finalize", &ascendcl::finalizeAscend);
    m.def("device_count", &ascendcl::Device::getDeviceCount);
    m.def("set_device", [](int id) { ascendcl::Device dev(id); dev.setActive(); });

    // Tensor transfer helpers
    m.def("tensor_to_ascend", &tensor_to_ascend, "Move PyTorch tensor to NPU");
    m.def("device_to_tensor", &device_to_tensor, "Move NPU memory back to PyTorch tensor");

    // Expose DeviceMemory class for Python
    py::class_<ascendcl::DeviceMemory, std::shared_ptr<ascendcl::DeviceMemory>>(m, "DeviceMemory")
        .def(py::init<size_t, bool>(), py::arg("size"), py::arg("huge_page") = true)
        .def("data", &ascendcl::DeviceMemory::getData)
        .def("size", &ascendcl::DeviceMemory::getSize)
        .def("aligned_size", &ascendcl::DeviceMemory::getAlignedSize);
}