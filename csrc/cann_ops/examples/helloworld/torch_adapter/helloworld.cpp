#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <memory>
#include "library_loader.h"
#include "aclnn_sgl_helloworld.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {
struct OpApi {
    decltype(&aclnnSglHelloworldGetWorkspaceSize) query;
    decltype(&aclnnSglHelloworld) launch;
};

const OpApi& opApi() {
    static const OpApi api = [] {
        auto query = sgl::adapter::resolve<decltype(OpApi::query)>(
            "aclnnSglHelloworldGetWorkspaceSize");
        auto launch = sgl::adapter::resolve<decltype(OpApi::launch)>(
            "aclnnSglHelloworld");
        return OpApi{query, launch};
    }();
    return api;
}

using TensorDescriptor = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
TensorDescriptor describe(const at::Tensor& tensor) {
    const int64_t storageSize = tensor.numel();
    auto* descriptor = aclCreateTensor(
        tensor.sizes().data(), tensor.dim(), ACL_FLOAT16,
        tensor.strides().data(), 0, ACL_FORMAT_ND,
        &storageSize, 1, tensor.data_ptr());
    TORCH_CHECK(descriptor != nullptr, "aclCreateTensor failed");
    return TensorDescriptor(descriptor, aclDestroyTensor);
}

at::Tensor helloworld(const at::Tensor& x, const at::Tensor& y) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "helloworld requires NPU tensors");
    TORCH_CHECK(x.device() == y.device(), "x and y must be on the same device");
    TORCH_CHECK(x.scalar_type() == at::kHalf && y.scalar_type() == at::kHalf,
                "helloworld supports float16 only");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                "helloworld requires contiguous tensors");
    for (const auto& tensor : {x, y}) {
        const auto format = at_npu::native::get_npu_format(tensor);
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                    "helloworld requires base-format tensors");
    }
    TORCH_CHECK(x.numel() > 0 && x.numel() % 128 == 0 && x.numel() <= UINT32_MAX,
                "numel must be a positive multiple of 128 and fit uint32");
    const c10::DeviceGuard guard(x.device());
    auto z = at::empty_like(x);
    auto ax = describe(x);
    auto ay = describe(y);
    auto az = describe(z);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto& api = opApi();
    const auto status = api.query(
        ax.get(), ay.get(), az.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == 0, "aclnnSglHelloworldGetWorkspaceSize failed: ", status);
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                               x.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // Retain tensors while the framework's host task queue submits the launch.
    auto launch = [x, y, z, workspace, workspaceSize, executor, stream, api]() -> int {
        return api.launch(
            workspaceSize ? workspace.data_ptr() : nullptr,
            workspaceSize, executor, stream);
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglHelloworld", launch);
    return z;
}
}
TORCH_LIBRARY_FRAGMENT(sgl_kernel_npu, m) {
    m.def("helloworld(Tensor x, Tensor y) -> Tensor");
}
TORCH_LIBRARY_IMPL(sgl_kernel_npu, PrivateUse1, m) {
    m.impl("helloworld", TORCH_FN(helloworld));
}
