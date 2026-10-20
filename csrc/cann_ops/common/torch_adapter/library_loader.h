#pragma once

#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <filesystem>

namespace sgl::adapter {
// The adapter lives in <package>/lib; retain the vendor handle until process exit.
inline void* loadOpApi() {
    static void* handle = [] {
        Dl_info info{};
        TORCH_CHECK(dladdr(reinterpret_cast<void*>(&loadOpApi), &info),
                    "Cannot locate the CANN adapter library");
        auto path = std::filesystem::path(info.dli_fname).parent_path().parent_path() /
                    "vendors/sgl_kernel_npu/op_api/lib/libcust_opapi.so";
        void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(library, "Cannot load ", path.string(), ": ", dlerror());
        return library;
    }();
    return handle;
}

template <typename Function>
Function resolve(const char* name) {
    auto function = reinterpret_cast<Function>(dlsym(loadOpApi(), name));
    TORCH_CHECK(function, "Missing ACLNN symbol: ", name);
    return function;
}
}  // namespace sgl::adapter
