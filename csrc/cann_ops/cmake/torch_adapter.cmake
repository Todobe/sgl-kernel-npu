find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# Discover dependencies using the same Python interpreter used for packaging.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE TORCH_NPU_PATH OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN installation")
set(SGL_CANN_VENDOR_PATH "" CACHE PATH "Installed sgl_kernel_npu vendor")
if(NOT EXISTS "${SGL_CANN_VENDOR_PATH}/op_api/include/aclnn_sgl_helloworld.h")
    message(FATAL_ERROR "Build/install the CANN vendor before the adapter")
endif()
find_library(TORCH_NPU_LIBRARY torch_npu
    PATHS "${TORCH_NPU_PATH}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(ASCENDCL_LIBRARY ascendcl
    PATHS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)
find_library(NNOPBASE_LIBRARY nnopbase
    PATHS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)

add_library(sgl_kernel_npu_adapter SHARED ${SGL_TORCH_ADAPTER_SOURCES})
target_compile_features(sgl_kernel_npu_adapter PRIVATE cxx_std_17)
# Preserve the ABI selected by the installed PyTorch (not the compiler default).
separate_arguments(TORCH_COMPILE_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(sgl_kernel_npu_adapter PRIVATE ${TORCH_COMPILE_FLAGS})
target_compile_definitions(sgl_kernel_npu_adapter PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_include_directories(sgl_kernel_npu_adapter PRIVATE
    "${PROJECT_SOURCE_DIR}/common/torch_adapter"
    ${TORCH_INCLUDE_DIRS}
    "${TORCH_NPU_PATH}/include"
    "${ASCEND_HOME_PATH}/include"
    "${SGL_CANN_VENDOR_PATH}/op_api/include"
    ${Python3_INCLUDE_DIRS})
target_link_libraries(sgl_kernel_npu_adapter PRIVATE
    ${TORCH_LIBRARIES}
    "${TORCH_NPU_LIBRARY}"
    "${ASCENDCL_LIBRARY}"
    "${NNOPBASE_LIBRARY}"
    ${CMAKE_DL_LIBS})
# Runtime imports torch/torch_npu first; CANN comes from its set_env.sh.
# Do not embed the build machine's Python or vendor directories in the wheel.
set_target_properties(sgl_kernel_npu_adapter PROPERTIES
    INSTALL_RPATH "" INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS sgl_kernel_npu_adapter LIBRARY DESTINATION lib)
