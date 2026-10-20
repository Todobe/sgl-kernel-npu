# Helloworld CANN pilot validation

Date: 2026-10-20.

## Unified package and per-operator layout (latest)

- Clean source/build directory: `/home/fby/cann-layout-clean-20261020`.
- CANN host/kernel sources and adapter were rebuilt from the reorganized source
  tree. The compatible legacy library was reused from the container's installed
  `sgl_kernel_npu` via `SGL_LEGACY_LIBRARY`; the full legacy build was not rerun.
- The final wheel was installed with `pip --no-deps --target installed`.
  Existing system packages and system OPP were not modified.
- Result: **22 passed**, 14 upstream PyTorch deprecation warnings, 7.06 seconds.
  Coverage: auto/ascendc across six shapes, alias registration, non-default stream
  and repeated calls, six invalid-input cases, storage offsets, old/new coexistence.
- Wheel includes both `lib/libsgl_kernel_npu.so` and
  `lib/libsgl_kernel_npu_adapter.so`, vendor artifacts and the public operator.
  `_build_features.py` enables import-time adapter loading. No `._*` metadata or
  old `sgl_kernel_npu_cann` package is included.
- Adapter has no RPATH/RUNPATH referring to the build directory.
- Local artifacts and logs: `output/cann_ops/unified/` and its `validation/`.

SHA256:

```text
7613d722c707b6cdee02f5a3e29fcc425de2ae3bbaa9a798296b44bdb41cdac2  custom_opp_ubuntu_aarch64.run
110e92f73b2c940203c830c03fe1c6b110a836d1eeede3b8c60af4f555c876ac  sgl_kernel_npu-2026.9.0-cp312-cp312-linux_aarch64.whl
```

The following sections retain the historical independent-pilot validation.

## Environment

- Server: `113.46.15.88`, container: `test-fby`.
- Fresh build directory: `/home/fby/cann-helloworld-clean-20261020`.
- Architecture: Linux AArch64; A3 device exposed via `ASCEND_RT_VISIBLE_DEVICES=6`.
- CANN: `/usr/local/Ascend/cann-9.1.0`.
- Python: 3.12.13; PyTorch: `2.10.0+cpu`; torch_npu: `2.10.0.post6`.
- Original `sgl_kernel_npu`: 2026.9.0 (retained unchanged).

## Procedure and result

1. Copy only the new checked-in source files into a fresh directory.
2. Source CANN environment and run `bash build_cann_ops.sh`.
3. Build the binary/package CMake targets and the independent Python wheel.
4. Install the wheel using `pip install --no-deps --target installed`.
5. Run tests with `PYTHONPATH` pointing at that installation and
   `SGL_TEST_LEGACY=1`.

Result: **15 passed in 6.89 seconds**. The 14 warnings are upstream
`torch.jit.script_method` deprecation warnings.

| Coverage | Result |
|---|---|
| FP16 addition for six shapes (128 through 1,048,576 elements) | Exact match to torch addition |
| Non-default NPU stream and ten chained launches | Passed |
| dtype, shape, contiguous layout, alignment, empty input, mixed device checks | Rejected as expected |
| Contiguous view with nonzero storage offset | Passed |
| Old `torch.ops.npu.helloworld` and new namespace in one process | Exact match |

Neither the original package nor the system CANN OPP installation was replaced.
The test container was initially stopped and was started for validation.

## Artifacts

Server directory: `/home/fby/cann-helloworld-clean-20261020/output/cann_ops/`.
Local copies are in the repository's ignored `output/cann_ops/` directory.

| File | SHA256 |
|---|---|
| `custom_opp_ubuntu_aarch64.run` | `50e89bc969b3eb6a141c36d8e330d8545d0f08770501640b43cb4e6af284a714` |
| `sgl_kernel_npu_cann-0.1.0-cp312-cp312-linux_aarch64.whl` | `5736d2b2267c8b2bf8aa251980a4221de88941986a678381939584d9b57a759d` |

Local raw logs: `output/cann_ops/validation/build.log` and `test.log`.
The wheel is specific to this Python/platform/framework environment.

## Adaptation notes and remaining scope

- Using torch_npu's internal op-plugin `EXEC_NPU_CMD` header produced an
  unresolved `NPUStorageImpl` RTTI symbol in this environment. Disabling RTTI
  is not a solution because PyTorch headers require it.
- The final adapter uses generated ACLNN declarations, ACL tensor descriptors,
  and `OpCommand::RunOpApi`. It resolves its own `libcust_opapi.so` by absolute
  vendor path once and retains the handle.
- This avoids depending on the SDK's internal target naming or a global
  `libcust_opapi.so` search order.
- A2/A5, other SDK/framework versions, autograd, graph capture and torch.compile
  are not validated. No AICPU kernel has been added.
- This pilot does not yet join the main wheel, root build target selection,
  or the Python operators registry.

## Follow-up: CMake adapter build

The adapter build was migrated from `CppExtension` to an independent CMake
`SHARED` target on 2026-10-20. Its output is now
`libsgl_kernel_npu_adapter.so`; `setup.py` only packages prebuilt files.

- Fresh server build: `/home/fby/cann-helloworld-cmake-20261020`.
- Same container, toolkit, framework versions and test device as above.
- CMake discovers dependencies via the packaging Python interpreter and explicitly
  matches `torch._C._GLIBCXX_USE_CXX11_ABI`.
- The installed library does not contain build-machine RPATH/RUNPATH entries.
- The installed wheel contains the new library and vendor, not the old `_C*.so`.
- The first clean CMake build passed all 15 NPU tests (6.72 seconds).
- After adding the explicit ABI definition, rebuilding and installing into a
  new target directory again passed all 15 tests (6.89 seconds).
- This follow-up's artifacts/logs are kept separately in local
  `output/cann_ops/cmake/`; the earlier hashes above refer to the original
  CppExtension version.

Final CMake wheel SHA256:
`e0c581b82ce8ebc576062d533fdd01e34cbe6687ba79ff0dfcb0241197825816`.
