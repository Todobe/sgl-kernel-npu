# CANN operator migration

The first migrated operator is A3 FP16 helloworld. Legacy sources and `build.sh`
are retained; the new build produces a unified `sgl_kernel_npu` wheel.

## Layout and build path

```text
build_cann_ops.sh
├─ csrc/cann_ops/CMakeLists.txt (SGL_CANN_STAGE=vendor)
│  ├─ examples/helloworld/CMakeLists.txt: explicit source list
│  ├─ cmake/cann_package.cmake: OpDef → generated ACLNN/prototype/tiling
│  └─ examples/helloworld/op_kernel: device binary → custom_opp*.run
├─ fresh build/package-XXXXXX staging directory
│  ├─ install .run → vendors/sgl_kernel_npu/
│  └─ copy legacy libsgl_kernel_npu.so → lib/
├─ csrc/cann_ops/CMakeLists.txt (SGL_CANN_STAGE=adapter)
│  └─ cmake/torch_adapter.cmake
│     └─ per-op torch_adapter + common/torch_adapter → lib/adapter.so
└─ python/sgl_kernel_npu/setup.py (SGL_CANN_ARTIFACTS=staging)
   └─ Python sources + both libraries + vendor → sgl_kernel_npu wheel
```

The adapter's full filename is `libsgl_kernel_npu_adapter.so`.
Vendor generation precedes adapter compilation because the latter needs the
generated ACLNN header. Separate CMake build directories share one source entry.
`common/torch_adapter/library_loader.h` resolves the package-local ACLNN library;
FP16 argument conversion and execution remain operator-local until a reusable
contract is needed. No empty generic helpers are introduced.

## Build

Requires Linux AArch64, CANN ASC development tooling, compatible PyTorch/torch_npu,
Python development headers, CMake >= 3.20, setuptools and wheel.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build_cann_ops.sh
```

By default the script also invokes `build.sh -a kernels Ascend910_9382` to build
the legacy library. To reuse an existing compatible A3 library:

```bash
SGL_LEGACY_LIBRARY=/absolute/path/libsgl_kernel_npu.so bash build_cann_ops.sh
```

Outputs under `output/cann_ops/`:

- `custom_opp*.run`: CANN vendor package.
- `sgl_kernel_npu-*.whl`: unified package; no separate `.run` installation needed.

Optional variables: `CANN_BUILD_DIR`, `CANN_ADAPTER_BUILD_DIR`, `CANN_OUTPUT_DIR`,
`MAX_JOBS`, `PYTHON`. Vendor build defaults to `csrc/cann_ops/build`, adapter build
to its `adapter/` subdirectory. Fresh packaging staging is retained for inspection.
No system OPP directory is modified.

In CANN mode, setup.py packages prebuilt artifacts without compiling C++; its
generated `_build_features.py` enables eager adapter loading. The source default
is disabled, preserving the legacy-only build. A CANN-enabled wheel fails import
if required native artifacts are missing. The two wheels are alternative builds
of the same distribution, not packages to install side by side.

## Call path

Package import prepares `ASCEND_CUSTOM_OPP_PATH`, imports torch/torch_npu and loads
both native libraries. Adapter loading registers all compiled PyTorch schemas.

```python
from sgl_kernel_npu.operators.examples.helloworld import helloworld

z = helloworld(x, y)                 # auto → ascendc
z = helloworld(x, y, impl="ascendc")
```

The public operator dispatches through the existing registry, then calls
`torch.ops.sgl_kernel_npu.helloworld`, the C++ adapter, and ACLNN.
Both aliases declare A3 support. Legacy `torch.ops.npu.helloworld` remains usable.

Inputs: same-device, same-shape, contiguous base-format FP16 NPU tensors;
positive numel divisible by 128 and <= UINT32_MAX. Output is a new tensor x+y.
This migration does not add AICPU, graph conversion, autograd or performance
validation. `examples/helloworld/helloworld.json` is an msopgen specification,
not an input to the normal build.

## Installed-wheel verification

```bash
python -m pip install --no-deps --target /absolute/isolated/install output/cann_ops/*.whl
PYTHONPATH=/absolute/isolated/install SGL_TEST_LEGACY=1 \
    python -m pytest tests/cann_ops/test_helloworld.py -v
```

Tests cover auto/ascendc registration, exact FP16 results, stream execution,
storage offsets, invalid inputs and old/new namespace coexistence.
