#!/usr/bin/env bash
# Independent A3 pilot: leave build.sh and the legacy targets unchanged.
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${ASCEND_HOME_PATH:?Source the CANN set_env.sh before building}"
BUILD_ROOT="${CANN_BUILD_DIR:-$ROOT/csrc/cann_ops/build}"
ADAPTER_BUILD="${CANN_ADAPTER_BUILD_DIR:-$BUILD_ROOT/adapter}"
OUTPUT="${CANN_OUTPUT_DIR:-$ROOT/output/cann_ops}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$BUILD_ROOT" "$OUTPUT"
cmake -S "$ROOT/csrc/cann_ops" -B "$BUILD_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DASCEND_COMPUTE_UNIT=ascend910_93 \
    -DASCEND_PYTHON_EXECUTABLE="$(command -v "$PYTHON")" \
    -DENABLE_SOURCE_PACKAGE=ON -DENABLE_BINARY_PACKAGE=ON \
    -DENABLE_CROSS_COMPILE=OFF
cmake --build "$BUILD_ROOT" --target binary -j "${MAX_JOBS:-4}"
cmake --build "$BUILD_ROOT" --target package -j "${MAX_JOBS:-4}"
mapfile -t packages < <(find "$BUILD_ROOT" -maxdepth 1 -name 'custom_opp*.run')
if [[ ${#packages[@]} != 1 ]]; then
    echo "Expected exactly one custom_opp .run in $BUILD_ROOT" >&2
    exit 1
fi
cp "${packages[0]}" "$OUTPUT/"
PACKAGE_DIR="$ROOT/python/sgl_kernel_npu"
# Fresh staging prevents artifacts from a previous build entering this wheel.
STAGING=$(mktemp -d "$BUILD_ROOT/package-XXXXXX")
mkdir -p "$STAGING/lib"
if [[ -z "${SGL_LEGACY_LIBRARY:-}" ]]; then
    bash "$ROOT/build.sh" -a kernels Ascend910_9382
    SGL_LEGACY_LIBRARY="$PACKAGE_DIR/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
fi
test -f "$SGL_LEGACY_LIBRARY"
cp "$SGL_LEGACY_LIBRARY" "$STAGING/lib/"
bash "${packages[0]}" --quiet --install-path="$STAGING"
cmake -S "$ROOT/csrc/cann_ops" -B "$ADAPTER_BUILD" \
    -DSGL_CANN_STAGE=adapter \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$(command -v "$PYTHON")" \
    -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
    -DSGL_CANN_VENDOR_PATH="$STAGING/vendors/sgl_kernel_npu" \
    -DCMAKE_INSTALL_PREFIX="$STAGING"
cmake --build "$ADAPTER_BUILD" -j "${MAX_JOBS:-4}"
cmake --install "$ADAPTER_BUILD"
(
    cd "$PACKAGE_DIR"
    SGL_CANN_ARTIFACTS="$STAGING" "$PYTHON" setup.py build --build-base "$STAGING/python" \
        bdist_wheel --dist-dir "$OUTPUT"
)
echo "CANN package and wheel: $OUTPUT"
