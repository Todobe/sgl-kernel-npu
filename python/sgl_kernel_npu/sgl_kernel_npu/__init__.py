import os
import pathlib
from functools import lru_cache, wraps

from ._build_features import CANN_OPS

if CANN_OPS:
    _vendor = pathlib.Path(__file__).parent / "vendors" / "sgl_kernel_npu"
    if not _vendor.is_dir():
        raise ImportError("Bundled CANN vendor is missing; reinstall the wheel")
    _paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
        [str(_vendor)] + [p for p in _paths if p and p != str(_vendor)]
    )

import torch
import torch_npu


def _load_sgl_kernel_npu():
    npu_path = pathlib.Path(__file__).parents[0]
    so_path = os.path.join(npu_path, "lib", "libsgl_kernel_npu.so")
    torch.ops.load_library(so_path)


_load_sgl_kernel_npu()

if CANN_OPS:
    torch.ops.load_library(str(
        pathlib.Path(__file__).parent / "lib" / "libsgl_kernel_npu_adapter.so"
    ))
