"""CANN operator registration and public dispatch."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("examples.helloworld", "auto", supported_generations={Generation.A3})
@register_impl("examples.helloworld", "ascendc", supported_generations={Generation.A3})
def _ascendc_impl(x, y):
    return torch.ops.sgl_kernel_npu.helloworld(x, y)


@register_operator("examples.helloworld")
def helloworld(x, y, *, impl="auto"):
    """Return x + y; auto selects Ascend C.

    Args:
        x (torch.Tensor): Contiguous NPU float16 tensor of shape [...];
            numel must be a positive multiple of 128 and fit uint32.
        y (torch.Tensor): Same shape, dtype and device as x; contiguous.
        impl (str): "auto" or "ascendc".
    """
    return get_impl_by_name("examples.helloworld", impl)(x, y)
