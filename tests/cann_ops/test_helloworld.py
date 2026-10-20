"""Run against an installed wheel, not the source package."""
import os

import pytest
import torch
import torch_npu

from sgl_kernel_npu.operators.examples.helloworld import helloworld
from sgl_kernel_npu.operators.utils.registry import get_impl, list_impls


@pytest.mark.parametrize("shape", [(128,), (384,), (2048,), (3, 128), (32, 64), (1048576,)])
@pytest.mark.parametrize("impl", ["auto", "ascendc"])
def test_add(shape, impl):
    x = torch.randn(shape, device="npu", dtype=torch.float16)
    y = torch.randn_like(x)
    z = helloworld(x, y, impl=impl)
    torch.testing.assert_close(z, x + y, rtol=0, atol=0)
    assert z.shape == x.shape
    assert z.dtype == x.dtype
    assert z.device == x.device
    assert z.data_ptr() != x.data_ptr()


def test_registration():
    assert set(list_impls(helloworld)) == {"auto", "ascendc"}
    assert get_impl(helloworld, "auto") is get_impl(helloworld, "ascendc")
    assert callable(torch.ops.sgl_kernel_npu.helloworld)


def test_nondefault_stream_and_repeated_calls():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        x = torch.ones(2048, device="npu", dtype=torch.float16)
        for _ in range(10):
            x = helloworld(x, x)
    stream.synchronize()
    torch.testing.assert_close(x, torch.full_like(x, 1024), rtol=0, atol=0)


@pytest.mark.parametrize("case", ["dtype", "shape", "noncontiguous", "unaligned", "empty", "device"])
def test_invalid_inputs(case):
    x = torch.ones((16, 128), device="npu", dtype=torch.float16)
    y = torch.ones_like(x)
    if case == "dtype":
        x, y = x.float(), y.float()
    elif case == "shape":
        y = y.reshape(-1)
    elif case == "noncontiguous":
        x, y = x.t(), y.t()
    elif case == "unaligned":
        x, y = x.flatten()[:129], y.flatten()[:129]
    elif case == "empty":
        x, y = x[:0], y[:0]
    elif case == "device":
        y = y.cpu()
    with pytest.raises(RuntimeError):
        helloworld(x, y)


def test_storage_offset():
    x = torch.randn(2176, device="npu", dtype=torch.float16)[128:]
    y = torch.randn_like(x)
    torch.testing.assert_close(helloworld(x, y), x + y, rtol=0, atol=0)


@pytest.mark.skipif(
    os.environ.get("SGL_TEST_LEGACY") != "1",
    reason="Set SGL_TEST_LEGACY=1 with the original sgl_kernel_npu installed",
)
def test_legacy_coexistence():
    import sgl_kernel_npu

    x = torch.randn(2048, device="npu", dtype=torch.float16)
    y = torch.randn_like(x)
    old = torch.ops.npu.helloworld(x, y)
    new = helloworld(x, y)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
