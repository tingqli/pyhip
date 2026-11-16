"""Tests for pyhip.contrib.flydsl.host_launch (@flydsl_kernel).

    pytest -q tests/flydsl/test_host_launch.py        # 2+ GPUs also cover multi-device
    python tests/flydsl/test_host_launch.py           # host-overhead comparison
"""

import time
import warnings

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import pytest
import torch
from flydsl.expr import const_expr, gpu

from pyhip.contrib.flydsl import host_launch as hl
from pyhip.contrib.flydsl.host_launch import cdiv, flydsl_kernel, kernel_name, ptr

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _global(address, dtype, align):
    return fx.inttoptr(fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, align),
                       fx.Int64(address))


@flydsl_kernel(heuristics={"block": lambda k: 64 if k["mul"] < 0 else 256})
def axpb(*, mul, has_y, block):
    """out[i] = x[i] * mul + bias (+ y[i]) for i < n, int32."""

    @flyc.kernel(name=kernel_name("test_axpb", m=mul, y=has_y, b=block),
                 known_block_size=(block, 1, 1))
    def kernel(x: fx.Int64, y: fx.Int64, out: fx.Int64, i32_n: fx.Int32, i32_bias: fx.Int32):
        i = fx.Int32(gpu.block_id("x")) * block + fx.Int32(gpu.thread_id("x"))
        if i < i32_n:
            v = fx.ptr_load(fx.add_offset(_global(x, fx.Int32, 4), i)) * mul + i32_bias
            if const_expr(has_y):
                v = v + fx.ptr_load(fx.add_offset(_global(y, fx.Int32, 4), i))
            fx.ptr_store(v, fx.add_offset(_global(out, fx.Int32, 4), i))

    @flyc.jit
    def launch(x: fx.Int64, y: fx.Int64, out: fx.Int64, i32_n: fx.Int32, i32_bias: fx.Int32,
               stream: fx.Stream):
        grid = (i32_n + (block - 1)) // block
        kernel(x, y, out, i32_n, i32_bias).launch(grid=(fx.Int64(grid), 1, 1),
                                                  block=(block, 1, 1), stream=stream)

    return launch


def axpb_op(x, bias=0, y=None, mul=3):
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        axpb(ptr(x, torch.int32, numel=n), ptr(y, torch.int32, numel=n),
             ptr(out, torch.int32, numel=n), n, bias, mul=mul, has_y=y is not None)
    return out


def axpb_ref(x, bias=0, y=None, mul=3):
    out = x * mul + bias
    return out if y is None else out + y


def _ints(n, device="cuda"):
    return torch.randint(-1000, 1000, (n,), dtype=torch.int32, device=device)


def test_matches_torch_and_runtime_sizes_do_not_recompile():
    for mul in (3, -2):
        for has_y in (False, True):
            for n in (1, 63, 256, 1000, 70001):
                x, y = _ints(n), _ints(n) if has_y else None
                torch.testing.assert_close(axpb_op(x, 7, y, mul), axpb_ref(x, 7, y, mul),
                                           rtol=0, atol=0)
    keys = {dict(k)["mul"]: dict(k)["block"] for k in axpb.variants}
    assert len(axpb.variants) == 4 and keys == {3: 256, -2: 64}


def test_empty_input_launches_nothing():
    assert axpb_op(_ints(0)).numel() == 0


def test_key_normalization():
    exe = axpb.get(mul=3, has_y=False)
    assert axpb.get(has_y=False, mul=3) is exe
    assert axpb.get(mul=np.int64(3), has_y=False) is exe
    assert axpb.get(mul=3, has_y=False, block=256) is exe  # the heuristic's value
    assert axpb.get(mul=-2, has_y=False).name == "axpb[block=64,has_y=False,mul=-2]"
    assert hl._freeze("k", [1, [2, 3]]) == (1, (2, 3))
    with pytest.raises(TypeError, match="not a compile-time value"):
        axpb.get(mul=torch.tensor(3), has_y=False)


def test_bound_specialization():
    exe = axpb.get(mul=5, has_y=False)
    x = _ints(300)
    out = torch.empty_like(x)
    exe(x, None, out, x.numel(), -4)
    torch.testing.assert_close(out, axpb_ref(x, -4, mul=5), rtol=0, atol=0)


def test_call_without_runtime_args_raises():
    with pytest.raises(TypeError, match=r"axpb\.get"):
        axpb(mul=3, has_y=False)


def test_side_stream():
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x = _ints(5000)
        out = axpb_op(x, 1)
    s.synchronize()
    torch.testing.assert_close(out, axpb_ref(x, 1), rtol=0, atol=0)


def test_cuda_graph():
    x, y = _ints(4096), _ints(4096)
    axpb_op(x, 2, y)  # compile outside the capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = axpb_op(x, 2, y)
    x.copy_(_ints(4096))
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, axpb_ref(x, 2, y), rtol=0, atol=0)


def test_preload_then_launch():
    axpb.preload(mul=11, has_y=True)
    assert axpb.get(mul=11, has_y=True).cf is None  # compiled, nothing launched
    x, y = _ints(777), _ints(777)
    torch.testing.assert_close(axpb_op(x, 0, y, 11), axpb_ref(x, 0, y, 11), rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_tensors_on_another_device():
    exe = axpb.get(mul=3, has_y=False)
    for cur, dev in ((0, 1), (1, 0), (1, 1)):
        with torch.cuda.device(cur):
            x = _ints(2048, f"cuda:{dev}")
            out = axpb_op(x, 9)
            assert torch.cuda.current_device() == cur
        torch.testing.assert_close(out, axpb_ref(x, 9), rtol=0, atol=0)
    assert set(exe._others) | {exe.dev} == {0, 1}


def test_check_mode(monkeypatch):
    monkeypatch.setattr(hl, "CHECK", True)
    x = _ints(256)
    torch.testing.assert_close(axpb_op(x, 1), axpb_ref(x, 1), rtol=0, atol=0)
    out = torch.empty_like(x)
    n = x.numel()
    with pytest.raises(TypeError, match="dtype torch.float32"):
        axpb_op(x.float())
    with pytest.raises(ValueError, match="not contiguous"):
        axpb_op(_ints(512)[::2])
    with pytest.raises(ValueError, match="expected a GPU tensor"):
        axpb_op(x.cpu())
    with pytest.raises(ValueError, match="kernel touches 256"):
        axpb(ptr(x, numel=n), None, ptr(out[:100], numel=n), n, 0, mul=3, has_y=False)
    with pytest.raises(OverflowError, match="Int32"):
        axpb(x, None, out, 2**31, 0, mul=3, has_y=False)
    with pytest.raises(TypeError, match="argument order"):
        axpb(x, None, n, out, 0, mul=3, has_y=False)
    with pytest.raises(TypeError, match="called with 4 args"):
        axpb(x, None, out, n, mul=3, has_y=False)
    if torch.cuda.device_count() > 1:
        with pytest.raises(ValueError, match="earlier tensors on cuda:0"):
            axpb(x, None, out.to("cuda:1"), n, 0, mul=3, has_y=False)


@flydsl_kernel
def bad_signature(*, which):
    @flyc.kernel
    def kernel(x: fx.Int64):
        pass

    if which == "unannotated":
        @flyc.jit
        def launch(x, stream: fx.Stream):
            kernel(x).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    elif which == "pointer":
        @flyc.jit
        def launch(x: fx.Pointer, stream: fx.Stream):
            kernel(x).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    elif which == "no_stream":
        @flyc.jit
        def launch(x: fx.Int64):
            kernel(x).launch(grid=(1, 1, 1), block=(64, 1, 1))
    else:
        return kernel
    return launch


@pytest.mark.parametrize("which", ["unannotated", "pointer", "no_stream"])
def test_unsupported_launch_signature(which):
    with pytest.raises(TypeError, match="is not supported"):
        bad_signature(_ints(1), which=which)


def test_builder_must_return_a_jit_function():
    with pytest.raises(TypeError, match="must return a @flyc.jit function"):
        bad_signature.get(which="kernel")


def test_variant_warning_and_clear():
    noisy = flydsl_kernel(axpb.builder, warn_after=2)
    noisy.get(mul=1, has_y=False, block=256)
    with pytest.warns(UserWarning, match="2 compiled variants"):
        noisy.get(mul=2, has_y=False, block=256)
    noisy.clear()
    assert not noisy.variants
    quiet = flydsl_kernel(axpb.builder, warn_after=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for mul in range(40):
            quiet.get(mul=mul, has_y=False, block=256)


def _is_gfx942():
    return torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942")


@pytest.mark.skipif(not torch.cuda.is_available() or not _is_gfx942(), reason="fp8 FNUZ")
def test_wraps_an_existing_compile_function_bit_exact():
    """Adopting the decorator for an existing ``functools.cache`` compile function."""
    from pyhip.contrib.flydsl.a8w4.kernels import row_quant

    row_quant_kernel = flydsl_kernel(row_quant._compile_row_quant.__wrapped__)

    def quantize_rows(x, valid=None):
        rows, cols = x.shape
        q = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
        s = torch.empty(rows, device=x.device, dtype=torch.float32)
        if rows:
            row_quant_kernel(ptr(x, torch.bfloat16, 8), ptr(q, torch.float8_e4m3fnuz, 4),
                             ptr(s, torch.float32), ptr(valid, torch.int32), rows, cdiv(rows, 4),
                             cols=cols, dynamic_rows=valid is not None)
        return q, s

    for rows, cols in ((1, 256), (37, 2048), (300, 4096)):
        x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
        for valid in (None, torch.tensor([rows // 2], device="cuda", dtype=torch.int32)):
            q_old, s_old = row_quant.quantize_rows(x, valid)
            q_new, s_new = quantize_rows(x, valid)
            k = rows if valid is None else rows // 2
            assert torch.equal(q_old[:k].view(torch.uint8), q_new[:k].view(torch.uint8))
            assert torch.equal(s_old[:k], s_new[:k])


def _host_us(fn, batch=256, rounds=40):
    # Short batches with an untimed sync in between, so the GPU queue never fills up and
    # back-pressures the host; median over rounds.
    times = []
    for r in range(rounds + 5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(batch):
            fn()
        if r >= 5:
            times.append((time.perf_counter() - t0) / batch * 1e6)
    torch.cuda.synchronize()
    return sorted(times)[len(times) // 2]


def bench_host_overhead():
    """Host time per launch of a tiny kernel."""
    x = _ints(256)
    out = torch.empty_like(x)
    exe = axpb.get(mul=3, has_y=False)
    exe(x, None, out, 256, 0)
    cf, xp, op = exe.cf, x.data_ptr(), out.data_ptr()
    cur = torch.cuda.current_stream
    raw = torch._C._cuda_getCurrentRawStream
    rows = [
        ("CompiledFunction, raw stream", lambda: cf(xp, 0, op, 256, 0, raw(0))),
        ("CompiledFunction, torch.cuda.current_stream()",
         lambda: cf(x.data_ptr(), 0, out.data_ptr(), 256, 0, cur())),
        ("axpb.get(...) bound", lambda: exe(x, None, out, 256, 0)),
        ("axpb(..., **key)", lambda: axpb(x, None, out, 256, 0, mul=3, has_y=False)),
        ("axpb_op (alloc + ptr + launch)", lambda: axpb_op(x, 0)),
        ("@flyc.jit called directly", lambda: exe.launch(xp, 0, op, 256, 0, raw(0))),
    ]
    for name, fn in rows:
        print(f"{_host_us(fn):7.2f} us  {name}")


if __name__ == "__main__":
    bench_host_overhead()
