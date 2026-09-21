# SPDX-License-Identifier: MIT
"""Standalone quantizer tests: no imports from model tests or GEMM kernels."""

from importlib.util import find_spec

import pytest
import torch
import quantizer as quant


POLICIES = (
    "no_quant", "bf16", "fp8_ptpc", "fp8_per_tensor", "fp8_blockscale",
    "a16w8_per_channel", "a16w8_per_tensor", "a16w8_blockscale",
    "fp8_per_token_per_tensor", "a16w4", "a8w4", "a4w4",
    "int8_ptpc", "int8_smoothquant", "fp8_int4_ptpc",
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU unavailable")
    return request.param


def data(shape, device):
    return torch.randn(shape, generator=torch.Generator().manual_seed(43)).to(device)


def same_bytes(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    torch.testing.assert_close(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8), atol=0, rtol=0)


def require_aiter():
    if find_spec("aiter") is None:
        pytest.skip("Aiter is required for quantization")
    # A broken installed package is an error, not missing optional coverage.
    import aiter


def require_policy(name):
    if name not in ("no_quant", "bf16"):
        require_aiter()


def options(name, x):
    kwargs = {"fp8_dtype": torch.float8_e4m3fn}
    if name == "int8_smoothquant":
        kwargs["smooth_scale"] = torch.ones(x.shape[-1], device=x.device)
    return kwargs


@pytest.mark.parametrize("name", POLICIES)
def test_uniform_interface_and_no_mutation(name, device):
    require_policy(name)
    policy = getattr(quant, name)
    assert quant.get_quantizer(name.replace("_", "-")) is policy
    a = data((3, 2, 128), device)
    w = data((2, 128, 256), device).transpose(-1, -2)  # noncontiguous [E,N,K]
    for side, x in (("a", a), ("w", w)):
        original = x.clone()
        q, scale = getattr(policy, "apply_" + side)(x, **options(name, x))
        reconstructed = getattr(policy, "dequant_" + side)(q, scale)
        assert reconstructed.shape == x.shape and reconstructed.dtype == torch.float32
        assert (q.is_contiguous() or name == "no_quant") and q.device == x.device
        assert reconstructed.device == x.device and torch.isfinite(reconstructed).all()
        if scale is not None:
            assert scale.is_contiguous() and scale.device == x.device
        assert not reconstructed.requires_grad
        torch.testing.assert_close(x, original, atol=0, rtol=0)
        # Sanity only; exact rounding/scale/layout are checked separately below.
        relative_mse = (reconstructed - x).square().sum() / x.square().sum()
        assert relative_mse.item() < (0.12 if name == "fp8_int4_ptpc" else 0.03)


@pytest.mark.parametrize("name", POLICIES)
@pytest.mark.parametrize("empty", [False, True])
def test_zero_and_empty(name, empty, device):
    require_policy(name)
    policy = getattr(quant, name)
    for side, shape in (("a", (0 if empty else 3, 2, 128)), ("w", (0 if empty else 2, 128, 128))):
        x = torch.zeros(shape, device=device)
        q, scale = getattr(policy, "apply_" + side)(x, **options(name, x))
        restored = getattr(policy, "dequant_" + side)(q, scale)
        if name == "fp8_per_tensor" and side == "a":
            # Aiter's per-tensor reference does not replace scale=0.
            torch.testing.assert_close(scale, torch.zeros_like(scale), atol=0, rtol=0)
            torch.testing.assert_close(restored, torch.full_like(x, float("nan")), equal_nan=True)
            continue
        torch.testing.assert_close(restored, x, atol=0, rtol=0)
        if scale is not None:
            if scale.dtype == torch.float8_e8m0fnu:
                from aiter.utility.fp4_utils import e8m0_to_f32
                scale = e8m0_to_f32(scale)
            assert torch.isfinite(scale).all() and (scale > 0).all()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
def test_fp8_rows_and_per_expert_tensor(dtype, device):
    require_aiter()
    x = data((2, 4, 128), device)
    x[0] *= 0.125
    limit = torch.finfo(dtype).max
    for policy, dims, side in (
        (quant.fp8_ptpc, (-1,), "a"), (quant.fp8_ptpc, (-1,), "w"),
        (quant.fp8_per_tensor, (0, 1, 2), "a"), (quant.fp8_per_tensor, (-2, -1), "w"),
        (quant.fp8_per_token_per_tensor, (-1,), "a"),
        (quant.fp8_per_token_per_tensor, (-2, -1), "w"),
    ):
        q, scale = getattr(policy, "apply_" + side)(x, fp8_dtype=dtype)
        expected = x.abs().amax(dim=dims, keepdim=True) / limit
        if policy is quant.fp8_per_tensor and side == "a":
            expected = expected.reshape(1)
        torch.testing.assert_close(scale, expected, atol=0, rtol=0)
        same_bytes(q, (x / expected).to(dtype))


def test_blockscale_axes(device):
    require_aiter()
    from aiter.ops.quant import pertoken_quant

    x = data((2, 256, 384), device)
    x[0, :128, :128] = 0
    x[:, 128:, :128] *= 1e-38
    q, scale = quant.fp8_blockscale.apply_w(x, fp8_dtype=torch.float8_e4m3fn)
    assert scale.shape == (2, 2, 3)
    dequant = quant.fp8_blockscale.dequant_w(q, scale)
    for e in range(2):
        for n in range(2):
            for k in range(3):
                tile = x[e, n*128:(n+1)*128, k*128:(k+1)*128]
                qt, expected = pertoken_quant(tile.reshape(1, -1), quant_dtype=torch.float8_e4m3fn)
                qt, expected = qt.reshape(128, 128), expected.reshape(())
                torch.testing.assert_close(scale[e,n,k], expected, atol=0, rtol=0)
                same_bytes(q[e,n*128:(n+1)*128,k*128:(k+1)*128], qt)
                torch.testing.assert_close(dequant[e,n*128:(n+1)*128,k*128:(k+1)*128], qt.float()*expected, atol=0, rtol=0)
    a = data((3, 2, 256), device)
    a[0] = 0
    a[1] *= 1e-38
    aq, sa = quant.fp8_blockscale.apply_a(a, fp8_dtype=torch.float8_e4m3fn)
    expected_q, expected = pertoken_quant(a.view(3,2,2,128), quant_dtype=torch.float8_e4m3fn)
    same_bytes(aq, expected_q.reshape_as(aq))
    torch.testing.assert_close(sa, expected.squeeze(-1), atol=0, rtol=0)
    torch.testing.assert_close(quant.fp8_blockscale.dequant_a(aq,sa), (aq.float().view(3,2,2,128)*sa[...,None]).view_as(a), atol=0, rtol=0)


def test_int8_and_int4_truncation(device):
    require_aiter()
    x = torch.tensor([[127., 2.5, 1.5, -2.5, -1.5, -127.]], device=device)
    expected = torch.tensor([[127, 2, 1, -2, -1, -127]], dtype=torch.int8, device=device)
    for fn in (quant.int8_ptpc.apply_a, quant.int8_ptpc.apply_w):
        q, scale = fn(x)
        torch.testing.assert_close(q, expected, atol=0, rtol=0)
        assert scale.item() == 1
    w = torch.tensor([[7., 2.5, 1.5, -2.5, -1.5, -7.]], device=device)
    q, scale = quant.fp8_int4_ptpc.apply_w(w)
    torch.testing.assert_close(q, torch.tensor([[7,2,1,-2,-1,-7]],dtype=torch.int8,device=device),atol=0,rtol=0)
    assert scale.item() == 1 and q.shape == w.shape  # unpacked, not MXFP4


def test_mx_packing_scale_and_fp8_rounding(device):
    require_aiter()
    # Each group's amax equals the format maximum, so scale=1 (E8M0 byte127).
    x = torch.tensor([[0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.] * 2], device=device)
    q, scale = quant.a4w4.apply_a(x)
    expected_bytes = torch.tensor([[0x10,0x32,0x54,0x76,0x98,0xBA,0xDC,0xFE] * 2], dtype=torch.uint8, device=device)
    torch.testing.assert_close(q.view(torch.uint8),expected_bytes,atol=0,rtol=0)
    assert scale.view(torch.uint8).item() == 127
    torch.testing.assert_close(quant.a4w4.dequant_a(q,scale),x,atol=0,rtol=0)
    for dtype,maximum in ((torch.float8_e4m3fn,448.),(torch.float8_e4m3fnuz,240.)):
        # Halfway E4M3 values round to even: 1.0625 -> 1; 1.1875 -> 1.25.
        x = torch.tensor([[maximum,1.0625,1.1875,-1.0625] * 8],device=device)
        q,scale = quant.a8w4.apply_a(x,fp8_dtype=dtype)
        assert scale.view(torch.uint8).item() == 127
        torch.testing.assert_close(q.float()[0,:4],torch.tensor([maximum,1.,1.25,-1.],device=device),atol=0,rtol=0)
    with pytest.raises(ValueError): quant.a8w4.dequant_a(q,torch.ones_like(scale,dtype=torch.float32))


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
def test_input_dtypes_and_autograd_boundary(dtype,device):
    require_aiter()
    x = data((2,128),device).to(dtype).requires_grad_()
    for policy in (quant.bf16,quant.fp8_ptpc,quant.fp8_blockscale,quant.int8_ptpc):
        q,s = policy.apply_a(x,fp8_dtype=torch.float8_e4m3fn)
        assert not q.requires_grad and (s is None or not s.requires_grad)
        assert torch.isfinite(policy.dequant_a(q,s)).all()


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
def test_no_quant_exact_identity(dtype,device):
    x = data((2,64,128),device).to(dtype).transpose(-1,-2).requires_grad_()
    assert quant.get_quantizer("no-quant") is quant.no_quant
    for side in ("a","w"):
        result = getattr(quant.no_quant,"apply_"+side)(x)
        q,scale = result
        assert q is x and scale is None and q.requires_grad
        ref = getattr(quant.no_quant,"dequant_"+side)(*result)
        assert not ref.requires_grad and ref.dtype == torch.float32
        torch.testing.assert_close(ref,x.float(),atol=0,rtol=0)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
@pytest.mark.parametrize("magnitude", [0.0, 1e-45, 1e-38, 1e-15, 1.0])
def test_fp8_boundaries_match_aiter(dtype, magnitude, device):
    require_aiter()
    from aiter.ops.quant import pertoken_quant, per_tensor_quant

    x = data((2, 4, 128), device) * magnitude
    x[0] = 0
    for fn, reference, operand in (
        (quant.fp8_ptpc.apply_a, pertoken_quant, x),
        (quant.fp8_per_tensor.apply_a, per_tensor_quant, x),
        (quant.fp8_per_tensor.apply_w, pertoken_quant, x.flatten(-2)),
        (quant.fp8_blockscale.apply_a, pertoken_quant, x.reshape(2, 4, 1, 128)),
    ):
        q, scale = fn(x, fp8_dtype=dtype)
        expected_q, expected_scale = reference(operand, quant_dtype=dtype)
        same_bytes(q, expected_q.reshape(q.shape))
        torch.testing.assert_close(scale, expected_scale.reshape(scale.shape), rtol=0, atol=0)


def test_smoothquant_and_output_quant(device):
    require_aiter()
    from aiter.ops.quant import pertoken_quant

    a, w = data((3,2,128), device), data((2,128,128), device)
    routes = torch.tensor([[0,1],[1,0],[1,1]], device=device)
    scales = torch.stack((torch.full((128,),2.,device=device),torch.full((128,),0.5,device=device)))
    sa, sw = scales[routes], scales[:,None,:]
    for side, x, smooth, transformed in (("a",a,sa,a*sa),("w",w,sw,w/sw)):
        original = x.clone()
        quant.validate_input(x, smooth_scale=smooth)
        q, s = getattr(quant.int8_smoothquant,"apply_"+side)(x,smooth_scale=smooth)
        qr, sr = (pertoken_quant(x, x_scale=smooth, quant_dtype=torch.int8) if side == "a"
              else pertoken_quant(transformed, quant_dtype=torch.int8))
        same_bytes(q,qr)
        torch.testing.assert_close(s,sr,atol=0,rtol=0)
        torch.testing.assert_close(x,original,atol=0,rtol=0)
    out = data((3,2,128),device)
    q,s = quant.int8_smoothquant.apply_output(out)
    qr,sr = pertoken_quant(out.reshape(3,2,4,32), quant_dtype=torch.int8)
    same_bytes(q,qr.reshape_as(out))
    torch.testing.assert_close(s,sr.squeeze(-1),atol=0,rtol=0)
    restored = quant.int8_smoothquant.dequant_output(q,s)
    torch.testing.assert_close(restored,(q.float().reshape(3,2,4,32)*s[...,None]).reshape_as(out),atol=0,rtol=0)


@pytest.mark.parametrize("magnitude", [0.0, 1.0, 1e-15])
def test_aiter_reference_parity(device, magnitude):
    require_aiter()
    from aiter.ops.quant import pertoken_quant, per_1x32_f4_quant, per_1x32_f8_scale_f8_quant
    from aiter import dtypes
    x = (data((3,2,128),device) * magnitude).to(torch.bfloat16)
    q,s = quant.fp8_ptpc.apply_a(x,fp8_dtype=dtypes.fp8)
    qr,sr = pertoken_quant(x,quant_dtype=dtypes.fp8)
    same_bytes(q,qr)
    torch.testing.assert_close(s,sr,atol=0,rtol=0)
    for fn,ref,kwargs in (
        (quant.a4w4.apply_a,per_1x32_f4_quant,{}),
        (quant.a8w4.apply_a,per_1x32_f8_scale_f8_quant,{"scale_type":dtypes.fp8_e8m0}),
    ):
        q,s = fn(x,fp8_dtype=dtypes.fp8)
        qr,sr = ref(x,**kwargs)
        same_bytes(q,qr)
        same_bytes(s,sr.reshape(s.shape))
    for fn, maximum in ((quant.int8_ptpc.apply_a, None), (quant.int8_ptpc.apply_w, None),
                        (quant.fp8_int4_ptpc.apply_w, 7)):
        q,s = fn(x)
        qr,sr = pertoken_quant(x,quant_dtype=torch.int8,dtypeMax=maximum)
        same_bytes(q,qr)
        torch.testing.assert_close(s,sr,atol=0,rtol=0)


def test_mx_hip_parity():
    if not torch.cuda.is_available():
        pytest.skip("GPU unavailable")
    require_aiter()
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip
    x = data((8,128),"cuda").to(torch.bfloat16)
    for fn,dtype in ((quant.a4w4.apply_a,dtypes.fp4x2),(quant.a8w4.apply_a,dtypes.fp8)):
        q,s = fn(x,fp8_dtype=dtypes.fp8)
        qr,sr = per_1x32_mx_quant_hip(x,quant_dtype=dtype,shuffle=False,scale_type=dtypes.fp8_e8m0)
        same_bytes(q,qr)
        same_bytes(s,sr)


def require_hip():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm GPU")
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("HIP quantization validation targets gfx950")
    require_aiter()


@pytest.mark.parametrize("name,mode,dtype", [
    ("fp8_ptpc", "per_Token", torch.float8_e4m3fn),
    ("fp8_blockscale", "per_1x128", torch.float8_e4m3fn),
    ("fp8_per_token_per_tensor", "per_Token", torch.float8_e4m3fn),
    ("fp8_int4_ptpc", "per_Token", torch.float8_e4m3fn),
    ("int8_ptpc", "per_Token", torch.int8),
    ("a8w4", "per_1x32", torch.float8_e4m3fn),
    ("a4w4", "per_1x32", torch.float4_e2m1fn_x2),
])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_hip_activation_matches_aiter(name, mode, dtype, input_dtype):
    require_hip()
    from aiter.ops.quant import get_hip_quant
    from aiter.ops.enum import QuantType

    policy = getattr(quant, name)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for shape in ((7, 512), (7, 2, 256)):
            for magnitude in (0.0, 1.0, 1e-15):
                x = (data(shape, "cuda") * magnitude).to(input_dtype)
                x[0] = 0
                original = x.clone()
                q, scale = policy.apply_a(x, backend="hip")
                operand, kwargs = x, dict(quant_dtype=dtype)
                if mode == "per_1x32":
                    operand = x.view(-1, x.shape[-1])
                    kwargs.update(scale_type=torch.float8_e8m0fnu, shuffle=False)
                expected_q, expected_scale = get_hip_quant(getattr(QuantType, mode))(operand, **kwargs)
                same_bytes(q, expected_q.view(q.shape))
                same_bytes(scale, expected_scale.view(scale.shape))
                torch.testing.assert_close(x, original, rtol=0, atol=0)
        empty = torch.empty((0, 2, 256), dtype=input_dtype, device="cuda")
        q, scale = policy.apply_a(empty, backend="hip")
        assert q.shape == (0, 2, 128 if name == "a4w4" else 256)
        assert scale.shape == (0, 2, 8 if mode == "per_1x32" else 2 if mode == "per_1x128" else 1)
    stream.synchronize()


@pytest.mark.parametrize("routed", [False, True])
def test_hip_fused_smoothquant_and_output(routed):
    require_hip()
    from aiter.ops.quant import smooth_per_token_scaled_quant, get_hip_quant
    from aiter.ops.enum import QuantType

    x = data((7, 2, 256), "cuda").bfloat16()
    x[0] = 0
    smooth = torch.rand(x.shape if routed else (256,), device="cuda") + 0.5
    q, scale = quant.int8_smoothquant.apply_a(x, smooth_scale=smooth, backend="hip")
    expected_q, expected_scale = torch.empty_like(q), torch.empty_like(scale)
    mapping = torch.arange(14, dtype=torch.int32, device="cuda") if routed else None
    smooth_per_token_scaled_quant(expected_q, x, expected_scale, smooth.view(-1, 256),
                                  smooth_scale_map=mapping)
    same_bytes(q, expected_q)
    same_bytes(scale, expected_scale)
    q, scale = quant.int8_smoothquant.apply_output(x, backend="hip")
    expected_q, expected_scale = get_hip_quant(QuantType.per_Token)(x.view(7, 2, 8, 32), quant_dtype=torch.int8)
    same_bytes(q, expected_q.view_as(q))
    same_bytes(scale, expected_scale.squeeze(-1))


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_per_tensor_uses_torch(input_dtype, device, monkeypatch):
    require_aiter()
    from aiter.ops import quant as aiter_quant

    monkeypatch.setattr(aiter_quant, "get_hip_quant", lambda *a, **k: pytest.fail("per-tensor selected HIP"))
    x = data((7, 2, 258), device).to(input_dtype)
    for value in (x, x[..., ::2], x[:0], torch.zeros_like(x)):
        original = value.clone()
        empty_scale = torch.zeros(1, device=device) if value.numel() == 0 else None
        expected_q, expected_scale = aiter_quant.per_tensor_quant(
            value, scale=empty_scale, quant_dtype=torch.float8_e4m3fn)
        for backend in ("torch", "hip"):
            # The generic HIP preference cannot select a per-tensor HIP implementation.
            q, scale = quant.fp8_per_tensor.apply_a(value, fp8_dtype=torch.float8_e4m3fn, backend=backend)
            same_bytes(q, expected_q)
            same_bytes(scale, expected_scale)
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["fp8_ptpc", "fp8_blockscale", "a8w4", "a4w4", "int8_ptpc", "int8_smoothquant"])
def test_hip_graph_updates_input(name):
    require_hip()
    policy = getattr(quant, name)
    x = data((7, 2, 256), "cuda").bfloat16()
    kwargs = {**options(name, x), "backend": "hip"}
    policy.apply_a(x, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, scale = policy.apply_a(x, **kwargs)
    x.mul_(2)
    graph.replay()
    expected_q, expected_scale = policy.apply_a(x, **kwargs)
    same_bytes(q, expected_q)
    same_bytes(scale, expected_scale)


def test_hip_rejects_unsupported_inputs():
    x = torch.ones((2, 128))
    with pytest.raises(ValueError, match="backend"):
        quant.fp8_ptpc.apply_a(x, backend="unknown")
    with pytest.raises(ValueError, match="backend"):
        quant.fp8_per_tensor.apply_a(x, backend="flydsl")
    with pytest.raises(ValueError, match="transpose_scale"):
        quant.fp8_per_tensor.apply_a(x, transpose_scale=True)
    with pytest.raises(ValueError, match="ROCm"):
        quant.fp8_ptpc.apply_a(x, backend="hip")
    with pytest.raises(ValueError, match="transpose_scale"):
        quant.fp8_blockscale.apply_a(x, transpose_scale=True)
    assert quant.no_quant.apply_a(x, backend="hip")[0] is x
    require_hip()
    with pytest.raises(ValueError, match="FP16/BF16"):
        quant.fp8_ptpc.apply_a(x.cuda(), backend="hip")
    x = x.cuda().bfloat16()
    with pytest.raises(ValueError, match="contiguous"):
        quant.fp8_ptpc.apply_a(x[:, ::2], backend="hip")
    with pytest.raises(ValueError, match="32"):
        quant.fp8_ptpc.apply_a(x[:, :127].contiguous(), backend="hip")
    with pytest.raises(ValueError, match="128"):
        quant.fp8_blockscale.apply_a(x[:, :96].contiguous(), backend="hip")
    with pytest.raises(ValueError, match="native FP8"):
        quant.fp8_ptpc.apply_a(x, backend="hip", fp8_dtype=torch.float8_e4m3fnuz)
    with pytest.raises(ValueError, match="transposed"):
        quant.fp8_ptpc.apply_a(x, backend="hip", transpose_scale=True)


def test_invalid_scale_and_grouping():
    require_aiter()
    with pytest.raises(ValueError): quant.fp8_blockscale.apply_a(torch.ones(2,129))
    with pytest.raises(ValueError): quant.fp8_blockscale.apply_w(torch.ones(2,127,128))
    with pytest.raises(ValueError): quant.a8w4.apply_a(torch.ones(2,33))
    with pytest.raises(ValueError): quant.a16w4.apply_w(torch.ones(2,33))
    with pytest.raises(ValueError): quant.int8_smoothquant.apply_output(torch.ones(2,33))
    with pytest.raises(ValueError): quant.fp8_ptpc.apply_a(torch.ones(2,32),smooth_scale=torch.ones(32))
    with pytest.raises(ValueError): quant.int8_smoothquant.apply_a(torch.ones(2,32))
    with pytest.raises(TypeError): quant.int8_smoothquant.apply_a(torch.ones(2,32),smooth_scale=1.0)
    with pytest.raises(TypeError): quant.int8_smoothquant.apply_a(torch.ones(2,32),smooth_scale=torch.ones(32,dtype=torch.int8))
    with pytest.raises(ValueError): quant.int8_smoothquant.apply_a(torch.ones(2,32),smooth_scale=torch.ones(3,2,32))
    for value in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError): quant.validate_input(torch.ones(2,32),smooth_scale=torch.full((32,),value))
    with pytest.raises(ValueError): quant.validate_input(torch.full((2,32),float("nan")))
    q,s = quant.fp8_blockscale.apply_a(torch.ones(3,2,256))
    with pytest.raises(ValueError): quant.fp8_blockscale.dequant_a(q,s.reshape(6,2))


@pytest.mark.parametrize("name",[name for name in POLICIES if name not in ("bf16","no_quant")])
def test_graph_updates_input(name):
    if not torch.cuda.is_available():
        pytest.skip("GPU unavailable")
    require_policy(name)
    policy = getattr(quant,name)
    x = data((2,128,128),"cuda").to(torch.bfloat16)
    kwargs = options(name,x)
    policy.apply_a(x,**kwargs); policy.apply_w(x,**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qa,sa = policy.apply_a(x,**kwargs)
        qw,sw = policy.apply_w(x,**kwargs)
    x.mul_(2)
    graph.replay()
    ea,esa = policy.apply_a(x,**kwargs)
    ew,esw = policy.apply_w(x,**kwargs)
    for actual,expected in ((qa,ea),(qw,ew),(sa,esa),(sw,esw)):
        if actual is not None:
            same_bytes(actual,expected)

