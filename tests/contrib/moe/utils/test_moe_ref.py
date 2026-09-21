# SPDX-License-Identifier: MIT
"""Independent reference checks; run as a script for a quantization-error table."""

import argparse
import math
from importlib.util import find_spec

import pytest
import torch

import moe_ref
import quantizer

POLICIES = tuple(
    name for name in quantizer.__all__
    if isinstance(getattr(quantizer, name), quantizer.Quantizer)
)
ACTIVATIONS = ("no", "silu", "gelu", "gelu_tanh", "swiglu", "situv2")


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU unavailable")
    return request.param


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def require_policy(name):
    if name not in ("no_quant", "bf16"):
        if find_spec("aiter") is None:
            pytest.skip("Aiter is required for quantization")
        import aiter  # An installed but broken dependency is a test error.


def make_case(device, *, tokens=7, h=128, i=128, e=3, topk=2, gated=True, seed=43):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(shape):
        return torch.randn(shape, generator=generator).to(device)

    x = rand((tokens, h))
    w1 = rand((e, (2 if gated else 1) * i, h)) / math.sqrt(h)
    w2 = rand((e, h, i)) / math.sqrt(i)
    weights = rand((tokens, topk)).softmax(dim=-1)
    ids = torch.randint(e, (tokens, topk), generator=generator).to(device)
    return x, w1, w2, weights, ids


def exact_projection_case(device, *, both=False, **kwargs):
    case = make_case(device, **kwargs)
    for weight in case[1:3] if both else case[1:2]:
        e, n, k = weight.shape
        rows = torch.arange(n, device=device)
        experts = torch.arange(e, device=device)[:, None]
        columns = (rows[None, :] * 17 + experts * 13) % k
        values = weight[experts, rows[None, :], columns].clone() * math.sqrt(k)
        weight.zero_()
        weight[experts, rows[None, :], columns] = values
    return case


def smooth_args(case):
    x, w1, w2, _, _ = case
    e = w1.shape[0]
    s1 = torch.linspace(0.7, 1.3, e * x.shape[-1], device=x.device).reshape(e, 1, -1)
    s2 = torch.linspace(1.2, 0.8, e * w2.shape[-1], device=x.device).reshape(e, 1, -1)
    return dict(a1_smooth_scale=s1, a2_smooth_scale=s2)


def formula(projected, activation, *, gate_mode="separated", limit=None, beta=1.0, linear_beta=1.0, native=False):
    def unary(x):
        if activation == "silu":
            if native:
                return torch.nn.functional.silu(x)
            return x * x.sigmoid()
        if activation == "gelu":
            if native:
                return torch.nn.functional.gelu(x)
            return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
        if activation == "gelu_tanh":
            if native:
                return torch.nn.functional.gelu(x, approximate="tanh")
            return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))
        return x

    if gate_mode == "none":
        return unary(projected)
    n = projected.shape[-1] // 2
    gate, up = (projected[..., 0::2], projected[..., 1::2]) if gate_mode == "interleave" else (
        projected[..., :n], projected[..., n:]
    )
    if activation == "swiglu" and limit is None:
        limit = 7.0
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    if activation == "swiglu":
        return gate * (1.702 * gate).sigmoid() * (up + 1)
    if activation == "situv2":
        if native:
            return (beta * (gate / beta).tanh() * gate.sigmoid()) * (linear_beta * (up / linear_beta).tanh())
        return beta * (gate / beta).tanh() * gate.sigmoid() * linear_beta * (up / linear_beta).tanh()
    return unary(gate) * up


@torch.no_grad()
def dense_oracle(case, name, activation, *, quantize_weights=True, gate_mode="separated",
                 swiglu_limit=None, beta=1.0, linear_beta=1.0, doweight_stage1=False,
                 intermediate_dtype=None, route_dtype=None, output_dtype=torch.float32,
                 quantize_output=False, bias1=None, bias2=None, a1_smooth_scale=None,
                 a2_smooth_scale=None, stage1=None):
    """Small-shape oracle: full per-route weight gather + batched matmul.

    Deliberately different from the implementation's expert/chunk loops. Apply
    each ordinary A/W quantizer to its entire logical tensor, never to chunks.
    """
    x, w1, w2, weights, ids = case
    policy = quantizer.get_quantizer(name)
    active, safe = ids >= 0, ids.clamp_min(0).long()
    smooth = a1_smooth_scale is not None
    if smooth:
        s1 = a1_smooth_scale.reshape(-1, 1, x.shape[-1]).expand(w1.shape[0], 1, -1)
        s2 = a2_smooth_scale.reshape(-1, 1, w2.shape[-1]).expand(w1.shape[0], 1, -1)
    else:
        s1 = s2 = None

    def weight(value, scale):
        if quantize_weights:
            return policy.dequant_w(*policy.apply_w(value, fp8_dtype=torch.float8_e4m3fn, smooth_scale=scale))
        return value.float() if scale is None else value.float() / scale

    w1, w2 = weight(w1, s1), weight(w2, s2)
    if smooth:
        a = x[:, None, :].expand(-1, ids.shape[1], -1)
        a = policy.dequant_a(*policy.apply_a(a, smooth_scale=s1[:, 0][safe]))
    else:
        a = policy.dequant_a(*policy.apply_a(x, fp8_dtype=torch.float8_e4m3fn))[:, None, :]
        a = a.expand(-1, ids.shape[1], -1)
    projected = (a.unsqueeze(-2) @ w1[safe].transpose(-1, -2)).squeeze(-2)
    if doweight_stage1:
        projected *= weights[..., None].float()
    if bias1 is not None:
        projected += bias1[safe].float()
    mid = formula(projected, activation, gate_mode=gate_mode, limit=swiglu_limit,
                  beta=beta, linear_beta=linear_beta, native=True)
    mid = torch.where(active[..., None], mid, 0)
    if stage1 is not None:
        # Dense bmm vs grouped mm may differ by FP32 ulps. First independently
        # validate that boundary, then test downstream quantization using the
        # SAME values; otherwise a halfway FP8/BF16/INT8 value can cross bins.
        # No tolerance is widened to absorb a whole quantization-bin mismatch.
        torch.testing.assert_close(stage1, mid, rtol=2e-5, atol=3e-6)
        mid = stage1
    if intermediate_dtype is not None:
        mid = mid.to(intermediate_dtype).float()
    a2 = policy.dequant_a(*policy.apply_a(
        mid, fp8_dtype=torch.float8_e4m3fn, smooth_scale=s2[:, 0][safe] if smooth else None
    ))
    out = (a2.unsqueeze(-2) @ w2[safe].transpose(-1, -2)).squeeze(-2)
    if bias2 is not None:
        out += bias2[safe].float()
    if not doweight_stage1:
        out *= weights[..., None].float()
    if quantize_output:
        out = quantizer.int8_smoothquant.dequant_output(*quantizer.int8_smoothquant.apply_output(out))
    if route_dtype is not None:
        out = out.to(route_dtype).float()
    return torch.where(active[..., None], out, 0).sum(dim=1).to(output_dtype)


def build(case, name="no_quant", activation="silu", **kwargs):
    x, w1, w2, _, ids = case
    return moe_ref.get(TP=2, model_dim=x.shape[-1], inter_dim=w2.shape[-1] * 2,
                       experts=w1.shape[0], topk=ids.shape[-1], quant_scheme_str=name,
                       activation=activation, fp8_dtype=torch.float8_e4m3fn, **kwargs)


def capture_stage1(monkeypatch, case, name, activation, *, quantize_weights=True, **runtime):
    factory = moe_ref._make_activation
    parts = []

    def recording_factory(*args):
        fn = factory(*args)

        def apply(projected):
            value = fn(projected)
            # Check the nonlinear formula independently BEFORE quantization.
            torch.testing.assert_close(value, formula(projected, activation), rtol=2e-5, atol=3e-6)
            parts.append(value.clone())
            return value

        return apply

    with monkeypatch.context() as patch:
        patch.setattr(moe_ref, "_make_activation", recording_factory)
        actual = build(case, name, activation, quantize_weights=quantize_weights)(*case, **runtime)
    x, w1, w2, _, ids = case
    mid = x.new_zeros((*ids.shape, w2.shape[-1]))
    pieces = iter(parts)
    for e in range(w1.shape[0]):
        mask = ids == e
        if bool(mask.any()):
            mid[mask] = next(pieces)
    assert next(pieces, None) is None
    return actual, mid


@pytest.mark.parametrize("name,activation", [
    ("no_quant", "silu"), ("fp8_per_tensor", "gelu"),
    ("fp8_blockscale", "swiglu"), ("a8w4", "situv2"), ("int8_smoothquant", "silu"),
])
def test_all_policies_and_activations(name, activation, device, record_property, monkeypatch):
    require_policy(name)
    case = make_case(device)
    kwargs = smooth_args(case) if name == "int8_smoothquant" else {}
    actual, mid = capture_stage1(monkeypatch, case, name, activation, **kwargs)
    expected = dense_oracle(case, name, activation, stage1=mid, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()

    uncompressed, mid_uncompressed = capture_stage1(
        monkeypatch, case, name, activation, quantize_weights=False, **kwargs
    )
    expected_uncompressed = dense_oracle(case, name, activation, quantize_weights=False, stage1=mid_uncompressed, **kwargs)
    torch.testing.assert_close(uncompressed, expected_uncompressed, rtol=2e-5, atol=3e-6)
    full_precision = dense_oracle(case, "no_quant", activation)
    record_property("calc_diff_no_weight_compression", moe_ref.calc_diff(actual, uncompressed))
    record_property("calc_diff_no_quant", moe_ref.calc_diff(actual, full_precision))
    # Compression loss is reported, not used to loosen the oracle tolerance.


@pytest.mark.parametrize("name,activation", [(name, "silu") for name in POLICIES]
                         + [("no_quant", act) for act in ACTIVATIONS if act != "silu"])
def test_independent_end_to_end_exact_projection(name, activation, device):
    require_policy(name)
    # One nonzero per W1 row makes its reduction exact irrespective of GEMM
    # grouping. W2 remains fully dense. No implementation intermediates are
    # reused by this independent full-pipeline comparison.
    case = exact_projection_case(device)
    kwargs = smooth_args(case) if name == "int8_smoothquant" else {}
    actual = build(case, name, activation)(*case, **kwargs)
    expected = dense_oracle(case, name, activation, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("name", ["no_quant", "fp8_ptpc", "fp8_blockscale", "a4w4", "int8_smoothquant"])
def test_prequantized_weights_match_raw(name, device):
    require_policy(name)
    case = make_case(device)
    x, w1, w2, weights, ids = case
    kwargs = smooth_args(case) if name == "int8_smoothquant" else {}
    policy = quantizer.get_quantizer(name)
    q1, s1 = policy.apply_w(w1, fp8_dtype=torch.float8_e4m3fn, smooth_scale=kwargs.get("a1_smooth_scale"))
    q2, s2 = policy.apply_w(w2, fp8_dtype=torch.float8_e4m3fn, smooth_scale=kwargs.get("a2_smooth_scale"))
    op = build(case, name)
    expected = op(*case, **kwargs)
    actual = op(x, q1, q2, weights, ids, w1_scale=s1, w2_scale=s2, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("activation", ("no", "silu", "gelu", "gelu_tanh"))
def test_ungated_activations(activation, device):
    case = make_case(device, h=8, i=5, gated=False)
    actual = build(case, activation=activation, gate_mode="none")(*case)
    expected = dense_oracle(case, "no_quant", activation, gate_mode="none")
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_interleaved_layout_and_mock_gate(activation, device):
    case = make_case(device, h=8, i=5)
    x, w1, w2, weights, ids = case
    bias1 = torch.linspace(-1, 1, w1.shape[0] * w1.shape[1], device=device).reshape(w1.shape[:2])
    expected = build(case, activation=activation)(*case, bias1=bias1)
    gate, up = w1.chunk(2, dim=1)
    packed = torch.stack((gate, up), dim=2).flatten(1, 2)
    b_gate, b_up = bias1.chunk(2, dim=1)
    packed_bias = torch.stack((b_gate, b_up), dim=2).flatten(1, 2)
    actual = build(case, activation=activation, gate_mode="interleave")(
        x, packed, w2, weights, ids, bias1=packed_bias
    )
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(build(case, activation=activation, gate_mode="mock_gate_only")(
        *case, bias1=bias1), expected, rtol=0, atol=0)


@pytest.mark.parametrize("activation", ("silu", "swiglu", "situv2"))
@pytest.mark.parametrize("limit", (None, 0.0, 0.5, 7.0))
def test_activation_limits_and_situ_parameters(activation, limit, device):
    # Force projections beyond the clamp boundary (ordinary small random tests
    # cannot distinguish the repository's two SiTU conventions).
    case = make_case(device, tokens=2, h=4, i=4, e=2)
    x, w1, w2, weights, ids = case
    w1.mul_(10)
    opts = dict(swiglu_limit=limit, beta=4.0, linear_beta=25.0)
    actual = build(case, activation=activation, **opts)(*case)
    expected = dense_oracle(case, "no_quant", activation, **opts)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-6)


@pytest.mark.parametrize("doweight", (False, True))
def test_route_weight_and_bias_order(doweight, device):
    x = torch.tensor([[2., -1.], [1., 3.]], device=device)
    w1 = torch.tensor([[[1., 2.], [-1., 1.], [2., 1.], [1., -1.]]], device=device)
    w2 = torch.tensor([[[2., 1.], [1., -2.]]], device=device)
    weights = torch.tensor([[0.2], [1.5]], device=device)
    ids = torch.zeros((2, 1), dtype=torch.int64, device=device)
    case = (x, w1, w2, weights, ids)
    bias1 = torch.tensor([[1., -2., 3., 0.5]], device=device)
    bias2 = torch.tensor([[0.25, -0.75]], device=device)
    actual = build(case, doweight_stage1=doweight)(*case, bias1=bias1, bias2=bias2)
    expected = dense_oracle(case, "no_quant", "silu", doweight_stage1=doweight, bias1=bias1, bias2=bias2)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    other = build(case, doweight_stage1=not doweight)(*case, bias1=bias1, bias2=bias2)
    assert moe_ref.calc_diff(actual, other) > 0.01  # before/after SiLU are not interchangeable


@pytest.mark.parametrize("name", ("no_quant", "fp8_per_tensor", "int8_smoothquant"))
def test_duplicate_inactive_masked_routes_and_chunks(name, device):
    # INT8 truncation can amplify FP32 projection ulps across an integer bin.
    # Keep this routing/chunk test independent of GEMM reduction order.
    factory = exact_projection_case if name == "int8_smoothquant" else make_case
    case = factory(device, tokens=5, e=3, topk=4)
    x, w1, w2, weights, _ = case
    global_ids = torch.tensor([[1,1,3,-1],[5,3,1,5],[0,2,4,-1],[5,5,5,1],[3,1,0,5]], device=device)
    mask = torch.tensor([0,1,0,1,0,1], dtype=torch.int32, device=device)
    local_map = torch.tensor([-1,0,-1,1,-1,2], device=device)
    local_ids = torch.where(global_ids >= 0, local_map[global_ids.clamp_min(0)], -1)
    local_case = (x, w1, w2, weights, local_ids)
    kwargs = smooth_args(case) if name == "int8_smoothquant" else {}
    expected = dense_oracle(local_case, name, "silu", **kwargs)
    for chunk in (1, 3, 256):
        actual = build(local_case, name, token_chunk_size=chunk)(
            x, w1, w2, weights, global_ids, expert_mask=mask, **kwargs
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        assert torch.equal(actual[2], torch.zeros_like(actual[2]))


def test_whole_tensor_activation_scale_domains(device):
    case = exact_projection_case(device, tokens=6, e=3)
    x, w1, w2, weights, ids = case
    x[-1].mul_(100)  # Unrouted token still belongs to the first A scale domain.
    ids[-1].fill_(-1)
    w1[1].mul_(5)
    ids[0] = 0
    ids[1] = 1
    expected = dense_oracle(case, "fp8_per_tensor", "silu")
    for chunk in (1, 2, 256):
        actual = build(case, "fp8_per_tensor", token_chunk_size=chunk)(*case)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    trimmed = tuple(t[:-1] if n in (0, 3, 4) else t for n, t in enumerate(case))
    changed = build(trimmed, "fp8_per_tensor")(*trimmed)
    assert moe_ref.calc_diff(changed, expected[:-1]) > 1e-6


@pytest.mark.parametrize("shared_shape", ("k", "1k", "11k", "ek", "e1k"))
def test_smoothquant_scale_shapes(shared_shape, device):
    case = make_case(device, tokens=5)
    kwargs = smooth_args(case)
    for key, value in tuple(kwargs.items()):
        if shared_shape == "k":
            kwargs[key] = value[0, 0]
        elif shared_shape == "1k":
            kwargs[key] = value[0]
        elif shared_shape == "11k":
            kwargs[key] = value[:1]
        elif shared_shape == "ek":
            kwargs[key] = value[:, 0]
    expected = dense_oracle(case, "int8_smoothquant", "gelu", **kwargs)
    actual = build(case, "int8_smoothquant", "gelu", token_chunk_size=1)(*case, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    ones = {key: torch.ones_like(value) for key, value in kwargs.items()}
    actual = build(case, "int8_smoothquant", "gelu")(*case, **ones)
    expected = build(case, "int8_ptpc", "gelu")(*case)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", ("no_quant", "fp8_ptpc", "int8_smoothquant"))
def test_explicit_rounding_and_output_compression(name, device):
    # Output INT8 truncation also requires an exact second projection for the
    # bitwise bmm/mm comparison; dense GEMMs are covered separately.
    case = exact_projection_case(device, both=True) if name == "int8_smoothquant" else make_case(device)
    kwargs = smooth_args(case) if name == "int8_smoothquant" else {}
    opts = dict(intermediate_dtype=torch.bfloat16, route_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16, quantize_output=name == "int8_smoothquant")
    actual = build(case, name, **opts)(*case, **kwargs)
    expected = dense_oracle(case, name, "silu", **opts, **kwargs)
    # Both algorithms enter the same explicit BF16 rounding boundaries.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16


@pytest.mark.parametrize("name", ["no_quant", "fp8_per_tensor", "a4w4", "int8_smoothquant"])
@pytest.mark.parametrize("tokens", (0, 3))
def test_empty_zero_and_inactive(name, tokens, device):
    require_policy(name)
    case = make_case(device, tokens=tokens)
    x, w1, w2, weights, ids = case
    x.zero_()
    kwargs = smooth_args(case) if name == "int8_smoothquant" else {}
    op = build(case, name)
    output = op(*case, **kwargs)
    assert output.shape == x.shape
    if name == "fp8_per_tensor" and tokens:
        # The upstream per-tensor Torch reference preserves 0/0 as NaN.
        assert torch.isnan(output).all()
    else:
        assert torch.equal(output, torch.zeros_like(output))
    if tokens:
        ids.fill_(-1)
        x.fill_(1)
        output = op(*case, bias1=torch.ones_like(w1[..., 0]), bias2=torch.ones_like(w2[..., 0]), **kwargs)
        assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_noncontiguous_no_mutation_and_output_buffer(dtype, device):
    original = make_case(device, h=12, i=7)
    case = []
    for index, value in enumerate(original):
        if index < 3:
            value = value.to(dtype).transpose(-1, -2).contiguous().transpose(-1, -2)
            value.requires_grad_()
        case.append(value)
    snapshots = [t.detach().clone() for t in case]
    op = build(case, output_dtype=dtype)
    out = torch.empty_like(case[0], memory_format=torch.contiguous_format)
    actual = op(*case, output=out)
    assert actual is out and not actual.requires_grad
    expected = dense_oracle(case, "no_quant", "silu", output_dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=2e-5 if dtype == torch.float32 else 0, atol=2e-6 if dtype == torch.float32 else 0)
    for a, b in zip(case, snapshots):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    alias = case[0].detach().contiguous()
    with pytest.raises(ValueError, match="storage"):
        op(alias, *case[1:], output=alias)


def test_tp_shards_add_to_full_problem(device):
    case = make_case(device, tokens=3, h=8, i=12)
    x, w1, w2, weights, ids = case
    full = moe_ref.get(1, 8, 12, 3, 2)(*case)
    partials = []
    gate, up = w1.chunk(2, dim=1)
    for rank in range(3):
        shard_w1 = torch.cat((gate[:, rank*4:(rank+1)*4], up[:, rank*4:(rank+1)*4]), dim=1)
        shard_w2 = w2[:, :, rank*4:(rank+1)*4]
        partials.append(moe_ref.get(3, 8, 12, 3, 2)(x, shard_w1, shard_w2, weights, ids))
    torch.testing.assert_close(sum(partials), full, rtol=3e-6, atol=5e-7)
    with pytest.raises(ValueError, match="weight1"):
        moe_ref.get(3, 8, 12, 3, 2)(*case)  # do not silently slice global weights


def test_fp32_inside_autocast_and_precision_restoration(device):
    case = make_case(device, h=8, i=8)
    op = build(case)
    expected = op(*case)
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium")
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            actual = op(*case)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.get_float32_matmul_precision() == "medium"
        invalid = build(case, "fp8_blockscale")
        with pytest.raises(ValueError, match="128"):
            invalid(*case)
        assert torch.get_float32_matmul_precision() == "medium"
    finally:
        torch.set_float32_matmul_precision(previous)


def test_calc_diff_definition_and_finite_checks(device):
    a = torch.tensor([1., 2.], device=device)
    b = torch.tensor([2., 4.], device=device)
    assert moe_ref.calc_diff(a, b) == pytest.approx(0.2)
    assert moe_ref.calc_diff(a, -a) == 2
    assert moe_ref.calc_diff(a, a) == 0
    z = torch.zeros_like(a)
    assert moe_ref.calc_diff(z, z) == 0
    assert moe_ref.calc_diff(z, a) == 1
    assert moe_ref.calc_diff(z[:0], z[:0]) == 0
    with pytest.raises(ValueError):
        moe_ref.calc_diff(a, b[:, None])
    with pytest.raises(TypeError):
        moe_ref.calc_diff(a.long(), b.long())
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            moe_ref.calc_diff(a, torch.full_like(a, value))


@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float8_e4m3fnuz))
def test_explicit_fp8_formats(dtype, device):
    case = make_case(device)
    x, w1, w2, weights, ids = case
    op = moe_ref.get(1, 128, 128, 3, 2, "fp8_ptpc", fp8_dtype=dtype)
    q1, s1 = quantizer.fp8_ptpc.apply_w(w1, fp8_dtype=dtype)
    q2, s2 = quantizer.fp8_ptpc.apply_w(w2, fp8_dtype=dtype)
    torch.testing.assert_close(op(*case), op(x, q1, q2, weights, ids, w1_scale=s1, w2_scale=s2), rtol=0, atol=0)


def accuracy_report(device, activations, policies, tokens=17, seed=43):
    """Report loss; not a benchmark or an optimized-kernel acceptance test."""
    case = make_case(device, tokens=tokens, seed=seed)
    print(f"device={device}; TP=2 H=128 global_I=256 local_I=128 E=3 topk=2 tokens={tokens} seed={seed}")
    print("A/W source=FP32; intermediate/route/output=FP32; gate_mode=separated; weights at stage2")
    print("| activation | quantizer | calc_diff vs uncompressed W (same A policy) | calc_diff vs no_quant |")
    print("|---|---|---:|---:|")
    for activation in activations:
        full = build(case, activation=activation)(*case)
        for name in policies:
            kwargs = smooth_args(case) if name == "int8_smoothquant" else {}
            actual = build(case, name, activation)(*case, **kwargs)
            raw_w = build(case, name, activation, quantize_weights=False)(*case, **kwargs)
            print(f"| {activation} | {name} | {moe_ref.calc_diff(actual, raw_w):.9g} | {moe_ref.calc_diff(actual, full):.9g} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--activation", nargs="+", choices=ACTIVATIONS, default=["silu"])
    parser.add_argument("--quant", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    torch.set_num_threads(1)
    accuracy_report(args.device, args.activation, args.quant, args.tokens, args.seed)