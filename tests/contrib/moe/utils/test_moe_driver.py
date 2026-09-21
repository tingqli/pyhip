# SPDX-License-Identifier: MIT
"""Existing driver validation matrix, using config -> (prepare, run)."""

import importlib
import inspect
import sys

import pytest
import torch

if __package__:
    from . import moe_driver as driver, moe_ref, quantizer, tune_aiter
else:
    import moe_driver as driver
    import moe_ref
    import quantizer
    import tune_aiter


def _config(H=512, I=256, E=4, topk=2, quant="no_quant", activation="silu", **kwargs):
    return driver.MOEconfig(H, I, E, topk, quant, activation, **{"preshuffle": True, **kwargs})


def _inputs(H=128, I=128, E=4, topk=2, M=7, device="cpu", seed=43):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((M, H), generator=gen, device=device).bfloat16()
    w1 = (torch.randn((E, 2 * I, H), generator=gen, device=device) / H**0.5).bfloat16()
    w2 = (torch.randn((E, H, I), generator=gen, device=device) / I**0.5).bfloat16()
    values, ids = torch.randn((M, E), generator=gen, device=device).topk(topk, dim=-1)
    return x, w1, w2, values.softmax(-1), ids.int()


def _gpu():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm GPU")
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("this validation matrix targets gfx950")


def _oracle(config, w1, w2, **options):
    dtype = options.pop("output_dtype", torch.float32)
    prepare, run = driver.ref(config._replace(preshuffle=False, output_dtype=dtype), **options)
    return run, prepare(w1, w2), dtype


def _expected(oracle, x, ti, tw):
    run, weights, dtype = oracle
    return run(x, weights, ti, tw, torch.empty_like(x, dtype=dtype))


@pytest.mark.parametrize("quant", ["no_quant", "fp8_ptpc", "fp8_blockscale"])
def test_reference_prepare_contract(quant):
    x, w1, w2, tw, ti = _inputs()
    config = _config(128, 128, quant=quant, preshuffle=False, output_dtype=torch.float32)
    prepare, run = driver.ref(config)
    weights = prepare(w1, w2)
    expected = moe_ref.get(1, 128, 128, 4, 2, quant)(x, w1, w2, tw, ti)
    output = torch.empty_like(x, dtype=config.output_dtype)
    assert run(x, weights, ti, tw, output) is output
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert weights["w1"].data_ptr() != w1.data_ptr() and weights["w2"].data_ptr() != w2.data_ptr()
    assert not hasattr(w2, "is_shuffled")
    _, other = driver.ref(config)
    copied = {**weights, "w1s": driver._copy(weights["w1s"])}
    torch.testing.assert_close(other(x, copied, ti, tw, output), expected, rtol=0, atol=0)


def test_reference_gate_pair_and_output():
    x, w1, w2, tw, ti = _inputs()
    prepare, run = driver.ref(_config(128, 128, preshuffle=False))
    pair = w1[:, :128].contiguous(), w1[:, 128:].contiguous()
    a, b = prepare(w1, w2), prepare(pair, w2)
    expected = run(x, a, ti, tw, torch.empty_like(x)).clone()
    output = torch.full_like(x, float("nan"))
    assert run(x, b, ti, tw, output) is output
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    # Preparing another set must not replace weights captured by the first call.
    w1.zero_()
    prepare(w1, w2)
    torch.testing.assert_close(run(x, a, ti, tw, output), expected, rtol=0, atol=0)


@pytest.mark.parametrize("compress", [True, False])
def test_reference_weight_baseline(compress):
    x, w1, w2, tw, ti = _inputs()
    config = _config(128, 128, quant="fp8_ptpc", preshuffle=False, output_dtype=torch.float32)
    prepare, run = driver.ref(config, quantize_weights=compress)
    expected = moe_ref.get(1, 128, 128, 4, 2, "fp8_ptpc", quantize_weights=compress)(x, w1, w2, tw, ti)
    actual = run(x, prepare(w1, w2), ti, tw, torch.empty_like(expected))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reference_duplicate_and_empty_routes():
    x, w1, w2, tw, ti = _inputs()
    ti[:, 1] = ti[:, 0]
    config = _config(128, 128, preshuffle=False, output_dtype=torch.float32)
    prepare, run = driver.ref(config)
    weights = prepare(w1, w2)
    driver.validate_routes(config, ti, tw, sorted_routes=False)
    expected = moe_ref.get(1, 128, 128, 4, 2)(x, w1, w2, tw, ti)
    torch.testing.assert_close(run(x, weights, ti, tw, torch.empty_like(expected)), expected)
    assert run(x[:0], weights, ti[:0], tw[:0], torch.empty_like(expected[:0])).shape == (0, 128)


def test_removed_driver_options_are_explicit():
    with pytest.raises(NotImplementedError, match="SmoothQuant"):
        driver.ref(_config(quant="int8_smoothquant", preshuffle=False))
    for name in ("doweight_stage1", "fp8_dtype", "gate_mode"):
        with pytest.raises(TypeError):
            _config(**{name: None})


@pytest.mark.parametrize("name,config_updates,tiles", [
    ("jit_fused", {"quant_scheme": "fp8_ptpc"}, {}),
    ("jit_fused", {"inter_dim_tp": 512}, {}),
    ("jit_splitk", {}, {"block_m": 256}),
    ("jit_splitk", {"quant_scheme": "a8w4"}, {}),
    ("jit_splitk", {"quant_scheme": "fp8_blockscale"}, {}),
    ("jit_splitk", {"quant_scheme": "a16w8_blockscale"}, {}),
    ("jit_splitk", {"quant_scheme": "fp8_ptpc"}, {"block_m": 64}),
    ("jit_batch1", {"model_dim": 128}, {}),
    ("fly_decode", {}, {"block_m": 32}),
    ("fly_splitk", {"quant_scheme": "fp8_blockscale"}, {}),
    ("prefill_bf16", {"quant_scheme": "fp8_ptpc"}, {}),
    ("prefill_fp8", {"quant_scheme": "a16w8_per_tensor"}, {}),
    ("prefill_fp8", {"quant_scheme": "fp8_per_tensor", "inter_dim_tp": 192}, {"sort_block_m": 32}),
    ("prefill_fp8", {"quant_scheme": "fp8_ptpc"}, {"down_path": "8x1", "stage2_blockn": 128, "down_output_padding_bytes": 0}),
    ("prefill_bf16", {"activation": "situv2"}, {}),
    ("aiter", {"quant_scheme": "int8_ptpc"}, {}),
])
def test_unsupported_contracts_are_explicit(name, config_updates, tiles):
    with pytest.raises(NotImplementedError):
        getattr(driver, name)(_config()._replace(**config_updates), **tiles)


def test_prepare_rejects_quantized_or_shuffled_sources():
    _, w1, w2, _, _ = _inputs()
    prepare, _ = driver.ref(_config(128, 128, preshuffle=False))
    with pytest.raises(ValueError, match="weight1"):
        prepare(w1.float(), w2)
    w1.is_shuffled = True
    with pytest.raises(ValueError, match="unshuffled"):
        prepare(w1, w2)


def test_batch_limit_and_route_preconditions():
    config = _config()
    _, run = driver.jit_batch1(config)
    x, w1, w2, tw, ti = _inputs(512, 256)
    with pytest.raises(NotImplementedError, match="tokens"):
        run(x, dict(w1=w1, w2=w2), ti, tw, torch.empty_like(x))
    ti[:, 1] = ti[:, 0]
    driver.validate_routes(config, ti, tw, sorted_routes=False)
    with pytest.raises(ValueError, match="distinct"):
        driver.validate_routes(config, ti, tw)
    ti[0, 0] = 4
    with pytest.raises(ValueError, match="topk_ids"):
        driver.validate_routes(config, ti, tw, sorted_routes=False)


def test_registry_and_user_example():
    config = _config(1024, 256)
    prepare, run = driver.registry["jit_splitk_64_128_True"](config)
    assert callable(prepare) and callable(run) and run.activation_path == "bf16"
    with pytest.raises(NotImplementedError, match="wrong first K-group scale"):
        driver.registry["jit_splitk_64_128_True"](config._replace(quant_scheme="fp8_blockscale"))
    for name, factory in driver.registry.items():
        assert name == factory.__name__ and list(inspect.signature(factory).parameters) == ["config"]
    assert "jit_splitk_64_128_False" not in driver.registry
    _, run = driver.registry["jit_blockscale_256_256_tiled_True"](_config(4096, 1536, quant="fp8_blockscale"))
    assert run.activation_path == "native_fp8_blockscale"
    for name in driver.__all__:
        assert hasattr(driver, name)


def test_tuned_situ_contract_is_explicit():
    with pytest.raises(NotImplementedError, match="beta=4"):
        driver.aiter(_config(4096, 256, quant="a16w4", activation="situv2"), tuned_config={})


# Same complete-kernel matrix, including dynamic A quantization and reductions.
GPU_CASES = [
    ("jit_splitk", "no_quant", {}, 512, 256, 7),
    ("jit_splitk", "fp8_ptpc", {}, 512, 256, 17),
    ("jit_splitk", "fp8_per_tensor", {"block_m": 32}, 512, 256, 7),
    ("jit_splitk", "no_quant", {"block_m": 64}, 1024, 256, 65),
    ("jit_splitk", "a16w8_per_channel", {"block_m": 32}, 1024, 256, 7),
    ("jit_splitk", "a16w4", {}, 1024, 256, 7),
    ("jit_batch1", "no_quant", {}, 512, 256, 1),
    ("jit_batch1", "fp8_ptpc", {}, 512, 256, 1),
    ("jit_batch", "no_quant", {}, 512, 256, 7),
    ("jit_batch", "fp8_ptpc", {}, 512, 256, 17),
    ("jit_fused", "no_quant", {}, 512, 256, 7),
    ("jit_loopn", "fp8_ptpc", {}, 1024, 256, 17),
    ("jit_loopn", "fp8_ptpc", {"atomic_write": False}, 1024, 256, 17),
    ("jit_mxfp4", "a4w4", {}, 1024, 256, 65),
    ("fly_splitk", "no_quant", {}, 512, 256, 17),
    ("fly_splitk", "fp8_ptpc", {}, 512, 256, 17),
    ("fly_splitk", "a16w4", {}, 1024, 256, 17),
    ("fly_decode", "no_quant", {}, 512, 256, 7),
    ("fly_decode", "fp8_per_tensor", {}, 512, 256, 7),
    ("fly_decode", "a16w4", {}, 1024, 256, 7),
    ("prefill_bf16", "no_quant", {}, 512, 256, 65),
    ("prefill_fp8", "fp8_ptpc", {}, 512, 256, 65),
    ("prefill_fp8", "fp8_per_tensor", {}, 512, 256, 65),
    ("prefill_fp8", "fp8_per_token_per_tensor", {}, 512, 256, 65),
    ("prefill_fp8", "fp8_ptpc", {"down_path": "1x4_64x256", "stage2_blockn": 256, "down_output_padding_bytes": 32}, 512, 256, 65),
    ("prefill_fp8", "fp8_ptpc", {"sort_block_m": 256, "stage1_blockm": 64, "down_path": "8x1", "stage2_blockn": 128, "down_output_padding_bytes": 64}, 512, 192, 65),
    ("prefill_fp8", "fp8_per_tensor", {"down_path": "8x1_compact", "stage2_blockn": 128, "down_output_padding_bytes": 128}, 512, 320, 129),
]


@pytest.mark.parametrize("name,quant,kwargs,H,I,M", GPU_CASES,
                         ids=[f"{i:02d}-{c[0]}-{c[1]}" for i, c in enumerate(GPU_CASES)])
def test_gpu_complete_pipeline(name, quant, kwargs, H, I, M):
    _gpu()
    config = _config(H, I, quant=quant)
    prepare, run = getattr(driver, name)(config, **kwargs)
    x, w1, w2, tw, ti = _inputs(H, I, M=M, device="cuda")
    weights = prepare(w1, w2)
    oracle = _oracle(config, w1, w2, intermediate_dtype=torch.bfloat16)
    out = torch.full_like(x, float("nan"))
    for iteration in range(2):
        driver.validate_routes(config, ti, tw)
        actual = run(x, weights, ti, tw, out)
        assert actual is out
        diff = moe_ref.calc_diff(actual, _expected(oracle, x, ti, tw))
        assert diff < (0.02 if quant in ("a16w4", "a4w4") else 0.002), (name, quant, iteration, diff)
        out.fill_(float("nan"))
        x = (x.float() * -0.75).bfloat16()
        ti = (ti + 1) % 4
        tw = tw.flip(-1).contiguous()


@pytest.mark.parametrize("quant", ["fp8_ptpc", "fp8_per_tensor", "fp8_blockscale", "fp8_per_token_per_tensor"])
@pytest.mark.parametrize("magnitude", [0.0, 1.0, 1e-15, 1e-30, 1e-38])
def test_gpu_dynamic_quant_matches_aiter(quant, magnitude):
    _gpu()
    from aiter.fused_moe import get_quant
    from aiter.ops.enum import QuantType
    from aiter.ops.quant import get_torch_quant

    mode = {"fp8_ptpc": QuantType.per_Token, "fp8_per_tensor": QuantType.per_Tensor,
            "fp8_blockscale": QuantType.per_1x128, "fp8_per_token_per_tensor": QuantType.per_Token}[quant]
    for shape in ((7, 512), (7, 2, 256)):
        x = (torch.randn(shape, device="cuda") * magnitude).bfloat16()
        x[0] = 0
        actual_q, actual_s = driver._quant_a(quantizer.get_quantizer(quant), x)
        backend = get_torch_quant if quant == "fp8_per_tensor" else get_quant
        q, s = backend(mode)(x, quant_dtype=torch.float8_e4m3fn)
        torch.testing.assert_close(actual_q.view(torch.uint8), q.view(torch.uint8), atol=0, rtol=0)
        torch.testing.assert_close(actual_s, s, atol=0, rtol=0)


@pytest.mark.parametrize("quant", ["no_quant", "fp8_ptpc", "fp8_per_tensor", "fp8_blockscale"])
def test_gpu_aiter_public_default(quant):
    _gpu()
    config = _config(4096, 256, quant=quant)
    x, w1, w2, tw, ti = _inputs(4096, 256, device="cuda")
    prepare, run = driver.aiter(config)
    weights = prepare(w1, w2)
    module = importlib.import_module("aiter.fused_moe")
    old_resolver = module.get_2stage_cfgs
    output = torch.empty_like(x)
    assert run(x, weights, ti, tw, output) is output
    assert module.get_2stage_cfgs is old_resolver and module.kernel_bench_callable is None
    oracle = _oracle(config, w1, w2, intermediate_dtype=torch.bfloat16)
    assert moe_ref.calc_diff(output, _expected(oracle, x, ti, tw)) < 0.002


@pytest.mark.parametrize("quant", ["a16w4", "a8w4", "a4w4"])
def test_gpu_aiter_mx_contract(quant):
    _gpu()
    config = _config(4096, 256, quant=quant, activation="situv2", beta=4.0, linear_beta=25.0)
    x, w1, w2, tw, ti = _inputs(4096, 256, device="cuda")
    prepare, run = driver.aiter(config)
    y = run(x, prepare(w1, w2), ti, tw, torch.empty_like(x))
    md = next(iter(run.metadata.values()))
    oracle = _oracle(config, w1, w2, intermediate_dtype=None if md.fuse_quant else torch.bfloat16)
    assert moe_ref.calc_diff(y, _expected(oracle, x, ti, tw)) < 0.002


def _tuned_row(**updates):
    # Catalog-valid CK configurations; synthetic timings, not perf evidence.
    spec = tune_aiter._spec(1, 4096, 256, 4, 2, 8, "no_quant", "silu")
    row = {**tune_aiter._untuned_row(spec, "gfx950", 256),
           **dict(zip(tune_aiter._RESULTS, (32, 0, 1,
               "moe_ck2stages_gemm1_256x32x64x128_1x4_TypeCast_v1_Nswizzle0_Quant0_MulRoutedWeight0_silu_B16_B16_B16",
               "0%", 1,
               "moe_ck2stages_gemm2_256x32x64x64_1x4_TypeCast_v1_Nswizzle0_Quant0_MulRoutedWeight1_B16_B16_B16",
               "0%", 2, 0, 0, 0, 0, 0)))}
    row.update(updates)
    return row


def test_gpu_aiter_serial_config_switch_and_first_call_check(monkeypatch):
    _gpu()
    module = importlib.import_module("aiter.fused_moe")
    first = _tuned_row()
    latest = _tuned_row(kernelName2="moe_ck2stages_gemm2_256x32x128x128_1x4_TypeCast_v1_Nswizzle0_Quant0_MulRoutedWeight1_B16_B16_B16")
    old_resolver = module.get_2stage_cfgs
    config = _config(4096, 256)
    (pa, a), (pb, b) = (driver.aiter(config, tuned_config=row) for row in (latest, first))
    x, w1, w2, tw, ti = _inputs(4096, 256, device="cuda")
    wa, wb = pa(w1, w2), pb(w1, w2)
    expected = _expected(_oracle(config, w1, w2, intermediate_dtype=torch.bfloat16), x, ti, tw)
    for run, weights, row in ((a, wa, latest), (b, wb, first), (a, wa, latest)):
        output = run(x, weights, ti, tw, torch.empty_like(x))
        assert moe_ref.calc_diff(output, expected) < 0.002
        assert driver._kernel_name(run.metadata[(x.device, 8)].stage2, 2) == row["kernelName2"]
        assert module.get_2stage_cfgs is old_resolver and module.kernel_bench_callable is None
    # On a steady call, get_2stage_cfgs must not be replaced by the audit wrapper.
    fused = module.fused_moe
    with monkeypatch.context() as patch:
        resolve = module.get_2stage_cfgs
        def observe(*args, **kwargs):
            assert module.get_2stage_cfgs is observe
            return resolve(*args, **kwargs)
        patch.setattr(module, "get_2stage_cfgs", observe)
        torch.testing.assert_close(a(x, wa, ti, tw, torch.empty_like(x)), output)
    assert module.fused_moe is fused
    with pytest.raises(ValueError, match="tuned token"):
        a(x[:1], wa, ti[:1], tw[:1], torch.empty_like(x[:1]))


@pytest.mark.parametrize("family,quant,options", [
    ("jit_splitk", "fp8_ptpc", {}), ("fly_decode", "fp8_ptpc", {}),
    ("prefill_fp8", "fp8_ptpc", {}), ("prefill_fp8", "fp8_per_tensor", {}),
    ("aiter", "fp8_ptpc", {}),
    ("jit_blockscale", "fp8_blockscale", {"down_path": "persistent"}),
    ("jit_blockscale", "fp8_blockscale", {"down_path": "tiled"}),
])
def test_gpu_current_stream_zero_input(family, quant, options):
    _gpu()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        config = _config(4096, 256, quant=quant)
        x, w1, w2, tw, ti = _inputs(4096, 256, device="cuda")
        x.zero_()
        prepare, run = getattr(driver, family)(config, **options)
        output = run(x, prepare(w1, w2), ti, tw, torch.empty_like(x))
        if quant == "fp8_per_tensor":
            expected = _expected(_oracle(config, w1, w2, intermediate_dtype=torch.bfloat16), x, ti, tw)
            assert torch.isnan(expected).all()
            torch.testing.assert_close(output.float(), expected, rtol=0, atol=0, equal_nan=True)
        else:
            torch.testing.assert_close(output.float(), torch.zeros_like(output.float()), rtol=0, atol=0)
    stream.synchronize()


@pytest.mark.parametrize("family", ["fly_decode", "fly_splitk", "prefill_bf16", "prefill_fp8"])
@pytest.mark.parametrize("activation,limit", [("swiglu", 7.0), ("situv2", 7.0)])
def test_gpu_fly_activation_contract(family, activation, limit):
    _gpu()
    quant = "fp8_ptpc" if family == "prefill_fp8" else "no_quant"
    config = _config(512, 256, quant=quant, activation=activation, swiglu_limit=limit, beta=4.0, linear_beta=25.0)
    x, w1, w2, tw, ti = _inputs(512, 256, M=33, device="cuda")
    prepare, run = getattr(driver, family)(config)
    actual = run(x, prepare(w1, w2), ti, tw, torch.empty_like(x))
    expected = _expected(_oracle(config, w1, w2, intermediate_dtype=torch.bfloat16), x, ti, tw)
    assert moe_ref.calc_diff(actual, expected) < 0.002


@pytest.mark.parametrize("quant", ["a4w4", "a8w4"])
def test_gpu_dynamic_mx_roundup_policy(quant):
    _gpu()
    from aiter.ops.quant import get_hip_quant
    from aiter.ops.enum import QuantType

    x = torch.randn((19, 256), device="cuda").bfloat16()
    x[0] = 0
    policy = quantizer.get_quantizer(quant)
    q, s = policy.apply_a(x)
    actual_q, actual_s = get_hip_quant(QuantType.per_1x32)(x, quant_dtype=q.dtype, scale_type=torch.float8_e8m0fnu, shuffle=False)
    torch.testing.assert_close(actual_q.view(torch.uint8), q.view(torch.uint8), atol=0, rtol=0)
    torch.testing.assert_close(policy.dequant_a(actual_q, actual_s), policy.dequant_a(q, s), atol=0, rtol=0)


def test_gpu_compact_builds_full_and_tail_every_call(monkeypatch):
    _gpu()
    compact = importlib.import_module("pyhip.contrib.flydsl.moe_gemm_2stage.gemm2_8x1_compact")
    allocate, seen = compact.allocate_task_buffers, []
    def observe(*args, **kwargs):
        result = allocate(*args, **kwargs)
        seen.append(result)
        return result
    monkeypatch.setattr(compact, "allocate_task_buffers", observe)
    config = _config(quant="fp8_ptpc")
    x, w1, w2, tw, ti = _inputs(512, 256, M=21017, device="cuda")
    prepare, run = driver.prefill_fp8(config, stage2_blockn=128, down_path="8x1_compact", down_output_padding_bytes=64)
    weights = prepare(w1, w2)
    oracle = _oracle(config, w1, w2, intermediate_dtype=torch.bfloat16)
    out = run(x, weights, ti, tw, torch.empty_like(x))
    full, tail = seen[-1][2].tolist()
    assert full > 0 and tail > 0
    assert moe_ref.calc_diff(out, _expected(oracle, x, ti, tw)) < 0.002
    ti[:, 0], ti[:, 1] = 0, 1
    assert moe_ref.calc_diff(run(x, weights, ti, tw, out), _expected(oracle, x, ti, tw)) < 0.002
    assert len(seen) == 2 and seen[-1][2].tolist() != [full, tail]


@pytest.mark.parametrize("family", ["jit_batch1", "fly_decode"])
def test_gpu_direct_route_duplicates(family):
    _gpu()
    config = _config()
    x, w1, w2, tw, ti = _inputs(512, 256, M=1 if family == "jit_batch1" else 33, device="cuda")
    ti[:, 1] = ti[:, 0]
    driver.validate_routes(config, ti, tw, sorted_routes=False)
    prepare, run = getattr(driver, family)(config)
    actual = run(x, prepare(w1, w2), ti, tw, torch.empty_like(x))
    expected = _expected(_oracle(config, w1, w2, intermediate_dtype=torch.bfloat16), x, ti, tw)
    assert moe_ref.calc_diff(actual, expected) < 0.002


@pytest.mark.parametrize("family,quant,kwargs,H,I", [
    ("jit_splitk", "no_quant", {"block_m": 32, "block_n": 32, "down_bn": 32}, 512, 256),
    ("jit_splitk", "fp8_ptpc", {"block_m": 64, "block_n": 64}, 512, 384),
    ("jit_fused", "no_quant", {"block_m": 32, "block_n": 64}, 1024, 384),
    ("jit_mxfp4", "a4w4", {"block_m": 128}, 1024, 256),
    ("fly_splitk", "a16w4", {"gate_mode": "separated"}, 1024, 256),
    ("prefill_bf16", "no_quant", {"stage1_blockn": 256, "stage1_tile_k": 128}, 512, 256),
    ("prefill_fp8", "fp8_ptpc", {"stage1_blockn": 256, "stage1_tile_k": 256}, 512, 256),
    ("prefill_fp8", "fp8_ptpc", {"sort_block_m": 256, "stage1_blockm": 32, "stage2_blockn": 128, "down_path": "8x1", "down_output_padding_bytes": 0}, 512, 640),
])
def test_gpu_additional_tiles(family, quant, kwargs, H, I):
    _gpu()
    config = _config(H, I, quant=quant)
    x, w1, w2, tw, ti = _inputs(H, I, M=129, device="cuda")
    prepare, run = getattr(driver, family)(config, **kwargs)
    actual = run(x, prepare(w1, w2), ti, tw, torch.empty_like(x))
    expected = _expected(_oracle(config, w1, w2, intermediate_dtype=torch.bfloat16), x, ti, tw)
    assert moe_ref.calc_diff(actual, expected) < 0.002


@pytest.mark.parametrize("updates,tiles", [
    ({}, {"block_m": 64}), ({}, {"block_n": 128}), ({"preshuffle": False}, {}),
    ({"quant_scheme": "a16w8_blockscale"}, {}), ({"activation": "gelu"}, {}),
    ({}, {"down_path": "unknown"}), ({}, {"num_oc_splits": 3}), ({}, {"persistent_workers": 0}),
    ({"model_dim": 384}, {}), ({"inter_dim_tp": 384}, {}), ({"inter_dim_tp": 128}, {}), ({}, {"down_bn": 128}),
    ({}, {"down_path": "tiled", "persistent_workers": 256}),
    ({}, {"down_path": "tiled", "num_oc_splits": 2}), ({}, {"down_path": "tiled", "down_bn": 128}),
])
def test_jit_blockscale_range_checks(updates, tiles):
    with pytest.raises((ValueError, NotImplementedError)):
        driver.jit_blockscale(_config(1024, 256, quant="fp8_blockscale")._replace(**updates), **tiles)


@pytest.mark.parametrize("family,quant,activation,kwargs", [
    ("aiter", "bf16", "silu", {}), ("aiter", "a8w4", "situv2", {"gate_mode": "interleave"}),
    ("jit_splitk", "fp8_ptpc", "silu", {}), ("jit_batch1", "bf16", "silu", {}),
    ("jit_batch", "bf16", "silu", {}), ("jit_fused", "bf16", "silu", {}),
    ("jit_loopn", "fp8_ptpc", "silu", {}), ("jit_mxfp4", "a4w4", "silu", {}),
    ("fly_splitk", "a16w4", "silu", {}), ("fly_decode", "fp8_per_tensor", "silu", {}),
    ("prefill_bf16", "bf16", "silu", {}), ("prefill_fp8", "fp8_ptpc", "silu", {}),
    ("jit_blockscale", "fp8_blockscale", "silu", {}),
])
def test_gpu_preshuffle_is_prepare_only(family, quant, activation, kwargs, monkeypatch):
    _gpu()
    shuffle = importlib.import_module("aiter.ops.shuffle")
    original, calls = shuffle.shuffle_weight, []
    def observe(value, *args, **options):
        result = original(value, *args, **options)
        calls.append((value, result))
        return result
    monkeypatch.setattr(shuffle, "shuffle_weight", observe)
    config = _config(1024, 256, quant=quant, activation=activation)
    x, w1, w2, tw, ti = _inputs(1024, 256, M=1, device="cuda")
    prepare, run = getattr(driver, family)(config, **kwargs)
    weights = prepare(w1, w2)
    assert len(calls) == 2
    for prepared, (_, shuffled) in zip((weights["w1"], weights["w2"]), calls):
        assert prepared is shuffled and prepared.is_shuffled is True
    assert not getattr(w1, "is_shuffled", False) and not getattr(w2, "is_shuffled", False)
    q1, _ = quantizer.get_quantizer(quant).apply_w(w1)
    torch.testing.assert_close(calls[0][0].view(torch.uint8), q1.view(torch.uint8), rtol=0, atol=0)
    with pytest.raises(ValueError, match="unshuffled"):
        prepare(original(w1), w2)
    count = len(calls)
    run(x, weights, ti, tw, torch.empty_like(x))
    torch.cuda.synchronize()
    assert len(calls) == count


@pytest.mark.parametrize("preshuffle", [False, True])
def test_gpu_aiter_a4w4_layouts(preshuffle, monkeypatch):
    _gpu()
    shuffle = importlib.import_module("aiter.ops.shuffle")
    original, count = shuffle.shuffle_weight, []
    def observe(*args, **kwargs):
        count.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(shuffle, "shuffle_weight", observe)
    config = _config(4096, 256, quant="a4w4", preshuffle=preshuffle)
    x, w1, w2, tw, ti = _inputs(4096, 256, M=33, device="cuda")
    prepare, run = driver.aiter(config)
    weights = prepare(w1, w2)
    assert len(count) == (2 if preshuffle else 0)
    assert bool(getattr(weights["w1"], "is_shuffled", False)) is preshuffle
    if not preshuffle:
        q, _ = quantizer.a4w4.apply_w(w1)
        torch.testing.assert_close(weights["w1"].view(torch.uint8), q.view(torch.uint8), rtol=0, atol=0)
    oracle = _oracle(config, w1, w2, intermediate_dtype=torch.bfloat16)
    for _ in range(2):
        actual = run(x, weights, ti, tw, torch.empty_like(x))
        assert moe_ref.calc_diff(actual, _expected(oracle, x, ti, tw)) < 0.002
        ti = (ti + 1) % 4


@pytest.mark.parametrize("shape", [(1, 1024), (17, 1024), (17, 2, 256), (33, 3, 384)])
@pytest.mark.parametrize("magnitude", [1.0, 1e-15, 1e-30])
def test_gpu_blockscale_transposed_activation_scales(shape, magnitude):
    _gpu()
    from aiter.fused_moe import get_quant
    from aiter.ops.enum import QuantType
    x = (torch.randn(shape, device="cuda") * magnitude).bfloat16()
    x[0] = 0
    x.reshape(-1, shape[-1])[1:, :128] = 0
    aq, scale = driver._quant_a(quantizer.fp8_blockscale, x, transpose_scale=True)
    expected_q, expected_scale = get_quant(QuantType.per_1x128)(x, quant_dtype=torch.float8_e4m3fn)
    restored = scale.reshape(shape[-1] // 128, -1).t().reshape(expected_scale.shape)
    torch.testing.assert_close(aq.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(restored, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("H,I,M,kwargs", [
    (1024, 256, 1, {}), (1024, 128, 17, {"down_path": "tiled"}),
    (1024, 128, 129, {"block_m": 128, "down_path": "tiled"}),
    (1024, 256, 257, {"block_m": 128, "num_oc_splits": 1, "persistent_workers": 1}),
    (4096, 256, 513, {"num_oc_splits": 4}), (1024, 128, 7, {"down_path": "tiled"}),
    (1024, 384, 257, {"down_path": "tiled", "block_m": 128}), (4096, 1536, 33, {"down_path": "tiled"}),
])
def test_gpu_jit_blockscale_complete(H, I, M, kwargs):
    _gpu()
    config = _config(H, I, quant="fp8_blockscale")
    x, w1, w2, tw, ti = _inputs(H, I, M=M, device="cuda")
    k1 = torch.pow(2.0, torch.arange(H // 128, device="cuda") % 7 - 3).repeat_interleave(128)
    k2 = torch.pow(2.0, torch.arange(I // 128, device="cuda") % 5 - 2).repeat_interleave(128)
    n1 = torch.pow(2.0, torch.arange(2 * I // 128, device="cuda") % 3 - 1).repeat_interleave(128)
    w1 = (w1.float() * k1 * n1[None, :, None]).bfloat16()
    w2 = (w2.float() * k2).bfloat16()
    prepare, run = driver.jit_blockscale(config, **kwargs)
    weights = prepare(w1, w2)
    oracle = _oracle(config, w1, w2, intermediate_dtype=torch.bfloat16, route_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
    out = torch.full_like(x, float("nan"))
    for _ in range(2):
        actual = run(x, weights, ti, tw, out)
        assert actual is out
        assert moe_ref.calc_diff(actual, _expected(oracle, x, ti, tw)) < 0.002
        out.fill_(float("nan"))
        x = (-0.5 * x.float()).bfloat16()
        ti = (ti + 1) % 4
        tw = tw.flip(-1).contiguous()


def test_gpu_jit_blockscale_queue_lifecycle(monkeypatch):
    _gpu()
    module = importlib.import_module("pyhip.contrib.moe_gemm_8wave")
    zeros, launch, sort = torch.zeros, module.moe_gemm_8wave_down, driver._sort
    counters, valid_counts, used = [], [], []
    def track_zeros(*args, **kwargs):
        tensor = zeros(*args, **kwargs)
        if tensor.numel() == 1 and tensor.dtype == torch.int32 and tensor.device.type == "cuda":
            counters.append(tensor)
        return tensor
    def track_sort(*args, **kwargs):
        result = sort(*args, **kwargs)
        valid_counts.append(result[3])
        return result
    def track_launch(*args):
        counter = next(t for t in counters if t.data_ptr() == args[-1])
        assert counter.item() == 0
        used.append(counter)
        return launch(*args)
    monkeypatch.setattr(torch, "zeros", track_zeros)
    monkeypatch.setattr(module, "moe_gemm_8wave_down", track_launch)
    monkeypatch.setattr(driver, "_sort", track_sort)
    config = _config(1024, 256, quant="fp8_blockscale")
    block_m, splits, workers = 128, 2, 1
    prepare, run = driver.jit_blockscale(config, block_m=block_m, num_oc_splits=splits, persistent_workers=workers)
    x, w1, w2, tw, ti = _inputs(1024, 256, M=4097, device="cuda")
    weights = prepare(w1, w2)
    oracle = _oracle(config, w1, w2, intermediate_dtype=torch.bfloat16, route_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
    for M in (4097, 1, 129, 4097):
        ids = ti[:M].clone()
        ids[:, 0], ids[:, 1] = 0, 1
        actual = run(x[:M], weights, ids, tw[:M], torch.empty_like(x[:M]))
        assert moe_ref.calc_diff(actual, _expected(oracle, x[:M], ids, tw[:M])) < 0.002
        assert used[-1].item() == valid_counts[-1][0].item() // block_m * splits + workers
    assert len(used) == 4 and len({t.data_ptr() for t in used}) == 4


def test_gpu_jit_blockscale_model_shape():
    _gpu()
    H, I, E, K = 4096, 128, 400, 20
    config = _config(H, I, E, K, "fp8_blockscale")
    x, w1, w2, tw, ti = _inputs(H, I, E, K, M=33, device="cuda")
    prepare, run = driver.jit_blockscale(config, down_path="tiled")
    weights = prepare(w1, w2)
    oracle = _oracle(config, w1, w2, intermediate_dtype=torch.bfloat16, route_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
    for M in (1, 33):
        actual = run(x[:M], weights, ti[:M], tw[:M], torch.empty_like(x[:M]))
        assert moe_ref.calc_diff(actual, _expected(oracle, x[:M], ti[:M], tw[:M])) < 0.002


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))