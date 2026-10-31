"""The Aiter-compatible A8W4 API and its real FlyDSL autotune/cache lifecycle."""

import inspect
import json

import pytest
import torch

import pyhip
from pyhip.ops.gemm import tuned_gemm_a8w4 as api

PYHIP = "pyhip_mxfp8_4w_256x256x128_pl1_so0_as0"
AITer = "flydsl_mxpsh_32x32x128_F8_F4_B16_w0_x0_sk1"


def _make(M, N, K, seed=0):
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip
    from aiter.ops.shuffle import shuffle_mxfp8fp4_b, shuffle_scale
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    torch.manual_seed(seed)

    def quant(rows, scale, dtype):
        x = (torch.randn((rows, K), device="cuda") * scale).to(torch.bfloat16)
        q, s = per_1x32_mx_quant_hip(
            x, quant_dtype=dtype, scale_type=dtypes.fp8_e8m0, shuffle=False,
        )
        return q, s.view(torch.uint8)

    a, sa = quant(M, 0.75, dtypes.fp8)
    b, sb = quant(N, 3.0, dtypes.fp4x2)
    a_ref = a.float() * e8m0_to_f32(sa).repeat_interleave(32, dim=1)
    b_ref = mxfp4_to_f32(b) * e8m0_to_f32(sb).repeat_interleave(32, dim=1)
    return dict(
        A=a, B=shuffle_mxfp8fp4_b(b), ScaleA=shuffle_scale(sa), ScaleB=shuffle_scale(sb),
        ref=(a_ref @ b_ref.T).bfloat16(),
    )


def _run(d, **kwargs):
    return api.gemm_a8w4_mxfp8(
        d["A"], d["B"], d["ScaleA"], d["ScaleB"], a_preshuffle=False, **kwargs,
    )


def _args(d, splitk=None, aiter_only=False):
    output = torch.empty_like(d["ref"])
    return api._make_tune_args(
        d["A"], d["B"], d["ScaleA"], d["ScaleB"], output, splitk, aiter_only,
    )


def _limited_configs(**args):
    result = []
    if api._aiter_fits(
        api.aiter_kernel(AITer, args["splitk"]), args["M"], args["N"], args["K"],
    ):
        result.append(api.Config(_impl="aiter", kernel_name=AITer))
    if not args["aiter_only"] and args["splitk"] in (None, 1):
        if api.PYHIP_KERNELS[0].supports(args["M"], args["N"], args["K"]):
            result.append(api.Config(_impl="pyhip", kernel_name=PYHIP))
    return result


def _tuner(configs=_limited_configs, do_bench=None):
    return api.autotune(
        configs=configs, key=api._AUTOTUNE_KEYS, warmup=1, rep=2,
        prune_configs_by=api._prune_invalid_configs,
        artifact_name=api._ARTIFACT_NAME, do_bench=do_bench,
    )(api._gemm_wrapper)


@pytest.fixture
def tuning(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("FLYDSL_AUTOTUNE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    monkeypatch.setattr(api, "record_dispatch", True)
    monkeypatch.setattr(api, "last_dispatch", None)
    tuner = _tuner()
    monkeypatch.setattr(api, "_autotuned_gemm", tuner)
    return tuner


def test_signature_matches_aiter():
    import aiter

    ours = inspect.signature(api.gemm_a8w4_mxfp8)
    theirs = inspect.signature(aiter.gemm_a8w4_mxfp8)
    assert [(p.name, p.kind, p.default) for p in ours.parameters.values()] == [
        (p.name, p.kind, p.default) for p in theirs.parameters.values()
    ]
    assert ours.return_annotation == theirs.return_annotation


def test_passthrough_off_gfx950(monkeypatch):
    calls = []
    sentinel = torch.empty(0)
    monkeypatch.setattr(api, "get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(
        api, "_aiter_gemm_a8w4_mxfp8", lambda *args: calls.append(args) or sentinel,
    )
    monkeypatch.setattr(api, "_autotuned_gemm", None)
    args = (torch.empty(2, 128), torch.empty(16, 64), torch.empty(32, 4), torch.empty(32, 4))
    assert api.gemm_a8w4_mxfp8(*args, kernelName="k", splitk=2) is sentinel
    assert calls == [(*args, api.dtypes.bf16, True, "k", 2)]


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize("rows,K", [(256, 512), (512, 768), (4096, 6144), (256, 16384)])
def test_scale_repack_matches_reference(rows, K):
    scale_a = torch.randint(0, 256, (rows + 256, K // 32), device="cuda", dtype=torch.uint8)
    scale_b = torch.randint(0, 256, (512, K // 32), device="cuda", dtype=torch.uint8)
    got_a, got_b = api._pyhip_scales(scale_a, scale_b, rows, 300, K)
    assert torch.equal(got_a, api._pyhip_scale_reference(scale_a, rows, K))
    assert torch.equal(got_b, api._pyhip_scale_reference(scale_b, 512, K))


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize("shape", [(1, 64, 512), (300, 272, 768), (257, 512, 512), (33, 64, 384)])
def test_reference_decodes_aiter_layout(shape):
    d = _make(*shape)
    result = api._torch_reference(*(d[n] for n in ("A", "B", "ScaleA", "ScaleB")))
    torch.testing.assert_close(result, d["ref"], rtol=0, atol=0)


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize("shape", [(1, 512, 6144), (300, 272, 768), (257, 512, 512), (1024, 1024, 2048)])
@pytest.mark.parametrize("name", [PYHIP, ""])
def test_backends_match_reference(tuning, shape, name):
    d = _make(*shape)
    out = _run(d, kernelName=name)
    assert out.shape == d["ref"].shape and out.dtype == torch.bfloat16
    assert pyhip.calc_diff(out, d["ref"]) <= api._DIFF_LIMIT
    assert api.last_dispatch["source"] == ("explicit" if name else "autotune")
    if name or shape[1] % 32:
        assert api.last_dispatch["_impl"] == "pyhip"


@pytest.mark.usefixtures("cdna4_device")
def test_candidates_are_legal_and_scope_sensitive():
    d = _make(32, 64, 512)
    args = _args(d)
    configs = api._configs(**args)
    assert {c.kwargs["_impl"] for c in configs} == {"aiter", "pyhip"}
    assert len({c.kwargs["kernel_name"] for c in configs}) == len(configs)
    for config in configs:
        impl, _ = api._explicit(config.kwargs["kernel_name"], 32, 64, 512, None)
        assert impl == config.kwargs["_impl"]
    split = api._configs(**_args(d, splitk=2))
    assert split and all(c.kwargs["_impl"] == "aiter" for c in split)
    assert all(api.aiter_kernel(c.kwargs["kernel_name"]).split_k == 2 for c in split)
    assert all(c.kwargs["_impl"] == "aiter" for c in api._configs(**_args(d, aiter_only=True)))
    ragged = api._configs(**_args(_make(32, 272, 512)))
    assert ragged and all(c.kwargs["_impl"] == "pyhip" for c in ragged)


@pytest.mark.usefixtures("cdna4_device")
def test_invalid_candidate_is_not_benchmarked(tuning, monkeypatch, caplog):
    d = _make(32, 64, 512)
    good = api.Config(_impl="aiter", kernel_name=AITer)
    bad = api.Config(_impl="pyhip", kernel_name=PYHIP)

    def wrong_result(kernel, A, B, ScaleA, ScaleB, out, M, N, K):
        out.zero_()

    monkeypatch.setattr(api, "_run_pyhip", wrong_result)
    monkeypatch.setattr(tuning, "configs", [bad, good])
    checked = []

    def measure(fn, **kwargs):
        fn()
        checked.append(api.last_dispatch["_impl"])
        return 1.0

    monkeypatch.setattr(tuning, "_do_bench", measure)
    out = _run(d)
    assert checked == ["aiter"]
    assert "rejected: calc_diff=" in caplog.text
    assert pyhip.calc_diff(out, d["ref"]) <= api._DIFF_LIMIT


@pytest.mark.usefixtures("cdna4_device")
def test_nonfinite_reference_and_all_rejected(tuning, monkeypatch):
    d = _make(32, 64, 512)
    d["ScaleA"].fill_(255)
    with pytest.raises(RuntimeError, match="reference contains nonfinite"):
        _run(d)
    d = _make(32, 64, 512)
    monkeypatch.setattr(tuning, "configs", [api.Config(_impl="pyhip", kernel_name="bad")])
    with pytest.raises(RuntimeError, match="No A8W4 candidate passed"):
        _run(d)


@pytest.mark.usefixtures("cdna4_device")
def test_cache_disk_and_artifact_reuse(tuning, monkeypatch, tmp_path):
    configs = [
        api.Config(_impl="pyhip", kernel_name=PYHIP),
        api.Config(_impl="aiter", kernel_name=AITer),
    ]
    measured = []

    def measure(fn, **kwargs):
        fn()
        measured.append(api.last_dispatch["_impl"])
        return 1.0 if api.last_dispatch["_impl"] == "pyhip" else 2.0

    def unexpected(*args, **kwargs):
        pytest.fail("cache/artifact hit must not search or benchmark")

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(api, "_autotuned_gemm", _tuner(configs, measure))
    d = _make(32, 64, 512)
    out = _run(d)
    assert measured == ["pyhip", "aiter"]
    assert api.last_dispatch["_impl"] == "pyhip"
    assert pyhip.calc_diff(out, d["ref"]) <= api._DIFF_LIMIT
    artifacts = list((tmp_path / "artifacts").glob(f"{api._ARTIFACT_NAME}-*.json"))
    assert len(artifacts) == 1
    exported = json.loads(artifacts[0].read_text())
    assert exported["config"] == configs[0].to_dict()
    assert not any("ptr" in name for name in exported["config"])

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    monkeypatch.setattr(api._autotuned_gemm, "configs", unexpected)
    monkeypatch.setattr(api._autotuned_gemm, "_do_bench", unexpected)
    fresh = _make(32, 64, 512, seed=4)
    cached = _run(fresh)
    assert cached.data_ptr() != out.data_ptr()
    assert pyhip.calc_diff(cached, fresh["ref"]) <= api._DIFF_LIMIT

    # A fresh object first loads the regular disk cache, without consulting artifacts.
    disk = _tuner(unexpected, unexpected)
    monkeypatch.setattr(disk, "_load_artifact", unexpected)
    monkeypatch.setattr(api, "_autotuned_gemm", disk)
    assert pyhip.calc_diff(_run(fresh), fresh["ref"]) <= api._DIFF_LIMIT

    # With an empty regular cache, the actual FlyDSL artifact is the only possible hit.
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(api, "_autotuned_gemm", _tuner(unexpected, unexpected))
    assert pyhip.calc_diff(_run(fresh), fresh["ref"]) <= api._DIFF_LIMIT
    assert api.last_dispatch["_impl"] == "pyhip"
    monkeypatch.setattr(api, "_WARMED_KERNELS", set())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before graph capture"):
        _run(fresh)


@pytest.mark.usefixtures("cdna4_device")
def test_cache_keys_cover_shape_options_and_environment(tuning, monkeypatch):
    d = _make(32, 64, 512)

    def key(call, **kwargs):
        return tuning._make_key((), _args(call, **kwargs))

    original = key(d)
    assert key(_make(32, 64, 512, seed=3)) == original
    assert key(_make(33, 64, 512)) != original
    assert key(d, splitk=2) != original
    assert key(d, aiter_only=True) != original
    monkeypatch.setenv("AITER_TEST_POLICY", "changed")
    assert key(d) != original


@pytest.mark.usefixtures("cdna4_device")
def test_force_retune_and_cached_errors_propagate(tuning, monkeypatch):
    d = _make(32, 64, 512)
    measured = []

    def measure(fn, **kwargs):
        fn()
        measured.append(1)
        return 1.0

    monkeypatch.setattr(tuning, "configs", [api.Config(_impl="pyhip", kernel_name=PYHIP)])
    monkeypatch.setattr(tuning, "_do_bench", measure)
    _run(d)
    _run(d)
    assert len(measured) == 1
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    _run(d)
    assert len(measured) == 2
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")

    def fail(*args):
        raise RuntimeError("cached kernel failed")

    monkeypatch.setattr(api, "_run_pyhip", fail)
    with pytest.raises(RuntimeError, match="cached kernel failed"):
        _run(d)
    assert len(measured) == 2


@pytest.mark.usefixtures("cdna4_device")
def test_explicit_bypass_and_invalid_calls(tuning, monkeypatch):
    d = _make(256, 512, 512)
    monkeypatch.setattr(api, "_autotuned_gemm", None)
    assert pyhip.calc_diff(_run(d, kernelName=PYHIP), d["ref"]) <= api._DIFF_LIMIT
    with pytest.raises(ValueError, match="unknown"):
        _run(d, kernelName="not_a_kernel")
    with pytest.raises(ValueError, match="does not split"):
        _run(d, kernelName=PYHIP, splitk=2)
    for split in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            _run(d, splitk=split)
    with pytest.raises(NotImplementedError, match="a_preshuffle=False"):
        api.gemm_a8w4_mxfp8(d["A"], d["B"], d["ScaleA"], d["ScaleB"])
    with pytest.raises(NotImplementedError, match="output dtype"):
        _run(d, dtype=torch.float16)
    with pytest.raises(ValueError, match="ScaleA"):
        _run(dict(d, ScaleA=d["ScaleA"][:128]))
    with pytest.raises(NotImplementedError):
        api._explicit(AITer, 256, 272, 512, None)


@pytest.mark.usefixtures("cdna4_device")
def test_empty_rows_do_not_tune(tuning, monkeypatch):
    d = _make(1, 64, 512)
    d["A"] = d["A"][:0]
    monkeypatch.setattr(api, "_autotuned_gemm", None)
    out = _run(d)
    assert out.shape == (0, 64) and out.dtype == torch.bfloat16
    assert api.last_dispatch["source"] == "empty"


@pytest.mark.usefixtures("cdna4_device")
def test_cached_backend_must_match_kernel(tuning, monkeypatch):
    d = _make(32, 64, 512)
    tuning.cache[tuning._make_key((), _args(d))] = api.Config(_impl="aiter", kernel_name=PYHIP)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid cached metadata must fail before launch, without retuning")

    monkeypatch.setattr(tuning, "configs", unexpected)
    monkeypatch.setattr(api, "_run_pyhip", unexpected)
    monkeypatch.setattr(api, "_run_aiter", unexpected)
    with pytest.raises(ValueError, match="does not match"):
        _run(d)


@pytest.mark.usefixtures("cdna4_device")
def test_reject_device_switch_before_launch(tuning, monkeypatch):
    d = _make(32, 64, 512)
    monkeypatch.setattr(api, "_native_device", torch.device("cuda:99"))
    with pytest.raises(RuntimeError, match="one GPU per process"):
        _run(d)


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize("name", [PYHIP, "flydsl_mxpsh_32x64x256_F8_F4_B16_w0_x0_sk4", ""])
def test_cuda_graph_replay(tuning, name):
    d = _make(512, 512, 6144)
    _run(d, kernelName=name)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            out = _run(d, kernelName=name)
    stream.synchronize()
    fresh = _make(512, 512, 6144, seed=1)
    for key in ("A", "B", "ScaleA", "ScaleB"):
        d[key].copy_(fresh[key])
    graph.replay()
    torch.cuda.synchronize()
    assert pyhip.calc_diff(out, fresh["ref"]) <= api._DIFF_LIMIT


@pytest.mark.usefixtures("cdna4_device")
def test_cold_capture_cannot_tune(tuning, monkeypatch):
    d = _make(32, 64, 512)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before graph capture"):
        api._prune_invalid_configs([], _args(d))
    with pytest.raises(RuntimeError, match="before graph capture"):
        api._configs(**_args(d))


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize("name", [PYHIP, AITer])
def test_cold_explicit_capture_requires_warmup(tuning, monkeypatch, name):
    d = _make(32, 64, 512)
    monkeypatch.setattr(api, "_WARMED_KERNELS", set())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before graph capture"):
        _run(d, kernelName=name)


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize("name", [PYHIP, ""])
def test_torch_compile_custom_op(tuning, name):
    compiled = torch.compile(
        lambda A, B, SA, SB: api.gemm_a8w4_mxfp8(
            A, B, SA, SB, a_preshuffle=False, kernelName=name,
        ),
        backend="eager", fullgraph=True, dynamic=True,
    )
    for m in (256, 257):
        d = _make(m, 512, 512)
        out = compiled(d["A"], d["B"], d["ScaleA"], d["ScaleB"])
        assert pyhip.calc_diff(out, d["ref"]) <= api._DIFF_LIMIT


@pytest.mark.usefixtures("cdna4_device")
@pytest.mark.parametrize("samples", [2, 3])
def test_benchmark_uses_runtime_tuner_and_checks_timed_outputs(tuning, monkeypatch, tmp_path, samples):
    from types import SimpleNamespace
    from benchmarks.gemm import tune_gemm_a8w4 as benchmark

    measured = []

    def measure(fn, **kwargs):
        fn()
        impl = api.last_dispatch["_impl"]
        measured.append(impl)
        return 1.0 if impl == "pyhip" else 2.0

    monkeypatch.setattr(tuning, "_do_bench", measure)
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))
    row = benchmark.run_case(
        (32, 64, 512),
        SimpleNamespace(retune=True, check_only=False, splitk=None, buffers=2, samples=samples),
    )
    assert measured == ["aiter", "aiter", "pyhip"]
    assert row["status"] == "complete"
    assert row["choices"]["aiter"]["_impl"] == "aiter"
    assert row["choices"]["auto"]["_impl"] == "pyhip"
    assert len(list((tmp_path / "artifacts").glob("*.json"))) == 2
    assert len(row["raw"]) == 2 * samples
    assert {r["buffer"] for r in row["raw"]} == {0, 1}
    assert len({r["output_pointer"] for r in row["raw"]}) == 2 * samples
    benchmark.print_table([row])
    for backend in ("aiter", "auto"):
        assert len(row["correctness"][backend]) == samples
        assert all(r["calc_diff"] <= api._DIFF_LIMIT for r in row["correctness"][backend])
        times = [r["us"] for r in row["raw"] if r["backend"] == backend]
        assert row[backend]["min_us"] == min(times)


@pytest.mark.usefixtures("cdna4_device")
def test_benchmark_checks_earlier_samples_of_reused_buffers(tuning, monkeypatch):
    from types import SimpleNamespace
    from benchmarks.gemm import tune_gemm_a8w4 as benchmark

    original = benchmark.run
    corrupted = False

    def run(buf, backend, splitk):
        nonlocal corrupted
        out = original(buf, backend, splitk)
        if pyhip.cudaPerf.nested_depth and not corrupted:
            out.zero_()
            corrupted = True
        return out

    monkeypatch.setattr(benchmark, "run", run)
    with pytest.raises(AssertionError, match="calc_diff"):
        benchmark.run_case(
            (32, 64, 512),
            SimpleNamespace(retune=False, check_only=False, splitk=None, buffers=2, samples=3),
        )
    assert corrupted


@pytest.mark.usefixtures("cdna4_device")
def test_benchmark_check_only_does_not_use_cuda_perf(tuning, monkeypatch, tmp_path):
    from benchmarks.gemm import tune_gemm_a8w4 as benchmark

    def unexpected(*args, **kwargs):
        pytest.fail("check-only must not instantiate cudaPerf")

    monkeypatch.setattr(pyhip, "cudaPerf", unexpected)
    path = tmp_path / "checked.json"
    benchmark.main(["--shapes", "32,64,512", "--check-only", "--output", str(path)])
    report = json.loads(path.read_text())
    assert report["status"] == "complete"
    assert report["rows"][0]["status"] == "checked"
    assert report["rows"][0]["raw"] == []


@pytest.mark.parametrize("shapes", [[], ["0,64,512"], ["32,64"], ["32,64,512,1"], ["32,16,512"], ["32,64,127"]])
def test_benchmark_rejects_invalid_shapes(shapes):
    from benchmarks.gemm import tune_gemm_a8w4 as benchmark

    with pytest.raises(ValueError):
        benchmark.parse_shapes(shapes)


def test_benchmark_deduplicates_shape_sources(tmp_path):
    from benchmarks.gemm import tune_gemm_a8w4 as benchmark

    path = tmp_path / "shapes.csv"
    path.write_text("M,N,K\n32,64,512\n4096,4096,8192\n")
    assert benchmark.parse_shapes(["32,64,512"], path) == [(32, 64, 512), (4096, 4096, 8192)]
