"""完整 tuned MoE 和 benchmark 调优前置流程的回归。"""

import csv
import inspect
from types import SimpleNamespace

import pytest
import torch
import aiter

from benchmarks.moe import bench_tuned_moe as bench
from benchmarks.moe.bench_tuned_moe import prepare
from pyhip import calc_diff
from pyhip.ops.moe import tuned_moe as tm


@pytest.fixture(autouse=True)
def _cuda_default():
    if not torch.cuda.is_available():
        pytest.skip("requires ROCm GPU")
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    yield
    torch.set_default_device(previous)


def _block_call(tokens=17, hidden=512, inter=256, shuffled=(True, True)):
    args = SimpleNamespace(
        dtype="fp8", quant="block", seed=8, gate_mode="separated", preshuffle="off",
        routing="balanced", activation="silu", beta=None, linear_beta=None, swiglu_limit=None,
    )
    model = dict(HIDDEN_SIZE=hidden, INTER_SIZE=inter, TP=1, E=4, TOPK=2)
    call, _ = prepare(model, tokens, args)
    # 不只覆盖 benchmark 的小输入；扩大幅度以检验 SiLU 和中间激活量化。
    call["hidden_states"].mul_(50)
    from aiter.ops.shuffle import shuffle_weight

    for name, enabled in zip(("w1", "w2"), shuffled):
        if enabled:
            call[name] = shuffle_weight(call[name])
    return call


def _check(result, call, reference):
    assert result is call["output"]
    assert result.shape == reference.shape and result.dtype == reference.dtype
    assert torch.isfinite(result).all()
    assert calc_diff(reference, result) <= .02


def _tune_arguments(call):
    bound = inspect.signature(tm.fused_moe).bind(**call)
    bound.apply_defaults()
    return tm._make_tune_args(bound.arguments)


def _native_configs(call):
    return [config for config in tm._configs(**_tune_arguments(call))
            if config.all_kwargs()["_impl"] == "jit_blockscale"]


def test_aiter_signature():
    assert inspect.signature(tm.fused_moe) == inspect.signature(tm._aiter_fused_moe)


def test_dispatch_recording(monkeypatch):
    assert tm.record_dispatch is False
    call = _block_call()
    tuning = _tune_arguments(call)
    stale = {"_impl": "previous"}
    monkeypatch.setattr(tm, "last_dispatch", stale)
    monkeypatch.setattr(tm, "_run_jit", lambda *args: call["output"])
    monkeypatch.setattr(tm, "_aiter_fused_moe", lambda *args, **kwargs: kwargs["output"])
    tm._fmoe_wrapper(**tuning, _impl="jit_batch")
    assert tm.last_dispatch is stale

    monkeypatch.setattr(tm, "record_dispatch", True)
    tm._fmoe_wrapper(**tuning, _impl="jit_batch", tile_m_gate=32, tile_n_down=128, block_n=1024)
    assert tm.last_dispatch == dict(_impl="jit_batch", tile_m_gate=32, tile_m_down=16,
                                    tile_n_gate=64, tile_n_down=128, block_n=1024,
                                    decode_alg="splitk", down_path="default", padding=None, num_oc_splits=1)
    assert all(value is None or type(value) in (str, int, float, bool)
               for value in tm.last_dispatch.values())
    tm._fmoe_wrapper(**tuning, _impl="aiter")
    assert tm.last_dispatch == {"_impl": "aiter"}
    tm.last_dispatch = stale
    # 不支持本地调优的参数直接转交 Aiter，也必须覆盖旧记录。
    assert tm.fused_moe(**call, hidden_pad=128) is call["output"]
    assert tm.last_dispatch == {"_impl": "aiter"}
    monkeypatch.setattr(tm, "record_dispatch", False)
    tm.last_dispatch = stale
    tm.fused_moe(**call, hidden_pad=128)
    assert tm.last_dispatch is stale


def test_dispatch_winner_cache_and_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "configs"))
    monkeypatch.setattr(tm, "record_dispatch", True)
    monkeypatch.setattr(tm, "last_dispatch", None)
    call = _block_call()
    dispatches = []

    def run(*args):
        dispatches.append(tm.last_dispatch["_impl"])
        return call["output"]

    def do_bench(fn, **kwargs):
        fn()
        return 1.0 if tm.last_dispatch["_impl"] == "jit_batch" else 2.0

    def tuner(configs):
        return tm.autotune(configs=configs, key=["batch_bucket", "model_key"], do_bench=do_bench,
                           artifact_name="pyhip_dispatch_test")(tm._fmoe_wrapper)

    def unexpected_search(*args, **kwargs):
        pytest.fail("cache/artifact hit must not search candidates")

    monkeypatch.setattr(tm, "_run_jit", run)
    monkeypatch.setattr(tm, "_autotuned_fmoe", tuner([
        tm.Config(_impl="jit_batch", block_n=1024), tm.Config(_impl="jit_splitk")]))
    assert tm.fused_moe(**call) is call["output"]
    # 最后一个被计时的候选不是 winner；正式调用必须覆盖它。
    assert dispatches == ["jit_batch", "jit_splitk", "jit_batch"]
    winner = tm.last_dispatch.copy()
    assert winner["_impl"] == "jit_batch" and winner["block_n"] == 1024
    assert len(list((tmp_path / "configs").glob("pyhip_dispatch_test-*.json"))) == 1

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    monkeypatch.setattr(tm._autotuned_fmoe, "configs", unexpected_search)
    dispatches.clear()
    tm.last_dispatch = None
    tm.fused_moe(**call)
    assert dispatches == ["jit_batch"] and tm.last_dispatch == winner

    # 新 autotuner + 空普通缓存目录，只能从离线 artifact 命中。
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(tm, "_autotuned_fmoe", tuner(unexpected_search))
    dispatches.clear()
    tm.last_dispatch = None
    tm.fused_moe(**call)
    assert dispatches == ["jit_batch"] and tm.last_dispatch == winner


@pytest.mark.parametrize("failed, previous_record", [(False, False), (False, True), (True, False), (True, True)])
def test_benchmark_dispatch_recording(monkeypatch, failed, previous_record):
    call = _block_call()
    model = dict(HIDDEN_SIZE=512, INTER_SIZE=256, TP=1, E=4, TOPK=2)
    args = SimpleNamespace(dtype="fp8", gate_mode="separated", preshuffle="on", routing="balanced", seed=8,
                           activation="silu", swiglu_limit=None, beta=None, linear_beta=None,
                           retune=False, tune_aiter=None, check_only=False, rounds=1)
    monkeypatch.setattr(tm, "record_dispatch", previous_record)
    monkeypatch.setattr(tm, "last_dispatch", {"_impl": "stale"})
    monkeypatch.setattr(tm, "_autotuned_fmoe", object())  # benchmark 不能查询 autotuner 内部状态。
    monkeypatch.setattr(bench, "prepare", lambda *args: (call, "block"))
    monkeypatch.setattr(tm, "_torch_reference", lambda call: torch.zeros_like(call["hidden_states"]))

    def aiter_call(**call):
        assert not tm.record_dispatch
        return call["output"].zero_()

    def tuned_call(**call):
        assert tm.record_dispatch and tm.last_dispatch is None
        tm.last_dispatch = {"_impl": "recorded"}
        if failed:
            raise RuntimeError("intentional dispatch failure")
        return call["output"].zero_()

    def measure(*args):
        assert not failed and not tm.record_dispatch
        tm.last_dispatch["_impl"] = "later"  # row 保存独立快照，不能跟着全局容器变化。
        return dict(samples_us=[1.0], mean_us=1.0)

    monkeypatch.setattr(tm, "_aiter_fused_moe", aiter_call)
    monkeypatch.setattr(tm, "fused_moe", tuned_call)
    monkeypatch.setattr(bench, "measure", measure)
    row = bench.run_case("test", model, 17, args)
    assert row["status"] == ("NOT_COMPARABLE" if failed else "PASS")
    assert row["winner"] == (None if failed else {"_impl": "recorded"})
    assert tm.record_dispatch is previous_record


@pytest.mark.parametrize("dtype, quant, expected", [
    ("bf16", "model", aiter.QuantType.No),
    ("fp8", "ptpc", aiter.QuantType.per_Token),
    ("fp8", "per_tensor", aiter.QuantType.per_Tensor),
    ("fp8", "block", aiter.QuantType.per_1x128),
    ("mxfp4", "model", aiter.QuantType.per_1x32),
])
def test_aiter_tuning_shape(dtype, quant, expected):
    args = SimpleNamespace(dtype=dtype, quant=quant, activation="silu", gate_mode="separated")
    model = bench.MOE_MODELS["qwen35_397B_k256"]
    shape = bench._aiter_shape(model, 3, args)
    assert shape["token"] == 4
    assert shape["model_dim"] == 4096 and shape["inter_dim"] == 256
    assert shape["q_type"] == expected
    # dtype 是输出类型，不是量化权重类型。
    assert shape["dtype"] == torch.bfloat16
    assert shape["q_dtype_a"] == shape["q_dtype_w"] == bench._moe_types(model, args)[1]
    assert shape["use_g1u1"] and not shape["doweight_stage1"]


def test_aiter_tuning_prepass(monkeypatch, tmp_path):
    import aiter.fused_moe as fm
    from aiter.jit.core import AITER_CONFIGS

    args = SimpleNamespace(dtype="fp8", quant="model", activation="silu", gate_mode="separated",
                           tokens=[3, 4, 64, 64], tune_aiter=tmp_path, routing="balanced", seed=8,
                           beta=None, linear_beta=None, swiglu_limit=None)
    models = ["hy3", "qwen35_35B_k256"]
    calls = []

    def fake_tuner(command, *, cwd, env, check):
        assert check and command[-3:] == ["--all", "--mp", "1"]
        assert env["TUNE_ONLY"] == "" and env["TUNE_MOE_KERNEL_REGEX"] == ""
        assert env["TUNE_MOE_EXPERT_BALANCE"] == "True"
        with (tmp_path / "untuned.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4  # 两个模型 × 两个 Aiter bucket，重复 M 不重复调优。
        assert {row["q_type"] for row in rows} == {"QuantType.per_Tensor", "QuantType.per_Token"}
        assert {row["inter_dim"] for row in rows} == {"192", "256"}  # 不改 Hy3 的实际 I。
        calls.append(rows)
        for row in rows:
            row.update(gfx=fm.get_gfx_runtime(), cu_num=fm.get_cu_num(), block_m=32, ksplit=0,
                       kernelName1=f"moe_ck2stages_test_{len(calls)}_gemm1",
                       kernelName2=f"moe_ck2stages_test_{len(calls)}_gemm2", run_1stage=0, us=1)
        if len(calls) == 3:
            rows.pop()  # 模拟官方 tuner 未为全部 shape 写入结果。
        with (tmp_path / "tuned.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    monkeypatch.setattr(bench, "subprocess", SimpleNamespace(run=fake_tuner))
    monkeypatch.setenv("AITER_CONFIG_FMOE", "previous-config.csv")
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    # 相同路径调优两次仍要刷新 Python caches；不读取上一次结果。
    for run in (1, 2):
        with bench.tune_aiter(models, args) as path:
            assert path == str(tmp_path / "tuned.csv")
            assert AITER_CONFIGS.AITER_CONFIG_FMOE_FILE == path
            assert all(row["kernelName1"] == f"moe_ck2stages_test_{run}_gemm1"
                       for row in fm.cfg_2stages[0].values())
        assert fm.cfg_2stages is None
        assert bench.os.environ["AITER_CONFIG_FMOE"] == "previous-config.csv"
        assert bench.os.environ["AITER_BYPASS_TUNE_CONFIG"] == "1"
    with pytest.raises(RuntimeError, match="no valid tuned config"):
        with bench.tune_aiter(models, args):
            pytest.fail("must not benchmark with a missing tuned shape")
    assert fm.cfg_2stages is None
    assert bench.os.environ["AITER_CONFIG_FMOE"] == "previous-config.csv"


@pytest.mark.parametrize("quant_type", [aiter.QuantType.per_128x128, aiter.QuantType.per_1x128])
def test_blockscale_reference(quant_type):
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

    call = _block_call(shuffled=(False, False))
    call["quant_type"] = quant_type
    x, w1, w2 = (call[name] for name in ("hidden_states", "w1", "w2"))
    weights, ids = call["topk_weight"], call["topk_ids"]
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x128)
    aq, a_scale = quantize(x, quant_dtype=w1.dtype)
    mid = torch_moe_stage1(aq, w1, w2, weights, ids, dtype=x.dtype, quant_type=quant_type,
                           a1_scale=a_scale, w1_scale=call["w1_scale"])
    dq, d_scale = quantize(mid.view(-1, mid.shape[-1]), quant_dtype=w2.dtype)
    reference = torch_moe_stage2(dq, w1, w2, weights, ids, dtype=x.dtype, quant_type=quant_type,
                                a2_scale=d_scale, w2_scale=call["w2_scale"])
    actual = tm._torch_reference(call)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("tokens, hidden, inter, shuffled", [
    (1, 512, 128, (False, False)),
    (17, 512, 128, (True, False)),
    (33, 512, 128, (True, True)),
    (33, 512, 256, (False, True)),
    (513, 512, 256, (True, True)),
    (129, 384, 384, (True, True)),
    (33, 128, 128, (False, False)),
    (17, 512, 512, (True, True)),
])
def test_blockscale_candidates(tokens, hidden, inter, shuffled):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native block-scale 8-wave kernels require gfx950")
    call = _block_call(tokens, hidden, inter, shuffled)
    tuning = _tune_arguments(call)
    configs = _native_configs(call)
    assert configs
    reference = tm._torch_reference(call)
    before = {name: call[name].view(torch.uint8).clone()
              for name in ("w1", "w2", "w1_scale", "w2_scale")}
    for config in configs:
        call["output"].fill_(float("nan"))
        _check(tm._fmoe_wrapper(**tuning, **config.all_kwargs()), call, reference)
    for name, value in before.items():
        assert torch.equal(call[name].view(torch.uint8), value)
    assert tuple(bool(getattr(call[name], "is_shuffled", False)) for name in ("w1", "w2")) == shuffled


def test_blockscale_config_constraints(monkeypatch):
    call = _block_call()
    call["block_size_M"] = 128
    configs = _native_configs(call)
    if torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        assert configs
    assert all(config.all_kwargs()["tile_m_down"] == 128 for config in configs)
    assert all(config.all_kwargs().get("down_path") != "persistent" for config in configs)
    # gfx942 可继续走现有 split-K/Aiter，不能发射 gfx950 的 FP8 指令。
    call["w1"] = call["w1"].float().to(torch.float8_e4m3fnuz)
    call["w2"] = call["w2"].float().to(torch.float8_e4m3fnuz)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(
        name="AMD Instinct MI300X", gcnArchName="gfx942", multi_processor_count=304,
    ))
    assert not _native_configs(call)


def test_blockscale_tuning_cache(monkeypatch, tmp_path):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "0")
    monkeypatch.setattr(tm, "record_dispatch", True)
    monkeypatch.setattr(tm, "last_dispatch", None)
    call = _block_call(tokens=33, shuffled=(True, False))
    call["block_size_M"] = 128
    reference = tm._torch_reference(call)
    _check(tm.fused_moe(**call), call, reference)
    winner = tm.last_dispatch.copy()
    assert any(all(winner[name] == value for name, value in config.all_kwargs().items())
               for config in tm._configs(**_tune_arguments(call)))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")

    def unexpected_search(*args, **kwargs):
        pytest.fail("a cached call must not search configs again")

    monkeypatch.setattr(tm._autotuned_fmoe, "configs", unexpected_search)
    # 同 shape 的数据/路由更新不能被缓存吞掉；out 必须重新覆盖。
    call["hidden_states"].mul_(-.75)
    call["topk_ids"].copy_((call["topk_ids"] + 1) % 4)
    call["topk_weight"].mul_(.5)
    call["output"].fill_(float("nan"))
    reference = tm._torch_reference(call)
    tm.last_dispatch = None
    _check(tm.fused_moe(**call), call, reference)
    assert tm.last_dispatch == winner
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = tm.fused_moe(**call)
    call["output"].fill_(float("nan"))
    tm.last_dispatch = None
    graph.replay()
    _check(result, call, reference)
    assert tm.last_dispatch is None  # replay 不执行 Python dispatcher。


@pytest.mark.parametrize("splits", [0, 1, 2, 4])
def test_blockscale_stream_graph(splits):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    call = _block_call(tokens=33, hidden=1024, inter=128)
    config = next(config for config in _native_configs(call)
                  if config.all_kwargs().get("num_oc_splits", 0) == splits)
    tuning = _tune_arguments(call)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            tm._fmoe_wrapper(**tuning, **config.all_kwargs())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            tm._fmoe_wrapper(**tuning, **config.all_kwargs())
            result = tm._fmoe_wrapper(**tuning, **config.all_kwargs())
    stream.synchronize()
    for zero in (False, True, False):
        call["hidden_states"].fill_(0.0 if zero else .05)
        call["topk_ids"].copy_((call["topk_ids"] + 1) % 4)
        reference = tm._torch_reference(call)
        call["output"].fill_(float("nan"))
        graph.replay()
        _check(result, call, reference)