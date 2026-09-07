"""CPU-only performance matrix transcribed from the original attention reports.

Workloads, measurement protocols and baseline eligibility are separate. A
four-wave or gfx950 result must never become an eight-wave gfx942 acceptance.
"""

from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
import math


@dataclass(frozen=True)
class Protocol:
    timer: str = "profiler"
    warmup: int = 100
    rounds: int = 5
    iterations: int = 100
    buffers: int = 1
    sample_warmup: int = 20

    def __post_init__(self):
        if self.timer not in ("profiler", "events"):
            raise ValueError(f"unknown timer: {self.timer}")
        if min(self.warmup, self.sample_warmup) < 0 or min(self.rounds, self.buffers, self.iterations) < 1:
            raise ValueError("warmups must be nonnegative; rounds, buffers and iterations must be positive")
        if self.timer == "profiler" and (self.buffers != 1 or self.iterations < 2):
            raise ValueError("profiler requires one buffer and at least two iterations; use events for buffer rotation")


PROFILER = Protocol()
NATIVE_FP8 = Protocol(warmup=1200)
ROTATING_EVENTS = Protocol("events", 10, 1, 50, 10, 0)
H3_EVENTS = Protocol("events", 3, 1, 10, 10, 0)
BRANCH = "23cc6d1e95b1611493e21232bef5d9962b7b73c9"
PA_REPORT = f"{BRANCH}:tests/flydsl/pa_4wave/README.md"
PA_MATRIX = f"{BRANCH}:tests/flydsl/pa_4wave/benchmark_readme.py:workloads"
SWA_REPORT = f"{BRANCH}:tests/flydsl/pa_1wave/README.md"
FP8_REPORT = "tests/flydsl/pa_8wave/tests/flydsl/pa_8wave/new_native_gfx942_results.md:S5"


@dataclass(frozen=True)
class Workload:
    name: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    dq: int = 192
    dv: int = 128
    heads: int = 16
    kv_heads: int = 1
    page: int = 64
    causal: bool = False
    window: int = -1
    sink: bool = False
    scale_mode: str = "per-token"
    input_kind: str = "bf16-source"
    seed: int = 20260905
    source: str = PA_MATRIX
    protocol: Protocol = PROFILER
    backends: tuple[str, ...] = ()

    @property
    def flops(self):
        pairs = 0
        for q, k in zip(self.q_lens, self.kv_lens):
            if not self.causal:
                pairs += q * k
            else:
                for row in range(q):
                    diagonal = k - q + row
                    left = max(0, diagonal - self.window) if self.window >= 0 else 0
                    pairs += max(0, min(k, diagonal + 1) - left)
        return 2 * self.heads * pairs * (self.dq + self.dv)

    def unsupported(self, backend):
        if self.backends and backend.name not in self.backends:
            return "workload is specific to a different backend"
        if self.window >= 0 and not self.causal:
            return "windowed attention requires bottom-right causal mode"
        if backend.name == "swa_bf16":
            if self.window < 0 or not self.causal:
                return "single-wave backend requires causal SWA"
        elif self.window >= 0 or self.sink:
            if backend.arch != "gfx950":
                return "this full-MHA backend does not support SWA/sink"
        if backend.name != "bf16_942" and self.page != 64:
            return "backend supports page64 only (do not relabel another page size)"
        if backend.name != "bf16_942" and self.dv != 128:
            return "backend supports V128 only"
        if not backend.empty_kv and any(q > 0 and k == 0 for q, k in zip(self.q_lens, self.kv_lens)):
            return "original BF16 pipeline requires nonempty active KV"
        if self.causal and not backend.causal_short_kv and any(k < q for q, k in zip(self.q_lens, self.kv_lens)):
            return "original BF16 causal pipeline requires KV>=Q per sequence"
        return None

    def to_dict(self):
        return {**asdict(self), "effective_flops": self.flops}


def documented_workloads():
    result = [Workload("fp8_native_410t", (10240,), (2560,), input_kind="native-cast",
        seed=20260906, source=FP8_REPORT, protocol=NATIVE_FP8,
        backends=("fp8_942", "fp8_942_register"))]
    for dq in (128, 192):
        result += [Workload(f"full_d{dq}_p64", (10240,), (2583,), dq=dq),
                   Workload(f"causal32k_d{dq}_p64", (32768,), (32768,), dq=dq, causal=True)]
        for kv in (32768, 65536, 131072):
            result.append(Workload(f"swa_kv{kv}_d{dq}", (16384,), (kv,), dq=dq,
                                   causal=True, window=128, sink=True, source=SWA_REPORT))
        for window in (0, 16, 64, 512, 1024):
            result.append(Workload(f"swa_window{window}_d{dq}", (16384,), (131072,), dq=dq,
                                   causal=True, window=window, sink=True, source=SWA_REPORT))
        for q in (256, 2048, 4096, 65536):
            result.append(Workload(f"swa_query{q}_d{dq}", (q,), (131072,), dq=dq,
                                   causal=True, window=128, sink=True, source=SWA_REPORT))
    result += [Workload("full_d192_p32", (10240,), (2583,), page=32),
               Workload("causal32k_d192_p32", (32768,), (32768,), page=32, causal=True)]
    for page in (32, 64):
        result += [Workload(f"batch4_d192_p{page}", (10240,)*4, (2560,)*4, page=page),
                   Workload(f"batch4_d128_h1_p{page}", (10240,)*4, (2560,)*4, dq=128, heads=1, page=page),
                   Workload(f"singlehead40k_p{page}", (40960,), (40960,), dq=128, heads=1, page=page),
                   Workload(f"singlehead_causal32k_p{page}", (32768,), (32768,), dq=128, heads=1, page=page, causal=True),
                   Workload(f"h3_p{page}", (63225,7), (63225,7), dq=128, heads=14, kv_heads=14, page=page)]
    # Preserve the distinct historical gfx942 *timer* too, not just its shapes.
    result += [Workload("historical_bf16_250t_shape", (40960,), (40960,), dq=128, heads=1, page=32,
                        source=PA_REPORT + ":2026-08-10", protocol=ROTATING_EVENTS, backends=("bf16_942",)),
               Workload("historical_bf16_d192_events", (10240,), (2583,), page=32,
                        source=PA_REPORT + ":2026-08-10", protocol=ROTATING_EVENTS, backends=("bf16_942",)),
               Workload("historical_h3_events", (63225,7), (63225,7), dq=128, heads=14, kv_heads=14, page=32,
                        source=PA_REPORT + ":2026-08-10", protocol=H3_EVENTS, backends=("bf16_942",))]
    assert len({case.name for case in result}) == len(result)
    return tuple(result)


@dataclass(frozen=True)
class Baseline:
    workload: str
    backend: str
    arch: str
    gpu: str
    tflops: float
    microseconds: float
    source: str
    ptl: str | None = None
    flydsl: str | None = None
    minimum_tflops: float | None = None
    torch_version: str | None = None
    hip_version: str | None = None


BASELINES = (
    Baseline("fp8_native_410t", "fp8_942", "gfx942", "MI308X", 413.983500680022, 648.420663043478,
             FP8_REPORT, "VECTOR,F8", "0.2.2", 400.0, "2.12.1+rocm7.2", "7.2.53211"),
    Baseline("historical_bf16_250t_shape", "historical_4wave_static", "gfx942", "MI308X", 250.952, 3422.933,
             PA_REPORT + ":1135"),
    Baseline("historical_bf16_d192_events", "historical_4wave_static", "gfx942", "MI308X", 204.653, 1323.445,
             PA_REPORT + ":1129"),
    Baseline("swa_kv131072_d128", "swa_bf16", "gfx950", "MI350X", 239.194, 72.38507070707067, SWA_REPORT + ":149",
             torch_version="2.9.1+rocm7.2.0.git7e1940d4", hip_version="7.2.26015-fc0010cf6a"),
    Baseline("swa_kv131072_d192", "swa_bf16", "gfx950", "MI350X", 253.215, 85.47121212121208, SWA_REPORT + ":154",
             torch_version="2.9.1+rocm7.2.0.git7e1940d4", hip_version="7.2.26015-fc0010cf6a"),
)


def baseline_status(workload, backend_name, environment, measured_tflops=None, *, protocol=None, isolated=True, config="auto"):
    if measured_tflops is not None and (not math.isfinite(measured_tflops) or measured_tflops <= 0):
        raise ValueError("measured TFLOPS must be finite and positive")
    originals = {case.name: case for case in documented_workloads()}
    records = []
    for baseline in BASELINES:
        if baseline.workload != workload.name:
            continue
        reasons = []
        original = originals[baseline.workload]
        fields = ("q_lens", "kv_lens", "dq", "dv", "heads", "kv_heads", "page", "causal", "window",
                  "sink", "scale_mode", "input_kind", "seed")
        changed = [name for name in fields if getattr(workload, name) != getattr(original, name)]
        if changed:
            reasons.append(f"documented shape/input protocol changed: {', '.join(changed)}")
        if (protocol or workload.protocol) != original.protocol:
            reasons.append("measurement protocol overridden; not an exact documented reproduction")
        if not isolated:
            reasons.append("GPU isolation was not enforced")
        if config != "auto":
            reasons.append("explicit tile variant is not the documented default")
        if backend_name != baseline.backend:
            reasons.append(f"baseline is {baseline.backend}, not {backend_name}")
        if environment.get("arch") != baseline.arch or baseline.gpu not in environment.get("gpu", ""):
            reasons.append(f"requires {baseline.gpu}/{baseline.arch}")
        if baseline.ptl:
            gpu_limits = environment.get("limits", {}).get("gpu_data", [])
            active = gpu_limits[0].get("limit", {}) if gpu_limits else {}
            if active.get("ptl_state") != "Enabled" or active.get("ptl_format") != baseline.ptl:
                reasons.append(f"requires PTL Enabled/{baseline.ptl}")
        status = "unmatched" if reasons else "not_measured" if measured_tflops is None else "measured"
        if not reasons and measured_tflops is not None and baseline.minimum_tflops is not None:
            status = "passed" if measured_tflops >= baseline.minimum_tflops else "failed"
        compiler_matches = None if baseline.flydsl is None else environment.get("flydsl") == baseline.flydsl
        runtime = (("torch", baseline.torch_version), ("hip", baseline.hip_version))
        runtime_matches = (None if any(value is None for _, value in runtime) else
                           all(environment.get(name) == value for name, value in runtime))
        records.append({**asdict(baseline), "status": status, "mismatch_reasons": reasons,
                        "measured_tflops": measured_tflops,
                        "compiler_matches_original": compiler_matches,
                        "runtime_matches_original": runtime_matches,
                        "exact_environment_match": not reasons and compiler_matches is True and runtime_matches is True,
                        "latency_ratio_vs_documented": None if measured_tflops is None or reasons else baseline.tflops / measured_tflops})
    return records


def select_workloads(patterns=()):
    cases = documented_workloads()
    if not patterns:
        return cases
    unknown = [pattern for pattern in patterns if not any(fnmatchcase(case.name, pattern) for case in cases)]
    if unknown:
        raise ValueError(f"unknown documented case patterns: {unknown}")
    return tuple(case for case in cases if any(fnmatchcase(case.name, pattern) for pattern in patterns))