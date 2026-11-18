"""Cold-cache GPU timing (torch): what a call costs inside a model, where every layer
reads other weights and nothing stays in the caches between two calls of a layer
(the rules: skill ``gpu-timing-protocol``; evidence: a8w4 experiments/log_20261115.md 3.3).

    Ws = copies(W)                                   # NCOPIES copies to rotate through
    nxt = itertools.cycle(Ws).__next__
    us = time_cold(lambda: fn(nxt()))                # median GPU us, host overhead excluded

    timer = Timer(Ws)                                # fn takes the copy to read
    res = interleaved({"a": fa, "a#2": fa, "b": fb}, timer, rounds=5, iters=16)
    band = noise_band(res["a"], res["a#2"])
    verdict(delta_pct, band)                         # "~" / "faster" / "slower"

Every timed call is cold in the one way that matches a model: a 512 MB read evicts L2,
a GPU sleep lets the host enqueue the call (host overhead excluded), and the call reads
the next of >= MIN_COPIES input copies. There is no other way to time: inputs that every
call re-reads stay in the 256 MB MALL, which no flush evicts, and cached loads of them
looked 5-13% faster than they are (a write flush instead leaves dirty lines: 9-48%
slower); timing without a flush depends on what the previous call left in the caches.
"""

import statistics

import torch

__all__ = ["NCOPIES", "MIN_COPIES", "MIN_NOISE", "flush_cache", "copies", "time_cold",
           "Timer", "interleaved", "noise_band", "verdict"]

NCOPIES = 8  # input copies timed calls cycle through (8 x the a8w4 EP8 / TP8 weights: 3+ GB)
MIN_COPIES = 8  # 2 copies still favored cached weight loads at 16 tokens (log_20261115 3.3)
MIN_NOISE = 0.3  # % : below this a difference is never called

_FLUSH = {}


def flush_cache():
    """Read 512 MB: evicts L2 with clean lines (a read-modify-write flush leaves dirty
    lines whose write-back then steals HBM bandwidth). It does not evict the lines that
    every timed call re-reads from the 256 MB MALL (a 2 GB read does not either): the
    same inputs read with cached loads stay there, so time with ``copies``."""
    if "buf" not in _FLUSH:
        _FLUSH["buf"] = torch.ones(128 << 20, dtype=torch.int32, device="cuda")
    _FLUSH["buf"].amax()


def copies(x, n=NCOPIES):
    """``n`` copies of ``x`` (a tensor, or a dict whose tensor values are cloned), ``x``
    first. A timed closure that takes the next copy per call reads inputs that no recent
    call touched, as consecutive layers of a model do (with one copy, cached weight loads
    looked 5-13% faster than they are)."""
    if torch.is_tensor(x):
        return [x] + [x.clone() for _ in range(n - 1)]
    return [x] + [{k: v.clone() if torch.is_tensor(v) else v for k, v in x.items()}
                  for _ in range(n - 1)]


def _times(fn, iters, warmup):
    """GPU times (us) of ``iters`` cold calls of ``fn()``: L2 flush, then a GPU sleep
    that lets the host enqueue all of fn's launches before the start event, so host
    overhead is excluded (as in a CUDA graph replay)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        flush_cache()
        torch.cuda._sleep(2_000_000)  # ~1 ms of GPU spin
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    return ts


def time_cold(fn, iters=20, warmup=3):
    """Median GPU time (us) of ``fn()`` with cold caches; ``fn`` takes the next input
    copy itself (see the module docstring)."""
    ts = sorted(_times(fn, iters, warmup))
    return ts[len(ts) // 2]


class Timer:
    """Cold call times of ``fn(W)``, each call on the next of the input copies ``Ws``
    (Ws[0]: the original)."""

    def __init__(self, Ws):
        self.Ws, self._i = Ws, 0

    def _next(self):
        self._i = (self._i + 1) % len(self.Ws)
        return self.Ws[self._i]

    def times(self, fn, iters, warmup=3):
        """``iters`` GPU times (us) of ``fn(W)``, host launch overhead excluded."""
        return _times(lambda: fn(self._next()), iters, warmup)


def interleaved(fns, timer, rounds, iters):
    """{label: [median of each round]}: each round times every variant once; the order
    reverses every other round (ABBA), so clock and power drift hit all variants alike."""
    res = {n: [] for n in fns}
    for rnd in range(rounds):
        for n in (list(fns) if rnd % 2 == 0 else list(fns)[::-1]):
            res[n].append(statistics.median(timer.times(fns[n], iters)))
    return res


def noise_band(base, dup=None):
    """% band within which a difference to ``base`` (round medians) is noise: the dup's
    |delta|, half the base's round spread, at least MIN_NOISE."""
    b = statistics.median(base)
    band = (max(base) - min(base)) / b * 50 if len(base) > 1 else 0.0
    if dup:
        band = max(band, abs(statistics.median(dup) / b - 1) * 100)
    return max(band, MIN_NOISE)


def verdict(delta, noise):
    return "~" if abs(delta) <= noise else ("faster" if delta < 0 else "slower")
