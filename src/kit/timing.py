"""Cold-cache GPU timing (torch): what a call costs inside a model, where every layer
reads other weights and nothing stays in the caches between two calls of a layer
(the rules: skill ``gpu-timing-protocol``; evidence: a8w4 experiments/log_20261115.md 3.3).

    Ws = copies(W)                                   # NCOPIES copies to rotate through
    nxt = itertools.cycle(Ws).__next__
    us = time_cold(lambda: fn(nxt()))                # median GPU us, host overhead excluded

    timer = Timer(Ws)                                # protocols, fn takes the copy to read
    res = interleaved({"a": fa, "a#2": fa, "b": fb}, ["coldrot"], timer, rounds=5, iters=16)
    band = noise_band(res["coldrot", "a"], res["coldrot", "a#2"])
    verdict(delta_pct, band)                         # "~" / "faster" / "slower"

Protocols. Only ``coldrot`` with >= MIN_COPIES copies matches a model; the others exist
to measure the bias:

  coldrot  512 MB read flush + GPU sleep; each call reads the next of the copies
  rotate   the copies back to back, no flush (depends on the previous call's MALL)
  cold     flush + sleep, the same inputs every call: the MALL keeps them, so code
           that reads them through the cache looks 5-13% faster than it is
  bigread  2 GB read flush + sleep, the same inputs (the same bias as cold)
  wflush   512 MB write + read flush + sleep, the same inputs (dirty lines: cached
           reads look 9-48% slower)
  warm     no flush, the same inputs
"""

import statistics

import torch

__all__ = ["NCOPIES", "PROTOCOLS", "MIN_COPIES", "MIN_NOISE", "flush_cache", "copies",
           "time_cold", "Timer", "interleaved", "noise_band", "verdict"]

NCOPIES = 8  # input copies timed calls cycle through (8 x the a8w4 EP8 / TP8 weights: 3+ GB)
PROTOCOLS = ("coldrot", "rotate", "cold", "bigread", "wflush", "warm")
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


def time_cold(fn, iters=20, warmup=3, flush=True):
    """Median GPU time (us) of ``fn`` with cold caches. A GPU-side sleep after the
    flush lets the host enqueue all of fn's launches before the start event, so
    host overhead is excluded (as in a CUDA graph replay)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(iters)]
    ts = []
    for e0, e1 in ev:
        if flush:
            flush_cache()
        torch.cuda._sleep(2_000_000)  # ~1 ms of GPU spin
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


class Timer:
    """Call times of ``fn(W)`` under a protocol; ``Ws``: input copies, Ws[0] the original."""

    def __init__(self, Ws):
        self.Ws, self._i, self._buf = Ws, 0, {}

    def _next(self):
        self._i = (self._i + 1) % len(self.Ws)
        return self.Ws[self._i]

    def _pre(self, protocol):
        if protocol in ("coldrot", "cold"):
            flush_cache()
        elif protocol == "wflush":
            if "w" not in self._buf:
                self._buf["w"] = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
            self._buf["w"].zero_()
            flush_cache()
        elif protocol == "bigread":
            if "r" not in self._buf:
                self._buf["r"] = torch.ones(512 << 20, dtype=torch.int32, device="cuda")
            self._buf["r"].amax()
        else:
            assert protocol == "warm", protocol

    def times(self, fn, protocol, iters, warmup=3):
        """``iters`` GPU times (us) of ``fn(W)``; host launch overhead excluded (a GPU sleep
        lets the host enqueue fn before the start event)."""
        if protocol == "rotate":
            return self._rotate(fn, iters)
        for _ in range(warmup):
            fn(self.Ws[0])
        torch.cuda.synchronize()
        out = []
        for _ in range(iters):
            W = self._next() if protocol == "coldrot" else self.Ws[0]
            self._pre(protocol)
            torch.cuda._sleep(2_000_000)  # ~1 ms
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn(W)
            e1.record()
            torch.cuda.synchronize()
            out.append(e0.elapsed_time(e1) * 1e3)
        return out

    def _rotate(self, fn, iters):
        out, n = [], len(self.Ws)
        for p in range(-(-iters // n) + 1):  # pass 0: warm-up
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
            torch.cuda._sleep(4_000_000)
            ev[0].record()
            for i, W in enumerate(self.Ws):
                fn(W)
                ev[i + 1].record()
            torch.cuda.synchronize()
            if p:
                out += [ev[i].elapsed_time(ev[i + 1]) * 1e3 for i in range(n)]
        return out


def interleaved(fns, protocols, timer, rounds, iters):
    """{(protocol, label): [median of each round]}: each round times every (protocol,
    variant) once; the variant order reverses every other round (ABBA), so clock and
    power drift hit all variants alike."""
    res = {(p, n): [] for p in protocols for n in fns}
    for rnd in range(rounds):
        names = list(fns) if rnd % 2 == 0 else list(fns)[::-1]
        for p in protocols:
            for n in names:
                res[p, n].append(statistics.median(timer.times(fns[n], p, iters)))
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
