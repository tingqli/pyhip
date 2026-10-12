"""Capture-free, correlated QSA selections and a small-row CPU FP32 oracle.

The profile names are qualitative historical-layer analogues, not fitted
capture statistics. At M12000/H12, the mixed preset has about 17% complete
causal-prefix rows; the remaining rows mix high- and low-sharing windows. Only
the measured planner counts establish the actual union/direct split; different
heads, lengths and window boundaries change it.
"""

import math

import numpy as np
import torch

from tests.ops.qsa._attention import _metadata


PROFILES = {
    "l3like": dict(sharing_fraction=0.25, high_shared_blocks=480,
                   low_shared_blocks=64, high_window_blocks=768,
                   low_window_blocks=4096),
    "l47like": dict(sharing_fraction=0.29, high_shared_blocks=496,
                    low_shared_blocks=96, high_window_blocks=640,
                    low_window_blocks=4096),
}
PROFILE_NAMES = (*PROFILES, "mixed")
WINDOW_ROWS = 128
DENSE_ROWS = 2051


def make_case(rows, heads, *, seed=20260929, profile="mixed", sharing_fraction=None):
    """Return CPU inputs; all buffers/implementations can clone identical values.

Each sparse row has 512 distinct complete four-token blocks and its causal
0--3 token tail. A window keeps a common core while its remaining blocks vary
by row. High-sharing windows draw from a narrow recent region; low-sharing
windows use a broader region. A seeded, balanced schedule avoids depending on
a lucky draw of a handful of high-sharing windows.
"""
    if rows < 1 or heads not in (12, 6, 3) or profile not in PROFILE_NAMES:
        raise ValueError("Require positive rows, H12/H6/H3 and a named synthetic profile")
    if not 0 <= seed < 2**32:
        raise ValueError("Require 0 <= seed < 2**32")
    components = tuple(PROFILES) if profile == "mixed" else (profile,)
    fraction = (sum(PROFILES[name]["sharing_fraction"] for name in components) / len(components)
                if sharing_fraction is None else sharing_fraction)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("sharing_fraction must be finite and within [0, 1]")

    rng = np.random.default_rng(seed)
    phase = float(rng.random())
    indices = np.full((rows, 2051), -1, dtype=np.int32)
    dense = min(rows, DENSE_ROWS)
    for row in range(dense):
        indices[row, :row + 1] = np.arange(row + 1)
    windows = []
    for window, first in enumerate(range(dense, rows, WINDOW_ROWS)):
        end = min(first + WINDOW_ROWS, rows)
        high = math.floor((window + 1) * fraction + phase) > math.floor(window * fraction + phase)
        name = components[window % len(components)]
        settings = PROFILES[name]
        sharing = "high" if high else "low"
        common_count = settings[f"{sharing}_shared_blocks"]
        span = settings[f"{sharing}_window_blocks"]
        available = (first + 1) // 4
        start = max(0, available - span)
        common = rng.choice(np.arange(start, available), common_count, replace=False)
        for row in range(first, end):
            visible = row + 1
            blocks = visible // 4
            pool = np.arange(start, blocks)
            private = pool[~np.isin(pool, common)]
            chosen = np.concatenate((common, rng.choice(private, 512 - common_count, replace=False)))
            rng.shuffle(chosen)
            indices[row, :2048] = (chosen[:, None] * 4 + np.arange(4)).reshape(-1)
            indices[row, 2048:2048 + visible % 4] = np.arange(blocks * 4, visible)
        windows.append(dict(first=first, rows=end - first, profile=name, sharing=sharing,
                            common_blocks=common_count, candidate_window_blocks=span))

    # Generate H12 before taking local heads so TP2/4/8 share the same Q/K/V
    # values, not just a selection seed. This is synthetic local work, not TP.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((rows, 12, 256), generator=generator, dtype=torch.bfloat16)
    k = torch.randn((rows, 1, 256), generator=generator, dtype=torch.bfloat16)
    v = torch.randn(k.shape, generator=generator, dtype=torch.bfloat16)
    value = _metadata(q[:, :heads].contiguous(), k, v, torch.from_numpy(indices), (rows,), (0,))
    high_rows = sum(window["rows"] for window in windows if window["sharing"] == "high")
    config = dict(
        profile=profile, seed=seed, sharing_fraction=fraction, window_rows=WINDOW_ROWS,
        components={name: dict(PROFILES[name]) for name in components}, windows=windows,
        high_sharing_rows=high_rows, low_sharing_rows=rows - dense - high_rows,
        actual_high_sharing_fraction=high_rows / (rows - dense) if rows > dense else None,
        complete_prefix_rows=dense, complete_blocks=512, block_tokens=4, causal_tail="0..3",
        q_shape=list(value.q.shape), kv_shape=list(k.shape), prefix=0,
        note="Synthetic correlation approximates historical selections; profile parameters are assumptions, "
             "not measured capture statistics. Use the reported actual union/direct counts.",
    )
    return value, config


@torch.no_grad()
def reference_rows(value, rows):
    """CPU FP32 QK, scaling, softmax and PV; retain only the requested output rows."""
    if value.q.device.type != "cpu" or value.k.shape[1] != 1:
        raise ValueError("The synthetic reference expects CPU tensors and one KV head")
    result = torch.empty((len(rows), value.q.shape[1], 256), dtype=torch.float32)
    for index, row in enumerate(rows):
        selected = value.indices[row]
        selected = selected[selected >= 0].long()
        scores = value.q[row].float() @ value.k[selected, 0].float().T
        result[index] = (scores * value.scale).softmax(-1) @ value.v[selected, 0].float()
    return result