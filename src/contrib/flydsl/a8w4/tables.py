# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reports on tuned tables (``python -m a8w4.tune compare | readme | stats``).

  compare  per (shape, act, tokens): winners of two tables, and what keeping the old
           winner costs in the new table (after a retune or a protocol change)
  readme   README 7.2 / 7.3 / 7.4 (SiLU rows) regenerated from the table, plus the
           numbers 7.6 quotes; --write updates README.md, --check fails when it is stale
  stats    SiTUv2 vs SiLU: time ratio of the same config, points whose winner differs

``readme`` keeps the mem bounds of the README rows (they depend on the routing sample, not
on the kernels; 7.4's are recovered as us / x mem) and recomputes the fp8 bound and TFLOPS:
gemm1 4 * rows * I * H flops at 1080 TFLOPS, gemm2 2 * rows * I * H at 900, rows = tokens *
topk * E_local / E_global (EP8: the sampled local rows behind the old TFLOPS column where
that is precise, >= 50 TFLOPS).
"""

import csv
import os
import statistics
import sys
from collections import defaultdict

from pyhip.kit import table

from .moe import DEFAULT_CSV, MOE, shape_dims

README = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
H, TOPK = 4096, 8
README_SHAPES = {"tp8": (256, 256, 256), "tp4": (512, 256, 256), "ep8": (2048, 32, 256)}


def _shape_name(inter, experts, experts_global):
    for name, s in README_SHAPES.items():
        if s == (int(inter), int(experts), int(experts_global)):
            return name
    return f"I{inter}xE{experts}/{experts_global}"


def compare(old_path, new_path):
    """``kit.table.compare``, points named and ordered by shape, act and tokens."""
    return table.compare(
        old_path, new_path,
        label=lambda p: (f"{_shape_name(p['inter_dim'], p['experts'], p['experts_global'])} "
                         f"{p['act']:6s} {int(p['num_tokens']):5d}"),
        order=lambda p: (_shape_name(p["inter_dim"], p["experts"], p["experts_global"]),
                         p["routing"], p["act"], int(p["num_tokens"])))


# =============================================================================
# README 7.2 - 7.4
# =============================================================================


def _silu_points(table):
    t = defaultdict(dict)
    with open(table) as fh:
        for r in csv.DictReader(fh):
            if (r.get("act") or "silu") != "silu":
                continue
            for name, (i, e, eg) in README_SHAPES.items():
                if (int(r["inter_dim"]), int(r["experts"]), int(r["experts_global"])) == (i, e, eg):
                    t[(name, int(r["num_tokens"]))][r["config"]] = float(r["us"])
    return t


def _section(text, head):
    """(start, end) of the first markdown table after the heading line ``head``."""
    i = text.index(head)
    s = text.index("\n|", i) + 1
    e = text.index("\n\n", s)
    return s, e


def _rows_of(text, head):
    s, e = _section(text, head)
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in text[s:e].splitlines()]


def readme(table=DEFAULT_CSV, path=README, write=False, check=False):
    t = _silu_points(table)
    text = open(path).read()
    rows_backed = {}  # (name, tokens) -> local rows backed out of the old README row

    def rows(name, tokens):
        i, e, eg = README_SHAPES[name]
        return rows_backed.get((name, tokens), tokens * TOPK * e / eg)

    def fp8_us(name, tokens):
        i, r = README_SHAPES[name][0], rows(name, tokens)
        return (4 * r * i * H / 1080e12 + 2 * r * i * H / 900e12) * 1e6

    def tflops(name, tokens, us):
        return 6 * rows(name, tokens) * README_SHAPES[name][0] * H / us / 1e6

    def best(d, pred=lambda c: True):
        c = [k for k in d if pred(k)]
        return min(c, key=d.get) if c else None

    out = {}
    head = "### 7.2 TP8"
    old = _rows_of(text, head)
    lines = old[:2]
    for r in old[2:]:
        n, mem = int(r[0]), float(r[6])
        d = t[("tp8", n)]
        b = best(d)
        us = d[b]
        f = lambda p: d[best(d, p)]  # noqa: E731
        lines.append([str(n), f"`{b}`", f"{us:.1f}", f"{f(lambda c: c.startswith('tp2k')):.1f}",
                      f"{f(lambda c: c.startswith('tp1k ')):.1f}", f"{d['tp1kq bcm1']:.1f}",
                      r[6], f"{us / mem:.2f}", f"{us / (mem + fp8_us('tp8', n)):.2f}",
                      f"{tflops('tp8', n, us):.0f}"])
    out[head] = lines

    head = "### 7.3 EP8"
    old = _rows_of(text, head)
    lines = old[:2]
    for r in old[2:]:
        n, mem = int(r[0]), float(r[6])
        if float(r[9]) >= 50:
            rows_backed[("ep8", n)] = float(r[9]) * float(r[2]) / (6 * README_SHAPES["ep8"][0] * H) * 1e6
        d = t[("ep8", n)]
        b = best(d)
        us = d[b]
        tp = d[best(d, lambda c: c.startswith("tp"))]
        ep1k = f"{d['ep1k m512']:.1f}" if "ep1k m512" in d else "—"
        lines.append([str(n), f"`{b}`", f"{us:.1f}", ep1k, f"{tp:.1f}", f"{tp / us:.2f}", r[6],
                      f"{us / mem:.2f}", f"{us / (mem + fp8_us('ep8', n)):.2f}",
                      f"{tflops('ep8', n, us):.0f}"])
    out[head] = lines

    head = "### 7.4 TP4"  # transposed
    old = _rows_of(text, head)
    toks = [int(x) for x in old[0][1:]]
    mems = [float(u) / float(x) for u, x in zip(old[3][1:], old[4][1:])]
    cfg, us, xm, xmf = [old[2][0]], [old[3][0]], [old[4][0]], [old[5][0]]
    for n, mem in zip(toks, mems):
        d = t[("tp4", n)]
        b = best(d)
        p = b.split()
        cfg.append(f"2k {p[1]}" if p[0] == "tp2k" else f"1k {p[2]}" if p[0] == "tp1k" else b)
        us.append(f"{d[b]:.1f}")
        xm.append(f"{d[b] / mem:.2f}")
        xmf.append(f"{d[b] / (mem + fp8_us('tp4', n)):.2f}")
    out[head] = [old[0], old[1], cfg, us, xm, xmf]

    new_text = text
    for head, lines in out.items():
        s, e = _section(new_text, head)
        raw = new_text[s:e].splitlines()[:2]
        body = raw + ["| " + " | ".join(r) + " |" for r in lines[2:]]
        new_text = new_text[:s] + "\n".join(body) + new_text[e:]
        if not check:
            print(head)
            print("\n".join(body))
    if check:
        # only the cells read from the table: the bound columns are backed out of the README's
        # own rounded numbers, so they may move in the last digit on every regeneration
        diffs = []
        for head, lines in out.items():
            old = _rows_of(text, head)
            if head.startswith("### 7.4"):
                cells = [(i, j) for i in (2, 3) for j in range(1, len(old[i]))]
            else:
                cells = [(i, j) for i in range(2, len(old)) for j in range(1, 6)]
            diffs += [f"{head} {old[i][0]} / {old[0][j]}: README {old[i][j]}, table {lines[i][j]}"
                      for i, j in cells if old[i][j] != lines[i][j]]
        print("\n".join(diffs) if diffs else f"README 7.2-7.4 match {table}")
        if diffs:
            print("regenerate: python -m a8w4.tune readme --write")
        return 1 if diffs else 0
    if write:
        with open(path, "w") as fh:
            fh.write(new_text)
        print(f"written: {path}")

    def rng(xs):
        return f"{min(xs):.2f}-{max(xs):.2f}"

    pts = sorted({n for _, n in t})
    for name, head in (("tp8", "### 7.2 TP8"), ("ep8", "### 7.3 EP8")):
        rr = _rows_of(new_text, head)[2:]
        mid = [min(t[(name, int(r[0]))].values()) / float(r[6]) for r in rr if 64 <= int(r[0]) <= 512]
        big = [min(t[(name, int(r[0]))].values()) / (float(r[6]) + fp8_us(name, int(r[0])))
               for r in rr if int(r[0]) >= 2048]
        print(f"7.6 {name}: 64-512 tokens x mem {rng(mid)}; >= 2048 x (mem+fp8) {rng(big)}")
    ep = [(n, t[("ep8", n)]) for n in pts]
    tp_slow = [d[best(d, lambda c: c.startswith("tp"))] / min(d.values()) for _, d in ep]
    e1k = [d["ep1k m512"] / min(d.values()) for n, d in ep if "ep1k m512" in d and n <= 512]
    print(f"7.6 ep8: TP family / best {rng(tp_slow)}; ep1k / best (<= 512) {rng(e1k)}")
    for name in ("tp8", "tp4"):
        g = [t[(name, n)]["tp2k bm64 g1bcm1 g2base"] / min(t[(name, n)].values()) for n in pts]
        print(f"7.6 {name}: default tp2k bm64 g1bcm1 g2base / best {rng(g)}")
    from .kernels.ep_gemm import AUTO

    for n in pts:
        d = t[("ep8", n)]
        c = next(nm for lim, nm in AUTO if lim is None or n <= lim)
        r = d[f"ep3k {c}"] / min(d.values())
        if r > 1.02:
            print(f"ep8 {n}: AUTO {c} / best ({min(d, key=d.get)}) {r:.3f}")
    return 0


# =============================================================================
# SiTUv2 vs SiLU
# =============================================================================


def stats(table=DEFAULT_CSV):
    rows = {}
    with open(table) as fh:
        for r in csv.DictReader(fh):
            key = tuple(int(r[k]) for k in ("model_dim", "inter_dim", "experts",
                                            "experts_global", "topk"))
            rows.setdefault((key, r.get("act") or "silu"), {}).setdefault(
                int(r["num_tokens"]), {})[r["config"]] = float(r["us"])
    for (shape, act) in sorted(rows):
        if act != "silu" or (shape, "situv2") not in rows:
            continue
        si, sv = rows[(shape, "silu")], rows[(shape, "situv2")]
        ratios, gain = [], []
        for t in sorted(si.keys() & sv.keys()):
            ratios += [sv[t][c] / si[t][c] for c in si[t].keys() & sv[t].keys()]
            w_si, w_sv = min(si[t], key=si[t].get), min(sv[t], key=sv[t].get)
            if w_si != w_sv and w_si in sv[t]:
                gain.append((t, w_si, w_sv, 100 * (sv[t][w_si] / sv[t][w_sv] - 1)))
        q = statistics.quantiles(sorted(ratios), n=20)
        print(f"{shape} situv2/silu, same config: median {statistics.median(ratios):.3f}, "
              f"p5 {q[0]:.3f}, p95 {q[-1]:.3f} (n={len(ratios)}); {len(gain)} of {len(si)} "
              "points pick another config:")
        for t, a, b, g in gain:
            print(f"  {t:6d}: silu {a:28s} situv2 {b:28s} situv2 gain {g:+.2f}%")
        for tok in (4096,):
            a, b = (MOE.pick(shape_dims(shape, tok, act), capturing=True, table=table)
                    for act in ("silu", "situv2"))
            print(f"  capture {tok}: silu {MOE.name_of(a)} | situv2 {MOE.name_of(b)}")


if __name__ == "__main__":
    sys.exit("use: python -m a8w4.tune compare | readme | stats")
