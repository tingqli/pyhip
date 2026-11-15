"""Rewrite README 7.2 / 7.3 / 7.4 (silu rows) from configs/a8w4_moe_tuned.csv and print the
numbers quoted in 7.6. The mem bounds stay as in the README (they depend on the routing
sample, not on the kernels; 7.4's are recovered as us / x mem); the fp8 bound and TFLOPS
are recomputed: gemm1 4 * rows * I * H flops at 1080 TFLOPS, gemm2 2 * rows * I * H at 900,
rows = tokens * topk * E_local / E_global, or for EP8 the sampled local rows behind the old
TFLOPS column where that is precise (>= 50 TFLOPS).

    cd src/contrib/flydsl && python a8w4/experiments/20261115_retune2/readme_tables.py [--write]
"""
import argparse
import csv
import os
import re
from collections import defaultdict

A8 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
README = os.path.join(A8, "README.md")
TABLE = os.path.join(A8, "configs", "a8w4_moe_tuned.csv")
H, TOPK = 4096, 8
SHAPES = {"tp8": (256, 256, 256), "tp4": (512, 256, 256), "ep8": (2048, 32, 256)}  # I, E, E_g


def load():
    t = defaultdict(dict)
    for r in csv.DictReader(open(TABLE)):
        if r["act"] != "silu":
            continue
        for name, (i, e, eg) in SHAPES.items():
            if (int(r["inter_dim"]), int(r["experts"]), int(r["experts_global"])) == (i, e, eg):
                t[(name, int(r["num_tokens"]))][r["config"]] = float(r["us"])
    return t


ROWS = {}  # (name, tokens) -> local rows backed out of the old README row


def rows(name, tokens):
    i, e, eg = SHAPES[name]
    return ROWS.get((name, tokens), tokens * TOPK * e / eg)


def fp8_us(name, tokens):
    i = SHAPES[name][0]
    r = rows(name, tokens)
    return (4 * r * i * H / 1080e12 + 2 * r * i * H / 900e12) * 1e6


def tflops(name, tokens, us):
    return 6 * rows(name, tokens) * SHAPES[name][0] * H / us / 1e6


def best(d, pred=lambda c: True):
    c = [k for k in d if pred(k)]
    return min(c, key=d.get) if c else None


def section(text, head):
    """(start, end) of the first markdown table after the heading line ``head``."""
    i = text.index(head)
    s = text.index("\n|", i) + 1
    e = text.index("\n\n", s)
    return s, e


def rows_of(text, head):
    s, e = section(text, head)
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in text[s:e].splitlines()]


def raw_head(text, head):
    s, e = section(text, head)
    return text[s:e].splitlines()[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    t = load()
    text = open(README).read()
    out = {}

    # 7.2 TP8
    head = "### 7.2 TP8"
    old = rows_of(text, head)
    lines = old[:2]
    for r in old[2:]:
        n, mem = int(r[0]), float(r[6])
        d = t[("tp8", n)]
        b = best(d)
        f = lambda p: d[best(d, p)]  # noqa: E731
        us = d[b]
        lines.append([str(n), f"`{b}`", f"{us:.1f}", f"{f(lambda c: c.startswith('tp2k')):.1f}",
                      f"{f(lambda c: c.startswith('tp1k ')):.1f}", f"{d['tp1kq bcm1']:.1f}",
                      r[6], f"{us / mem:.2f}", f"{us / (mem + fp8_us('tp8', n)):.2f}",
                      f"{tflops('tp8', n, us):.0f}"])
    out[head] = lines

    # 7.3 EP8
    head = "### 7.3 EP8"
    old = rows_of(text, head)
    lines = old[:2]
    for r in old[2:]:
        n, mem = int(r[0]), float(r[6])
        if float(r[9]) >= 50:
            ROWS[("ep8", n)] = float(r[9]) * float(r[2]) / (6 * SHAPES["ep8"][0] * H) * 1e6
        d = t[("ep8", n)]
        b = best(d)
        us = d[b]
        tp = d[best(d, lambda c: c.startswith("tp"))]
        ep1k = f"{d['ep1k m512']:.1f}" if "ep1k m512" in d else "—"
        lines.append([str(n), f"`{b}`", f"{us:.1f}", ep1k, f"{tp:.1f}", f"{tp / us:.2f}", r[6],
                      f"{us / mem:.2f}", f"{us / (mem + fp8_us('ep8', n)):.2f}",
                      f"{tflops('ep8', n, us):.0f}"])
    out[head] = lines

    # 7.4 TP4 (transposed)
    head = "### 7.4 TP4"
    old = rows_of(text, head)
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

    for head, lines in out.items():
        s, e = section(text, head)
        new = raw_head(text, head) + ["| " + " | ".join(r) + " |" for r in lines[2:]]
        text = text[:s] + "\n".join(new) + text[e:]
        print(head)
        print("\n".join(new))
    if a.write:
        open(README, "w").write(text)

    # numbers quoted in 7.6
    def rng(xs):
        return f"{min(xs):.2f}-{max(xs):.2f}"

    pts = sorted({n for _, n in t})
    for name in ("tp8", "ep8"):
        mid = [min(t[(name, n)].values()) / float(r[6]) for r in rows_of(text, "### 7.2 TP8" if
               name == "tp8" else "### 7.3 EP8")[2:] for n in [int(r[0])] if 64 <= n <= 512]
        big = [min(t[(name, n)].values()) / (float(r[6]) + fp8_us(name, n))
               for r in rows_of(text, "### 7.2 TP8" if name == "tp8" else "### 7.3 EP8")[2:]
               for n in [int(r[0])] if n >= 2048]
        print(f"{name}: 64-512 tokens x mem {rng(mid)}; >= 2048 x (mem+fp8) {rng(big)}")
    ep = [(n, t[("ep8", n)]) for n in pts]
    tp_slow = [d[best(d, lambda c: c.startswith("tp"))] / min(d.values()) for _, d in ep]
    e1k = [d["ep1k m512"] / min(d.values()) for n, d in ep if "ep1k m512" in d and n <= 512]
    print(f"ep8: TP family / best {rng(tp_slow)}; ep1k / best (<= 512) {rng(e1k)}")
    for name in ("tp8", "tp4"):
        g = [t[(name, n)]["tp2k bm64 g1bcm1 g2base"] / min(t[(name, n)].values()) for n in pts]
        print(f"{name}: default tp2k bm64 g1bcm1 g2base / best {rng(g)}")
    import sys
    sys.path.insert(0, os.path.dirname(A8))
    from a8w4.kernels.ep_gemm import AUTO
    for n in pts:
        d = t[("ep8", n)]
        cfg = next(nm for lim, nm in AUTO if lim is None or n <= lim)
        r = d[f"ep3k {cfg}"] / min(d.values())
        if r > 1.02:
            print(f"ep8 {n}: AUTO {cfg} / best ({min(d, key=d.get)}) {r:.3f}")


if __name__ == "__main__":
    main()
