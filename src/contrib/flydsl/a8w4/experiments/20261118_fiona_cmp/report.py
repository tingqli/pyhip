#!/usr/bin/env python
"""The tables of ``experiments/log_20261118b.md``, from the json files ``cmp_fiona.py``
wrote (sglang_env.json, default_env.json, tiny.json in this directory):

    python a8w4/experiments/20261118_fiona_cmp/report.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# FionaZZ92/aiter 28e6b92 docs/mimo_mega_decode_ep8.csv (a CUDA graph of 8 calls on the same
# weights, 25 replays, max rank, median of 18 samples): M/rank -> (ms of "MORI EPv1 + A8W4",
# ms of "MegaMoEV2 decode")
PUBLISHED = {
    12: (0.6286345100402833, 0.4041815376281738),
    24: (0.6546002960205078, 0.4380696487426758),
    48: (0.6613036346435547, 0.41765062332153324),
    80: (0.5369794273376465, 0.49990001678466794),
    96: (0.6223001480102539, 0.49986494064331055),
    192: (0.6276252746582032, 0.8565995407104492),
    256: (0.8828724288940429, 1.065962104797363),
    304: (1.1333230209350584, 1.2584975051879883),
    418: (1.1135042190551756, 1.5262197113037108),
    512: (1.3259856414794922, 1.7780789184570311),
}


def rows(name):
    return json.load(open(os.path.join(HERE, name)))["rows"]


def small(d):
    """(label, value) of the mega_small entry of a protocol / accuracy dict."""
    return next((k, v) for k, v in d.items() if k.startswith("mega_small"))


def md(head, body, align=None):
    align = align or ["---:"] * len(head)
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in body])


def perf_table(rs, proto):
    body, ratios = [], []
    for r in rs:
        if proto not in r:
            continue
        t = r[proto]
        fi = t["fiona"][0]
        lab, sm = small(t)
        mg = t["mega"][0] if "mega" in t else None
        best = min(sm[0], mg) if mg else sm[0]
        ratios.append((fi / sm[0], fi / best))
        body.append([str(r["M"]), str(r["total"]), f"{fi:.1f}",
                     f"{sm[0]:.1f}" + ("*" if "t4096" in lab else ""),
                     f"{mg:.1f}" if mg else "-", f"{fi / sm[0]:.2f}x", f"{fi / best:.2f}x",
                     f"{100 * (t['fiona#2'][0] / fi - 1):+.1f}%"])
    head = ["M/rank", "tokens", "Fiona", "mega_small", "mega", "Fiona / mega_small",
            "Fiona / pyhip 最快", "Fiona#2"]
    rng = (f"Fiona / mega_small {min(a for a, _ in ratios):.2f}–{max(a for a, _ in ratios):.2f}x, "
           f"Fiona / pyhip 最快 {min(b for _, b in ratios):.2f}–{max(b for _, b in ratios):.2f}x")
    return md(head, body), rng


def env_table(sg, de, proto):
    by = {r["M"]: r for r in sg}
    body = []
    for r in de:
        a = by.get(r["M"])
        if not a or proto not in a or proto not in r:
            continue
        f1, f2 = a[proto]["fiona"][0], r[proto]["fiona"][0]
        s1, s2 = small(a[proto])[1][0], small(r[proto])[1][0]
        body.append([str(r["M"]), f"{f1:.1f}", f"{f2:.1f}", f"{100 * (f2 / f1 - 1):+.1f}%",
                     f"{s1:.1f}", f"{s2:.1f}", f"{100 * (s2 / s1 - 1):+.1f}%",
                     r["fiona_cfg"]])
    head = ["M/rank", "Fiona sglang env", "Fiona aiter 默认", "差", "mega_small (job 1)",
            "mega_small (job 2)", "差", "aiter 默认下 Fiona 的 config"]
    return md(head, body, ["---:"] * 7 + ["---"])


def acc_table(rs):
    body = []
    for r in rs:
        a = r["accuracy"]
        mg = a.get("mega")
        body.append([str(r["M"]), f"{a['fiona'][0]:.2e} / {a['fiona'][1]:.4f}",
                     "{:.2e} / {:.4f}".format(*small(a)[1][:2]),
                     f"{mg[0]:.2e} / {mg[1]:.4f}" if mg else "-",
                     "yes" if all(v[2] for v in a.values()) else "NO"])
    return md(["M/rank", "Fiona cos / relL2", "mega_small cos / relL2", "mega cos / relL2",
               "全部有限"], body)


def repro_table(sg):
    body = []
    for r in sg:
        if r["M"] not in PUBLISHED or "graph8w" not in r:
            continue
        base, pub = (1e3 * v for v in PUBLISHED[r["M"]])
        here = r["graph8w"]["fiona"][0]
        body.append([str(r["M"]), f"{pub:.1f}", f"{here:.1f}", f"{100 * (here / pub - 1):+.1f}%",
                     f"{base:.1f}", f"{small(r['graph8w'])[1][0]:.1f}"])
    return md(["M/rank", "发布的 MegaMoEV2", "这里 graph8w Fiona", "差",
               "发布的 MORI EPv1 + A8W4", "这里 graph8w mega_small"], body)


def main():
    sg, de, ti = rows("sglang_env.json"), rows("default_env.json"), rows("tiny.json")
    for name, rs, protos in (("sglang_env.json", sg, ("cold", "graph8", "graph8w")),
                             ("tiny.json", ti, ("cold", "graph8"))):
        for p in protos:
            t, rng = perf_table(rs, p)
            print(f"\n#### {name}: {p} (us)\n\n{t}\n\n{rng}")
    for p in ("cold", "graph8w"):
        print(f"\n#### default_env.json vs sglang_env.json: {p} (us; two jobs)\n\n"
              f"{env_table(sg, de, p)}")
    print(f"\n#### accuracy vs fp32 (max over ranks)\n\n{acc_table(ti + sg)}")
    print(f"\n#### FionaZZ92 docs/mimo_mega_decode_ep8.csv vs graph8w here (us)\n\n"
          f"{repro_table(sg)}")


if __name__ == "__main__":
    main()
