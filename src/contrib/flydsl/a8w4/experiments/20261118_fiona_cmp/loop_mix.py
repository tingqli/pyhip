#!/usr/bin/env python
"""Static instruction mix of the MFMA loops of a FlyDSL ``21_final_isa.s``.

    python loop_mix.py <21_final_isa.s> [...]

A loop = the instructions from a label to a backward branch to it (innermost first).
Per loop with MFMAs: instruction counts by class and per MFMA.
"""
import collections
import re
import sys

CLASSES = [
    ("mfma", r"^v_mfma"),
    ("v_perm", r"^v_perm_b32"),
    ("v_fma/mul/add f32", r"^v_(fma|fmac|mul|add|sub|pk_fma|pk_mul|pk_add)_f32"),
    ("v_cvt", r"^v_cvt"),
    ("v_bit/int", r"^v_(and|or|xor|lshl|lshr|ashr|bfe|bfi|alignbit|alignbyte|not|add_u32|add_co|"
                  r"addc|sub_u32|sub_co|mad_u|mul_lo|mul_hi|lshl_add|lshl_or|and_or|add3|or3|"
                  r"xad|bcnt|cndmask|max_i|min_i|max_u|min_u)"),
    ("v_mov/accvgpr", r"^v_(mov|accvgpr)"),
    ("v_other", r"^v_"),
    ("ds_bpermute", r"^ds_bpermute"),
    ("ds_read", r"^ds_read"),
    ("ds_write", r"^ds_write"),
    ("buffer/global load", r"^(buffer|global)_load"),
    ("buffer/global store", r"^(buffer|global)_store"),
    ("s_waitcnt", r"^s_waitcnt"),
    ("s_barrier", r"^s_barrier"),
    ("s_other", r"^s_"),
]


def parse(path):
    lines, labels = [], {}
    for ln in open(path):
        s = ln.strip()
        m = re.match(r"^([.\w$]+):", s)
        if m:
            labels[m[1]] = len(lines)
            continue
        if not s or s.startswith((";", "//", ".")):
            continue
        lines.append(s.split(";")[0].strip())
    return lines, labels


def classify(ins):
    op = ins.split()[0]
    for name, rx in CLASSES:
        if re.match(rx, op):
            return name
    return "other"


def loops(lines, labels):
    out = []
    for i, ins in enumerate(lines):
        m = re.match(r"^s_(cbranch_\w+|branch)\s+([.\w$]+)", ins)
        if m and m[2] in labels and labels[m[2]] <= i:
            out.append((labels[m[2]], i))
    return out


def main():
    for path in sys.argv[1:]:
        lines, labels = parse(path)
        print(f"== {path}")
        rows = []
        for a, b in loops(lines, labels):
            c = collections.Counter(classify(x) for x in lines[a:b + 1])
            if c["mfma"]:
                rows.append((a, b, c))
        # keep innermost loops (no other MFMA loop strictly inside)
        inner = [r for r in rows if not any(o is not r and r[0] <= o[0] and o[1] <= r[1]
                                            and (o[0], o[1]) != (r[0], r[1]) for o in rows)]
        for a, b, c in inner:
            n = c["mfma"]
            valu = sum(v for k, v in c.items() if k.startswith("v_"))
            print(f"loop lines {a}-{b}: {b - a + 1} instructions, {n} MFMA, "
                  f"VALU (all v_* but MFMA) {valu} = {valu / n:.2f} / MFMA")
            for name, _ in CLASSES + [("other", None)]:
                if c[name]:
                    print(f"   {name:22s} {c[name]:6d}   {c[name] / n:6.2f} / MFMA")


if __name__ == "__main__":
    main()
