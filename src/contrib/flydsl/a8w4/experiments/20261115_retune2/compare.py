"""Winners of two tuned tables per (shape, act, tokens): config, us, and the new table's
time of the old winner (what keeping the old choice would cost under the new protocol).

    python compare.py old.csv new.csv
"""
import csv
import sys
from collections import defaultdict


def load(p):
    t = defaultdict(dict)
    for r in csv.DictReader(open(p)):
        k = (r["model_dim"], r["inter_dim"], r["experts"], r["topk"], r["act"], int(r["num_tokens"]))
        t[k][r["config"]] = float(r["us"])
    return t


old, new = load(sys.argv[1]), load(sys.argv[2])
NAMES = {("256", "256"): "tp8", ("512", "256"): "tp4", ("2048", "32"): "ep8"}
for k in sorted(new, key=lambda k: (NAMES.get((k[1], k[2]), k[1]), k[4], k[5])):
    case = NAMES.get((k[1], k[2]), f"I{k[1]}xE{k[2]}")
    nw = min(new[k], key=new[k].get)
    if k not in old:
        print(f"{case} {k[4]:6s} {k[5]:5d}  new {nw:28s} {new[k][nw]:8.1f}  (no old row)")
        continue
    ow = min(old[k], key=old[k].get)
    keep = new[k].get(ow)
    gain = f"{(new[k][nw] / keep - 1) * 100:+5.1f}%" if keep else "  n/a"
    flag = "" if ow == nw else "  *"
    print(f"{case} {k[4]:6s} {k[5]:5d}  old {ow:28s} {old[k][ow]:8.1f}  new {nw:28s} "
          f"{new[k][nw]:8.1f}  old-cfg-now {keep or float('nan'):8.1f} {gain}{flag}")
