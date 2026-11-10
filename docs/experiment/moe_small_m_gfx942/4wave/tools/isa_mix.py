"""Static instruction mix of the K loop per wave and K block, from the compiled ISA of one slice-count variant
(FLYDSL_DUMP_IR=1): counts between the loop header label and its backward branch, divided by the K blocks the loop
body covers (ring = 2). Compile only (one launch on a tiny input).

usage: FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir> FLYDSL_RUNTIME_ENABLE_CACHE=0 python isa_mix.py <impl> <gateup|down> <S>
  impl: cur | c64 | old8 | <module file>
"""
import collections
import glob
import importlib.util
import os
import re
import sys

import torch
from aiter import dtypes

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"c64": "moe_small_m_4wave_64col.py", "old8": "moe_small_m_8wave_sort128.py", "down128": "moe_small_m_4wave_down128.py"}
impl, which, S = sys.argv[1], sys.argv[2], int(sys.argv[3])
path = (importlib.util.find_spec("pyhip.contrib.flydsl.moe_small_m_a8w8_blockscale_gfx942").origin if impl == "cur"
        else os.path.join(HERE, "../variants", FILES[impl]) if impl in FILES else impl)
# Exactly one slice-count variant: the copy drops the check that the variant list ends at 8.
source = open(path).read().replace("assert variants[-1] == 8 and variants[0] >= 1, variants", "pass")
spec = importlib.util.spec_from_loader("pyhip.contrib.flydsl._mix", loader=None)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "pyhip.contrib.flydsl"
sys.modules[spec.name] = mod
exec(compile(source, path, "exec"), mod.__dict__)
orig = mod.compile_moe_small_m
mod.compile_moe_small_m = lambda *a, **k: orig(*a, slice_variants=(S,), **k)
block_m = 128 if impl == "old8" else 256
I, H, TOPK, E, T = 2048, 4096, 8, 1, 256
sid = torch.full((block_m,), T | (TOPK << 24), device="cuda", dtype=torch.int32)
sid[: S * 16] = torch.arange(S * 16, device="cuda", dtype=torch.int32)
eid = torch.zeros(1, device="cuda", dtype=torch.int32)
nv = torch.tensor([block_m], device="cuda", dtype=torch.int32)
a2 = torch.zeros((block_m, I), device="cuda", dtype=dtypes.fp8)
a2s = torch.ones((I // getattr(mod, "A2_SCALE_COLUMNS", 128), block_m), device="cuda")
if which == "gateup":
    a = torch.zeros(T, H, device="cuda", dtype=dtypes.fp8)
    w1 = torch.zeros(E, 2 * I, H, device="cuda", dtype=dtypes.fp8)
    mod.moe_gateup_small_m(a, w1, torch.ones(H // 128, T, device="cuda"), torch.ones(E, 2 * I // 128, H // 128, device="cuda"), sid, eid, nv, a2, a2s, TOPK)
else:
    w2 = torch.zeros(E, H, I, device="cuda", dtype=dtypes.fp8)
    out = torch.zeros(T, H, device="cuda", dtype=torch.bfloat16)
    extra = (None,) if impl == "old8" else ()
    mod.moe_down_small_m(a2, w2, a2s, torch.ones(E, H // 128, I // 128, device="cuda"), sid, torch.ones(block_m, device="cuda"), eid, nv, out, TOPK, *extra)
torch.cuda.synchronize()

lines = open(sorted(glob.glob(os.environ["FLYDSL_DUMP_DIR"] + "/*/22_final_isa.s"))[-1]).read().splitlines()
labels = {line.rstrip(":"): i for i, line in enumerate(lines) if re.match(r"^\.LBB\d+_\d+:", line)}
loops = []
for i, line in enumerate(lines):
    m = re.match(r"\s*s_cbranch_\w+\s+(\.LBB\d+_\d+)", line)
    if m and m.group(1) in labels and labels[m.group(1)] < i:
        loops.append((labels[m.group(1)], i))
# The K loop: the backward branch around the most MFMAs.
start, end = max(loops, key=lambda se: sum("v_mfma" in line for line in lines[se[0]:se[1]]))
mix = collections.Counter()
for line in lines[start + 1 : end + 1]:
    op = line.strip().split(" ")[0]
    if not op or op.startswith((";", ".")):
        continue
    group = ("mfma" if op.startswith("v_mfma") else "buffer_load" if op.startswith("buffer_load") else "ds_read" if op.startswith(("ds_read", "ds_bpermute"))
             else "ds_write" if op.startswith("ds_write") else "s_barrier" if op == "s_barrier" else "s_waitcnt" if op == "s_waitcnt"
             else "valu" if op.startswith("v_") else "salu/branch/nop")
    mix[group] += 1
k_blocks = 2
print(f"{impl:6s} {which:6s} S={S}: per wave and K block " + ", ".join(f"{g} {mix[g] / k_blocks:g}" for g in
      ("mfma", "buffer_load", "ds_read", "ds_write", "s_barrier", "s_waitcnt", "valu", "salu/branch/nop")))
