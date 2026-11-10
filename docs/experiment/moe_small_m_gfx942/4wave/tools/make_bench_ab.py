"""Write bench_ab.py: tests/contrib/moe/bench_moe_small_m_gfx942.py plus an "old8" path, the previous 8-wave design
end to end (aiter moe_sorting(128) zeroing out, then variants/moe_small_m_8wave_sort128.fused_moe_small_m with BF16
atomics), checked against the per-128 asm reference, and a "c64" path, the 4-wave version with 64-column GateUp
tiles (variants/moe_small_m_4wave_64col.py) through its mixed dispatch (moe_sorting(256) zeroing out, small-M
GateUp with A2 per (row, 64), BM256 GateUp of big blocks, both Downs), not checked (per-64 A2 differs from the
per-128 reference by more than its threshold). Everything else (timing protocol, gates, JSON) is the repository
benchmark's, so all paths are measured in one session.

usage: python make_bench_ab.py && HIP_VISIBLE_DEVICES=1 python bench_ab.py --paths asm small old8 c64 ...
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.normpath(os.path.join(HERE, "../../../../../tests/contrib/moe"))
OLD8 = os.path.normpath(os.path.join(HERE, "../variants/moe_small_m_8wave_sort128.py"))
C64 = os.path.normpath(os.path.join(HERE, "../variants/moe_small_m_4wave_64col.py"))

s = open(os.path.join(TESTS, "bench_moe_small_m_gfx942.py")).read()


def patch(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


patch("import time\n", f"import time\n\nsys.path.insert(0, {TESTS!r})\n")
patch('choices=["asm", "bm256", "small", "split"])', 'choices=["asm", "bm256", "small", "split", "old8", "c64"])')
patch("    from pyhip.contrib.flydsl import moe_small_m_a8w8_blockscale_gfx942 as small\n",
      "    from pyhip.contrib.flydsl import moe_small_m_a8w8_blockscale_gfx942 as small\n"
      "    import importlib.util\n\n"
      "    def load_variant(name, path):\n"
      "        spec = importlib.util.spec_from_file_location(f'pyhip.contrib.flydsl._{name}', path)\n"
      "        module = importlib.util.module_from_spec(spec)\n"
      "        sys.modules[spec.name] = module\n"
      "        spec.loader.exec_module(module)\n"
      "        return module\n\n"
      f"    old8 = load_variant('old8', {OLD8!r})\n"
      f"    c64 = load_variant('c64', {C64!r})\n")
patch("        def path_small(c, big_blocks=\"bm256\"):",
      "        def path_old8(c):\n"
      "            s = sort(128, True)\n"
      "            mark(\"sort\")\n"
      "            old8.fused_moe_small_m(\n"
      "                c[\"a\"], c[\"a_scale_t\"], c[\"w1\"], c[\"w1_scale\"], c[\"w2\"], c[\"w2_scale\"], s[\"sorted_ids\"], s[\"sorted_weights\"],\n"
      "                s[\"sorted_expert_ids\"], s[\"num_valid_ids\"], out, TOPK, single_store=False, ring=2, b_aux=2, out_zeroed=True,\n"
      "            )\n\n"
      "        def path_c64(c):\n"
      "            s = sort(256, True)\n"
      "            mark(\"sort\")\n"
      "            blocks = min(s[\"sorted_expert_ids\"].numel(), max_sorted_blocks(tokens, TOPK, E))\n"
      "            e256 = s[\"sorted_expert_ids\"][:blocks]\n"
      "            a2 = torch.empty((blocks * 256, I), device=\"cuda\", dtype=dtypes.fp8)\n"
      "            a2_scale = torch.empty((I // 64, blocks * 256), device=\"cuda\")\n"
      "            c64.moe_gateup_small_m(c[\"a\"], c[\"w1\"], c[\"a_scale_t\"], c[\"w1_scale\"], s[\"sorted_ids\"], e256, s[\"num_valid_ids\"], a2, a2_scale, TOPK, 2, 2)\n"
      "            if tokens > 128:\n"
      "                a2_scale_t = torch.empty((I // 128, blocks * 256), device=\"cuda\")\n"
      "                moe_gateup_a8w8_blockscale_gfx942_lds(\n"
      "                    c[\"a\"], c[\"w1\"], c[\"a_scale_t\"], c[\"w1_scale\"], s[\"sorted_ids\"], e256, s[\"num_valid_ids\"], a2, layout=\"2x4_pf1\",\n"
      "                    out_scale_t=a2_scale_t, topk=TOPK, big_blocks_only=True)\n"
      "            c64.moe_down_small_m(a2, c[\"w2\"], a2_scale, c[\"w2_scale\"], s[\"sorted_ids\"], s[\"sorted_weights\"], e256, s[\"num_valid_ids\"], out, TOPK, 2, 2)\n"
      "            if tokens > 128:\n"
      "                moe_down_a8w8_blockscale_gfx942_lds(\n"
      "                    a2, c[\"w2\"], a2_scale_t, c[\"w2_scale\"], s[\"sorted_ids\"], s[\"sorted_weights\"], e256, s[\"num_valid_ids\"], out,\n"
      "                    layout=\"4x2_pf1\", reduce=\"atomic\", zero_out=False, topk=TOPK, big_blocks_only=True)\n\n"
      "        def path_small(c, big_blocks=\"bm256\"):")
patch("        if args.kernels:\n            s256 = sort(256, False)",
      "        if \"old8\" in args.paths:\n"
      "            variants[\"old8\"] = (None, path_old8, expected, True)\n"
      "        if \"c64\" in args.paths:\n"
      "            variants[\"c64\"] = (None, path_c64, None, True)\n"
      "        if args.kernels:\n            s256 = sort(256, False)")
patch('            for k in asm_names if "asm" in args.paths else []:\n',
      '            if "old8" in args.paths:\n                s128 = sort(128, False)\n                blocks128 = min(s128["sorted_expert_ids"].numel(), old8.max_sorted_blocks(tokens, TOPK, E))\n                e128 = s128["sorted_expert_ids"][:blocks128]\n                a2_128 = torch.empty((blocks128 * 128, I), device="cuda", dtype=dtypes.fp8)\n                a2s_128 = torch.empty((I // 128, blocks128 * 128), device="cuda")\n                variants["k: sort128Z"] = (None, lambda c: sort(128, True), None, False)\n                variants["k: gateup_old8"] = (None, lambda c: old8.moe_gateup_small_m(\n                    c["a"], c["w1"], c["a_scale_t"], c["w1_scale"], s128["sorted_ids"], e128, s128["num_valid_ids"], a2_128, a2s_128,\n                    TOPK, 2, 2), None, True)\n                variants["k: down_old8"] = (None, lambda c: old8.moe_down_small_m(\n                    a2_128, c["w2"], a2s_128, c["w2_scale"], s128["sorted_ids"], s128["sorted_weights"], e128, s128["num_valid_ids"],\n                    out, TOPK, None, 2, 2), None, True)\n            for k in asm_names if "asm" in args.paths else []:\n')
patch('for ours in ("bm256", "small", "split"):', 'for ours in ("bm256", "small", "split", "old8", "c64"):')
patch('os.path.join(os.path.dirname(os.path.abspath(__file__)), "roofline_probe_gfx942.cpp")',
      repr(os.path.join(TESTS, "roofline_probe_gfx942.cpp")))
with open(os.path.join(HERE, "bench_ab.py"), "w") as f:
    f.write("# Generated by make_bench_ab.py from tests/contrib/moe/bench_moe_small_m_gfx942.py; do not edit.\n" + s)
print("wrote", os.path.join(HERE, "bench_ab.py"))
