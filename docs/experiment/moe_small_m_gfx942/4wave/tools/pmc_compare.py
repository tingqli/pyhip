"""Hardware counters of small-M kernels: rocprofv3 --pmc over run_one.py (the synthetic blocks of ab_kernels.py:
32 blocks of --rows routes), median over 3 launches, also per MFMA instruction. The SQ / TCC counters are global to
the GPU, so each counter group is a separate run started after 20 s of an idle GPU (GFX activity <= 2 %) and redone
if the activity sampled during it shows another job (two samples above 30 %).

usage: HIP_VISIBLE_DEVICES=2 python pmc_compare.py --impls c64 old8 cur --which gateup down --rows 32 128
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = [
    ["SQ_WAVES", "SQ_INSTS_MFMA", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_INSTS_LDS", "SQ_INSTS_VALU"],
    ["SQ_INSTS_SALU", "SQ_INSTS_SMEM", "SQ_LDS_BANK_CONFLICT", "SQ_BUSY_CYCLES"],
    ["SQ_WAVE_CYCLES", "SQ_WAIT_INST_ANY", "SQ_ACTIVE_INST_ANY", "SQ_VALU_MFMA_BUSY_CYCLES"],
    ["TCC_REQ", "TCC_EA0_RDREQ", "TCC_HIT", "TCC_MISS"],
]

parser = argparse.ArgumentParser()
parser.add_argument("--impls", nargs="+", default=["c64", "old8", "cur"])
parser.add_argument("--which", nargs="+", default=["gateup", "down"])
parser.add_argument("--rows", type=int, nargs="+", default=[32, 128])
parser.add_argument("--json", help="write the raw medians here")
args = parser.parse_args()
gpu = int(os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0])


def activity():
    import amdsmi

    amdsmi.amdsmi_init()
    handle = amdsmi.amdsmi_get_processor_handles()[gpu]
    return lambda: amdsmi.amdsmi_get_gpu_activity(handle)["gfx_activity"]


read_activity = activity()


def wait_idle(quiet_s=20.0):
    quiet_since = None
    while True:
        if read_activity() <= 2:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= quiet_s:
                return
        else:
            quiet_since = None
        time.sleep(0.1)


def collect(impl, which, rows, counters):
    """counter -> median value over the launches, and the median launch time (us)."""
    for attempt in range(50):
        wait_idle()
        samples, done = [], threading.Event()

        def sample():
            while not done.is_set():
                samples.append(read_activity())
                time.sleep(0.25)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        with tempfile.TemporaryDirectory() as out:
            cmd = ["rocprofv3", "--pmc", *counters, "--kernel-include-regex", "small_m", "--output-format", "csv", "-d",
                   out, "-o", "pmc", "--", sys.executable, os.path.join(HERE, "run_one.py"), "--impl", impl, "--which",
                   which, "--rows", str(rows), "--calls", "3"]
            subprocess.run(cmd, check=True, capture_output=True)
            done.set()
            sampler.join()
            if sum(s > 30 for s in samples) >= 2:
                print(f"  ({impl} {counters[0]}...: another job ran during the run, redone)", flush=True)
                continue
            values, times = collections.defaultdict(list), {}
            for row in csv.DictReader(open(glob.glob(os.path.join(out, "*counter_collection.csv"))[0])):
                values[row["Counter_Name"]].append(float(row["Counter_Value"]))
                times[row["Dispatch_Id"]] = (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1e3
        return {name: statistics.median(v) for name, v in values.items()}, statistics.median(times.values())
    raise RuntimeError("no clean run in 50 attempts")


raw = {}
for which in args.which:
    for rows in args.rows:
        print(f"== {which}, {rows} rows per block (32 blocks)", flush=True)
        results = {}
        for impl in args.impls:
            merged, launch_us = {}, []
            for group in GROUPS:
                values, us = collect(impl, which, rows, group)
                merged.update(values)
                launch_us.append(us)
            merged["us"] = statistics.median(launch_us)
            results[impl] = merged
        raw[f"{which}/{rows}"] = results
        if args.json:
            json.dump(raw, open(args.json, "w"), indent=1)
        mfma = {impl: r["SQ_INSTS_MFMA"] for impl, r in results.items()}
        rows_out = [
            ("launch us (median of the counter runs)", lambda r, m: f"{r['us']:.1f}"),
            ("waves", lambda r, m: f"{r['SQ_WAVES']:.0f}"),
            ("MFMA instructions", lambda r, m: f"{m:.3g}"),
            ("VMEM loads / MFMA", lambda r, m: f"{r['SQ_INSTS_VMEM_RD'] / m:.3f}"),
            ("VMEM stores+atomics / MFMA", lambda r, m: f"{r['SQ_INSTS_VMEM_WR'] / m:.4f}"),
            ("LDS instructions / MFMA", lambda r, m: f"{r['SQ_INSTS_LDS'] / m:.3f}"),
            ("VALU (incl. MFMA) / MFMA", lambda r, m: f"{r['SQ_INSTS_VALU'] / m:.3f}"),
            ("SALU / MFMA", lambda r, m: f"{r['SQ_INSTS_SALU'] / m:.3f}"),
            ("SMEM / MFMA", lambda r, m: f"{r['SQ_INSTS_SMEM'] / m:.4f}"),
            ("LDS bank-conflict cycles / LDS instr", lambda r, m: f"{r['SQ_LDS_BANK_CONFLICT'] / max(r['SQ_INSTS_LDS'], 1):.3f}"),
            ("wave cycles waiting on dependencies (%)", lambda r, m: f"{100 * r['SQ_WAIT_INST_ANY'] / r['SQ_WAVE_CYCLES']:.1f}"),
            ("wave cycles issuing (%)", lambda r, m: f"{100 * r['SQ_ACTIVE_INST_ANY'] / r['SQ_WAVE_CYCLES']:.1f}"),
            ("wave cycles / MFMA", lambda r, m: f"{r['SQ_WAVE_CYCLES'] / m:.2f}"),
            ("issuing wave cycles / MFMA", lambda r, m: f"{r['SQ_ACTIVE_INST_ANY'] / m:.2f}"),
            ("waiting wave cycles / MFMA", lambda r, m: f"{r['SQ_WAIT_INST_ANY'] / m:.2f}"),
            ("MFMA busy / SQ busy cycles", lambda r, m: f"{r['SQ_VALU_MFMA_BUSY_CYCLES'] / r['SQ_BUSY_CYCLES']:.2f}"),
            ("L2 requests / MFMA", lambda r, m: f"{r['TCC_REQ'] / m:.3f}"),
            ("L2 hit rate (%)", lambda r, m: f"{100 * r['TCC_HIT'] / max(r['TCC_HIT'] + r['TCC_MISS'], 1):.1f}"),
            ("HBM read requests (M)", lambda r, m: f"{r['TCC_EA0_RDREQ'] / 1e6:.2f}"),
        ]
        print(f"  {'':42s}" + "".join(f"{impl:>12s}" for impl in args.impls))
        for label, fmt in rows_out:
            print(f"  {label:42s}" + "".join(f"{fmt(results[impl], mfma[impl]):>12s}" for impl in args.impls), flush=True)
