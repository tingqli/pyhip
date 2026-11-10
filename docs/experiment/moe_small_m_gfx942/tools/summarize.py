"""Markdown tables (paths, kernels, roofline) from bench_moe_small_m_gfx942.py JSON reports.

Usage: python summarize.py final_ep_uniform.json [final_random.json ...] [--hbm-tbps 4.94] [--mfma-tflops 1157]
Roofline per call: bytes = weights + A1 (T*H FP8) + A2 written and read (2*routes*I FP8) + out zeroed and written
(2*T*H BF16); t_mem = bytes / HBM, t_mfma = 6 * routes * I * H / MFMA (FP8 sustained, power-limited clock).
"""
import argparse
import json
import statistics

I, H = 2048, 4096


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--hbm-tbps", type=float, default=4.94, help="1 GiB nontemporal read probe")
    parser.add_argument("--mfma-tflops", type=float, default=1157.0, help="2 s sustained FP8 MFMA probe")
    args = parser.parse_args()
    for path in args.reports:
        report = json.load(open(path))
        runs = report["runs"]
        print(f"\n### {path}: routing {runs[0]['routing']}, gates {'PASS' if report['valid'] else 'FAIL'}\n")
        print("| T | routes/expert (min-max) | asm auto | tuned asm (kernel) | bm256 | small | small vs tuned asm | small vs asm auto | small w/o sort | small per-round medians | vs tuned asm per round |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        rounds = report["args"]["rounds"]
        for run in runs:
            med = run["median_us"]
            tuned = run["tuned_asm"]
            rpe = run["routes_per_expert"]
            small = med.get("small")
            per = len(run["samples_us"]["small"]) // rounds
            round_meds = sorted(statistics.median(run["samples_us"]["small"][i * per : (i + 1) * per]) for i in range(rounds))
            line = [
                str(run["tokens"]), f"{min(rpe)}-{max(rpe)}" if isinstance(rpe, list) else str(rpe), f"{med['asm auto']:.1f}",
                f"{med[tuned]:.1f} ({tuned[4:]})", f"{med.get('bm256', float('nan')):.1f}", f"{small:.1f}",
                f"**{med[tuned] / small:.3f}x**", f"{med['asm auto'] / small:.3f}x", f"{run['without_sort_us']['small']:.1f}",
                "/".join(f"{m:.0f}" for m in round_meds),
                f"{med[tuned] / round_meds[-1]:.2f}-{med[tuned] / round_meds[0]:.2f}x",
            ]
            print("| " + " | ".join(line) + " |")
        print("\n| T | sort128 zeroing | GateUp small | Down small (atomic) | GateUp+Down | sort256 | GateUp BM256 | Down BM256 | best asm kernel |")
        print("|---|---|---|---|---|---|---|---|---|")
        for run in runs:
            med = run["median_us"]
            asm_kernels = {k: v for k, v in med.items() if k.startswith("k: asm")}
            best = min(asm_kernels, key=asm_kernels.get) if asm_kernels else None
            g, d = med.get("k: gateup_small"), med.get("k: down_small_atomic")
            if g is None:
                continue
            print(
                f"| {run['tokens']} | {med['k: sort128Z']:.1f} | {g:.1f} | {d:.1f} | {g + d:.1f} | {med['k: sort256NZ']:.1f} | "
                f"{med['k: gateup256']:.1f} | {med['k: down256']:.1f} | "
                + (f"{asm_kernels[best]:.1f} ({best[7:]}) |" if best else "- |")
            )
        print("\n| T | routes | GFLOP | MiB moved | t_mem us | t_mfma us | bound | max(t_mem, t_mfma) | t_mem + t_mfma | small GateUp+Down | vs max | vs sum |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for run in runs:
            tokens, routes, flops = run["tokens"], run["routes"], run["flops"]
            moved = run["weight_bytes"] + tokens * H + 2 * routes * I + 2 * tokens * H * 2
            t_mem = moved / (args.hbm_tbps * 1e6)
            t_mfma = flops / (args.mfma_tflops * 1e6)
            med = run["median_us"]
            kernels = med.get("k: gateup_small", float("nan")) + med.get("k: down_small_atomic", float("nan"))
            print(
                f"| {tokens} | {routes} | {flops / 1e9:.1f} | {moved / 2**20:.0f} | {t_mem:.1f} | {t_mfma:.1f} | "
                f"{'memory' if t_mem >= t_mfma else 'MFMA'} | {max(t_mem, t_mfma):.1f} | {t_mem + t_mfma:.1f} | {kernels:.1f} | "
                f"{kernels / max(t_mem, t_mfma):.2f}x | {kernels / (t_mem + t_mfma):.2f}x |"
            )


if __name__ == "__main__":
    main()
