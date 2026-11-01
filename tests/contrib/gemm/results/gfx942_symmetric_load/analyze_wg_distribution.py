import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def analyze(path, expected_wgs):
    occupancy = json.loads(path.read_text())
    targets = {int(key) for key, name in occupancy["dispatches"].items() if "gemm_a8w8_blockscale_8wave_gfx942" in name}
    assert len(targets) == 1
    records = []
    engine_records = []
    for engine, events in occupancy.items():
        if not engine.isdigit():
            continue
        events = sorted(event for event in events if event[5] in targets)
        if not events:
            continue
        start, end = events[0][0], events[-1][0]
        per_cu = defaultdict(list)
        for event in events:
            per_cu[int(event[1])].append(event)
        engine_cus = []
        for cu, cu_events in sorted(per_cu.items()):
            live = set()
            previous = start
            resident_cycles = 0
            starts = 0
            maximum_resident = 0
            first_start = None
            last_end = None
            for timestamp, _, simd, slot, entering, _ in cu_events:
                if live:
                    resident_cycles += timestamp - previous
                key = (int(simd), int(slot))
                if entering:
                    assert key not in live, (engine, cu, key, "duplicate wave start")
                    live.add(key)
                    starts += 1
                    first_start = timestamp if first_start is None else first_start
                else:
                    assert key in live, (engine, cu, key, "wave end without start")
                    live.remove(key)
                    last_end = timestamp
                maximum_resident = max(maximum_resident, len(live))
                previous = timestamp
            assert not live and starts % 8 == 0
            assert first_start is not None and last_end is not None
            assert 0 <= resident_cycles <= end - start
            item = {
                "shader_engine": int(engine), "cu": cu, "WG_count": starts // 8,
                "maximum_resident_waves": maximum_resident,
                "engine_horizon_cycles": end - start,
                "first_start_relative": first_start - start,
                "last_end_relative": last_end - start,
                "last_end_fraction": (last_end - start) / (end - start),
                "resident_cycles": resident_cycles,
                "no_resident_wave_cycles": end - start - resident_cycles,
                "trailing_idle_cycles": end - last_end,
                "trailing_idle_fraction": (end - last_end) / (end - start),
            }
            records.append(item)
            engine_cus.append(item)
        engine_records.append({
            "shader_engine": int(engine), "CUs": len(engine_cus), "horizon_cycles": end - start,
            "WG_histogram": dict(Counter(item["WG_count"] for item in engine_cus)),
            "resident_CU_time_fraction": sum(item["resident_cycles"] for item in engine_cus) / (len(engine_cus) * (end - start)),
        })
    assert len(records) == 80
    assert sum(item["WG_count"] for item in records) == expected_wgs
    denominator = sum(item["engine_horizon_cycles"] for item in records)
    grouped = {}
    for count in sorted({item["WG_count"] for item in records}):
        group = [item for item in records if item["WG_count"] == count]
        grouped[str(count)] = {
            "CU_count": len(group),
            "median_last_end_fraction": statistics.median(item["last_end_fraction"] for item in group),
            "median_trailing_idle_fraction": statistics.median(item["trailing_idle_fraction"] for item in group),
            "mean_resident_cycles_per_WG": statistics.mean(item["resident_cycles"] / count for item in group),
        }
    return {
        "source": str(path), "actual_WGs": expected_wgs, "physical_CUs": len(records),
        "WG_histogram": dict(Counter(item["WG_count"] for item in records)),
        "maximum_resident_waves_per_CU": max(item["maximum_resident_waves"] for item in records),
        "resident_CU_time_fraction": sum(item["resident_cycles"] for item in records) / denominator,
        "no_resident_CU_time_fraction": sum(item["no_resident_wave_cycles"] for item in records) / denominator,
        "trailing_idle_CU_time_fraction": sum(item["trailing_idle_cycles"] for item in records) / denominator,
        "by_WG_count": grouped, "engines": engine_records, "CUs": records,
        "scope": "Each SE uses its own first-to-last target-wave horizon; aggregate is CU-time weighted. Resident occupancy includes stalled waves and is not MFMA utilization. No cross-SE timestamp alignment is assumed.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = {"4096x4096": analyze(args.small, 256), "5120x4096": analyze(args.control, 320)}
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    for name, result in reports.items():
        print(name, json.dumps({key: value for key, value in result.items() if key not in ("CUs", "engines", "scope", "source")}))


if __name__ == "__main__":
    main()