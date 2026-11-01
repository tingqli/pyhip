import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import amdsmi
import torch
import pyhip
from pyhip.core.hiptools import _build_hip_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="4096,4096,16384;8192,8192,16384;16384,3584,6144")
    parser.add_argument("--layouts", default="0,1")
    parser.add_argument("--variants", default="baseline,direct,lds")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    amdsmi.amdsmi_init()
    handle = amdsmi.amdsmi_get_processor_handles()[0]
    gates = []

    def gate(stage):
        memory = amdsmi.amdsmi_get_gpu_vram_usage(handle)
        record = {
            "stage": stage, "bdf": amdsmi.amdsmi_get_gpu_device_bdf(handle),
            "activity": amdsmi.amdsmi_get_gpu_activity(handle), "memory": memory,
            "pids": [int(process["pid"]) for process in amdsmi.amdsmi_get_gpu_process_list(handle)],
            "ptl": amdsmi.amdsmi_get_gpu_ptl_state(handle),
            "formats": [amdsmi.AmdSmiPtlData(value).name for value in amdsmi.amdsmi_get_gpu_ptl_formats(handle)],
        }
        gates.append(record)
        print("GATE", json.dumps(record), flush=True)
        assert record["ptl"] and set(record["formats"]) == {"VECTOR", "F8"}
        assert memory["vram_used"] <= memory["vram_total"] * 0.2
        if stage == "entry":
            assert not record["pids"] and record["activity"]["gfx_activity"] <= 5
        else:
            assert len(record["pids"]) <= 1

    gate("entry")
    spec = importlib.util.spec_from_file_location("gemm_test", "/mywork/pyhip/tests/contrib/gemm/test_a8w8_f32_blockscale.py")
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    variants = args.variants.split(",")
    results = []
    symbol = "gemm_a8w8_blockscale_8wave_gfx942"

    for shape in args.shapes.split(";"):
        rows, columns, depth = map(int, shape.split(","))
        for layout in map(int, args.layouts.split(",")):
            preshuffle = bool(layout)
            torch.manual_seed(7)
            activation = (torch.rand((rows, depth), device="cuda") / 10).to(tests.dtypes.fp8)
            weight = (torch.rand((columns, depth), device="cuda") / 10).to(tests.dtypes.fp8)
            scale_a = torch.rand((rows, depth // 128), device="cuda")
            scale_b = torch.rand((pyhip.div_up(columns, 128), depth // 128), device="cuda")
            reference = tests.run_torch(activation, weight, scale_a, scale_b).cpu()
            if preshuffle:
                weight = tests.shuffle_weight(weight, layout=(16, 16))
            scale_a = scale_a.T.contiguous()
            buffers = [tuple(tensor.clone() for tensor in (activation, weight, scale_a, scale_b)) for _ in range(32)]
            output = torch.empty((rows, columns), dtype=torch.bfloat16, device="cuda")
            intervals = sorted((tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.itemsize) for group in buffers for tensor in group)
            assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
            kernels, artifacts = {}, {}
            for variant in variants:
                if variant == "baseline":
                    directory = Path('/tmp/gfx942_perf/cache_v25')
                    suffix = 'gemm_fp8_gfx942.py-10.co'
                else:
                    directory = Path('/tmp/gfx942_perf/symmetric_load/cache')
                    suffix = f'gemm_fp8_gfx942_{variant}.py-*.co'
                matches = list(directory.glob(f'*preshuffle_b={preshuffle}-N={columns}-K={depth}-*{suffix}'))
                assert len(matches) == 1, (variant, matches)
                path = matches[0]
                kernels[variant] = _build_hip_func(str(path), "", None, symbol)
                artifacts[variant] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

            def launch(variant, index):
                x, weights, activation_scale, weight_scale = buffers[index]
                kernels[variant]([pyhip.div_up(rows, 256) * pyhip.div_up(columns, 256)], [512], x, weights, output, activation_scale, weight_scale, rows)

            def validate(variant):
                with torch.device("cpu"):
                    actual = output.cpu()
                    assert tests.checkAllclose(reference, actual, msg=variant, tol_err_ratio=0) == 0
                    return pyhip.calc_diff(reference, actual, diff_thr=1e-5)

            samples, diffs = [], {}
            for sweep, order in enumerate((variants, list(reversed(variants)))):
                for variant in order:
                    for index in range(32):
                        launch(variant, index)
                    torch.cuda.synchronize()
                    validate(variant)
                    gate(f'{shape}:{layout}:{sweep}:{variant}:before')
                    for index in range(32):
                        begin = time.monotonic_ns()
                        with pyhip.cudaPerf(2 * rows * columns * depth, name=variant, verbose=0) as perf:
                            launch(variant, index)
                        samples.append({"sweep": sweep, "variant": variant, "buffer": index,
                                        "us": perf.latencies[0] * 1e6, "begin_ns": begin, "end_ns": time.monotonic_ns()})
                    diffs[variant] = validate(variant)
                    print('RAW_SAMPLES', json.dumps(samples), flush=True)
                    gate(f'{shape}:{layout}:{sweep}:{variant}:after')
            summary = {}
            for variant in variants:
                times = [sample['us'] for sample in samples if sample['variant'] == variant]
                duration = statistics.median(times)
                summary[variant] = {"median_us": duration, "median_tflops": 2 * rows * columns * depth / (duration * 1e6),
                                    "samples": len(times), "diff": diffs[variant],
                                    "sweep_medians_us": [statistics.median(sample['us'] for sample in samples if sample['variant'] == variant and sample['sweep'] == sweep) for sweep in (0, 1)]}
            result = {"shape": [rows, columns, depth], "preshuffle_b": preshuffle, "summary": summary,
                      "samples": samples, "artifacts": artifacts,
                      "input_pointers": [[tensor.data_ptr() for tensor in group] for group in buffers],
                      "output_pointer": output.data_ptr(), "A_B_working_set_bytes": 32 * (rows + columns) * depth}
            results.append(result)
            args.output.write_text(json.dumps({"protocol": "32 independent A/B buffers; forward/reverse full sweeps with full warmup before each; no immediate variant same-address reuse", "gates": gates, "results": results}, indent=1) + '\n')
            print('LOAD_COMPARISON', shape, preshuffle, json.dumps(summary), flush=True)
            del buffers, output, activation, weight, scale_a, scale_b, reference
    amdsmi.amdsmi_shut_down()


if __name__ == '__main__':
    main()