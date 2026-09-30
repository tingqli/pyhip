"""Read-only gated gfx942 QSA benchmark with rotating buffers and useful FLOPs.

candidate_run times the prepared dispatcher (the legacy label was flydsl_run).
plan_and_run also rebuilds dynamic plans in preallocated scratch; static dense
and direct metadata preparation is excluded, even when rebuilding is a no-op.
baseline_run times the frozen baseline with its own output allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import msgspec
import torch

from pyhip.testing.misc import cudaPerf
from tests.ops.gr_read.test_gr_read import (
    read_hardware,
    tensor_address,
    validate_hardware,
)

from . import baseline, dense, direct, implementation, kernel
from .inputs import default_spec, make_inputs, validate_inputs
from .reference import check_output


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _gate(*, gpu, amd_smi, folder, phase):
    snapshot = read_hardware(gpu, amd_smi)
    snapshot["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    props = torch.cuda.get_device_properties(gpu)
    bdf = (
        f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    )
    snapshot["runtime"] = {"name": props.name, "arch": props.gcnArchName, "bdf": bdf}
    _write(folder / f"hardware_{phase}.json", snapshot)
    assert (
        bdf.lower() == snapshot["card"]["PCI Bus"].lower()
    ), "Physical/runtime device mismatch"
    validate_hardware(snapshot)


def _clone(inputs):
    values = {}
    for field in inputs.__struct_fields__:
        value = getattr(inputs, field)
        values[field] = value.clone() if isinstance(value, torch.Tensor) else value
    return type(inputs)(**values)


def _prepare_buffers(*, args, spec):
    source = make_inputs(spec=spec, device=f"cuda:{args.gpu}")
    validate_inputs(inputs=source)
    useful_flops = int((source.indices >= 0).sum()) * source.q.shape[1] * 4 * 256
    buffers, errors = [], []
    for i in range(args.buffers):
        inputs = source if i == 0 else _clone(source)
        plan = implementation.prepare(
            inputs=inputs,
            mode=args.algorithm,
            query_tile=args.query_tile,
            grid_multiplier=args.grid_multiplier,
            max_union_inflation=1.5 if args.algorithm == "auto" else float("inf"),
            dense_limit=args.dense_limit,
            block_n=args.block_n,
        )
        base_plan = baseline.prepare(inputs=inputs)
        out = torch.full_like(inputs.q, float("nan"))
        expected = torch.full_like(inputs.q, float("nan"))
        full_out = torch.full_like(inputs.q, float("nan"))
        baseline.run(inputs=inputs, prepared=base_plan, out=expected)
        implementation.run(inputs=inputs, prepared=plan, out=out)
        implementation.rebuild_plan(inputs=inputs, plan=plan)
        implementation.run(inputs=inputs, prepared=plan, out=full_out)
        torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)
        torch.testing.assert_close(full_out, expected, rtol=0.02, atol=0.02)
        errors.append(
            check_output(inputs=inputs, output=out, sample_count=args.reference_rows)
        )
        buffers.append((inputs, plan, base_plan, out, expected, full_out))
    return buffers, useful_flops, errors


def _calls(buffers):
    functions = {"candidate_run": [], "plan_and_run": [], "baseline_run": []}
    for inputs, plan, base_plan, out, expected, full_out in buffers:

        def run(inputs=inputs, plan=plan, out=out):
            implementation.run(inputs=inputs, prepared=plan, out=out)

        def full(inputs=inputs, plan=plan, out=full_out):
            implementation.rebuild_plan(inputs=inputs, plan=plan)
            implementation.run(inputs=inputs, prepared=plan, out=out)

        def original(inputs=inputs, plan=base_plan, out=expected):
            baseline.run(inputs=inputs, prepared=plan, out=out)

        functions["candidate_run"].append(run)
        functions["plan_and_run"].append(full)
        functions["baseline_run"].append(original)
    return functions


def _selection_hashes(inputs):
    return {
        name: hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, tensor in (
            ("indices", inputs.indices),
            ("block_indices", inputs.block_indices),
        )
    }


def _audit_plan(inputs, plan):
    """Audit SparsePlan; only active tiles must publish compact membership.

    counts[:, 0] and active must be initialized for every tile. Inactive tiles
    are checked against the fully rebuilt dense_membership table instead of
    reading their unspecified compact blocks, membership, or score masks.
    """
    blocks = plan.blocks.cpu().numpy()
    bits = plan.membership.cpu().numpy()
    counts = plan.counts[:, 0].cpu().tolist()
    active = plan.active.cpu().tolist()
    dense_bits = plan.dense_membership.cpu().numpy() if not all(active) else None
    meta = plan.metadata.cpu().tolist()
    selected = inputs.block_indices.cpu().numpy()
    positions = inputs.query_positions.cpu().tolist()
    assert len(meta) == len(counts) == len(active) == plan.num_tiles
    assert all(flag in (0, 1) for flag in active), "Invalid union active flags"
    for tile, (first, rows, _, _, position0) in enumerate(meta):
        assert 0 < rows <= plan.query_tile <= 32
        expected = {}
        for local in range(rows):
            assert positions[first + local] == position0 + local
            chosen = selected[first + local]
            for block in chosen[chosen >= 0]:
                expected[int(block)] = expected.get(int(block), 0) | (1 << local)
            visible = positions[first + local] + 1
            if visible % 4:
                block = visible // 4
                expected[block] = expected.get(block, 0) | (1 << local)
        count = counts[tile]
        assert count == len(expected), f"Union count mismatch in tile {tile}"
        if active[tile]:
            assert count <= blocks.shape[1] and count <= bits.shape[1]
            actual = {
                int(b): int(m) & 0xFFFFFFFF
                for b, m in zip(blocks[tile, :count], bits[tile, :count])
            }
            assert len(actual) == count, f"Duplicate compact blocks in tile {tile}"
        else:
            assert dense_bits is not None
            row = dense_bits[tile]
            actual = {int(b): int(row[b]) & 0xFFFFFFFF for b in row.nonzero()[0]}
        assert actual == expected, f"Block membership mismatch in tile {tile}"
    return {
        "tiles": len(meta),
        "active_tiles": sum(active),
        "inactive_tiles": len(active) - sum(active),
        "exact_membership_verified": True,
        "counts_verified": True,
        "inactive_membership_source": "dense_membership",
    }


def _audit_dispatch(inputs, plan, dense_limit):
    """Check the host dense/sparse partition without assuming a DirectPlan layout."""
    counts = list(plan.dense.query_counts)
    expected = [
        min(length, max(0, dense_limit - prefix))
        for length, prefix in zip(inputs.spec.query_lens, inputs.spec.prefix_lens)
    ]
    assert counts == expected, "Dense query counts disagree with the host contract"
    assert plan.mode in ("auto", "direct", "union"), "Unknown dispatch mode"
    sparse_rows = sum(inputs.spec.query_lens) - sum(counts)
    result = {
        "mode": plan.mode,
        "dense_query_counts": counts,
        "dense_rows": sum(counts),
        "sparse_rows": sparse_rows,
        "static_partition_verified": True,
        "direct_plan_present": plan.direct is not None,
        "direct_mapping_audit": "output correctness; private mapping not inspected",
        "union": None,
    }
    if plan.union is None:
        assert sparse_rows == 0 or plan.direct is not None, "Uncovered sparse rows"
        return result

    union = plan.union
    assert 1 <= union.query_tile <= 32
    metadata, q0, k0 = [], 0, 0
    for length, prefix, count in zip(
        inputs.spec.query_lens, inputs.spec.prefix_lens, counts
    ):
        local = count
        while local < length:
            end = min(length, (local // union.query_tile + 1) * union.query_tile)
            metadata.append(
                [
                    q0 + local,
                    end - local,
                    k0,
                    prefix + length,
                    prefix + local,
                ]
            )
            local = end
        q0 += length
        k0 += prefix + length
    assert union.metadata.cpu().tolist() == metadata, "Sparse-only metadata mismatch"
    result["union"] = _audit_plan(inputs, union)
    return result


def _artifacts(folder):
    result = []
    caches = (
        ("union", kernel._COMPILED),
        ("direct", direct._COMPILED),
        ("dense_bounded", dense._BOUNDED_COMPILED),
        ("dense_native_linear", dense.native._COMPILED),
    )
    entries = (
        (name, key, compiled)
        for name, cache in caches
        for key, compiled in cache.items()
    )
    for i, (cache_name, cache_key, compiled) in enumerate(entries):
        text = compiled._keepalive.ir
        fields = {
            name: [int(v) for v in re.findall(rf"\b{name} = (\d+)", text)]
            for name in (
                "vgpr_count",
                "sgpr_count",
                "private_segment_fixed_size",
                "vgpr_spill_count",
                "sgpr_spill_count",
                "group_segment_fixed_size",
            )
        }
        for name in (
            "private_segment_fixed_size",
            "vgpr_spill_count",
            "sgpr_spill_count",
        ):
            assert fields[name] and not any(fields[name]), (name, fields[name])
        (folder / f"compiled_{i}.mlir").write_text(text)
        blobs = []
        for binary in re.findall(r'bin = "((?:\\.|[^"\\])*)"', text):
            data = bytearray()
            cursor = 0
            while cursor < len(binary):
                if binary[cursor] == "\\":
                    if binary[cursor + 1] in ("\\", '"'):
                        data.append(ord(binary[cursor + 1]))
                        cursor += 2
                    else:
                        data.append(int(binary[cursor + 1 : cursor + 3], 16))
                        cursor += 3
                else:
                    data.append(ord(binary[cursor]))
                    cursor += 1
            assert data[:4] == b"\x7fELF"
            name = f"compiled_{i}_{len(blobs)}.hsaco"
            (folder / name).write_bytes(data)
            blobs.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
        assert blobs, "Missing actual executed ELF in compiled IR"
        result.append(
            {
                "cache": cache_name,
                "specialization": repr(cache_key),
                "ir_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "code_objects": blobs,
                **fields,
            }
        )
    return result


def benchmark(*, args, name):
    folder = (
        args.output
        / f"{name}_tp{args.attention_tp}_{args.selection}_bq{args.query_tile}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "raw": [],
        "errors": [],
        "attention_tp": args.attention_tp,
        "selection_group": args.selection_group,
        "query_tile": args.query_tile,
        "grid_multiplier": args.grid_multiplier,
        "algorithm": args.algorithm,
        "dense_limit": args.dense_limit,
        "block_n": args.block_n,
        "buffers": args.buffers,
        "warmup": args.warmup,
        "samples": args.samples,
    }
    try:
        _gate(gpu=args.gpu, amd_smi=args.amd_smi, folder=folder, phase="before")
        spec = default_spec(
            name=name,
            query_tokens=args.query_tokens,
            prefix_tokens=args.prefix_tokens,
            attention_tp=args.attention_tp,
            selection=args.selection,
            selection_group=args.selection_group,
            seed=args.seed,
        )
        buffers, flops, errors = _prepare_buffers(args=args, spec=spec)
        functions = _calls(buffers)
        selection_hashes = [_selection_hashes(b[0]) for b in buffers]
        assert all(h == selection_hashes[0] for h in selection_hashes)
        report.update(
            spec=msgspec.to_builtins(spec),
            useful_flops=flops,
            correctness=errors,
            indices_sha256=selection_hashes[0]["indices"],
            block_indices_sha256=selection_hashes[0]["block_indices"],
            selection_sha256=selection_hashes,
            timing="cudaPerf, rotating allocations, median of all raw samples",
            scope={
                "candidate_run": "prepared dispatch, including dense/direct/union branches",
                "plan_and_run": (
                    "rebuild_plan + dispatch; preallocated scratch, no static metadata "
                    "preparation; direct-only rebuilding may be a no-op"
                ),
                "baseline_run": "frozen baseline with caller-owned output",
            },
            legacy_scope_note=(
                "candidate_run replaces flydsl_run; neither label implies pure union "
                "execution. Legacy wave timing is not emitted by this harness."
            ),
            allocations=[
                {
                    label: tensor_address(
                        t, output=label in ("out", "baseline_out", "full_out")
                    )
                    for label, t in {
                        "q": b[0].q,
                        "k": b[0].k,
                        "v": b[0].v,
                        "indices": b[0].indices,
                        "block_indices": b[0].block_indices,
                        "out": b[3],
                        "baseline_out": b[4],
                        "full_out": b[5],
                    }.items()
                }
                for b in buffers
            ],
        )
        for allocation in report["allocations"]:
            pointers = {
                allocation[label]["pointer"]
                for label in ("out", "baseline_out", "full_out")
            }
            assert (
                len(pointers) == 3
            ), "Each timing scope requires an independent output allocation"
        assert all(
            len({allocation[label]["pointer"] for allocation in report["allocations"]})
            == args.buffers
            for label in report["allocations"][0]
        ), "Rotating buffers must have independent allocations"
        union = buffers[0][1].union
        report["union"] = None
        if union is not None:
            counts = union.counts[:, 0].cpu().tolist()
            report["union"] = {
                "min": min(counts, default=None),
                "median": statistics.median(counts) if counts else None,
                "max": max(counts, default=None),
                "active_tiles": int(union.active.sum()),
                "total_tiles": len(counts),
                "query_tile": union.query_tile,
            }
        print(
            f"{name} TP{args.attention_tp} BQ{args.query_tile} "
            f"{args.selection} selection_group={args.selection_group} "
            f"indices_sha256={report['indices_sha256']} "
            f"block_indices_sha256={report['block_indices_sha256']}",
            flush=True,
        )
        for funcs in functions.values():
            for call in funcs:
                for _ in range(args.warmup):
                    call()
        torch.cuda.synchronize()
        for _, _, _, out, expected, full_out in buffers:
            torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)
            torch.testing.assert_close(full_out, expected, rtol=0.02, atol=0.02)
        report["artifacts"] = _artifacts(folder)
        report["artifacts_scope"] = (
            "Actual FlyDSL process-cache ELF/IR, including earlier TP/case entries "
            "in this process; this is not a per-case launch trace"
        )
        report["plan_audit"] = [
            _audit_dispatch(b[0], b[1], args.dense_limit) for b in buffers
        ]
        # Validate the actual warmed selection buffers, including direct-only runs.
        for inputs, *_ in buffers:
            validate_inputs(inputs=inputs)
        report["warmed_selection_sha256"] = [_selection_hashes(b[0]) for b in buffers]
        assert report["warmed_selection_sha256"] == selection_hashes
        _gate(gpu=args.gpu, amd_smi=args.amd_smi, folder=folder, phase="before_samples")
        timer = cudaPerf(name="qsa", verbose=0)
        if not timer.enable:
            raise RuntimeError("CUDAPERF disabled measurement")
        for sample in range(args.samples):
            buffer_id = sample % args.buffers
            order = list(functions) if sample % 2 == 0 else list(reversed(functions))
            for scope in order:
                wall = time.perf_counter()
                with timer:
                    functions[scope][buffer_id]()
                report["raw"].append(
                    {
                        "scope": scope,
                        "sample": sample,
                        "buffer": buffer_id,
                        "us": timer.latencies[-1] * 1e6,
                        "wall_us_including_timer_preamble": (time.perf_counter() - wall)
                        * 1e6,
                    }
                )
        for _, _, _, out, expected, full_out in buffers:
            torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)
            torch.testing.assert_close(full_out, expected, rtol=0.02, atol=0.02)
        report["post_sample_plan_audit"] = [
            _audit_dispatch(b[0], b[1], args.dense_limit) for b in buffers
        ]
        report["post_sample_selection_sha256"] = [
            _selection_hashes(b[0]) for b in buffers
        ]
        assert (
            report["post_sample_selection_sha256"] == selection_hashes
        ), "Timed calls modified the input selection"
        report["summary"] = {}
        for scope in functions:
            samples = [v["us"] for v in report["raw"] if v["scope"] == scope]
            median = statistics.median(samples)
            tflops = flops / (median * 1e6)
            target = (
                200.0
                if args.attention_tp in (2, 4) and scope != "baseline_run"
                else None
            )
            report["summary"][scope] = {
                "median_us": median,
                "effective_tflops": tflops,
                "target_tflops": target,
                "target_met": None if target is None else tflops >= target,
            }
            print(
                f"{name} TP{args.attention_tp} BQ{args.query_tile} "
                f"SG{args.selection_group} {args.selection} {scope}: "
                f"{median:.3f} us, {tflops:.2f} effective TFLOPS",
                flush=True,
            )
        report["complete"] = True
    except BaseException as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["versions"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": version("triton"),
            "flydsl": version("flydsl"),
        }
        root = Path(__file__).parent
        report["source_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.glob("*.py")
        }
        report["reference_source_sha256"] = {
            str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root.parent / "mha").glob("mha_pa_bf16*942.py")
        }
        (folder / "source").mkdir(exist_ok=True)
        for p in root.glob("*.py"):
            (folder / "source" / p.name).write_bytes(p.read_bytes())
        try:
            _gate(gpu=args.gpu, amd_smi=args.amd_smi, folder=folder, phase="after")
        except BaseException as exc:
            report["complete"] = False
            report["errors"].append(f"after gate: {type(exc).__name__}: {exc}")
            raise
        finally:
            _write(folder / "result.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("all", "no_prefix", "chunk_prefill"), default="all"
    )
    parser.add_argument("--query-tokens", type=int, default=12000)
    parser.add_argument("--prefix-tokens", type=int, default=12000)
    parser.add_argument("--attention-tp", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument(
        "--tp-list",
        type=int,
        nargs="+",
        choices=(1, 2, 4, 8),
        help="Override --attention-tp with a sweep, e.g. --tp-list 2 4",
    )
    parser.add_argument(
        "--query-tile",
        type=int,
        default=32,
        help="Requested union query tile; prepare may cap it for local head grouping",
    )
    parser.add_argument(
        "--selection-group",
        type=int,
        default=32,
        help="Shared-selection group size, independent of --query-tile",
    )
    parser.add_argument("--grid-multiplier", type=int, default=2)
    parser.add_argument(
        "--algorithm", choices=("auto", "direct", "union"), default="auto"
    )
    parser.add_argument("--dense-limit", type=int, default=2051)
    parser.add_argument("--block-n", type=int, choices=(32, 64), default=32)
    parser.add_argument(
        "--selection",
        choices=("independent", "shared", "recent"),
        default="independent",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amd-smi")
    parser.add_argument("--buffers", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--reference-rows", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.buffers, args.samples, args.warmup, args.reference_rows) < 1:
        parser.error("buffers/samples/warmup/reference-rows must be positive")
    if min(args.query_tile, args.selection_group, args.grid_multiplier) < 1:
        parser.error("query-tile/selection-group/grid-multiplier must be positive")
    if not 0 <= args.dense_limit <= 2051:
        parser.error("dense-limit must be in [0, 2051]")
    if args.tp_list is not None and len(set(args.tp_list)) != len(args.tp_list):
        parser.error("tp-list must not contain duplicates")
    if any(
        os.environ.get(k)
        for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    ):
        parser.error(
            "Use --gpu with unremapped devices so physical PTL gates match runtime"
        )
    torch.cuda.set_device(args.gpu)
    for attention_tp in args.tp_list or (args.attention_tp,):
        case_args = argparse.Namespace(**vars(args))
        case_args.attention_tp = attention_tp
        for name in (
            ("no_prefix", "chunk_prefill") if args.case == "all" else (args.case,)
        ):
            benchmark(args=case_args, name=name)


if __name__ == "__main__":
    main()
