# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Independent single-launch QSA cases; setup, resets and oracles are untimed."""

import math
import hashlib
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
import msgspec
import numpy as np
import torch
import triton

from pyhip.ops.qsa.flydsl import attention_direct_packed as packed
from tests.ops.qsa import _attention as helpers

runtime = helpers._runtime
SCOPES = ("recover", "compact", "order_masks_validate", "scatter_prepared",
          "dense", "union", "raw_direct", "pack", "packed_direct")
SENTINEL = -123


@flyc.jit
def _pack_only(K: fx.Tensor, V: fx.Tensor, PK: fx.Tensor, PV: fx.Tensor,
               N: fx.Constexpr[int], HK: fx.Constexpr[int], stream: fx.Stream):
    packed._pack(K, V, PK, PV, N, HK).launch(
        grid=((N // 4 * HK + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)


@flyc.jit
def _packed_only(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor,
                 BLOCKS: fx.Tensor, META: fx.Tensor, ACTIVE: fx.Tensor, QUERY_TILES: fx.Tensor,
                 H: fx.Constexpr[int], HK: fx.Constexpr[int], NQ: fx.Constexpr[int],
                 NK: fx.Constexpr[int], TASKS: fx.Constexpr[int], GATED: fx.Constexpr[bool],
                 SCALE: fx.Constexpr[float], stream: fx.Stream):
    packed._kernel(
        Q, K, V, O, BLOCKS, META, ACTIVE, QUERY_TILES, H, HK, NQ, NK, GATED, SCALE,
        value_attrs={"llvm.target_features": ir.Attribute.parse(
            '#llvm.target_features<["-packed-fp32-ops"]>')},
    ).launch(grid=(TASKS, 1, 1), block=(64, 1, 1), stream=stream)


def _packed_reference(k, v):
    """Logical BF16 permutations, independent of the device lane/shuffle code."""
    n, hk, dim = k.shape
    assert n % 4 == 0 and dim == 256 and v.shape == k.shape
    key = k.reshape(n // 4, 4, hk, 4, 2, 4, 8).permute(0, 3, 2, 4, 5, 1, 6)
    value = v.reshape(n // 4, 4, hk, 2, 16, 4, 2).permute(0, 2, 3, 5, 4, 6, 1)
    value = value.reshape(n // 4, hk, 4, 256).transpose(1, 2)
    return key.contiguous().view_as(k), value.contiguous().view_as(v)


def _order_options(plan, hk):
    slices = triton.cdiv(plan.query_tile * plan.group_padded, 128)
    tasks = plan.num_tiles * hk * slices
    size = min(4096, triton.next_power_of_2(plan.task_order.numel()))
    shift = max(1, (tasks - 1).bit_length())
    return SimpleNamespace(slices=slices, tasks=tasks, size=size, shift=shift,
                           wide=((plan.block_capacity // 16) << shift) + tasks - 1 >= 2**31,
                           ctas=triton.cdiv(plan.task_order.numel(), size),
                           warps=8 if size > 1024 else 4)


def _plan_reference(value, plan):
    """Exact valid-ABI recovery, forced-union membership, masks and task order."""
    tokens, positions = value.indices.cpu().numpy(), value.positions.cpu().tolist()
    recovered = np.full((len(positions), 512), -1, dtype=np.int32)
    for row, position in enumerate(positions):
        visible, count = position + 1, min((position + 1) // 4, 512)
        chosen = tokens[row, :count * 4:4] // 4
        assert len(np.unique(chosen)) == count and np.all((chosen >= 0) & (chosen < visible // 4))
        expanded = np.concatenate(((chosen[:, None] * 4 + np.arange(4)).ravel(),
                                   np.arange(visible - visible % 4, visible)))
        np.testing.assert_array_equal(tokens[row, :len(expanded)], expanded)
        assert np.all(tokens[row, len(expanded):] == -1)
        recovered[row, :count] = np.sort(chosen)

    dense = np.zeros(tuple(plan.dense_membership.shape), dtype=np.uint32)
    blocks = np.full(tuple(plan.blocks.shape), SENTINEL, dtype=np.int32)
    members = np.full_like(blocks, SENTINEL)
    counts = np.zeros((plan.num_tiles, 2), dtype=np.int32)
    masks = np.full(tuple(plan.score_masks.shape), SENTINEL, dtype=np.int32)
    query, quarter = np.arange(plan.query_tile, dtype=np.uint32)[:, None], np.arange(4)[None, :]
    for tile, (first, rows, _, length, position) in enumerate(plan.metadata.cpu().tolist()):
        for local in range(rows):
            selected = recovered[first + local]
            dense[tile, selected[selected >= 0]] |= np.uint32(1 << local)
            visible = positions[first + local] + 1
            if visible % 4:
                dense[tile, visible // 4] |= np.uint32(1 << local)
        ids = np.flatnonzero(dense[tile])
        common = (dense[tile, ids] == (1 << rows) - 1) & (ids * 4 + 3 <= position)
        ordered = np.concatenate((ids[common], ids[~common]))
        count = len(ordered)
        blocks[tile, :count] = ordered
        members[tile, :count] = dense[tile, ordered].view(np.int32)
        counts[tile] = count, int(common.sum()) // 16
        for nt in range(counts[tile, 1], math.ceil(count / 16)):
            bits = np.zeros((plan.query_tile, 4), dtype=np.uint32)
            for group in range(4):
                slot = nt * 16 + quarter * 2 + (group // 2) * 8 + group % 2
                safe = np.minimum(slot, count - 1)
                keep = (slot < count) & (query < rows)
                keep &= ((members[tile, safe].view(np.uint32) >> query) & 1) != 0
                for offset in range(4):
                    token = blocks[tile, safe] * 4 + offset
                    valid = keep & (token <= position + query) & (token < length)
                    bits |= valid.astype(np.uint32) << (group * 4 + offset)
            masks[tile, nt] = bits.view(np.int32)

    options = _order_options(plan, value.k.shape[1])
    order = np.full(plan.task_order.numel(), -1, dtype=np.int32)
    for start in range(0, len(order), options.size):
        ids = list(range(start, min(start + options.size, options.tasks)))
        ids.sort(key=lambda task: (-math.ceil(int(counts[(task % (options.tasks // value.k.shape[1]))
                                                       // options.slices, 0]) / 16), task))
        ids.extend([-1] * (options.size - len(ids)))
        for offset, task in enumerate(ids):
            row, column = divmod(start + offset, plan.grid)
            destination = row * plan.grid + (plan.grid - 1 - column if row % 2 else column)
            if destination < len(order):
                order[destination] = task
    return dict(recovered=recovered, errors=np.zeros(len(positions), dtype=np.int32),
                dense=dense.view(np.int32), blocks=blocks, membership=members, counts=counts,
                active=np.ones(plan.num_tiles, dtype=np.int32), order=order, masks=masks,
                valid=np.asarray(True))


def _buffer(case, name, shape, dtype):
    # Tail canaries preserve allocation-base addresses for performance buffers.
    size = math.prod(shape)
    storage = torch.full((size + 16,), SENTINEL, dtype=dtype, device=case.device)
    result = storage[:size].view(shape)
    case.tensors[name], case.guards[name] = result, storage[size:]
    return result


def _register(case, label, run, reset, check, symbol, *, flydsl=False):
    audited = None

    def invoke():
        with torch.cuda.device(case.device):
            case.compiled[label] = run()

    def verify():
        nonlocal audited
        check()  # Check this invocation's output, never a replacement launch.
        for name, guard in case.guards.items():
            assert torch.equal(guard, torch.full_like(guard, SENTINEL)), name
        compiled = case.compiled[label]
        assert compiled is not None, f"{label}: missing compiled kernel"
        if compiled is not audited:
            if flydsl:
                text = compiled._keepalive.ir
                assert re.findall(r'#gpu\.kernel_metadata<"([^"]+)"', text) == [symbol]
                pattern = lambda field: rf"\b{field}\s*=\s*(\d+)"
                evidence = dict(ir_sha256=hashlib.sha256(text.encode()).hexdigest())
            else:
                assert compiled.src.fn.fn.__name__ == symbol
                binary = compiled.asm["hsaco"]
                readelf = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm/bin/llvm-readelf"
                text = subprocess.run([str(readelf), "--notes", "-"], input=binary,
                                      stdout=subprocess.PIPE, check=True).stdout.decode()
                pattern = lambda field: rf"\.{field}:\s+(\d+)"
                evidence = dict(hsaco_sha256=hashlib.sha256(binary).hexdigest(),
                                dynamic_shared_bytes=int(compiled.metadata.shared))
            for field in ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"):
                values = re.findall(pattern(field), text)
                assert len(values) == 1 and int(values[0]) == 0, (symbol, field, values)
                evidence[field] = 0
            case.metadata["kernels"][label] = dict(symbol=symbol, launches_per_run=1, **evidence)
            audited = compiled

    case.runs[label], case.reset[label], case.checks[label] = invoke, reset, verify


def _flydsl(case, label, launcher, args, reset, check, symbol):
    compiled = None

    def run():
        nonlocal compiled
        current = (*args, torch.cuda.current_stream(case.device))
        if compiled is None:
            # Normal FlyDSL compile executes once; do not launch again here.
            compiled = flyc.compile(launcher, *current)
        else:
            compiled(*current)
        return compiled

    _register(case, label, run, reset, check, symbol, flydsl=True)


@torch.no_grad()
def make_case(buffer, device, *, tp_size=2, rows=68, prefix=30000):
    """Nine scopes; the three sparse attention paths share inputs and output addresses.

    Dense uses a separate one-request boundary case (67 written / 129 rows).
    A semantically empty second request forces canonical four-wave raw planning;
    packed planning uses the original single request. Neither path is gated.
    """
    if tp_size not in helpers.TP_SIZES or rows < 1 or prefix < 2051 or (rows + prefix) % 4:
        raise ValueError("Require TP2/4/8, positive rows, prefix >= 2051 and four-aligned KV length")
    if (rows + prefix) * 1024 > runtime.direct.MAX_PACKED_KV_BYTES:
        raise ValueError("Single-request KV exceeds the canonical packed scratch budget")
    device = torch.device(device)
    with torch.cuda.device(device):
        value = helpers._make_case((rows,), (prefix,), 24 // tp_size, device, seed=17 + buffer)
        workspace = runtime._Workspace(value.q, value.k, value.v, value.indices,
                                       value.query_lens, value.prefix_lens, value.scale)
        case = SimpleNamespace(device=device, runs={}, reset={}, checks={}, tensors={}, guards={},
                               flops={}, compiled={}, workspace=workspace, metadata=dict(
                                   tp_size=tp_size, local_heads=24 // tp_size, distributed_tp_run=False,
                                   source="synthetic; TP labels are local-head shapes, not service runs",
                                   rows=rows, prefix=prefix, seed=17 + buffer, kernels={},
                                   boundary="one kernel; setup/reset/reference/validation excluded",
                                   union="forced on: RHO=inf, PACKED_DIRECT=False",
                                   raw_direct="canonical four-wave plan with an additional empty request",
                                   packed_direct="ungated; pack excluded from packed_direct timing",
                                   attention_tolerance=dict(rtol=0.02, atol=0.02, reference="FP32, all written rows"),
                                   preparation="exact CPU plan; unwritten padding initialized to sentinel",
                                   guards="16 trailing elements at allocation-base output addresses",
                                   flop_definition="4*H*256*selected_tokens (QK+PV); excludes softmax/padding"))
        workspace.metadata["block_indices"] = _buffer(case, "recovered", (rows, 512), torch.int32)
        workspace.errors = _buffer(case, "errors", (rows,), torch.int32)
        workspace.valid = _buffer(case, "valid", (), torch.bool)
        inputs = workspace.bind(value.q, value.k, value.v, value.indices)
        plan = workspace.union
        assert plan is not None and not workspace.dense.calls
        workspace.direct = None  # Separate raw/packed plans below own their scratch.
        plan.max_union_inflation, plan.packed_direct = float("inf"), False
        fields = dict(dense="dense_membership", blocks="blocks", membership="membership",
                      counts="counts", active="active", order="task_order", masks="score_masks")
        for name, field in fields.items():
            original = getattr(plan, field)
            setattr(plan, field, _buffer(case, name, tuple(original.shape), original.dtype))
        expected = {name: torch.from_numpy(array.copy()).to(device)
                    for name, array in _plan_reference(value, plan).items()}
        seeds = dict(expected)
        options = _order_options(plan, value.k.shape[1])

        def fill(*names):
            for name in names:
                case.tensors[name].fill_(SENTINEL)

        def restore(*names):
            for name in names:
                case.tensors[name].copy_(seeds[name])

        def exact(*names):
            for name in names:
                actual = case.tensors[name]
                assert actual.shape == expected[name].shape and actual.dtype == expected[name].dtype, name
                assert torch.equal(actual, expected[name]), name

        def reset_recover():
            fill("recovered", "errors")
            plan.dense_membership.zero_()

        def recover():
            return runtime.prepare.attention_recover_scatter[(rows,)](
                inputs.indices, inputs.query_positions, inputs.kv_lens, inputs.query_sequence_ids,
                inputs.block_indices, workspace.errors, plan.dense_membership, plan.query_tiles,
                plan.metadata, plan.max_blocks, REVERSE=True, num_warps=1)

        def reset_compact():
            restore("dense")
            fill("blocks", "membership", "counts", "active")

        def compact():
            return runtime.prepare.attention_compact[(plan.num_tiles,)](
                plan.dense_membership, plan.blocks, plan.membership, plan.counts, plan.metadata, plan.active,
                plan.max_blocks, plan.block_capacity, min(1024, triton.next_power_of_2(plan.max_blocks)),
                plan.max_union_inflation, plan.packed_direct, CLEAR=True, REVERSE=True, num_warps=4)

        def check_compact():
            exact("blocks", "membership", "counts", "active")
            assert not bool(plan.dense_membership.any()), "compact must clear consumed scratch"

        def reset_order():
            restore("blocks", "membership", "counts", "active", "errors")
            fill("order", "masks")
            workspace.valid.fill_(False)

        def order():
            return runtime.prepare.attention_order_masks_validate[(options.ctas + plan.num_tiles * 4,)](
                workspace.errors, workspace.valid, plan.counts, plan.active, plan.task_order,
                rows, options.tasks, value.k.shape[1], options.slices, plan.grid, options.size,
                options.shift, options.wide, True, plan.blocks, plan.membership, plan.metadata,
                plan.score_masks, options.ctas, plan.query_tile, plan.block_capacity,
                options.warps * 64, 4, True, plan.num_tiles, num_warps=options.warps)

        def reset_scatter():
            restore("recovered")
            plan.dense_membership.zero_()

        def scatter():
            return runtime.prepare.attention_scatter_prepared[(plan.num_tiles, plan.query_tile)](
                inputs.block_indices, inputs.query_positions, plan.metadata, plan.dense_membership,
                plan.max_blocks, num_warps=4)

        for label, launch, reset, verify, symbol in (
            ("recover", recover, reset_recover, lambda: exact("recovered", "errors", "dense"),
             "attention_recover_scatter"),
            ("compact", compact, reset_compact, check_compact, "attention_compact"),
            ("order_masks_validate", order, reset_order, lambda: exact("order", "masks", "valid"),
             "attention_order_masks_validate"),
            ("scatter_prepared", scatter, reset_scatter, lambda: exact("dense", "recovered"),
             "attention_scatter_prepared"),
        ):
            _register(case, label, launch, reset, verify, symbol)

        output = _buffer(case, "output", tuple(value.q.shape), value.q.dtype)
        output_ref = helpers.reference(value, list(range(rows)))

        def reset_output():
            restore("recovered")
            output.fill_(math.nan)

        def check_output():
            torch.testing.assert_close(output.float(), output_ref, rtol=0.02, atol=0.02)

        dense_value = helpers._make_case((129,), (1984,), value.q.shape[1], device, seed=101 + buffer)
        dense_plan = runtime.dense.prepare(dense_value)
        assert len(dense_plan.calls) == 1, "dense.run would otherwise launch once per request"
        call = dense_plan.calls[0]
        dense_output = _buffer(case, "dense_output", tuple(dense_value.q.shape), dense_value.q.dtype)
        dense_ref = helpers.reference(dense_value, list(range(call.q_count)))

        def reset_dense():
            dense_output.fill_(SENTINEL)
            dense_output[:call.q_count].fill_(math.nan)

        def check_dense():
            torch.testing.assert_close(dense_output[:call.q_count].float(), dense_ref, rtol=0.02, atol=0.02)
            assert bool((dense_output[call.q_count:] == SENTINEL).all()), "noneligible dense rows overwritten"

        _flydsl(case, "dense", runtime.dense._bounded_launch,
                (dense_value.q[:call.q_count].view(-1), dense_value.k[:call.kv_count].view(-1),
                 dense_value.v[:call.kv_count].view(-1), dense_output[:call.q_count].view(-1),
                 call.cu_q, call.cu_k, value.q.shape[1], 1, call.kv_count, call.q_count,
                 float(value.scale), dense_plan.num_cus), reset_dense, check_dense,
                "attention_dense_bf16_d256_bounded")

        def reset_union():
            restore("blocks", "membership", "counts", "active", "order", "masks")
            reset_output()

        _flydsl(case, "union", runtime.union._launch,
                (value.q.view(-1), value.k.view(-1), value.v.view(-1), output.view(-1),
                 plan.metadata.view(-1), plan.blocks.view(-1), plan.score_masks.view(-1),
                 plan.counts.view(-1), plan.active, plan.task_order, value.q.shape[1], 1, rows,
                 value.k.shape[0], plan.block_capacity, plan.query_tile, plan.group_padded,
                 options.tasks, plan.grid, value.scale, False), reset_union, check_output,
                "attention_union_bf16_d256")

        raw_inputs = SimpleNamespace(**vars(inputs))
        raw_inputs.query_lens += (0,)
        raw_inputs.prefix_lens += (0,)
        raw = runtime.direct.prepare(inputs=raw_inputs)
        assert raw.packed_key is None and raw.query_tile == 4 and not raw.gated
        _flydsl(case, "raw_direct", runtime.direct._launch,
                (value.q.view(-1), value.k.view(-1), value.v.view(-1), output.view(-1),
                 raw.source_blocks.view(-1), raw.metadata.view(-1), raw.active, raw.query_tiles,
                 value.q.shape[1], 1, rows, value.k.shape[0], 32, raw.query_tile * 64,
                 raw.num_tiles, raw.gated, value.scale), reset_output, check_output,
                "attention_direct_bf16_d256")

        direct = runtime.direct.prepare(inputs=inputs)
        assert direct.packed_key is not None and direct.query_tile == 1 and not direct.gated
        direct = msgspec.structs.replace(
            direct, packed_key=_buffer(case, "packed_key", tuple(value.k.shape), value.k.dtype),
            packed_value=_buffer(case, "packed_value", tuple(value.v.shape), value.v.dtype))
        pk_ref, pv_ref = _packed_reference(value.k, value.v)

        def reset_pack():
            direct.packed_key.fill_(math.nan)
            direct.packed_value.fill_(math.nan)

        def check_pack():
            for actual, wanted in ((direct.packed_key, pk_ref), (direct.packed_value, pv_ref)):
                assert torch.equal(actual.view(torch.uint8), wanted.view(torch.uint8)), "packed KV byte layout"

        _flydsl(case, "pack", _pack_only,
                (value.k.view(-1), value.v.view(-1), direct.packed_key.view(-1), direct.packed_value.view(-1),
                 value.k.shape[0], 1), reset_pack, check_pack, "attention_pack_kv_bf16_d256")

        def reset_packed():
            restore("packed_key", "packed_value")
            reset_output()

        _flydsl(case, "packed_direct", _packed_only,
                (value.q.view(-1), direct.packed_key.view(-1), direct.packed_value.view(-1), output.view(-1),
                 direct.source_blocks.view(-1), direct.metadata.view(-1), direct.active, direct.query_tiles,
                 value.q.shape[1], 1, rows, value.k.shape[0], direct.num_tiles, direct.gated, value.scale),
                reset_packed, check_output, "attention_direct_bf16_d256")

        # Bootstrap and freeze actual, independently checked kernel outputs as
        # later scopes' inputs. No check overwrites a timed output with a reference.
        for label, names in (("recover", ("recovered", "errors", "dense")),
                             ("compact", ("blocks", "membership", "counts", "active")),
                             ("order_masks_validate", ("order", "masks")),
                             ("pack", ("packed_key", "packed_value"))):
            case.reset[label]()
            case.runs[label]()
            case.checks[label]()
            seeds.update({name: case.tensors[name].clone() for name in names})

        for name in ("q", "k", "v", "indices"):
            case.tensors[name] = getattr(value, name)
            case.tensors[f"dense_{name}"] = getattr(dense_value, name)
        case.tensors.update(positions=inputs.query_positions, lengths=inputs.kv_lens,
                            sequence_ids=inputs.query_sequence_ids, metadata=plan.metadata,
                            query_tiles=plan.query_tiles, raw_metadata=raw.metadata, packed_metadata=direct.metadata,
                            dense_cu_q=call.cu_q, dense_cu_k=call.cu_k, output_reference=output_ref,
                            dense_reference=dense_ref, packed_key_reference=pk_ref, packed_value_reference=pv_ref)
        case.tensors.update({f"seed_{name}": tensor for name, tensor in seeds.items()})
        sparse_flops = 4 * value.q.shape[1] * 256 * int((value.indices >= 0).sum().item())
        case.flops.update({name: sparse_flops for name in ("union", "raw_direct", "packed_direct")})
        case.flops["dense"] = 4 * value.q.shape[1] * 256 * sum(1984 + row + 1 for row in range(call.q_count))
        case.metadata.update(scopes=list(case.runs), q_shape=list(value.q.shape), kv_shape=list(value.k.shape),
                             dense=dict(rows=129, prefix=1984, written_rows=call.q_count),
                             query_tile=plan.query_tile, union_tiles=plan.num_tiles,
                             pack_bytes_read_write=4 * value.k.numel() * value.k.element_size())
        assert tuple(case.runs) == SCOPES
        return case