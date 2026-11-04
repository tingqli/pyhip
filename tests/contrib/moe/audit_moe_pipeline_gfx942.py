"""Audit of the gfx942 MoE GateUp / Down LDS pipeline (layouts 2x4_pf1, 4x2_pf1) on the final ISA.

Checks, per kernel, on the instruction stream LLVM actually emitted:
  - barrier pairing: every wave executes the same number of s_barrier; the group-conditional barriers that
    make waves 4-7 trail waves 0-3 by one barrier (prologue end) and re-pair them (loop exit);
  - LDS slot ordering across the two wave groups: every K-loop read of an A/B packet finds the version it
    expects from both groups complete in an earlier barrier epoch (RAW), and the next version of that packet
    is issued only in an epoch after the read completed (WAR); K-loop LDS ops all complete before the
    epilogue's first LDS access;
  - vmcnt / lgkmcnt coverage, re-simulated in hardware order (VMEM in issue order, DS in order, SMEM out of
    order): no instruction reads or overwrites a VGPR with a load still in flight;
  - VGPR staging lifetimes: global load -> wait -> LDS write, and LDS write -> LDS read leads in epochs.
An epoch is the interval between two consecutive s_barrier instances of the workgroup. Each kernel also gets
a swimlane chart of wave group 0 (prolog + K blocks 0 and 1, waits, barriers and the cross-group LDS
dependency each barrier enforces) on a nominal issue-cycle axis.

usage: python audit_moe_pipeline_gfx942.py --dump DIR                 (compile with IR dumps; needs a gfx942 GPU)
       python audit_moe_pipeline_gfx942.py --isa DIR [--plot OUTDIR]   (analyze DIR/<config>/<kernel>/22_final_isa.s)
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

# Shapes of the MiMo EP8 rank (tokens do not affect code generation).
T, I, H, TOPK, E = 1024, 2048, 4096, 8, 32
CONFIGS = {
    "gateup_2x4_pf1": dict(kernel="gateup", layout="2x4_pf1", kb=H // 128),
    "gateup_4x2_pf1": dict(kernel="gateup", layout="4x2_pf1", kb=H // 128),
    "down_2x4_pf1": dict(kernel="down", layout="2x4_pf1", kb=I // 128),
    "down_4x2_pf1": dict(kernel="down", layout="4x2_pf1", kb=I // 128),
}
PACKET_BYTES, B_BASE_BYTES = 8192, 32768
EPILOGUE = 10**6


@dataclass(eq=False)
class Instr:
    line: int
    mnem: str
    ops: list
    mods: dict
    kind: str = ""
    dst: frozenset = frozenset()
    src: frozenset = frozenset()
    vm: int = None
    lgkm: int = None

    def slot(self):
        """(tile, packet) of a K-loop A/B LDS access: the packet sits in the immediate offset."""
        offset = int(self.mods.get("offset", 0))
        return ("B" if offset >= B_BASE_BYTES else "A", offset % B_BASE_BYTES // PACKET_BYTES)


@dataclass(eq=False)
class Event:
    group: int
    instr: Instr
    epoch: int
    part: int  # -1 prologue, i: loop iteration i, EPILOGUE
    segment: int = -1  # barrier interval index inside the loop body
    complete: int = None  # epoch in which a wait retired this memory op
    retired: int = 0  # for waits: in-flight ops this wait retired
    retired_by: object = None  # the wait event that retired this memory op


def split_operands(text):
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += ch in "[("
        depth -= ch in "])"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur] if cur.strip() else parts


REG = re.compile(r"^([vsa])(?:(\d+)|\[(\d+):(\d+)\])$")


def vregs(token):
    m = REG.match(token)
    if not m or m.group(1) == "s":
        return set()
    if m.group(2) is not None:
        return {(m.group(1), int(m.group(2)))}
    return {(m.group(1), r) for r in range(int(m.group(3)), int(m.group(4)) + 1)}


def classify(m):
    if m.startswith(("buffer_load", "global_load", "flat_load", "scratch_load")):
        return "vmem_load"
    if m.startswith(("buffer_store", "global_store", "flat_store", "scratch_store", "global_atomic", "buffer_atomic", "flat_atomic")):
        return "vmem_store"
    if m.startswith(("ds_read", "ds_load")):
        return "ds_read"
    if m.startswith(("ds_write", "ds_store")):
        return "ds_write"
    if m.startswith("ds_"):
        return "ds_other"
    if m.startswith(("s_load", "s_buffer_load")):
        return "smem"
    if m == "s_waitcnt":
        return "waitcnt"
    if m == "s_barrier":
        return "barrier"
    if m.startswith("v_mfma"):
        return "mfma"
    if m.startswith("v_"):
        return "valu"
    if m.startswith(("s_cbranch", "s_branch")):
        return "branch"
    return "salu"


def make_instr(line, text):
    mnem, _, rest = text.partition(" ")
    ins = Instr(line, mnem, [], {})
    ins.kind = classify(mnem)
    if ins.kind == "waitcnt":
        for name, value in re.findall(r"(vmcnt|lgkmcnt|expcnt)\((\d+)\)", rest):
            if name == "vmcnt":
                ins.vm = int(value)
            elif name == "lgkmcnt":
                ins.lgkm = int(value)
        return ins
    for piece in split_operands(rest.strip()):
        words = piece.split()
        if not words:
            continue
        ins.ops.append(words[0])
        for word in words[1:]:
            key, _, value = word.partition(":")
            ins.mods[key] = value
    regs = [vregs(op) for op in ins.ops]
    if ins.kind in ("ds_write", "vmem_store"):
        ins.src = frozenset().union(*regs) if regs else frozenset()
    elif ins.kind in ("vmem_load", "ds_read", "ds_other", "mfma", "valu") and regs:
        ins.dst = frozenset(regs[0])
        ins.src = frozenset().union(*regs[1:]) if len(regs) > 1 else frozenset()
    return ins


def parse_isa(path):
    """Instructions of the kernel body and the positions of its .LBB labels."""
    instrs, labels, started = [], {}, False
    for number, raw in enumerate(open(path), 1):
        text = raw.split(";")[0].strip()
        if not text:
            continue
        if re.match(r"^[.\w$]+:$", text):
            name = text[:-1]
            if not name.startswith("."):
                started = True
            elif started:
                labels[name] = len(instrs)
            continue
        if text.startswith(".") or not started:
            continue
        instrs.append(make_instr(number, text))
        if instrs[-1].mnem == "s_endpgm":
            break
    return instrs, labels


def find_k_loop(instrs, labels):
    """(header, back-edge) of the backward branch whose body holds the most MFMAs."""
    best = None
    for i, ins in enumerate(instrs):
        if ins.kind == "branch" and ins.ops and ins.ops[0] in labels and labels[ins.ops[0]] <= i:
            head = labels[ins.ops[0]]
            mfmas = sum(x.kind == "mfma" for x in instrs[head:i])
            if best is None or mfmas > best[2]:
                best = (head, i, mfmas)
    assert best and best[2] > 0, "no K loop found"
    return best[0], best[1]


def barrier_conditions(instrs, labels):
    """Barrier index -> wave group that executes it, for `s_cmp_lg_u32 sN, g; s_cbranch_scc1 L; s_barrier; L:`."""
    conds = {}
    for i, ins in enumerate(instrs):
        if ins.kind != "barrier" or i < 2:
            continue
        br, cmp = instrs[i - 1], instrs[i - 2]
        if br.mnem == "s_cbranch_scc1" and labels.get(br.ops[0]) == i + 1 and cmp.mnem == "s_cmp_lg_u32":
            conds[i] = int(cmp.ops[1], 0)
    return conds


def simulate(instrs, head, back, iterations, conds, group):
    """Straight-line run of one wave of `group`: events, epochs, wait retirement and VGPR hazards."""
    parts = [(-1, range(0, head))] + [(it, range(head, back + 1)) for it in range(iterations)] + [(EPILOGUE, range(back + 1, len(instrs)))]
    events, vm, lgkm, hazards, epoch = [], [], [], [], 0

    def retire(queue, keep, wait, ordered=True):
        if not ordered and keep > 0:
            return 0
        count = 0
        while len(queue) > keep:
            done = queue.pop(0)
            done.complete, done.retired_by = epoch, wait
            count += 1
        return count

    for part, indices in parts:
        segment = 0
        for i in indices:
            ins = instrs[i]
            if ins.kind == "barrier":
                if conds.get(i, group) != group:
                    continue
                events.append(Event(group, ins, epoch, part, segment))
                epoch += 1
                segment += 1
                continue
            ev = Event(group, ins, epoch, part, segment)
            events.append(ev)
            if ins.kind == "waitcnt":
                if ins.vm is not None:
                    ev.retired += retire(vm, ins.vm, ev)
                if ins.lgkm is not None:
                    ev.retired += retire(lgkm, ins.lgkm, ev, ordered=not any(e.instr.kind == "smem" for e in lgkm))
                continue
            pending = {}
            for queue in (vm, lgkm):
                for e in queue:
                    for r in e.instr.dst:
                        pending[r] = e
            for r in ins.src:
                if r in pending:
                    hazards.append(("RAW", ev, pending[r]))
            for r in ins.dst:
                other = pending.get(r)
                # Loads on one counter return in order, so a younger load may target an in-flight load's VGPRs.
                in_order = other is not None and ((ins.kind == "vmem_load" and other in vm) or (ins.kind in ("ds_read", "ds_other") and other in lgkm))
                if other is not None and not in_order:
                    hazards.append(("WAW", ev, other))
            if ins.kind in ("vmem_load", "vmem_store"):
                vm.append(ev)
            elif ins.kind in ("ds_read", "ds_write", "ds_other", "smem"):
                lgkm.append(ev)
    return events, hazards


@dataclass
class Report:
    config: str
    kernel: str
    lines: list = field(default_factory=list)
    failures: list = field(default_factory=list)

    def out(self, text=""):
        self.lines.append(text)

    def fail(self, text):
        self.failures.append(text)
        self.lines.append("  FAIL " + text)


def lds_events(events, part_filter):
    return [e for e in events if e.instr.kind in ("ds_read", "ds_write") and part_filter(e.part)]


def check_lds(rep, groups):
    """RAW/WAR of every K-loop A/B packet read against both groups' writes (versions = write epochs)."""
    in_loop = lambda p: p != EPILOGUE  # noqa: E731
    writes = {}  # (group, slot) -> [first write event of each version]
    for g, events in groups.items():
        last = {}
        for e in lds_events(events, in_loop):
            if e.instr.kind == "ds_write":
                slot = e.instr.slot()
                versions = writes.setdefault((g, slot), [])
                if last.get(slot) != e.epoch:
                    versions.append([e])
                else:
                    versions[-1].append(e)
                last[slot] = e.epoch
    checked, raw_slack, war_slack = 0, [], []
    for g, events in groups.items():
        seen = {}
        for e in lds_events(events, in_loop):
            slot = e.instr.slot()
            if e.instr.kind == "ds_write":
                if seen.get(("w", slot)) != e.epoch:
                    seen[("v", slot)] = seen.get(("v", slot), 0) + 1
                seen[("w", slot)] = e.epoch
                continue
            version = seen.get(("v", slot), 0)
            checked += 1
            if version == 0:
                rep.fail(f"G{g} reads {slot} at line {e.instr.line} before any write")
                continue
            for other in groups:
                versions = writes.get((other, slot), [])
                if len(versions) < version:
                    rep.fail(f"G{g} line {e.instr.line} reads {slot} v{version}; G{other} wrote only {len(versions)}")
                    continue
                done = max((w.complete if w.complete is not None else 10**9) for w in versions[version - 1])
                raw_slack.append(e.epoch - done - 1)
                if done >= e.epoch:
                    rep.fail(f"RAW {slot}: G{g} line {e.instr.line} epoch {e.epoch} reads v{version}, G{other} write completes in epoch {done}")
                if len(versions) > version:
                    nxt = min(w.epoch for w in versions[version])
                    if e.complete is None or nxt <= e.complete:
                        rep.fail(f"WAR {slot}: G{g} line {e.instr.line} read completes in epoch {e.complete}, G{other} writes v{version + 1} in epoch {nxt}")
                    else:
                        war_slack.append(nxt - e.complete - 1)
    rep.out(f"LDS A/B packets: {checked} reads checked against both groups' writes; "
            f"RAW slack min {min(raw_slack)} epochs, WAR slack min {min(war_slack) if war_slack else '-'} epochs")
    # Epilogue LDS use (reuse of the A/B tiles) after every K-loop LDS op of both groups.
    loop_done = max((e.complete if e.complete is not None else 10**9) for events in groups.values() for e in lds_events(events, in_loop))
    epi = [e for events in groups.values() for e in lds_events(events, lambda p: p == EPILOGUE)]
    if epi:
        first = min(e.epoch for e in epi)
        rep.out(f"epilogue LDS: first access in epoch {first}, all K-loop LDS ops complete by epoch {loop_done}")
        if first <= loop_done:
            rep.fail(f"epilogue LDS access in epoch {first} overlaps K-loop LDS ops completing in epoch {loop_done}")
    else:
        rep.out("epilogue LDS: none")


def body_segments(events, iteration):
    segs = {}
    for e in events:
        if e.part == iteration:
            segs.setdefault(e.segment, []).append(e)
    return segs


def load_label(ev, consumer):
    if ev.instr.mnem.endswith("dwordx4") and consumer is not None:
        tile, packet = consumer.instr.slot()
        return f"{tile}{packet}"
    return "sA"


def consumers(events):
    """Each dwordx4 load event -> the ds_write event storing its VGPRs (same wave)."""
    result, open_loads = {}, []
    for e in events:
        if e.instr.kind == "vmem_load" and e.instr.mnem.endswith("dwordx4"):
            open_loads.append(e)
        elif e.instr.kind == "ds_write":
            for load in open_loads:
                if load.instr.dst and load.instr.dst <= e.instr.src:
                    result[id(load)] = e
                    open_loads.remove(load)
                    break
    return result


def stage_table(rep, events, cons):
    """Per loop-body barrier interval of one steady-state iteration (wave group 0)."""
    segs = body_segments(events, 1)
    rep.out(" seg | phase       | ds_write      | ds_read            | buffer_load (next K block)  | vm waits (retired) | lgkm waits | MFMA | VALU")
    for s in sorted(segs):
        evs = [e for e in segs[s] if e.instr.kind != "barrier"]
        # Interval 16 is the loop-counter tail, which shares its epoch with the next iteration's interval 0.
        if not evs or s >= 16:
            continue
        parity, stage, phase = s // 8, s % 8 // 2, "mem" if s % 2 == 0 else "mfma"
        w = [f"{t}{p}" for t, p in (e.instr.slot() for e in evs if e.instr.kind == "ds_write")]
        r = []
        for e in evs:
            if e.instr.kind == "ds_read":
                label = "{}{}".format(*e.instr.slot())
                if not r or r[-1][0] != label:
                    r.append([label, 0])
                r[-1][1] += 1
        loads = [load_label(e, cons.get(id(e))) for e in evs if e.instr.kind == "vmem_load"]
        vmw = [f"{e.instr.vm}({e.retired})" for e in evs if e.instr.kind == "waitcnt" and e.instr.vm is not None]
        lgw = [str(e.instr.lgkm) for e in evs if e.instr.kind == "waitcnt" and e.instr.lgkm is not None]
        mfma = sum(e.instr.kind == "mfma" for e in evs)
        valu = sum(e.instr.kind == "valu" for e in evs)
        rep.out(f" {s:3d} | b{parity} s{stage} {phase:4s} | {' '.join(w):13s} | {' '.join(f'{a}x{n}' for a, n in r):18s} | "
                f"{' '.join(loads):27s} | {' '.join(vmw):18s} | {' '.join(lgw):10s} | {mfma:4d} | {valu}")


def lifetimes(rep, events, cons):
    """Steady-state leads, in epochs, of the A/B staging pipeline (wave group 0, iteration 1)."""
    loads = [e for e in events if e.part == 1 and e.instr.kind == "vmem_load" and id(e) in cons]
    lead_wait = [e.complete - e.epoch for e in loads if e.complete is not None]
    lead_write = [cons[id(e)].epoch - e.epoch for e in loads]
    rep.out(f"VGPR staging (A/B dwordx4 loads, steady state): load->wait {min(lead_wait)}..{max(lead_wait)} epochs, "
            f"load->LDS write {min(lead_write)}..{max(lead_write)} epochs, {len(loads)} loads per iteration")
    scale = [e for e in events if e.part == 1 and e.instr.kind == "vmem_load" and id(e) not in cons]
    if scale:
        leads = [e.complete - e.epoch for e in scale if e.complete is not None]
        rep.out(f"scale_a loads: {len(scale)} per iteration, load->wait {min(leads)}..{max(leads)} epochs")
    # LDS write -> first read of that version by this group.
    first_read, lead = {}, []
    for e in events:
        if e.part == EPILOGUE or e.instr.kind not in ("ds_read", "ds_write"):
            continue
        slot = e.instr.slot()
        if e.instr.kind == "ds_write":
            first_read[slot] = e
        elif slot in first_read and first_read[slot] is not None:
            if first_read[slot].part == 1:
                lead.append(e.epoch - first_read[slot].epoch)
            first_read[slot] = None
    rep.out(f"LDS write -> first read (same group): {min(lead)}..{max(lead)} epochs")


def analyze(config, isa_path, plot_dir=None, mutate=None):
    cfg = CONFIGS[config]
    instrs, labels = parse_isa(isa_path)
    head, back = find_k_loop(instrs, labels)
    conds = barrier_conditions(instrs, labels)
    if mutate:
        instrs, labels, conds = mutate(instrs, labels, conds, head, back)
        head, back = find_k_loop(instrs, labels)
    rep = Report(config, os.path.basename(os.path.dirname(isa_path)))
    rep.out(f"== {config}: {cfg['kernel']} layout={cfg['layout']} KB={cfg['kb']} ({rep.kernel}, {isa_path})")
    body = instrs[head:back + 1]
    rep.out(f"K loop: ISA lines {instrs[head].line}-{instrs[back].line}, {sum(x.kind == 'barrier' for x in body)} barriers, "
            f"{sum(x.kind == 'mfma' for x in body)} MFMA, {sum(x.kind == 'ds_read' for x in body)} ds_read, "
            f"{sum(x.kind == 'ds_write' for x in body)} ds_write, {sum(x.kind == 'vmem_load' for x in body)} buffer_load per iteration (2 K blocks)")
    rep.out("group-conditional barriers: " + ", ".join(f"line {instrs[i].line} (G{g}, {'prologue' if i < head else 'loop exit'})" for i, g in sorted(conds.items())))
    groups, all_hazards = {}, []
    for g in (0, 1):
        events, hazards = simulate(instrs, head, back, cfg["kb"] // 2, conds, g)
        groups[g] = events
        all_hazards += hazards
    counts = {g: sum(e.instr.kind == "barrier" for e in ev) for g, ev in groups.items()}
    rep.out(f"s_barrier executed per wave: G0 {counts[0]}, G1 {counts[1]}")
    if counts[0] != counts[1]:
        rep.fail("wave groups execute different barrier counts (deadlock)")
    first = {g: next(e.epoch for e in ev if e.part == 0) for g, ev in groups.items()}
    rep.out(f"K loop entry epoch: G0 {first[0]}, G1 {first[1]} (G1 trails by {first[1] - first[0]} barrier)")
    if first[1] - first[0] != 1:
        rep.fail("expected waves 4-7 to trail waves 0-3 by one barrier")
    n_instr = sum(len(ev) for ev in groups.values())
    rep.out(f"vmcnt/lgkmcnt re-simulation: {n_instr} instructions over both groups, {len(all_hazards)} VGPR hazards")
    for kind, ev, src in all_hazards[:10]:
        rep.fail(f"{kind}: line {ev.instr.line} {ev.instr.mnem} uses a VGPR still loading from line {src.instr.line} {src.instr.mnem}")
    check_lds(rep, groups)
    cons = consumers(groups[0])
    lifetimes(rep, groups[0], cons)
    rep.out("steady-state iteration, wave group 0 (G1 runs each interval one barrier later):")
    stage_table(rep, groups[0], cons)
    rep.out("RESULT: " + ("PASS" if not rep.failures else f"FAIL ({len(rep.failures)})"))
    if plot_dir:
        plot_swimlane(config, cfg, groups, consumers, os.path.join(plot_dir, f"moe_{config}_swimlane.png"))
    return rep


def _drop(instrs, labels, conds, index):
    """Remove instruction `index`, shifting labels and conditional-barrier indices."""
    instrs = instrs[:index] + instrs[index + 1:]
    labels = {k: v - (v > index) for k, v in labels.items()}
    conds = {k - (k > index): g for k, g in conds.items() if k != index}
    return instrs, labels, conds


def _lag_g1(instrs, labels, conds, head, back):
    """Waves 4-7 trail by two barriers: duplicate both group-conditional barriers."""
    out, new_labels, new_conds, shift = [], {}, {}, 0
    inverse = {}
    for name, pos in labels.items():
        inverse.setdefault(pos, []).append(name)
    for i, ins in enumerate(instrs):
        for name in inverse.get(i, []):
            new_labels[name] = i + shift
        out.append(ins)
        if i in conds:
            new_conds[i + shift] = conds[i]
            out.append(ins)
            shift += 1
            new_conds[i + shift] = conds[i]
    for name, pos in labels.items():
        new_labels.setdefault(name, pos + shift)
    return out, new_labels, new_conds


def _no_vm_wait(instrs, labels, conds, head, back):
    """Drop the loop's first vmcnt wait that guards an LDS write of staged data."""
    for i in range(head, back):
        if instrs[i].kind == "waitcnt" and instrs[i].vm is not None and instrs[i + 1].kind == "ds_write" and instrs[i].vm > 0:
            return _drop(instrs, labels, conds, i)
    raise AssertionError("no vmcnt wait before an LDS write")


def _no_lgkm_wait(instrs, labels, conds, head, back):
    """Drop the loop's first lgkmcnt(0) in front of a barrier (LDS ops may cross the barrier)."""
    for i in range(head, back):
        if instrs[i].kind == "waitcnt" and instrs[i].lgkm == 0 and instrs[i + 1].kind == "barrier":
            return _drop(instrs, labels, conds, i)
    raise AssertionError("no lgkmcnt(0) before a barrier")


def _early_lds_write(instrs, labels, conds, head, back):
    """Hoist stage 1's first LDS write (and its vmcnt wait) above the barrier that ends stage 0."""
    barriers = [i for i in range(head, back) if instrs[i].kind == "barrier"]
    end_stage0 = barriers[1]
    write = next(i for i in range(end_stage0, back) if instrs[i].kind == "ds_write")
    wait = write - 1 if instrs[write - 1].kind == "waitcnt" else write
    moved = instrs[wait:write + 1]
    rest = instrs[:wait] + instrs[write + 1:]
    out = rest[:end_stage0] + moved + rest[end_stage0:]
    assert len(out) == len(instrs) and all(labels.get(k, 0) <= end_stage0 or labels[k] > write for k in labels)
    return out, labels, conds


MUTATIONS = {
    "waves 4-7 trail by 2 barriers": _lag_g1,
    "drop a vmcnt wait": _no_vm_wait,
    "drop an lgkmcnt(0) before s_barrier": _no_lgkm_wait,
    "LDS write one barrier early": _early_lds_write,
}


def self_test(isa_dir):
    """Each mutation of a real kernel must make the audit fail; the unmutated kernels must pass."""
    ok = True
    for config, cfg in CONFIGS.items():
        isa = os.path.join(isa_dir, config, f"{cfg['kernel']}_kernel_0", "22_final_isa.s")
        for name, mutate in MUTATIONS.items():
            rep = analyze(config, isa, mutate=mutate)
            ok &= bool(rep.failures)
            kinds = {}
            for failure in rep.failures:
                key = failure.split(":")[0].split(" ")[0]
                kinds[key] = kinds.get(key, 0) + 1
            print(f"self-test {config:16s} {name:38s}: {'detected' if rep.failures else 'MISSED'} {kinds}")
    return ok


TILE_COLOR = {"A": "#1f77b4", "B": "#ff7f0e"}
LOAD_COLOR = {"sA": "#2ca02c", "sB": "#9467bd", "ids": "#8c564b", "meta": "#7f7f7f", "kernarg": "#c7c7c7"}
PHASE_COLOR = {"prolog": "#e8eef5", "M": "#e3f2e3", "C": "#fde9e9", "wait": "#eeeeee"}
SYNC_COLOR = {"WAR": "#c0392b", "RAW": "#1e8449"}
VALU_COLOR = "#a8a8a8"


def _cost(ins):
    if ins.mnem == "s_nop":
        return 1 + int(ins.ops[0], 0)
    return 1 if ins.kind in ("salu", "branch", "waitcnt", "barrier") else 4


def timeline(groups, epochs):
    """Nominal issue cycles of both groups' events in epochs [0, epochs) and the barrier release times.

    Each SIMD runs one wave of each group, so the XDL pipe (16 cycles per 16x16x32 FP8 MFMA) and the VALU pipe
    (4 cycles) are shared; other instructions take 4 cycles (scalar 1). Memory latency is not modeled.
    """
    times, release, xdl, valu = {}, [0.0], 0.0, 0.0
    for epoch in range(epochs):
        queues = {g: [e for e in ev if e.epoch == epoch] for g, ev in groups.items()}
        t, idx = {g: release[-1] for g in groups}, {g: 0 for g in groups}
        while any(idx[g] < len(queues[g]) for g in groups):
            g = min((k for k in groups if idx[k] < len(queues[k])), key=lambda k: t[k])
            e = queues[g][idx[g]]
            idx[g] += 1
            if e.instr.kind == "mfma":
                start = max(t[g], xdl)
                xdl, t[g] = start + 16, start + 4
                times[id(e)] = (start, start + 16)
            elif e.instr.kind == "valu":
                start = max(t[g], valu)
                valu = t[g] = start + 4
                times[id(e)] = (start, start + 4)
            else:
                times[id(e)] = (t[g], t[g] + _cost(e.instr))
                t[g] += _cost(e.instr)
        release.append(max(t.values()))
    return times, release


def first_uses(events):
    """Load event id -> first event reading one of its VGPRs."""
    result, writer = {}, {}
    for e in events:
        for r in e.instr.src:
            w = writer.get(r)
            if w is not None and id(w) not in result:
                result[id(w)] = e
        for r in e.instr.dst:
            writer[r] = e
    return result


def load_class(e, uses, cons):
    if e.instr.kind == "smem":
        return "kernarg"
    if id(e) in cons:
        return cons[id(e)].instr.slot()[0]
    mnem = uses[id(e)].instr.mnem if id(e) in uses else ""
    if mnem.startswith("v_readfirstlane"):
        return "meta"
    if mnem.startswith("v_readlane"):
        return "sB"
    if mnem.startswith(("v_mul_f32", "v_fma", "v_fmac", "v_pk_mul")):
        return "sA"
    return "ids"


def computed_block(e):
    return 2 * e.part + e.segment // 8 if 0 <= e.part < EPILOGUE else None


def phase_of(events):
    """Header of one group's barrier interval: prolog / k<b>·s<s> M|C / wait."""
    loop = [e for e in events if 0 <= e.part < EPILOGUE and e.segment < 16]
    if loop:
        e = max(loop, key=lambda x: (x.part, x.segment))
        return ("M" if e.segment % 2 == 0 else "C"), f"k{computed_block(e)}·s{e.segment % 8 // 2} {'M' if e.segment % 2 == 0 else 'C'}"
    if any(e.instr.kind not in ("salu", "branch", "barrier") for e in events):
        return "prolog", "prolog"
    return "wait", "extra s_barrier"


def pack(intervals, gap):
    """Greedy row assignment so that bars in one row are at least `gap` cycles apart."""
    ends, rows = [], []
    for start, end in intervals:
        for r, last in enumerate(ends):
            if last + gap <= start:
                ends[r] = end
                rows.append(r)
                break
        else:
            ends.append(end)
            rows.append(len(ends) - 1)
    return rows, max(len(ends), 1)


def binding_pairs(groups):
    """Cross-group LDS dependencies with no spare barrier: (kind, slot, from event, to event)."""
    seq = {}  # (group, slot) -> [[writes], [reads]] per version
    for g, events in groups.items():
        last_write = {}
        for e in lds_events(events, lambda p: p != EPILOGUE):
            slot = e.instr.slot()
            versions = seq.setdefault((g, slot), [])
            if e.instr.kind == "ds_write":
                if last_write.get(slot) != e.epoch:
                    versions.append([[], []])
                last_write[slot] = e.epoch
                versions[-1][0].append(e)
            elif versions:
                versions[-1][1].append(e)
    pairs = []
    for (g, slot), versions in seq.items():
        for h in groups:
            if h == g:
                continue
            other = seq.get((h, slot), [])
            for v, (writes, _) in enumerate(versions):
                if v < len(other) and other[v][1] and all(w.complete is not None for w in writes):
                    done = max(writes, key=lambda w: w.complete)
                    read = other[v][1][0]
                    if read.epoch - done.complete - 1 == 0:
                        pairs.append(("RAW", slot, done, read))
                if v + 1 < len(other) and versions[v][1]:
                    read = max((r for r in versions[v][1] if r.complete is not None), key=lambda r: r.complete, default=None)
                    nxt = other[v + 1][0][0]
                    if read is not None and nxt.epoch - read.complete - 1 == 0:
                        pairs.append(("WAR", slot, read, nxt))
    return pairs


def plot_swimlane(config, cfg, groups, consumer_fn, path):
    """Wave group 0 (waves 0-3), prolog + K blocks 0 and 1, on the nominal-cycle axis of timeline(); one panel per K block."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    # G1 stays in the timeline: it shares the SIMD pipes and sets the barrier release times.
    epochs = next(e.epoch for e in groups[0] if e.part == 1)
    view = {g: [e for e in ev if e.epoch < epochs] for g, ev in groups.items()}
    times, release = timeline(view, epochs)
    events, span = view[0], release[-1]
    split = next(e.epoch for e in events if e.part == 0 and e.segment == 8)
    panels = [("prolog + kiter 0 (K block 0)", 0.0, release[split]), ("kiter 1 (K block 1)", release[split], span)]
    gap = release[split] / 160
    cons, uses = consumer_fn(groups[0]), first_uses(groups[0])
    retired = {}
    for e in groups[0]:
        if e.retired_by is not None:
            retired.setdefault(id(e.retired_by), set()).add("vm" if e.instr.kind.startswith("vmem") else "lgkm")

    def issue(e):
        return times[id(e)][0]

    def retire(e):
        return issue(e.retired_by) if e.retired_by is not None and id(e.retired_by) in times else span

    def merge(items):
        """(key, event) pairs -> bars (key, first issue, last retire, events) in issue order."""
        bars = {}
        for key, e in items:
            bars.setdefault(key, []).append(e)
        return sorted(((k, min(map(issue, evs)), max(map(retire, evs)), evs) for k, evs in bars.items()), key=lambda b: b[1])

    def packets(slots):
        slots, text = list(slots), []
        for tile in "AB":
            ps = sorted({p for t, p in slots if t == tile})
            if ps:
                text.append(f"{tile}{ps[0]}-{ps[-1]}" if len(ps) > 1 and ps[-1] - ps[0] == len(ps) - 1 else tile + ",".join(map(str, ps)))
        return " ".join(text)

    def sa_label(e):
        k = computed_block(e)
        if not cfg["layout"].startswith("2x4"):
            return "sA k0" if k is None else f"sA k{k + 1}"
        stage = e.segment % 8 // 2
        return "sA k0 p0" if k is None else (f"sA k{k} p{stage + 1}" if stage < 3 else f"sA k{k + 1} p0")

    def runs(kind, breaks, max_gap):
        out, cur = [], []
        for e in events:
            if e.instr.kind == kind:
                if cur and issue(e) - times[id(cur[-1])][1] > max_gap:
                    out.append(cur)
                    cur = []
                cur.append(e)
            elif cur and e.instr.kind in breaks:
                out.append(cur)
                cur = []
        return out + [cur] if cur else out

    loads, scalars = {"A": [], "B": []}, []
    for e in events:
        if e.instr.kind in ("vmem_load", "smem"):
            cls = load_class(e, uses, cons)
            if cls in TILE_COLOR:
                loads[cls].append(((e.epoch, 0 if e.part < 0 else min(computed_block(e) + 1, cfg["kb"] - 1)), e))
            else:
                label = sa_label(e) if cls == "sA" else (f"sB k0-{cfg['kb'] - 1}" if cls == "sB" else cls)
                scalars.append(((e.epoch, cls, label), e))
    load_bars = {tile: merge(items) for tile, items in loads.items()}
    scalar_bars = merge(scalars)
    write_bars = merge(((e.epoch, e.instr.slot()[0]), e) for e in events if e.instr.kind == "ds_write")
    read_bars = merge(((e.epoch, e.instr.slot()[0]), e) for e in events if e.instr.kind == "ds_read")
    mfma_runs = runs("mfma", ("waitcnt", "barrier", "ds_read", "ds_write", "vmem_load"), float("inf"))
    valu_runs = runs("valu", ("barrier",), 96)
    load_rows = {tile: pack([(s, x) for _, s, x, _ in load_bars[tile]], gap * 5) for tile in "AB"}
    scalar_rows, n_scalar = pack([(s, x) for _, s, x, _ in scalar_bars], gap * 5)
    # Every binding pair runs G1 -> G0 (G1 trails by one barrier); the barrier between them is the destination's epoch.
    sync, ring = {}, {}
    for kind, slot, src, dst in binding_pairs(view):
        if dst.group == 0:
            sync.setdefault((dst.epoch, kind), set()).add(slot)
            ring[id(dst)] = kind

    lanes = [("sync", "s_barrier\nG1 → G0 sync", 2), ("load", "buffer_load\nA / B packets", load_rows["A"][1] + load_rows["B"][1]),
             ("scalar", "scale + routing\nloads", n_scalar), ("write", "ds_write\nA / B", 2), ("read", "ds_read\nA / B", 2),
             ("mfma", "MFMA", 1), ("valu", "VALU", 1), ("wait", "s_waitcnt\nvm / lgkm", 2)]
    base, n, y = {}, {}, 0.0
    for lane, _, rows in reversed(lanes):
        base[lane], n[lane] = y, rows
        y += rows + 0.8
    header = y

    def lane_y(lane, row):
        return base[lane] + n[lane] - 0.5 - row

    write_pos = {id(e): (s, lane_y("write", "AB".index(tile))) for (_, tile), s, _, evs in write_bars for e in evs}

    def draw(ax, lo, hi):
        def bar_label(s, x, yy, text, **kw):
            if x >= lo and s <= hi:
                ax.text(max(s, lo + gap * 0.2), yy, text if s >= lo else "← " + text, clip_on=True, **kw)

        def point_text(x, yy, text, **kw):
            if lo <= x <= hi:
                ax.text(x, yy, text, clip_on=True, **kw)

        for epoch in range(epochs):
            evs = [e for e in events if e.epoch == epoch]
            phase, title = phase_of(evs)
            x0, x1 = release[epoch], release[epoch + 1]
            ax.add_patch(Rectangle((x0, -0.4), x1 - x0, header + 1.6, color=PHASE_COLOR[phase], zorder=0))
            point_text((x0 + x1) / 2, header + 0.6, title, ha="center", va="center", fontsize=10, fontweight="bold" if phase == "M" else "normal")
            arrive = max((issue(e) for e in evs if e.instr.kind == "barrier"), default=x1)
            if x1 - arrive > gap / 2:
                ax.add_patch(Rectangle((arrive, -0.4), x1 - arrive, header, facecolor="none", hatch="///", edgecolor="#c4c4c4", lw=0, zorder=0.5))
        for lane, _, _ in lanes:
            ax.axhline(base[lane] - 0.4, color="#d0d0d0", lw=0.6, zorder=0.6)
        for b in range(1, epochs + 1):
            ax.axvline(release[b], color="black", lw=1.0, ls=(0, (4, 3)), zorder=4)
            at_edge = "left" if release[b] - lo < gap * 4 else ("right" if hi - release[b] < gap * 4 else "center")
            point_text(release[b], header + 1.55, f"bar {b}", fontsize=8.5, ha=at_edge, va="center")
        for (b, kind), slots in sync.items():
            point_text(release[b] + gap * 0.3, lane_y("sync", 0 if kind == "WAR" else 1), f"{kind} {packets(slots)}", fontsize=9,
                       va="center", color=SYNC_COLOR[kind], fontweight="bold")
        offset = {"A": 0, "B": load_rows["A"][1]}
        for tile in "AB":
            for ((_, block), s, x, evs), row in zip(load_bars[tile], load_rows[tile][0]):
                yy = lane_y("load", offset[tile] + row)
                ax.plot([s, x], [yy, yy], color=TILE_COLOR[tile], lw=2.6, alpha=0.8, solid_capstyle="butt", zorder=2)
                ax.plot([s], [yy], marker=">", ms=6, color=TILE_COLOR[tile], zorder=3)
                bar_label(s, x, yy + 0.38, f"{packets(cons[id(e)].instr.slot() for e in evs)} k{block}", fontsize=8.5, color=TILE_COLOR[tile])
                for tx, ty in {write_pos[id(cons[id(e)])] for e in evs if id(cons[id(e)]) in write_pos}:
                    ax.plot([x, tx], [yy, ty], color=TILE_COLOR[tile], lw=0.7, alpha=0.5, zorder=1)
        for ((_, cls, text), s, x, evs), row in zip(scalar_bars, scalar_rows):
            yy = lane_y("scalar", row)
            ax.plot([s, x], [yy, yy], color=LOAD_COLOR[cls], lw=2.6, alpha=0.85, solid_capstyle="butt", zorder=2)
            ax.plot([s], [yy], marker=">", ms=6, color=LOAD_COLOR[cls], zorder=3)
            bar_label(s, x, yy + 0.38, text + (f" ×{len(evs)}" if len(evs) > 1 else ""), fontsize=8.5,
                      color="gray" if cls == "kernarg" else LOAD_COLOR[cls])
        for lane, bars, marker in (("write", write_bars, "s"), ("read", read_bars, "o")):
            for (epoch, tile), s, x, evs in bars:
                yy = lane_y(lane, "AB".index(tile))
                ax.plot([s, x], [yy, yy], color=TILE_COLOR[tile], lw=2.6, alpha=0.8, solid_capstyle="butt", zorder=2)
                ax.plot([s], [yy], marker=marker, ms=6, color=TILE_COLOR[tile], mec="black", mew=0.5, zorder=3)
                text = packets(e.instr.slot() for e in evs) + (f" ×{len(evs)}" if lane == "read" else "")
                bar_label(s, x, yy + 0.38, text, fontsize=8.5, color=TILE_COLOR[tile], fontweight="bold")
                for kind in {ring[id(e)] for e in evs if id(e) in ring}:
                    ax.plot([s], [yy], marker="o", ms=15, mfc="none", mec=SYNC_COLOR[kind], mew=1.8, zorder=4)
                    ax.plot([release[epoch], s], [lane_y("sync", 0 if kind == "WAR" else 1), yy], color=SYNC_COLOR[kind], lw=1.0, ls=":", zorder=3.5)
        for lane, found, color, text_color in (("mfma", mfma_runs, "#e8605c", "white"), ("valu", valu_runs, VALU_COLOR, "black")):
            yy = lane_y(lane, 0)
            for run in found:
                s0, s1 = issue(run[0]), times[id(run[-1])][1]
                ax.add_patch(Rectangle((s0, yy - 0.38), s1 - s0, 0.76, color=color, zorder=2))
                if s1 - s0 > gap * 1.5:
                    point_text((s0 + s1) / 2, yy, str(len(run)), ha="center", va="center", fontsize=8.5, color=text_color, fontweight="bold", zorder=3)
        level = [0, 0]
        for e in events:
            if e.instr.kind != "waitcnt":
                continue
            for row, value, name, color in ((0, e.instr.vm, "vm", "#d62728"), (1, e.instr.lgkm, "lgkm", "#1f4e9a")):
                if value is None:
                    continue
                x, yy = issue(e), lane_y("wait", row)
                alpha = 1.0 if name in retired.get(id(e), ()) else 0.3
                ax.plot([x, x], [yy - 0.32, yy + 0.32], color=color, lw=1.5, alpha=alpha, zorder=3)
                point_text(x + gap * 0.15, yy + (0.2 if level[row] % 2 == 0 else -0.2), f"{name}{value}", fontsize=8, va="center", color=color, alpha=alpha)
                level[row] += 1

    fig_w, left, right, unit = 30.0, 1.6, 0.3, 0.24
    scale = (fig_w - left - right) / max(hi - lo for _, lo, hi in panels)
    panel_h = (header + 2.6) * unit
    top_m, mid, bottom_m = 0.8, 0.9, 1.9
    fig_h = top_m + 2 * panel_h + mid + bottom_m
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, (name, lo, hi) in enumerate(panels):
        ax = fig.add_axes([left / fig_w, (bottom_m + (1 - i) * (panel_h + mid)) / fig_h, (hi - lo) * scale / fig_w, panel_h / fig_h])
        draw(ax, lo, hi)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.8, header + 1.8)
        ax.set_yticks([base[lane] + n[lane] / 2 for lane, _, _ in lanes], [label for _, label, _ in lanes], fontsize=10)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(name, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("nominal issue cycles", fontsize=9)
    held = "B" if cfg["layout"].startswith("2x4") else "A"
    streamed = "A" if held == "B" else "B"
    fig.text(left / fig_w, 1 - 0.35 / fig_h, f"MoE {cfg['kernel']} layout={cfg['layout']}: wave group 0 (waves 0-3), prolog + kiter 0 and 1, from the final gfx942 ISA.   "
             f"{held} held: all 4 packets read at stage 0 of each K block;  {streamed} streamed: packet s read at stage s", fontsize=13, va="center")
    fig.legend(
        handles=[
            Patch(color=TILE_COLOR["A"], label="A packet"), Patch(color=TILE_COLOR["B"], label="B packet"),
            Patch(color=LOAD_COLOR["sA"], label="scale_a"), Patch(color=LOAD_COLOR["sB"], label="B scales (all K blocks)"),
            Patch(color=LOAD_COLOR["ids"], label="sorted ids"), Patch(color=LOAD_COLOR["meta"], label="num_valid / expert"),
            Patch(color="#e8605c", label="MFMA run (count)"), Patch(color=VALU_COLOR, label="VALU run (count)"),
            Line2D([], [], color="#d62728", lw=1.5, label="vmcnt wait"), Line2D([], [], color="#1f4e9a", lw=1.5, label="lgkmcnt wait"),
            Line2D([], [], color="black", ls=(0, (4, 3)), label="s_barrier (all 8 waves)"),
            Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=SYNC_COLOR["WAR"], mew=1.8, label="WAR: G0 overwrites after G1's read"),
            Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=SYNC_COLOR["RAW"], mew=1.8, label="RAW: G0 reads after G1's write"),
            Patch(facecolor=PHASE_COLOR["M"], label="M: LDS write/read + loads"), Patch(facecolor=PHASE_COLOR["C"], label="C: MFMA + dequant"),
            Patch(facecolor="white", hatch="///", edgecolor="#999999", label="G0 waits at s_barrier for G1"),
        ],
        loc="lower left", bbox_to_anchor=(left / fig_w, 0.75 / fig_h), ncol=8, fontsize=9.5, frameon=False,
    )
    fig.text(
        left / fig_w, 0.12 / fig_h,
        "x: nominal issue cycles in ISA order (MFMA 16x16x32 FP8 = 16 cycles on the XDL pipe and VALU = 4 cycles, both shared with the G1 wave on the same SIMD; "
        "other instructions 4, scalar 1; memory latency not modeled). Wave group 1 (waves 4-7) runs the same instruction stream one barrier later, so every G0 "
        "M interval overlaps a G1 MFMA interval and vice versa.\nBars run from issue to the s_waitcnt that retires them; thin lines join a load to its ds_write; "
        "faded waits retire nothing. s_barrier lane: the cross-group LDS dependency that this barrier alone enforces (G1's last read / write of the packet "
        "completes right before it; the ringed G0 write / read follows right after it).",
        fontsize=9.5, va="bottom",
    )
    fig.savefig(path, dpi=100)
    plt.close(fig)


def dump_one(config, out_dir):
    """Compile one config with FlyDSL IR dumps (fresh runtime cache so the compiler really runs)."""
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = os.path.join(out_dir, config)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = tempfile.mkdtemp(prefix="moe_audit_cache_")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from aiter import dtypes

    cfg = CONFIGS[config]
    if cfg["kernel"] == "gateup":
        from test_gateup_a8w8_blockscale_gfx942 import SORT_BLOCK_M, make_inputs, run_kernel

        inputs = make_inputs(T, I, H, TOPK, E, "ep8")
        capacity = inputs["expert_ids"].numel() * SORT_BLOCK_M
        out = torch.empty((capacity, I), device="cuda", dtype=dtypes.fp8)
        run_kernel(inputs, out, layout=cfg["layout"], out_scale_t=torch.empty((I // 128, capacity), device="cuda"), topk=TOPK)
    else:
        from test_down_a8w8_blockscale_gfx942 import make_inputs, run_kernel

        out = torch.empty((T, H), device="cuda", dtype=torch.bfloat16)
        run_kernel(make_inputs(T, H, I, TOPK, E, "ep8"), out, layout=cfg["layout"], reduce="atomic", sorted_a2=True, single_store=True)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", help="compile every config with IR dumps into this directory")
    parser.add_argument("--dump-one", nargs=2, metavar=("CONFIG", "DIR"), help=argparse.SUPPRESS)
    parser.add_argument("--isa", help="directory holding <config>/<kernel>_0/22_final_isa.s")
    parser.add_argument("--plot", help="write a swimlane PNG per config into this directory")
    parser.add_argument("--report", help="also write the text report to this file")
    parser.add_argument("--self-test", action="store_true", help="check that mutated kernels (lagged waves, dropped waits) fail")
    args = parser.parse_args()
    if args.dump_one:
        dump_one(*args.dump_one)
        return
    if args.dump:
        for config in CONFIGS:
            subprocess.run([sys.executable, os.path.abspath(__file__), "--dump-one", config, args.dump], check=True)
        args.isa = args.isa or args.dump
    if not args.isa:
        parser.error("pass --isa or --dump")
    if args.self_test:
        sys.exit(0 if self_test(args.isa) else 1)
    if args.plot:
        os.makedirs(args.plot, exist_ok=True)
    lines, failed = [], False
    for config, cfg in CONFIGS.items():
        isa = os.path.join(args.isa, config, f"{cfg['kernel']}_kernel_0", "22_final_isa.s")
        rep = analyze(config, isa, args.plot)
        lines += rep.lines + [""]
        failed |= bool(rep.failures)
    text = "\n".join(lines)
    print(text)
    if args.report:
        with open(args.report, "w") as f:
            f.write(text + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
