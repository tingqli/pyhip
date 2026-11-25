"""Pipeline audit of the gfx942 FP8 block-scale 2x4-LDS GEMMs (pf1, pf2) by emulating their final ISA.

All 8 waves of one 256x256 workgroup run the instruction stream LLVM emitted, in s_barrier lockstep:
  - integer SALU/VALU run on concrete values (thread ids, kernel arguments, workgroup id), so every LDS and
    global address is exact per lane. vmcnt/lgkmcnt are re-simulated in hardware order (VMEM in issue order,
    LDS in order, SMEM out of order) and a load writes its registers only when a wait retires it;
  - loaded data is not a number but its provenance, the global byte address of each dword. It flows through
    VGPR staging, ds_write, LDS and ds_read into each MFMA, whose operands must be one B row and one A row over
    the same 32 k of one K block. MFMA results, scale products and accumulators carry (n, m, K block) tags:
    every FMA must retire a complete K-block partial times ScaleA[m, kb] * ScaleB[n / 128, kb] into the
    accumulator of the same (m, n), once per K block, and every BF16 store must write the accumulators of its
    own (m, n) after all K blocks, covering the 256x256 tile exactly once;
  - LDS is checked per dword across the 8 waves with the barrier-epoch happens-before order: an LDS op of one
    wave precedes an access of another wave iff the wait retiring it runs in an earlier epoch (RAW/WAR/WAW).
    The slack (spare barriers) of each cross-wave dependency is recorded;
  - registers: reading or overwriting a VGPR/SGPR whose load is in flight is a hazard, including the
    inline-asm s_buffer_load_dword whose latency LLVM cannot see;
  - barrier pairing: all waves execute the same number of s_barrier; the lead of waves 0-3 is measured.
An epoch is the interval between two consecutive s_barrier of the workgroup.

usage:
  python audit_gemm_2x4_lds_pipeline_gfx942.py --dump DIR      compile every config (COMPILE_ONLY, no launch)
  python audit_gemm_2x4_lds_pipeline_gfx942.py --isa DIR [--report FILE] [--self-test] [--config NAME ...]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

LANES, WAVES = 64, 8
M32, M64 = 0xFFFFFFFF, (1 << 64) - 1
LANE = np.arange(LANES, dtype=np.int64)
INF = 1 << 30
TILE, TILE_K, PACKET_BYTES, LDS_B, LDS_BYTES = 256, 128, 8192, 32768, 65536
FULL_K = (1 << TILE_K) - 1
# Each tensor gets its own 2^40-byte window, so a global address names its tensor.
BASE = {"A": 1 << 40, "B": 2 << 40, "C": 3 << 40, "SA": 4 << 40, "SB": 5 << 40, "KARG": 6 << 40}
WINDOW = {value >> 40: name for name, value in BASE.items()}
# Emulated problem: M = N = 4096 has no partial tile; workgroup 37 is an interior tile.
EMU_M, EMU_BLOCK = 4096, 37
# From this barrier epoch on both wave groups run K blocks >= 1, past the prologue's one-off LDS fill.
STEADY_EPOCH = 10

CONFIGS = {}
# KB = K / 128: 128 (steady state), 5 (odd: pf1 peels block 0), 1 (prologue + epilogue only).
for _variant in ("pf1", "pf2"):
    for _k in (16384, 640, 128):
        for _pre in (1, 0) if _k == 16384 else (1,):
            CONFIGS[f"{_variant}_n4096_k{_k}_pre{_pre}"] = dict(variant=_variant, N=4096, K=_k, preshuffle=bool(_pre))


@dataclass(frozen=True)
class Op:
    file: str  # v / a / s / imm / off / label
    start: int = 0
    count: int = 1
    value: object = 0
    neg: bool = False
    abs: bool = False

    def regs(self):
        base = {"v": 0, "a": 256, "s": 512}.get(self.file)
        return () if base is None else tuple(range(base + self.start, base + self.start + self.count))


SPECIAL = {"vcc": (106, 2), "vcc_lo": (106, 1), "vcc_hi": (107, 1), "exec": (126, 2), "exec_lo": (126, 1),
           "exec_hi": (127, 1), "m0": (124, 1), "flat_scratch": (102, 2)}
FLOAT_CONST = {"0.5": 0x3F000000, "-0.5": 0xBF000000, "1.0": 0x3F800000, "-1.0": 0xBF800000,
               "2.0": 0x40000000, "-2.0": 0xC0000000, "4.0": 0x40800000, "-4.0": 0xC0800000}
VCC = (512 + 106, 512 + 107)


def parse_op(tok):
    neg = absm = False
    if tok.startswith("-") and not re.match(r"^-(\d|0x)", tok):
        neg, tok = True, tok[1:]
    if len(tok) > 2 and tok[0] == tok[-1] == "|":
        absm, tok = True, tok[1:-1]
    m = re.match(r"^([vsa])(\d+)$", tok)
    if m:
        return Op(m.group(1), int(m.group(2)), 1, neg=neg, abs=absm)
    m = re.match(r"^([vsa])\[(\d+):(\d+)\]$", tok)
    if m:
        return Op(m.group(1), int(m.group(2)), int(m.group(3)) - int(m.group(2)) + 1, neg=neg, abs=absm)
    if tok in SPECIAL:
        return Op("s", *SPECIAL[tok])
    if tok == "off":
        return Op("off")
    if tok in FLOAT_CONST:
        return Op("imm", value=FLOAT_CONST[tok])
    if re.match(r"^-?(0x[0-9a-fA-F]+|\d+)$", tok):
        return Op("imm", value=int(tok, 0))
    if tok.startswith("."):
        return Op("label", value=tok)
    raise ValueError(f"unsupported operand {tok!r}")


@dataclass(eq=False)
class Instr:
    line: int
    mnem: str
    text: str
    ops: list
    mods: dict
    kind: str = ""
    dst: tuple = ()
    src: tuple = ()


def classify(m):
    if m.startswith(("buffer_load", "global_load")):
        return "vmem_load"
    if m.startswith(("buffer_store", "global_store")):
        return "vmem_store"
    if m.startswith("scratch_load"):
        return "scratch_load"
    if m.startswith("scratch_store"):
        return "scratch_store"
    if m.startswith(("ds_read", "ds_load")):
        return "ds_read"
    if m.startswith(("ds_write", "ds_store")):
        return "ds_write"
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
    if m == "s_endpgm":
        return "end"
    return "salu"


NO_DST_SALU = ("s_cmp", "s_bitcmp", "s_nop", "s_setprio", "s_sleep", "s_dcache", "s_icache", "s_trap", "s_sethalt", "s_ttrace")


def assign_regs(ins):
    regs = [op.regs() for op in ins.ops]

    def flat(groups):
        return tuple(r for group in groups for r in group)

    kind, m = ins.kind, ins.mnem
    if kind in ("vmem_store", "scratch_store", "ds_write"):
        ins.src = flat(regs)
    elif kind in ("vmem_load", "scratch_load", "ds_read", "smem", "mfma"):
        ins.dst, ins.src = regs[0], flat(regs[1:])
    elif kind == "valu":
        ndst = 2 if re.match(r"v_(add|sub|subrev|addc|subb|subbrev)_co_u32_e64", m) else 1
        ins.dst, ins.src = flat(regs[:ndst]), flat(regs[ndst:])
        if m.startswith(("v_fmac", "v_mac")):
            ins.src += regs[0]
        if m in ("v_cndmask_b32_e32", "v_addc_co_u32_e32", "v_subb_co_u32_e32"):
            ins.src += VCC
    elif kind == "salu":
        if m.startswith(NO_DST_SALU):
            ins.src = flat(regs)
        else:
            ins.dst, ins.src = regs[0], flat(regs[1:])
            if m in ("s_addk_i32", "s_mulk_i32"):
                ins.src += regs[0]
    elif kind == "branch" and m in ("s_cbranch_vccnz", "s_cbranch_vccz"):
        ins.src = VCC


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


def make_instr(line, text):
    mnem, _, rest = text.partition(" ")
    ins = Instr(line, mnem, text, [], {})
    ins.kind = classify(mnem)
    if ins.kind == "waitcnt":
        for name, value in re.findall(r"(vmcnt|lgkmcnt|expcnt)\((\d+)\)", rest):
            ins.mods[name] = int(value)
        return ins
    for piece in split_operands(rest.strip()):
        words = piece.split()
        if not words:
            continue
        ins.ops.append(parse_op(words[0]))
        for word in words[1:]:
            key, _, value = word.partition(":")
            ins.mods[key] = value
    assign_regs(ins)
    return ins


META = re.compile(r"^(?:-\s+)?\.?(amdhsa_\w+|vgpr_count|sgpr_count|agpr_count|vgpr_spill_count|sgpr_spill_count|"
                  r"private_segment_fixed_size|group_segment_fixed_size):?\s+(\S+)$")


@dataclass
class Program:
    path: str
    instrs: list
    labels: dict  # label -> instruction index
    meta: dict

    def target(self, ins):
        return self.labels[ins.ops[0].value]


def parse_isa(path):
    """Kernel body instructions, .LBB label positions and descriptor/metadata fields of one final ISA dump."""
    instrs, labels, meta, state = [], {}, {}, 0
    for number, raw in enumerate(open(path), 1):
        text = raw.split(";")[0].strip()
        if not text:
            continue
        found = META.match(text)
        if found:
            meta.setdefault(found.group(1), found.group(2))
            continue
        if re.match(r"^[.\w$]+:$", text):
            name = text[:-1]
            if state == 0 and not name.startswith("."):
                state = 1
            elif state == 1 and name.startswith(".Lfunc_end"):
                state = 2
            elif state == 1:
                labels[name] = len(instrs)
            continue
        if state == 1 and not text.startswith("."):
            instrs.append(make_instr(number, text))
    return Program(path, instrs, labels, meta)


# ---------------------------------------------------------------------------------------------------------------
# Values. A register holds a concrete value (np.int64[64] per-lane VGPR, int SGPR), a provenance tag, or None.


class G:
    """Dword loaded from global memory: its byte address per lane (VGPR) or uniform (SGPR); -1 = zero fill."""

    __slots__ = ("addr", "dec")

    def __init__(self, addr):
        self.addr, self.dec = addr, None


class P:
    """MFMA partial of C[m, n] (per lane of one result VGPR) over the k in `kmask` of K block `kb`; -1 = zero row."""

    __slots__ = ("n", "m", "kb", "kmask", "born")

    def __init__(self, n, m, kb, kmask, born=None):
        self.n, self.m, self.kb, self.kmask, self.born = n, m, kb, kmask, born


class S:
    """Scale product ScaleA[m, kb] * ScaleB[nb, kb]."""

    __slots__ = ("m", "nb", "kb")

    def __init__(self, m, nb, kb):
        self.m, self.nb, self.kb = m, nb, kb


class ACC:
    """FP32 accumulator of C[m, n] holding the scaled partials of the K blocks in `kbs`.

    `val` is a synthetic FP32 bit pattern per (m, n): the BF16 epilogue is executed on it, so the packed bits can be
    compared with the expected rounding and halfword placement.
    """

    __slots__ = ("n", "m", "kbs", "val")

    def __init__(self, n, m, kbs, val=None):
        self.n, self.m, self.kbs, self.val = n, m, kbs, val


class D:
    """Value computed from tagged inputs (BF16 packing): the accumulators it depends on and, when every input has
    one, its value computed from the accumulators' synthetic values."""

    __slots__ = ("atoms", "bad", "val")

    def __init__(self, atoms, bad, val=None):
        self.atoms, self.bad, self.val = atoms, bad, val


def concrete(x):
    return isinstance(x, (int, np.ndarray))


def vec(x):
    return np.full(LANES, x, np.int64) if isinstance(x, int) else x


def is_zero(x):
    return (isinstance(x, int) and x == 0) or (isinstance(x, np.ndarray) and not x.any())


def derive(*values):
    atoms, bad = {}, False
    for x in values:
        if concrete(x):
            continue
        if isinstance(x, D):
            atoms.update((id(a), a) for a in x.atoms)
            bad |= x.bad
        elif x is None:
            bad = True
        else:
            atoms[id(x)] = x
    return D(tuple(atoms.values()), bad)


def shadow(x):
    """Concrete stand-in of a value: itself if concrete, the synthetic value of an accumulator or derived value."""
    if concrete(x):
        return x
    return x.val if isinstance(x, (ACC, D)) else None


def with_shadow(tag, fn, args):
    """Attach to a derived tag the result of `fn` on the inputs' concrete stand-ins."""
    values = [shadow(a) for a in args]
    if isinstance(tag, D) and all(v is not None for v in values):
        with np.errstate(all="ignore"):
            tag.val = fn(*[vec(v) for v in values])
    return tag


def synthetic(n, m):
    """Deterministic normal FP32 bit patterns of magnitude 2^-15..2^16 per C element (m, n), any mantissa."""
    h = ((n * 0x9E3779B1) ^ (m * 0x85EBCA77)) & M32
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & M32
    h ^= h >> 12
    return (h >> 31) << 31 | (112 + (h >> 23) % 32) << 23 | (h & 0x7FFFFF)


def bf16_bias_trunc(bits):
    """pf1 / PyHIP: add the 0x8000 bit pattern as an FP32 denormal, keep the high half."""
    with np.errstate(all="ignore"):
        return from_f32(f32(bits) + np.array([0x8000], np.uint32).view(np.float32)[0]) >> 16


def bf16_rne(bits):
    """pf2: round to nearest even (the synthetic values are never NaN)."""
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


BF16_PACK = {"pf1": bf16_bias_trunc, "pf2": bf16_rne}


def mask_bits(mask):
    return ((np.uint64(mask) >> LANE.astype(np.uint64)) & np.uint64(1)).astype(bool)


def bits_mask(bits):
    return int(np.sum(np.left_shift(np.uint64(1), LANE[bits].astype(np.uint64)), dtype=np.uint64))


def compare(cc, ty, a, b):
    """64-lane mask of v_cmp_<cc>_<ty> on concrete per-lane values."""
    if ty == "f32":
        fa, fb = f32(a), f32(b)
        nan = np.isnan(fa) | np.isnan(fb)
        table = {"u": nan, "o": ~nan, "eq": fa == fb, "lg": ~nan & (fa != fb), "neq": fa != fb, "lt": fa < fb, "le": fa <= fb,
                 "gt": fa > fb, "ge": fa >= fb, "nlt": ~(fa < fb), "nle": ~(fa <= fb), "ngt": ~(fa > fb), "nge": ~(fa >= fb)}
        return bits_mask(table[cc])
    if ty == "i32":
        a, b = sx(a), sx(b)
    return bits_mask({"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[cc])


def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def sx(a):
    return np.where(a & 0x80000000, a - (1 << 32), a)


def f32(x):
    return np.asarray(x, np.int64).astype(np.uint32).view(np.float32)


def from_f32(f):
    return np.asarray(f, np.float32).view(np.uint32).astype(np.int64)


def lds_slot(byte):
    return ("A" if byte < LDS_B else "B"), int(byte % LDS_B // PACKET_BYTES)


@dataclass(eq=False)
class AOp:
    """One in-flight memory instruction instance of one wave."""

    wave: int
    ins: Instr
    queue: str  # vm / lgkm
    epoch: int
    seq: int
    writes: list = field(default_factory=list)  # (register key, value) applied when a wait retires the op
    complete: int = INF  # epoch of the retiring wait
    cseq: int = -1
    lds_id: int = -1
    info: dict = None


@dataclass(eq=False)
class Ev:
    """One executed instruction of a traced wave."""

    index: int
    epoch: int
    seq: int
    op: AOp = None
    retired: list = None
    info: dict = None


class Lds:
    """Per-dword LDS contents (provenance) and the last writer / readers, checked against barrier epochs."""

    def __init__(self):
        n = LDS_BYTES // 4
        self.src = np.full(n, -2, np.int64)  # global address of the dword, -1 zero fill, -2 never written, -3 not global
        self.w_op = np.full(n, -1, np.int64)
        self.r_op = np.full((n, WAVES), -1, np.int64)  # reads since the last write, per wave
        self.ops, self.complete, self.op_wave = [], np.full(4096, INF, np.int64), np.full(4096, -1, np.int64)
        self.stats, self.steady = {}, {}  # (kind, region, packet, cross-group) -> [pairs, min slack]; steady: consumer epoch >= STEADY_EPOCH
        self.violations, self.binding, self.uninit = [], [], 0
        self.collisions = []  # (op, dwords written by several lanes of one ds_write, of those with different data)

    def register(self, op):
        op.lds_id = len(self.ops)
        self.ops.append(op)
        if op.lds_id >= len(self.complete):
            self.complete = np.concatenate([self.complete, np.full(len(self.complete), INF, np.int64)])
            self.op_wave = np.concatenate([self.op_wave, np.full(len(self.op_wave), -1, np.int64)])
        self.op_wave[op.lds_id] = op.wave

    def _check(self, kind, consumer, epoch, producers, dwords):
        if not len(producers):
            return
        slack = epoch - self.complete[producers] - 1
        for pid in np.unique(producers):
            sel = producers == pid
            low = int(slack[sel].min())
            prod = self.ops[pid]
            region, packet = lds_slot(int(dwords[sel][0]) * 4)
            cross = prod.wave // 4 != consumer.wave // 4
            for table in (self.stats, self.steady) if consumer.epoch >= STEADY_EPOCH else (self.stats,):
                entry = table.setdefault((kind, region, packet, cross), [0, INF])
                entry[0] += 1
                entry[1] = min(entry[1], low)
            if low < 0:
                self.violations.append((kind, region, packet, prod, consumer, low))
            elif low == 0 and cross:
                self.binding.append((kind, region, packet, prod, consumer))

    def read(self, op, dwords, epoch):
        d = dwords.ravel()
        wo = self.w_op[d]
        self.uninit += int((wo < 0).sum())
        other = (wo >= 0) & (self.op_wave[np.maximum(wo, 0)] != op.wave)
        self._check("RAW", op, epoch, wo[other], d[other])
        self.r_op[d, op.wave] = op.lds_id
        return self.src[d].reshape(dwords.shape)

    def write(self, op, dwords, data, epoch):
        d = dwords.ravel()
        order = np.argsort(d, kind="stable")
        same = d[order][1:] == d[order][:-1]
        if same.any():
            values = data.ravel()[order]
            self.collisions.append((op, int(same.sum()), int((same & (values[1:] != values[:-1])).sum())))
        wo = self.w_op[d]
        other = (wo >= 0) & (self.op_wave[np.maximum(wo, 0)] != op.wave)
        self._check("WAW", op, epoch, wo[other], d[other])
        for wave in range(WAVES):
            if wave != op.wave:
                ro = self.r_op[d, wave]
                self._check("WAR", op, epoch, ro[ro >= 0], d[ro >= 0])
        self.src[d] = data.ravel()
        self.w_op[d] = op.lds_id
        self.r_op[d, :] = -1


# ---------------------------------------------------------------------------------------------------------------
# Scalar ALU. Each entry returns (result, SCC); KEEP leaves SCC unchanged.

KEEP = object()


def _nz(r):
    return r, int(r != 0)


def _bfe(a, b, signed):
    off, width = b & 31, (b >> 16) & 0x7F
    v = (a >> off) & ((1 << width) - 1) if width else 0
    if signed and width and v >> (width - 1) & 1:
        v -= 1 << width
    return v & M32


def _overflow(x):
    return int(not -(1 << 31) <= x < 1 << 31)


SALU2 = {
    "s_add_u32": lambda a, b, c: ((a + b) & M32, (a + b) >> 32),
    "s_sub_u32": lambda a, b, c: ((a - b) & M32, int(a < b)),
    "s_addc_u32": lambda a, b, c: ((a + b + c) & M32, (a + b + c) >> 32),
    "s_subb_u32": lambda a, b, c: ((a - b - c) & M32, int(a < b + c)),
    "s_add_i32": lambda a, b, c: ((a + b) & M32, _overflow(s32(a) + s32(b))),
    "s_sub_i32": lambda a, b, c: ((a - b) & M32, _overflow(s32(a) - s32(b))),
    "s_mul_i32": lambda a, b, c: ((a * b) & M32, KEEP),
    "s_mul_hi_u32": lambda a, b, c: ((a * b) >> 32, KEEP),
    "s_mul_hi_i32": lambda a, b, c: ((s32(a) * s32(b)) >> 32 & M32, KEEP),
    "s_lshl_b32": lambda a, b, c: _nz((a << (b & 31)) & M32),
    "s_lshr_b32": lambda a, b, c: _nz(a >> (b & 31)),
    "s_ashr_i32": lambda a, b, c: _nz((s32(a) >> (b & 31)) & M32),
    "s_and_b32": lambda a, b, c: _nz(a & b),
    "s_or_b32": lambda a, b, c: _nz(a | b),
    "s_xor_b32": lambda a, b, c: _nz(a ^ b),
    "s_andn2_b32": lambda a, b, c: _nz(a & ~b & M32),
    "s_orn2_b32": lambda a, b, c: _nz((a | ~b) & M32),
    "s_min_i32": lambda a, b, c: (a if s32(a) < s32(b) else b, int(s32(a) < s32(b))),
    "s_max_i32": lambda a, b, c: (a if s32(a) > s32(b) else b, int(s32(a) > s32(b))),
    "s_min_u32": lambda a, b, c: (min(a, b), int(a < b)),
    "s_max_u32": lambda a, b, c: (max(a, b), int(a > b)),
    "s_bfe_u32": lambda a, b, c: _nz(_bfe(a, b, False)),
    "s_bfe_i32": lambda a, b, c: _nz(_bfe(a, b, True)),
    "s_absdiff_i32": lambda a, b, c: _nz(abs(s32(a) - s32(b)) & M32),
}
for _shift in (1, 2, 3, 4):
    SALU2[f"s_lshl{_shift}_add_u32"] = lambda a, b, c, k=_shift: (((a << k) + b) & M32, int(((a << k) + b) >> 32 != 0))
SALU1 = {
    "s_not_b32": lambda a: _nz(~a & M32),
    "s_abs_i32": lambda a: _nz(abs(s32(a)) & M32),
    "s_brev_b32": lambda a: (int(f"{a:032b}"[::-1], 2), KEEP),
    "s_sext_i32_i16": lambda a: ((s32(a << 16) >> 16) & M32, KEEP),
    "s_sext_i32_i8": lambda a: ((s32(a << 24) >> 24) & M32, KEEP),
}
SALU64 = {
    "s_and_b64": lambda a, b: _nz(a & b),
    "s_or_b64": lambda a, b: _nz(a | b),
    "s_xor_b64": lambda a, b: _nz(a ^ b),
    "s_andn2_b64": lambda a, b: _nz(a & ~b & M64),
    "s_orn2_b64": lambda a, b: _nz((a | ~b) & M64),
}
CMP = {"eq": lambda a, b: a == b, "lg": lambda a, b: a != b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
       "lt": lambda a, b: a < b, "le": lambda a, b: a <= b}


def sext16(x):
    x &= 0xFFFF
    return x - 0x10000 if x >> 15 else x


class Wave:
    def __init__(self, emu, wid):
        self.emu, self.wid, self.group = emu, wid, wid // 4
        self.v, self.a, self.s = [None] * 256, [None] * 256, [None] * 128
        self.scc = None
        self.s[126] = self.s[127] = M32
        self.v[0] = LANE + LANES * wid
        meta, user = emu.prog.meta, 0
        for name, size in (("private_segment_buffer", 4), ("dispatch_ptr", 2), ("queue_ptr", 2)):
            user += size * int(meta.get(f"amdhsa_user_sgpr_{name}", 0))
        if int(meta.get("amdhsa_user_sgpr_kernarg_segment_ptr", 1)):
            self.s[user], self.s[user + 1] = BASE["KARG"] & M32, BASE["KARG"] >> 32
        if int(meta.get("amdhsa_system_sgpr_workgroup_id_x", 1)):
            self.s[int(meta.get("amdhsa_user_sgpr_count", 2))] = emu.block
        self.pc = self.seq = self.barriers = 0
        self.vm, self.lgkm, self.pending, self.scratch = [], [], {}, {}
        self.done = self.at_barrier = False
        self.trace = [] if wid in emu.trace_waves else None

    # --- registers -------------------------------------------------------------------------------------------
    def get(self, key):
        return self.v[key] if key < 256 else (self.a[key - 256] if key < 512 else self.s[key - 512])

    def put(self, key, value):
        if key < 256:
            self.v[key] = value
        elif key < 512:
            self.a[key - 256] = value
        else:
            self.s[key - 512] = value

    def rd(self, op):
        if op.file == "imm":
            return op.value & M32
        if op.file in ("v", "s", "a"):
            return self.get(op.regs()[0])
        return None

    def rd64(self, op):
        if op.file == "imm":
            return op.value & M64
        lo, hi = self.s[op.start], self.s[op.start + 1]
        if isinstance(lo, int) and isinstance(hi, int):
            return lo | hi << 32
        return lo if lo is hi else derive(lo, hi)

    def wr(self, op, value):
        self.put(op.regs()[0], value)

    def wr64(self, op, value):
        if isinstance(value, int):
            self.s[op.start], self.s[op.start + 1] = value & M32, value >> 32 & M32
        else:
            self.s[op.start] = self.s[op.start + 1] = value

    # --- asynchronous memory -----------------------------------------------------------------------------------
    def check_hazards(self, ins):
        pend = self.pending
        if not pend:
            return
        for key in ins.src:
            if key in pend:
                self.emu.hazard("RAW", self, ins, key, pend[key])
        for key in ins.dst:
            op = pend.get(key)
            if op is None:
                continue
            in_order = (ins.kind in ("vmem_load", "scratch_load") and op.queue == "vm") or (ins.kind == "ds_read" and op.ins.kind == "ds_read")
            if not in_order:
                self.emu.hazard("WAW", self, ins, key, op)

    def issue(self, ins, queue, writes, info=None):
        op = AOp(self.wid, ins, queue, self.emu.epoch, self.seq, writes, info=info)
        (self.vm if queue == "vm" else self.lgkm).append(op)
        for key, _ in writes:
            self.pending[key] = op
        return op

    def retire(self, op):
        op.complete, op.cseq = self.emu.epoch, self.seq
        if op.lds_id >= 0:
            self.emu.lds.complete[op.lds_id] = op.complete
        for key, value in op.writes:
            self.put(key, value)
            if self.pending.get(key) is op:
                del self.pending[key]

    def waitcnt(self, ins):
        out = []
        vm, lg = ins.mods.get("vmcnt"), ins.mods.get("lgkmcnt")
        if vm is not None:
            while len(self.vm) > vm:
                out.append(self.vm.pop(0))
        if lg is not None and len(self.lgkm) > lg:
            smem = sum(op.ins.kind == "smem" for op in self.lgkm)
            if lg == 0 or not smem:
                count = len(self.lgkm) - lg
                out += self.lgkm[:count]
                del self.lgkm[:count]
            else:
                # SMEM returns out of order: only the oldest LDS ops beyond the SMEM count are certain.
                done = [op for op in self.lgkm if op.ins.kind != "smem"][:max(0, len(self.lgkm) - lg - smem)]
                out += done
                self.lgkm = [op for op in self.lgkm if op not in done]
        for op in out:
            self.retire(op)
        return out

    # --- execution ---------------------------------------------------------------------------------------------
    def run(self):
        """Execute until the next s_barrier or s_endpgm."""
        prog, emu = self.emu.prog, self.emu
        while True:
            if self.pc >= len(prog.instrs):
                emu.fail("control", f"wave {self.wid} ran past the last instruction")
                self.done = True
                return
            index, ins = self.pc, prog.instrs[self.pc]
            self.pc += 1
            self.seq += 1
            emu.steps += 1
            if emu.steps > emu.max_steps:
                raise RuntimeError("step limit exceeded")
            self.check_hazards(ins)
            ev = Ev(index, emu.epoch, self.seq) if self.trace is not None else None
            kind = ins.kind
            if kind in ("barrier", "end"):
                if ev is not None:
                    self.trace.append(ev)
                self.at_barrier, self.done = kind == "barrier", kind == "end"
                return
            if kind == "waitcnt":
                retired = self.waitcnt(ins)
                if ev is not None:
                    ev.retired = retired
            elif kind == "branch":
                self.branch(ins)
            elif kind == "salu":
                self.salu(ins)
            elif kind == "valu":
                self.valu(ins)
            elif kind == "mfma":
                info = self.mfma(ins)
                if ev is not None:
                    ev.info = info
            else:
                op = self.memory(ins)
                if ev is not None:
                    ev.op = op
            if ev is not None:
                self.trace.append(ev)

    def branch(self, ins):
        m = ins.mnem
        if m == "s_branch":
            taken = True
        elif m in ("s_cbranch_scc0", "s_cbranch_scc1"):
            if self.scc is None:
                raise RuntimeError(f"line {ins.line}: branch on unknown SCC")
            taken = self.scc == (m == "s_cbranch_scc1")
        elif m in ("s_cbranch_vccz", "s_cbranch_vccnz"):
            vcc = self.rd64(Op("s", 106, 2))
            if not isinstance(vcc, int):
                raise RuntimeError(f"line {ins.line}: branch on unknown VCC")
            taken = (vcc != 0) == (m == "s_cbranch_vccnz")
        elif m in ("s_cbranch_execz", "s_cbranch_execnz"):
            taken = m == "s_cbranch_execnz"
        else:
            raise RuntimeError(f"line {ins.line}: unsupported branch {m}")
        if taken:
            self.pc = self.emu.prog.target(ins)

    def salu(self, ins):
        m, ops = ins.mnem, ins.ops
        if m in ("s_nop", "s_setprio", "s_sleep") or m.startswith(("s_dcache", "s_icache", "s_ttrace")):
            return
        cmp = re.match(r"s_cmpk?_(eq|lg|gt|ge|lt|le)_(i32|u32|u64)$", m)
        if cmp:
            cc, ty = cmp.groups()
            a, b = (self.rd64(ops[0]), self.rd64(ops[1])) if ty == "u64" else (self.rd(ops[0]), self.rd(ops[1]))
            if not (isinstance(a, int) and isinstance(b, int)):
                self.scc = None
                return
            if m.startswith("s_cmpk"):
                b = sext16(b) if ty == "i32" else b & 0xFFFF
                a = s32(a) if ty == "i32" else a
            elif ty == "i32":
                a, b = s32(a), s32(b)
            self.scc = int(CMP[cc](a, b))
            return
        if m in ("s_bitcmp0_b32", "s_bitcmp1_b32"):
            a, b = self.rd(ops[0]), self.rd(ops[1])
            self.scc = None if not (isinstance(a, int) and isinstance(b, int)) else int((a >> (b & 31) & 1) == (m[-5] == "1"))
            return
        if m in ("s_mov_b32", "s_movk_i32"):
            value = self.rd(ops[1])
            self.wr(ops[0], sext16(value) & M32 if m == "s_movk_i32" else value)
            return
        if m == "s_mov_b64":
            self.wr64(ops[0], self.rd64(ops[1]))
            return
        if m in ("s_cselect_b32", "s_cselect_b64"):
            if self.scc is None:
                self.wr(ops[0], None) if m.endswith("32") else self.wr64(ops[0], None)
                return
            pick = ops[1] if self.scc else ops[2]
            self.wr(ops[0], self.rd(pick)) if m.endswith("32") else self.wr64(ops[0], self.rd64(pick))
            return
        if m in ("s_addk_i32", "s_mulk_i32"):
            a, k = self.rd(ops[0]), sext16(ops[1].value)
            if not isinstance(a, int):
                self.wr(ops[0], None)
                self.scc = None
            elif m == "s_addk_i32":
                self.wr(ops[0], (a + k) & M32)
                self.scc = _overflow(s32(a) + k)
            else:
                self.wr(ops[0], (a * k) & M32)
            return
        if m in SALU2 or m in SALU1:
            args = [self.rd(op) for op in ops[1:3 if m in SALU2 else 2]]
            needs_scc = m in ("s_addc_u32", "s_subb_u32")
            if all(isinstance(x, int) for x in args) and not (needs_scc and self.scc is None):
                result, scc = SALU2[m](*args, self.scc or 0) if m in SALU2 else SALU1[m](*args)
                self.wr(ops[0], result)
                if scc is not KEEP:
                    self.scc = scc
            else:
                self.wr(ops[0], None)
                if not m.startswith("s_mul"):
                    self.scc = None
            return
        if m in SALU64 or m in ("s_lshl_b64", "s_lshr_b64", "s_not_b64"):
            a = self.rd64(ops[1])
            b = self.rd(ops[2]) if m in ("s_lshl_b64", "s_lshr_b64") else (self.rd64(ops[2]) if len(ops) > 2 else 0)
            if not (isinstance(a, int) and isinstance(b, int)):
                self.wr64(ops[0], derive(a, b))
                self.scc = None
                return
            if m == "s_lshl_b64":
                result, self.scc = _nz((a << (b & 63)) & M64)
            elif m == "s_lshr_b64":
                result, self.scc = _nz(a >> (b & 63))
            elif m == "s_not_b64":
                result, self.scc = _nz(~a & M64)
            else:
                result, self.scc = SALU64[m](a, b)
            self.wr64(ops[0], result)
            return
        raise RuntimeError(f"line {ins.line}: unsupported scalar instruction {ins.text}")

    # --- vector ALU --------------------------------------------------------------------------------------------
    def fsrc(self, op):
        """Float source with neg/abs applied; tags pass through unmodified."""
        x = self.rd(op)
        if not concrete(x) or not (op.neg or op.abs):
            return x
        f = f32(vec(x))
        f = np.abs(f) if op.abs else f
        return from_f32(-f if op.neg else f)

    def valu(self, ins):
        m, ops = ins.mnem, ins.ops
        base = m[:-4] if m.endswith(("_e32", "_e64")) else m
        if base.startswith("v_cmp"):
            self.vcmp(ins, base)
        elif base == "v_mov_b32":
            self.wr(ops[0], self.rd(ops[1]))
        elif base == "v_cndmask_b32":
            a, b = self.rd(ops[1]), self.rd(ops[2])
            mask = self.rd64(ops[3]) if m.endswith("_e64") else self.rd64(Op("s", 106, 2))
            if concrete(a) and concrete(b) and isinstance(mask, int):
                self.wr(ops[0], np.where(mask_bits(mask), vec(b), vec(a)))
            elif isinstance(a, G) and isinstance(b, G) and isinstance(mask, int):
                self.wr(ops[0], G(np.where(mask_bits(mask), vec(b.addr), vec(a.addr))))
            elif isinstance(mask, int) and mask in (0, M64):
                self.wr(ops[0], b if mask else a)
            else:
                res, sa, sb, sm = derive(a, b, mask), shadow(a), shadow(b), mask if isinstance(mask, int) else shadow(mask)
                if sa is not None and sb is not None and isinstance(sm, int):
                    res.val = np.where(mask_bits(sm), vec(sb), vec(sa))
                self.wr(ops[0], res)
        elif base == "v_readfirstlane_b32":
            x = self.rd(ops[1])
            if isinstance(x, np.ndarray):
                x = int(x[0])
            elif isinstance(x, G):
                x = G(int(x.addr[0]) if isinstance(x.addr, np.ndarray) else x.addr)
            self.wr(ops[0], x)
        elif base in ("v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32"):
            a, b = self.rd(ops[2]), self.rd(ops[3])
            if base == "v_subrev_co_u32":
                a, b = b, a
            if concrete(a) and concrete(b):
                a, b = vec(a), vec(b)
                r = a + b if base == "v_add_co_u32" else a - b
                self.wr(ops[0], r & M32)
                self.wr64(ops[1], bits_mask((r >> 32) != 0) if base == "v_add_co_u32" else bits_mask(r < 0))
            else:
                self.wr(ops[0], derive(a, b))
                self.wr64(ops[1], derive(a, b))
        elif base in ("v_fmac_f32", "v_mac_f32", "v_fma_f32"):
            if base == "v_fma_f32":
                a, b, c = self.fsrc(ops[1]), self.fsrc(ops[2]), self.fsrc(ops[3])
            else:
                a, b, c = self.fsrc(ops[1]), self.fsrc(ops[2]), self.rd(ops[0])
            if concrete(a) and concrete(b) and concrete(c):
                with np.errstate(all="ignore"):
                    r = f32(vec(a)).astype(np.float64) * f32(vec(b)) + f32(vec(c))
                self.wr(ops[0], from_f32(r.astype(np.float32)))
            else:
                self.wr(ops[0], self.emu.accumulate(self, ins, a, b, c))
        elif base in VALU_FLOAT:
            fn, arity = VALU_FLOAT[base]
            args = [self.fsrc(op) for op in ops[1:1 + arity]]
            if all(concrete(x) for x in args):
                with np.errstate(all="ignore"):
                    self.wr(ops[0], fn(*[vec(x) for x in args]))
            elif base == "v_mul_f32":
                self.wr(ops[0], with_shadow(self.emu.scale_product(ins, *args), fn, args))
            else:
                self.wr(ops[0], with_shadow(derive(*args), fn, args))
        elif base in VALU_INT:
            fn, arity = VALU_INT[base]
            args = [self.rd(op) for op in ops[1:1 + arity]]
            if all(concrete(x) for x in args):
                self.wr(ops[0], fn(*[vec(x) for x in args]))
            else:
                self.wr(ops[0], with_shadow(derive(*args), fn, args))
        else:
            raise RuntimeError(f"line {ins.line}: unsupported vector instruction {ins.text}")

    def vcmp(self, ins, base):
        found = re.match(r"v_cmp_(\w+?)_(f32|i32|u32)$", base)
        if not found:
            raise RuntimeError(f"line {ins.line}: unsupported compare {ins.text}")
        cc, ty = found.groups()
        a, b = self.fsrc(ins.ops[1]), self.fsrc(ins.ops[2])
        if not (concrete(a) and concrete(b)):
            res, sa, sb = derive(a, b), shadow(a), shadow(b)
            if sa is not None and sb is not None:
                res.val = compare(cc, ty, vec(sa), vec(sb))
            self.wr64(ins.ops[0], res)
            return
        self.wr64(ins.ops[0], compare(cc, ty, vec(a), vec(b)))

    def mfma(self, ins):
        d, a, b, c = ins.ops[:4]
        acc = None
        if c.file == "imm":
            if c.value:
                self.emu.fail("mfma", f"line {ins.line}: MFMA with non-zero literal accumulator")
        else:
            acc = [self.get(key) for key in c.regs()]
        results, info = self.emu.mfma(self, ins, [self.get(k) for k in a.regs()], [self.get(k) for k in b.regs()], acc)
        for key, value in zip(d.regs(), results):
            self.put(key, value)
        return info

    # --- memory ------------------------------------------------------------------------------------------------
    def rsrc(self, op):
        words = [self.s[op.start + i] for i in range(4)]
        if not all(isinstance(x, int) for x in words):
            raise RuntimeError(f"unknown buffer resource s[{op.start}:{op.start + 3}]")
        return words[0] | (words[1] & 0xFFFF) << 32, words[2]

    def memory(self, ins):
        kind, ops, emu = ins.kind, ins.ops, self.emu
        if kind == "smem":
            count = {"dword": 1, "dwordx2": 2, "dwordx4": 4, "dwordx8": 8, "dwordx16": 16}[ins.mnem.split("_load_")[1]]
            offset = self.rd(ops[2]) if len(ops) > 2 else 0
            if ins.mnem.startswith("s_load"):
                start = self.rd64(ops[1]) + offset
                values = [emu.kernarg_dword(start + 4 * i) for i in range(count)]
            else:
                base, records = self.rsrc(ops[1])
                values = [G(base + offset + 4 * i) if offset + 4 * i < records else 0 for i in range(count)]
            return self.issue(ins, "lgkm", list(zip(ops[0].regs(), values)), emu.describe(values[0]))
        if kind in ("vmem_load", "vmem_store"):
            count = {"dword": 1, "dwordx2": 2, "dwordx3": 3, "dwordx4": 4}[ins.mnem.split("_", 2)[2]]
            base, records = self.rsrc(ops[2])
            voff = self.rd(ops[1]) if "offen" in ins.mods else 0
            soff, imm = self.rd(ops[3]), int(ins.mods.get("offset", "0"), 0)
            if not (concrete(voff) and isinstance(soff, int)):
                raise RuntimeError(f"line {ins.line}: unknown buffer offset")
            rel = vec(voff) + imm
            addrs = [base + rel + soff + 4 * i for i in range(count)]
            # gfx9 raw buffers range-check voffset + inst_offset; soffset is not part of the check.
            inside = [rel + 4 * i < records for i in range(count)]
            if any(((rel + soff + 4 * i < records) != inside[i]).any() for i in range(count)):
                emu.note(f"line {ins.line}: range check depends on whether soffset is included")
            if kind == "vmem_load":
                values = [G(np.where(inside[i], addrs[i], -1)) for i in range(count)]
                return self.issue(ins, "vm", list(zip(ops[0].regs(), values)), emu.describe(values[0]))
            emu.store(self, ins, addrs, inside, [self.get(key) for key in ops[0].regs()])
            return self.issue(ins, "vm", [], {"tensor": "C"})
        if kind in ("ds_read", "ds_write"):
            count = {"b32": 1, "b64": 2, "b96": 3, "b128": 4}[ins.mnem.rsplit("_", 1)[1]]
            vaddr, data = (ops[0], ops[1]) if kind == "ds_write" else (ops[1], ops[0])
            addr = self.rd(vaddr)
            if not isinstance(addr, np.ndarray):
                raise RuntimeError(f"line {ins.line}: unknown LDS address")
            addr = addr + int(ins.mods.get("offset", "0"), 0)
            if (addr % 4).any() or (addr < 0).any() or (addr + 4 * count > LDS_BYTES).any():
                emu.fail("lds", f"line {ins.line}: LDS address out of range or misaligned")
                addr = np.clip(addr - addr % 4, 0, LDS_BYTES - 4 * count)
            dwords = (addr // 4)[:, None] + np.arange(count)
            op = AOp(self.wid, ins, "lgkm", emu.epoch, self.seq)
            emu.lds.register(op)
            if kind == "ds_write":
                src = np.stack([emu.lds_word(self.get(key)) for key in data.regs()], 1)
                emu.lds.write(op, dwords, src, emu.epoch)
            else:
                src = emu.lds.read(op, dwords, emu.epoch)
                op.writes = [(key, G(src[:, i].copy())) for i, key in enumerate(data.regs())]
            op.info = emu.describe_lds(addr, src)
            self.lgkm.append(op)
            for key, _ in op.writes:
                self.pending[key] = op
            return op
        if kind in ("scratch_load", "scratch_store"):
            count = {"dword": 1, "dwordx2": 2, "dwordx3": 3, "dwordx4": 4}[ins.mnem.split("_", 2)[2]]
            vaddr, data, saddr = (ops[1], ops[0], ops[2]) if kind == "scratch_load" else (ops[0], ops[1], ops[2])
            if vaddr.file != "off" or saddr.file != "off":
                raise RuntimeError(f"line {ins.line}: scratch addressing not supported")
            offset = int(ins.mods.get("offset", "0"), 0)
            if kind == "scratch_store":
                for i, key in enumerate(data.regs()):
                    self.scratch[offset + 4 * i] = self.get(key)
                return self.issue(ins, "vm", [], {"tensor": "scratch"})
            values = [self.scratch.get(offset + 4 * i) for i in range(count)]
            return self.issue(ins, "vm", list(zip(data.regs(), values)), {"tensor": "scratch"})
        raise RuntimeError(f"line {ins.line}: unsupported memory instruction {ins.text}")


def _brev(a):
    return np.array([int(f"{int(x):032b}"[::-1], 2) for x in a], np.int64)


def _perm(a, b, sel):
    combined = [(b >> (8 * i)) & 0xFF for i in range(4)] + [(a >> (8 * i)) & 0xFF for i in range(4)]
    sign = {8: (b >> 15) & 1, 9: (b >> 31) & 1, 10: (a >> 15) & 1, 11: (a >> 31) & 1}
    out = np.zeros(LANES, np.int64)
    for i in range(4):
        s = (sel >> (8 * i)) & 0xFF
        byte = np.where(s >= 13, 0xFF, 0)
        for value in range(8):
            byte = np.where(s == value, combined[value], byte)
        for value, bit in sign.items():
            byte = np.where(s == value, bit * 0xFF, byte)
        out |= byte << (8 * i)
    return out


VALU_INT = {
    "v_mov_b32": (lambda a: a, 1),
    "v_add_u32": (lambda a, b: (a + b) & M32, 2),
    "v_sub_u32": (lambda a, b: (a - b) & M32, 2),
    "v_subrev_u32": (lambda a, b: (b - a) & M32, 2),
    "v_add3_u32": (lambda a, b, c: (a + b + c) & M32, 3),
    "v_lshlrev_b32": (lambda a, b: (b << (a & 31)) & M32, 2),
    "v_lshrrev_b32": (lambda a, b: b >> (a & 31), 2),
    "v_ashrrev_i32": (lambda a, b: (sx(b) >> (a & 31)) & M32, 2),
    "v_and_b32": (lambda a, b: a & b, 2),
    "v_or_b32": (lambda a, b: a | b, 2),
    "v_xor_b32": (lambda a, b: a ^ b, 2),
    "v_not_b32": (lambda a: ~a & M32, 1),
    "v_or3_b32": (lambda a, b, c: a | b | c, 3),
    "v_and_or_b32": (lambda a, b, c: (a & b) | c, 3),
    "v_lshl_or_b32": (lambda a, b, c: ((a << (b & 31)) & M32) | c, 3),
    "v_lshl_add_u32": (lambda a, b, c: (((a << (b & 31)) & M32) + c) & M32, 3),
    "v_add_lshl_u32": (lambda a, b, c: (((a + b) & M32) << (c & 31)) & M32, 3),
    "v_xad_u32": (lambda a, b, c: ((a ^ b) + c) & M32, 3),
    "v_bfe_u32": (lambda a, b, c: (a >> (b & 31)) & ((np.int64(1) << (c & 31)) - 1), 3),
    "v_bfi_b32": (lambda a, b, c: (a & b) | (~a & M32 & c), 3),
    "v_min_i32": (lambda a, b: np.minimum(sx(a), sx(b)) & M32, 2),
    "v_max_i32": (lambda a, b: np.maximum(sx(a), sx(b)) & M32, 2),
    "v_min_u32": (np.minimum, 2),
    "v_max_u32": (np.maximum, 2),
    "v_mul_u32_u24": (lambda a, b: ((a & 0xFFFFFF) * (b & 0xFFFFFF)) & M32, 2),
    "v_mul_lo_u32": (lambda a, b: (a.astype(np.uint64) * b.astype(np.uint64) & np.uint64(M32)).astype(np.int64), 2),
    "v_bfrev_b32": (_brev, 1),
    "v_perm_b32": (_perm, 3),
}
VALU_FLOAT = {
    "v_mul_f32": (lambda a, b: from_f32(f32(a) * f32(b)), 2),
    "v_add_f32": (lambda a, b: from_f32(f32(a) + f32(b)), 2),
    "v_sub_f32": (lambda a, b: from_f32(f32(a) - f32(b)), 2),
    "v_subrev_f32": (lambda a, b: from_f32(f32(b) - f32(a)), 2),
    "v_max_f32": (lambda a, b: from_f32(np.fmax(f32(a), f32(b))), 2),
    "v_min_f32": (lambda a, b: from_f32(np.fmin(f32(a), f32(b))), 2),
    "v_rcp_f32": (lambda a: from_f32(np.float32(1.0) / f32(a)), 1),
    "v_rcp_iflag_f32": (lambda a: from_f32(np.float32(1.0) / f32(a)), 1),
    "v_cvt_f32_u32": (lambda a: from_f32(a.astype(np.float32)), 1),
    "v_cvt_f32_i32": (lambda a: from_f32(sx(a).astype(np.float32)), 1),
    "v_cvt_u32_f32": (lambda a: np.clip(np.nan_to_num(np.trunc(f32(a).astype(np.float64)), nan=0.0), 0, M32).astype(np.int64), 1),
    "v_cvt_i32_f32": (lambda a: np.clip(np.nan_to_num(np.trunc(f32(a).astype(np.float64)), nan=0.0), -(1 << 31), (1 << 31) - 1).astype(np.int64) & M32, 1),
}


# ---------------------------------------------------------------------------------------------------------------
# Workgroup emulation and data-path checks.


def tb_swizzle(block_id, M, N, m01=8, group_num=8):
    """pf1 workgroup -> (block_m, block_n), the integer formula of gemm_a8w8_blockscale_gfx942_common.tb_swizzle."""
    m0, n0 = (M + TILE - 1) // TILE, (N + TILE - 1) // TILE
    group_size = (m0 * n0 + group_num - 1) // group_num
    big_group_num = group_num - (group_size * group_num - m0 * n0)
    group_id_y, group_id_x = divmod(block_id, group_num)
    remap = group_id_x * group_size + group_id_y
    if group_id_x > big_group_num:
        remap += big_group_num - group_id_x
    idx_m0, idx_n0 = divmod(remap, n0)
    m0_mod_m01 = m0 - m0 // m01 * m01
    m01_adapt = m01 if idx_m0 < m0 - m0_mod_m01 else m0_mod_m01
    idx_m00 = idx_m0 // m01
    local = idx_n0 + (idx_m0 - idx_m00 * m01) * n0
    block_n = local // m01_adapt
    return local - block_n * m01_adapt + idx_m00 * m01, block_n


def expected_tile(cfg, M, block):
    if cfg["variant"] == "pf1":
        return tb_swizzle(block, M, cfg["N"])
    num_m, n_tiles = (M + TILE - 1) // TILE, (cfg["N"] + TILE - 1) // TILE
    group_tasks = 8 * n_tiles
    first_m = block // group_tasks * 8
    group_m = min(num_m - first_m, 8)
    return first_m + block % group_tasks % group_m, block % group_tasks // group_m


class Emulator:
    def __init__(self, prog, cfg, M=EMU_M, block=EMU_BLOCK, trace_waves=(0, 4)):
        self.prog, self.cfg = prog, cfg
        self.N, self.K, self.preshuffle = cfg["N"], cfg["K"], cfg["preshuffle"]
        self.KB, self.M, self.block = self.K // TILE_K, M, block
        block_m, block_n = expected_tile(cfg, M, block)
        self.row0, self.col0 = block_m * TILE, block_n * TILE
        assert self.row0 + TILE <= M and self.col0 + TILE <= self.N, "the audit emulates an interior tile"
        self.karg = {40: M}
        for i, name in enumerate(("A", "B", "C", "SA", "SB")):
            self.karg[8 * i], self.karg[8 * i + 4] = BASE[name] & M32, BASE[name] >> 32
        self.trace_waves = set(trace_waves)
        self.lds = Lds()
        self.epoch = self.steps = 0
        self.max_steps = 100_000_000
        self.failures, self.notes = defaultdict(list), []
        self.hazards = []
        self.mfma_count = self.fma_count = self.store_count = self.bits_checked = 0
        self.cover = []
        self.partial_age = []  # epochs between an MFMA chain's last MFMA and the FMA retiring it (traced waves)
        self.waves = [Wave(self, w) for w in range(WAVES)]

    # --- bookkeeping -------------------------------------------------------------------------------------------
    def fail(self, category, message):
        self.failures[category].append(message)

    def note(self, message):
        if message not in self.notes:
            self.notes.append(message)

    def hazard(self, kind, wave, ins, key, op):
        name = f"v{key}" if key < 256 else (f"a{key - 256}" if key < 512 else f"s{key - 512}")
        self.hazards.append((kind, wave.wid, ins, name, op))
        self.fail("register", f"{kind} {name}: wave {wave.wid} line {ins.line} '{ins.text}' while line {op.ins.line} "
                              f"'{op.ins.text}' (epoch {op.epoch}) is in flight")

    def kernarg_dword(self, addr):
        return self.karg.get(addr - BASE["KARG"])

    def lds_word(self, value):
        if isinstance(value, G):
            return vec(value.addr) if isinstance(value.addr, int) else value.addr
        self.fail("lds", f"LDS write of {type(value).__name__}, not data loaded from global memory")
        return np.full(LANES, -3, np.int64)

    # --- provenance --------------------------------------------------------------------------------------------
    @staticmethod
    def region(g):
        addr = g.addr if isinstance(g.addr, np.ndarray) else np.array([g.addr])
        windows = set((addr[addr >= 0] >> 40).tolist())
        return WINDOW.get(windows.pop()) if len(windows) == 1 else None

    def decode_bytes(self, tensor, q):
        """(row, k) of byte offsets q into A[M, K] or B[N, K] (plain or aiter shuffle_weight layout (16, 16))."""
        if tensor == "A" or not self.preshuffle:
            return q // self.K, q % self.K
        n0, k0 = q // (16 * self.K), q // 512 % (self.K // 32)
        return n0 * 16 + q // 16 % 16, k0 * 32 + q // 256 % 2 * 16 + q % 16

    def decode(self, g, tensor):
        if g.dec is None or g.dec[0] != tensor:
            addr = vec(g.addr) if isinstance(g.addr, int) else g.addr
            bad = (addr < 0) | ((addr >> 40) != BASE[tensor] >> 40)
            row, k = self.decode_bytes(tensor, addr[:, None] + np.arange(4) - BASE[tensor])
            g.dec = (tensor, (row, k, bad))
        return g.dec[1]

    def operand(self, regs, tensor, ins):
        """MFMA operand of 2 VGPRs: the row of each of the 16 lanes-in-group and the k of each (k group, byte)."""
        if not all(isinstance(g, G) for g in regs):
            self.fail("mfma", f"line {ins.line}: {tensor} operand holds {[type(g).__name__ for g in regs]}, not loaded data")
            return None
        parts = [self.decode(g, tensor) for g in regs]
        row = np.concatenate([p[0] for p in parts], 1)
        k = np.concatenate([p[1] for p in parts], 1)
        if any(p[2].any() for p in parts):
            self.fail("mfma", f"line {ins.line}: {tensor} operand holds zero fill, unwritten LDS or another tensor's data")
            return None
        lane_row = row[:, 0]
        if not (row == lane_row[:, None]).all() or not (lane_row.reshape(4, 16) == lane_row[:16]).all():
            self.fail("mfma", f"line {ins.line}: {tensor} operand lanes mix rows")
            return None
        k4 = k.reshape(4, 16, 8)
        if not (k4 == k4[:, :1, :]).all():
            self.fail("mfma", f"line {ins.line}: {tensor} operand k differs between the rows of one k group")
            return None
        return lane_row[:16], k4[:, 0, :]

    def mfma(self, wave, ins, src_a, src_b, acc):
        """D[n, m] += B[n, k] * A[m, k]: srcA is B data (rows n), srcB is A data (rows m)."""
        self.mfma_count += 1
        broken = [D((), True)] * 4
        ra, rb = self.operand(src_a, "B", ins), self.operand(src_b, "A", ins)
        if ra is None or rb is None:
            return broken, None
        (n_i, kg_a), (m_j, kg_b) = ra, rb
        ks = kg_a.ravel()
        blocks = np.unique(ks // TILE_K)
        if not (kg_a == kg_b).all() or len(np.unique(ks)) != 32 or len(blocks) != 1:
            self.fail("mfma", f"line {ins.line}: operands do not cover the same 32 k of one K block")
            return broken, None
        kb, mask = int(blocks[0]), 0
        for x in (ks % TILE_K).tolist():
            mask |= 1 << x
        m, out = m_j[LANE % 16], []
        for r in range(4):
            n, prev = n_i[4 * (LANE // 16) + r], 0
            if acc is not None:
                t = acc[r]
                if not (isinstance(t, P) and t.kb == kb and np.array_equal(t.n, n) and np.array_equal(t.m, m)) or t.kmask & mask:
                    self.fail("mfma", f"line {ins.line} wave {wave.wid}: accumulator {type(t).__name__} does not continue the chain of C[m, n] K block {kb}")
                    out.append(D((), True))
                    continue
                prev = t.kmask
            out.append(P(n, m, kb, prev | mask, (self.epoch, wave.seq)))
        return out, {"kb": kb, "m0": int(m_j.min()), "n0": int(n_i.min())}

    def scale_product(self, ins, a, b):
        if not (isinstance(a, G) and isinstance(b, G)) or {self.region(a), self.region(b)} != {"SA", "SB"}:
            return derive(a, b)
        sa, sb = (a, b) if self.region(a) == "SA" else (b, a)
        qa, qb = (vec(sa.addr) - BASE["SA"]) // 4, (vec(sb.addr) - BASE["SB"]) // 4
        kb_a, m = qa // self.M, qa % self.M
        nb, kb_b = qb // self.KB, qb % self.KB
        if len(np.unique(kb_a)) != 1 or len(np.unique(kb_b)) != 1 or len(np.unique(nb)) != 1 or kb_a[0] != kb_b[0]:
            self.fail("scale", f"line {ins.line}: ScaleA K blocks {sorted(set(kb_a.tolist()))} x ScaleB K blocks {sorted(set(kb_b.tolist()))}")
            return D((), True)
        return S(m, int(nb[0]), int(kb_a[0]))

    def accumulate(self, wave, ins, a, b, c):
        """FMA c + a * b on tags: a complete K-block partial times its scale, into the accumulator of its C[m, n]."""
        if is_zero(a) or is_zero(b):
            return c
        p, s = (a, b) if isinstance(a, P) else (b, a)
        where = f"line {ins.line} wave {wave.wid}"
        if not (isinstance(p, P) and isinstance(s, S)):
            self.fail("fma", f"{where}: FMA of {type(a).__name__} x {type(b).__name__}, expected partial x scale")
            return derive(a, b, c)
        self.fma_count += 1
        if wave.trace is not None:
            self.partial_age.append((wave.wid, p.born, (self.epoch, wave.seq)))
        errors = []
        if p.kmask != FULL_K:
            errors.append(f"partial covers {bin(p.kmask).count('1')} of 128 k")
        if p.kb != s.kb:
            errors.append(f"K block {p.kb} partial scaled with K block {s.kb} scales")
        if not (p.m == s.m).all():
            errors.append("ScaleA row differs from the partial's row")
        if not (p.n // 128 == s.nb).all():
            errors.append("ScaleB block differs from the partial's column block")
        if is_zero(c):
            new = ACC(p.n, p.m, (p.kb,), synthetic(p.n, p.m))
        elif isinstance(c, ACC):
            if not (np.array_equal(c.n, p.n) and np.array_equal(c.m, p.m)):
                errors.append("accumulator of another C element")
            if p.kb in c.kbs:
                errors.append(f"K block {p.kb} accumulated twice")
            new = ACC(c.n, c.m, c.kbs + (p.kb,), c.val)
        else:
            errors.append(f"accumulator holds {type(c).__name__}")
            new = derive(a, b, c)
        for error in errors:
            self.fail("fma", f"{where}: {error}")
        return new

    def store(self, wave, ins, addrs, inside, values):
        self.store_count += 1
        for d, value in enumerate(values):
            lanes = inside[d]
            if not lanes.any():
                continue
            addr = addrs[d][lanes]
            if ((addr >> 40) != BASE["C"] >> 40).any():
                self.fail("store", f"line {ins.line}: store outside C")
                continue
            q = (addr - BASE["C"]) // 2
            row, col = q // self.N, q % self.N
            atoms = value.atoms if isinstance(value, D) and not value.bad else ((value,) if isinstance(value, ACC) else None)
            if atoms is None:
                self.fail("store", f"line {ins.line}: stored dword {d} holds {type(value).__name__}")
                continue
            halves = []
            for j in (0, 1):
                hits = [x for x in atoms if isinstance(x, ACC) and (x.n[lanes] == col + j).all() and (x.m[lanes] == row).all()]
                if len(hits) != 1:
                    self.fail("store", f"line {ins.line} wave {wave.wid}: BF16 {2 * d + j} is not the accumulator of the C element it is stored to")
                    continue
                if tuple(sorted(hits[0].kbs)) != tuple(range(self.KB)):
                    self.fail("store", f"line {ins.line} wave {wave.wid}: stored accumulator holds {len(hits[0].kbs)} of {self.KB} K blocks")
                halves.append(BF16_PACK[self.cfg["variant"]](hits[0].val[lanes]))
            if len(atoms) != 2:
                self.fail("store", f"line {ins.line}: a BF16x2 dword packs {len(atoms)} values")
            packed = shadow(value)
            if packed is None:
                self.fail("store", f"line {ins.line}: BF16x2 dword {d} could not be evaluated on the accumulators' values")
            elif len(halves) == 2:
                self.bits_checked += int(lanes.sum()) * 2
                if ((vec(packed)[lanes] & M32) != (halves[0] | halves[1] << 16)).any():
                    self.fail("store", f"line {ins.line} wave {wave.wid}: BF16x2 dword {d} bits differ from the {self.cfg['variant']} "
                                       f"rounding of C[m, n] (low half) and C[m, n+1] (high half)")
            self.cover += [row * self.N + col, row * self.N + col + 1]

    def describe(self, value):
        """Labels of a loaded value: tensor, K block and the A/B packet, ScaleA packet or ScaleB side."""
        if not isinstance(value, G):
            return {"tensor": "KARG" if isinstance(value, int) else None}
        tensor = self.region(value)
        info = {"tensor": tensor}
        addr = vec(value.addr)
        if tensor in ("A", "B"):
            row, k = self.decode_bytes(tensor, addr - BASE[tensor])
            base = self.row0 if tensor == "A" else self.col0
            info.update(kb=int(k.min() // TILE_K), packet=int((row.min() - base) // 64), rows=(int(row.min()), int(row.max())))
        elif tensor == "SA":
            q = (addr - BASE["SA"]) // 4
            info.update(kb=int(q.min() // self.M), packet=int((q.min() % self.M - self.row0) // 64), rep=int((q.min() % self.M - self.row0) % 64 // 32))
        elif tensor == "SB":
            q = (addr - BASE["SB"]) // 4
            info.update(kb=int(q.min() % self.KB), side=int(q.min() // self.KB - self.col0 // 128))
        return info

    def describe_lds(self, addr, src):
        slots = sorted({lds_slot(int(x)) for x in np.unique(addr)})
        info = {"region": slots[0][0], "packet": slots[0][1], "slots": slots}
        valid = src[src >= 0]
        tensor = WINDOW.get(int(valid[0]) >> 40) if len(valid) else None
        if tensor in ("A", "B"):
            _, k = self.decode_bytes(tensor, valid - BASE[tensor])
            info.update(tensor=tensor, kb=sorted(set((k // TILE_K).tolist())))
        return info

    # --- driver ------------------------------------------------------------------------------------------------
    def run(self):
        while True:
            for wave in self.waves:
                if not wave.done and not wave.at_barrier:
                    wave.run()
            live = [w for w in self.waves if not w.done]
            if not live:
                break
            if len(live) != WAVES:
                self.fail("barrier", f"epoch {self.epoch}: s_barrier of waves {[w.wid for w in live]} after the other waves ended")
            self.epoch += 1
            for wave in live:
                wave.at_barrier = False
                wave.barriers += 1
        counts = [w.barriers for w in self.waves]
        if len(set(counts)) != 1:
            self.fail("barrier", f"s_barrier count per wave {counts}")
        for kind, region, packet, prod, cons, slack in self.lds.violations:
            self.fail("lds", f"{kind} {region}{packet}: wave {prod.wave} line {prod.ins.line} (epoch {prod.epoch}, retired in epoch "
                             f"{prod.complete if prod.complete < INF else 'never'}) vs wave {cons.wave} line {cons.ins.line} (epoch {cons.epoch})")
        if self.lds.uninit:
            self.fail("lds", f"{self.lds.uninit} dword reads of LDS never written")
        for op, count, conflicting in self.lds.collisions:
            if conflicting:
                self.fail("lds", f"wave {op.wave} line {op.ins.line} (epoch {op.epoch}): {conflicting} dwords written by several lanes "
                                 f"of one ds_write with different data")
            else:
                self.note(f"line {op.ins.line}: {count} dwords written by several lanes of one ds_write with identical data")
        cover = np.concatenate(self.cover) if self.cover else np.zeros(0, np.int64)
        ids, counts = np.unique(cover, return_counts=True)
        rows, cols = np.arange(self.row0, self.row0 + TILE), np.arange(self.col0, self.col0 + TILE)
        expected = np.sort((rows[:, None] * self.N + cols[None, :]).ravel())
        if not np.array_equal(ids, expected) or (counts != 1).any():
            self.fail("store", f"C coverage: {len(ids)} distinct elements stored (expected {len(expected)}), "
                               f"max {counts.max() if len(counts) else 0} stores per element")
        return self


# ---------------------------------------------------------------------------------------------------------------
# Analysis of an emulated run.


def kind_of(prog, ev):
    return prog.instrs[ev.index].kind


def group_barriers(emu, prog):
    """s_barrier instructions executed a different number of times by wave 0 (group 0) and wave 4 (group 1)."""
    counts = {}
    for wid in (0, 4):
        counts[wid] = defaultdict(int)
        for ev in emu.waves[wid].trace:
            if kind_of(prog, ev) == "barrier":
                counts[wid][ev.index] += 1
    return [(prog.instrs[i].line, counts[0].get(i, 0), counts[4].get(i, 0))
            for i in sorted(set(counts[0]) | set(counts[4])) if counts[0].get(i, 0) != counts[4].get(i, 0)]


def epoch_labels(emu, prog, wid):
    """epoch -> (phase, K block, stage): C = MFMA stage, M = LDS/memory stage, or prolog / epilog / idle."""
    by_epoch = defaultdict(list)
    for ev in emu.waves[wid].trace:
        by_epoch[ev.epoch].append(ev)
    first = min(e for e, evs in by_epoch.items() if any(ev.info for ev in evs))
    labels = {}
    for e in range(emu.epoch + 1):
        evs = by_epoch.get(e, [])
        mfma = [ev.info for ev in evs if ev.info]
        reads = [ev.op.info for ev in evs if ev.op is not None and kind_of(prog, ev) == "ds_read" and ev.op.info["region"] == "A"]
        if mfma:
            kb = max(set(i["kb"] for i in mfma), key=[i["kb"] for i in mfma].count)
            labels[e] = ("C", kb, (min(i["m0"] for i in mfma if i["kb"] == kb) - emu.row0) // 64)
        elif reads:
            labels[e] = ("M", reads[0].get("kb", [None])[0], reads[0]["packet"])
        elif any(kind_of(prog, ev) not in ("salu", "branch", "barrier", "end") for ev in evs):
            labels[e] = ("prolog" if e < first else "epilog", None, None)
        else:
            labels[e] = ("idle", None, None)
    return labels


def label_text(label):
    phase, kb, stage = label
    return f"k{kb}.s{stage} {phase}" if phase in ("C", "M") else phase


def fmt_lds(info):
    kb = info.get("kb")
    return f"{info['region']}{info['packet']}" + (f"(k{','.join(map(str, kb))})" if kb else "")


def fmt_load(info):
    t = info.get("tensor")
    if t in ("A", "B"):
        return f"{t}{info['packet']}(k{info['kb']})"
    if t == "SA":
        return f"sA{info['packet']}.{info['rep']}(k{info['kb']})"
    if t == "SB":
        return f"sB{info['side']}(k{info['kb']})"
    return {"KARG": "karg", "scratch": "scratch"}.get(t, str(t))


def runs(items):
    out = []
    for item in items:
        if out and out[-1][0] == item:
            out[-1][1] += 1
        else:
            out.append([item, 1])
    return " ".join(f"{a}x{n}" if n > 1 else a for a, n in out)


def stage_table(emu, prog, wid, kb):
    """Per epoch of K block kb for one traced wave: LDS writes/reads, loads, waits, MFMA and VALU counts."""
    labels = epoch_labels(emu, prog, wid)
    epochs = [e for e, label in labels.items() if label[1] == kb]
    by_epoch = defaultdict(list)
    for ev in emu.waves[wid].trace:
        by_epoch[ev.epoch].append(ev)
    rows = [" epoch | phase    | ds_write          | ds_read                          | loads issued (data K block)          | waits vm/lgkm (retired) | MFMA | VALU"]
    for e in epochs:
        evs = by_epoch[e]
        writes = runs(fmt_lds(ev.op.info) for ev in evs if kind_of(prog, ev) == "ds_write")
        reads = runs(fmt_lds(ev.op.info) for ev in evs if kind_of(prog, ev) == "ds_read")
        loads = " ".join(("s:" if kind_of(prog, ev) == "smem" else "") + fmt_load(ev.op.info) for ev in evs if kind_of(prog, ev) in ("vmem_load", "smem"))
        waits = []
        for ev in evs:
            ins = prog.instrs[ev.index]
            if ins.kind == "waitcnt":
                vm = sum(op.queue == "vm" for op in ev.retired)
                lg = len(ev.retired) - vm
                waits += ([f"vm{ins.mods['vmcnt']}({vm})"] if "vmcnt" in ins.mods else []) + ([f"lgkm{ins.mods['lgkmcnt']}({lg})"] if "lgkmcnt" in ins.mods else [])
        mfma = sum(kind_of(prog, ev) == "mfma" for ev in evs)
        valu = sum(kind_of(prog, ev) == "valu" for ev in evs)
        rows.append(f" {e:5d} | {label_text(labels[e]):8s} | {writes:17s} | {reads:32s} | {loads:36s} | {' '.join(waits):23s} | {mfma:4d} | {valu}")
    return rows


LOAD_CLASS = {"A": "A stage", "B": "B stage", "SA": "scale_a", "SB": "scale_b"}


def register_lifetimes(emu, prog, wid):
    """Live ranges in one traced wave of the registers written by global/LDS loads and MFMA chains.

    Each record: class, label, registers, def event (issue), ready (epoch, seq) when the retiring wait ran,
    last-use event and the number of reading instructions. A load's registers are reserved from issue.
    """
    live, records = {}, []
    for ev in emu.waves[wid].trace:
        ins = prog.instrs[ev.index]
        for key in set(ins.src):
            rec = live.get(key)
            if rec is not None:
                rec["last"] = ev
                rec["uses"] += 1
        for key in ins.dst:
            live.pop(key, None)
        rec = None
        if ins.kind in ("vmem_load", "smem") and ev.op.info.get("tensor") in LOAD_CLASS:
            info = ev.op.info
            rec = {"class": LOAD_CLASS[info["tensor"]], "label": fmt_load(info), "kb": info["kb"]}
        elif ins.kind == "ds_read":
            rec = {"class": f"{ev.op.info['region']} operand", "label": fmt_lds(ev.op.info), "kb": ev.op.info.get("kb", [None])[0]}
        elif ins.kind == "mfma" and ev.info and ins.ops[3].file == "imm":
            rec = {"class": "partial", "label": f"P k{ev.info['kb']}", "kb": ev.info["kb"]}
        elif ins.kind == "mfma" and ev.info:
            src = live.get(ins.ops[3].regs()[0])
            if src is not None and src["class"] == "partial":
                for key in ins.dst:
                    live[key] = src
                continue
        if rec is None:
            continue
        rec.update(regs=ins.dst, define=ev, ready=(ev.op.complete, ev.op.cseq) if ev.op is not None else (ev.epoch, ev.seq), last=None, uses=0)
        records.append(rec)
        for key in ins.dst:
            live[key] = rec
    return records


def report(config, cfg, prog, emu):
    meta = prog.meta
    lines = [f"== {config}: {cfg['variant']} N={cfg['N']} K={cfg['K']} (KB={emu.KB}) preshuffle_b={int(cfg['preshuffle'])}  ({prog.path})"]
    lines.append(f"resources: VGPR {meta.get('vgpr_count')} (spill {meta.get('vgpr_spill_count')}), AGPR {meta.get('agpr_count')}, "
                 f"SGPR {meta.get('sgpr_count')} (spill {meta.get('sgpr_spill_count')}), scratch {meta.get('private_segment_fixed_size')} B, "
                 f"LDS {meta.get('group_segment_fixed_size')} B")
    lines.append(f"emulated: workgroup {emu.block} of M={emu.M}, C tile rows {emu.row0}..{emu.row0 + TILE - 1} cols {emu.col0}..{emu.col0 + TILE - 1}; "
                 f"8 waves, {emu.steps} instructions, {emu.epoch} barrier epochs")
    counts = [w.barriers for w in emu.waves]
    lines.append(f"barriers: {counts[0]} s_barrier per wave (all waves {'equal' if len(set(counts)) == 1 else counts}); group-conditional: "
                 + ", ".join(f"line {line} (waves 0-3 x{a}, waves 4-7 x{b})" for line, a, b in group_barriers(emu, prog)))
    first = {wid: min(ev.epoch for ev in emu.waves[wid].trace if ev.info) for wid in (0, 4)}
    lines.append(f"stagger: first MFMA in epoch {first[0]} (waves 0-3) and {first[4]} (waves 4-7): waves 4-7 trail by {first[4] - first[0]} barrier")
    vm = sum(1 for h in emu.hazards if h[3].startswith("v"))
    lines.append(f"registers: {len(emu.hazards)} async hazards ({vm} VGPR, {len(emu.hazards) - vm} SGPR/AGPR) over all VMEM/LDS/SMEM loads incl. inline-asm s_buffer_load")
    st = emu.lds.stats
    lines.append(f"LDS: {len(emu.lds.ops)} ds ops of 8 waves checked per dword; {len(emu.lds.violations)} RAW/WAR/WAW violations, "
                 f"{emu.lds.uninit} reads of unwritten LDS")
    for table, cross, what in ((st, True, "across groups, all epochs"), (emu.lds.steady, True, f"across groups, epoch >= {STEADY_EPOCH}"),
                               (st, False, "within a group, all epochs")):
        parts = []
        for kind in ("RAW", "WAR", "WAW"):
            items = [f"{r}{p}:{table[(kind, r, p, cross)][1]}" for r in "AB" for p in range(4) if (kind, r, p, cross) in table]
            if items:
                parts.append(f"{kind} " + " ".join(items))
        lines.append(f"  min slack in barriers, {what}: " + ("; ".join(parts) or "-"))
    lines.append(f"data path: {emu.mfma_count} MFMA operand checks, {emu.fma_count} FMA retirements, {emu.store_count} BF16x2 stores "
                 f"({emu.bits_checked} BF16 values bit-checked on synthetic accumulators); "
                 f"C tile covered {'exactly once' if not emu.failures.get('store') else 'INCORRECTLY'}")
    lines.append(f"LDS writes: {len(emu.lds.collisions)} ds_write instructions with lanes writing the same dword")
    ages = [(f[0] - b[0]) for wid, b, f in emu.partial_age if wid == 0 and b]
    if ages:
        lines.append(f"partials (wave 0): retired {min(ages)}..{max(ages)} epochs after their last MFMA")
    for note in emu.notes:
        lines.append(f"note: {note}")
    kb = min(2, emu.KB - 1)
    lines.append(f"K block {kb}, wave 0 (group 0; group 1 runs every epoch one barrier later):")
    lines += stage_table(emu, prog, 0, kb)
    for category, messages in sorted(emu.failures.items()):
        for message in messages[:8]:
            lines.append(f"  FAIL [{category}] {message}")
        if len(messages) > 8:
            lines.append(f"  FAIL [{category}] ... {len(messages) - 8} more")
    lines.append("RESULT: " + ("PASS" if not emu.failures else f"FAIL ({sum(map(len, emu.failures.values()))})"))
    return lines


@dataclass
class Result:
    config: str
    cfg: dict
    prog: Program
    emu: Emulator
    lines: list

    @property
    def failed(self):
        return bool(self.emu.failures)


def audit(config, isa_path, cfg=None, mutate=None, M=EMU_M, block=EMU_BLOCK):
    cfg = cfg or CONFIGS[config]
    prog = parse_isa(isa_path)
    if mutate is not None:
        prog = mutate(prog)
    emu = Emulator(prog, cfg, M, block)
    try:
        emu.run()
    except RuntimeError as error:
        emu.fail("emulation", str(error))
        return Result(config, cfg, prog, emu, [f"== {config}", f"  FAIL [emulation] {error}", "RESULT: FAIL"])
    return Result(config, cfg, prog, emu, report(config, cfg, prog, emu))


# ---------------------------------------------------------------------------------------------------------------
# Mutations: each injects one schedule defect into the real ISA; the audit must report it.


def _edit(prog, index, remove=0, insert=()):
    """Replace prog.instrs[index:index + remove] by `insert`; labels at or after `index` move with the code after it."""
    instrs = prog.instrs[:index] + list(insert) + prog.instrs[index + remove:]
    delta = len(insert) - remove
    labels = {name: (pos + delta if pos >= index + remove else (index if pos > index else pos)) for name, pos in prog.labels.items()}
    return Program(prog.path, instrs, labels, prog.meta)


def k_loop(prog):
    """(head, back-edge) of the backward branch whose body holds the most MFMAs."""
    best = None
    for i, ins in enumerate(prog.instrs):
        if ins.kind == "branch" and ins.ops and ins.ops[0].file == "label" and prog.labels[ins.ops[0].value] <= i:
            head = prog.labels[ins.ops[0].value]
            count = sum(x.kind == "mfma" for x in prog.instrs[head:i])
            if best is None or count > best[2]:
                best = (head, i, count)
    assert best, "no K loop"
    return best[0], best[1]


def conditional_barriers(prog):
    """s_barrier guarded by `s_cbranch_scc1 L; s_barrier; L:` (executed by one wave group only)."""
    return [i for i, ins in enumerate(prog.instrs) if ins.kind == "barrier" and i and prog.instrs[i - 1].mnem == "s_cbranch_scc1"
            and prog.labels.get(prog.instrs[i - 1].ops[0].value) == i + 1]


def _lag2(prog):
    for i in reversed(conditional_barriers(prog)):
        prog = _edit(prog, i, 1, [prog.instrs[i], make_instr(prog.instrs[i].line, "s_barrier")])
    return prog


def _no_stagger(prog):
    for i in reversed(conditional_barriers(prog)):
        prog = _edit(prog, i, 1)
    return prog


def _drop_lgkm(prog):
    head, back = k_loop(prog)
    i = max(i for i in range(head, back) if prog.instrs[i].kind == "waitcnt" and prog.instrs[i].mods.get("lgkmcnt") == 0
            and prog.instrs[i + 1].kind == "barrier")
    return _edit(prog, i, 1)


def _drop_vm(prog):
    head, back = k_loop(prog)
    i = next(i for i in range(head, back) if prog.instrs[i].kind == "waitcnt" and "vmcnt" in prog.instrs[i].mods
             and prog.instrs[i + 1].kind == "ds_write")
    return _edit(prog, i, 1)


def _early_write(prog):
    head, back = k_loop(prog)
    barrier = [i for i in range(head, back) if prog.instrs[i].kind == "barrier"][1]
    write = next(i for i in range(barrier, back) if prog.instrs[i].kind == "ds_write")
    start = write - 1 if prog.instrs[write - 1].kind == "waitcnt" else write
    moved = prog.instrs[start:write + 1]
    prog = _edit(prog, start, len(moved))
    return _edit(prog, barrier, 0, moved)


def _early_use(prog):
    head, back = k_loop(prog)
    i = next((i for i in range(head, back) if prog.instrs[i].mnem.startswith("s_buffer_load")), None)
    if i is None:
        i = next(i for i in range(head, back) if prog.instrs[i].mnem == "buffer_load_dword")
    dst = prog.instrs[i].ops[0]
    reg = f"{dst.file}{dst.start}"
    return _edit(prog, i + 1, 0, [make_instr(prog.instrs[i].line, f"{'s_mov_b32' if dst.file == 's' else 'v_mov_b32_e32'} {reg}, {reg}")])


def _wrong_packet(prog):
    head, back = k_loop(prog)
    i = next(i for i in range(head, back) if prog.instrs[i].kind == "ds_read" and int(prog.instrs[i].mods.get("offset", "0"), 0) < 3 * PACKET_BYTES)
    ins = prog.instrs[i]
    offset = int(ins.mods.get("offset", "0"), 0) + PACKET_BYTES
    text = re.sub(r"\s+offset:\S+", "", ins.text) + f" offset:{offset}"
    return _edit(prog, i, 1, [make_instr(ins.line, text)])


def _lane_collision(prog):
    head, back = k_loop(prog)
    i = next(i for i in range(head, back) if prog.instrs[i].kind == "ds_write")
    reg = f"v{prog.instrs[i].ops[0].start}"
    return _edit(prog, i, 0, [make_instr(prog.instrs[i].line, f"v_and_b32_e32 {reg}, 0xffffff7f, {reg}")])


def _wrong_pack(prog):
    """Swap the halfwords of the epilogue's BF16x2 packing (v_perm selector or v_and_or mask held in an SGPR)."""
    head, back = k_loop(prog)
    for i in range(back, len(prog.instrs)):
        ins = prog.instrs[i]
        if ins.mnem in ("v_perm_b32", "v_and_or_b32"):
            op = ins.ops[3] if ins.mnem == "v_perm_b32" else ins.ops[2]
            if op.file == "s":
                value = 0x03020706 if ins.mnem == "v_perm_b32" else 0x0000FFFF
                return _edit(prog, i, 0, [make_instr(ins.line, f"s_mov_b32 s{op.start}, {value:#x}")])
    raise AssertionError("no BF16 packing instruction with an SGPR selector")


MUTATIONS = {
    "waves 4-7 trail by 2 barriers": (_lag2, True),
    "no stagger: both groups in phase": (_no_stagger, False),
    "drop the loop's last lgkmcnt(0) before s_barrier": (_drop_lgkm, True),
    "drop a vmcnt wait in front of ds_write": (_drop_vm, True),
    "ds_write one barrier early": (_early_write, True),
    "read a scale load's register right after issue": (_early_use, True),
    "ds_read 8 KiB off (wrong packet)": (_wrong_packet, True),
    "two lanes of one ds_write hit the same dword": (_lane_collision, True),
    "BF16 packing puts the wrong halfword": (_wrong_pack, True),
}


def isa_path(isa_dir, config):
    return os.path.join(isa_dir, config, "gemm_kernel_0", "22_final_isa.s")


def self_test(isa_dir, configs):
    ok = True
    for config in configs:
        for name, (mutate, expect_fail) in MUTATIONS.items():
            res = audit(config, isa_path(isa_dir, config), mutate=mutate)
            ok &= res.failed == expect_fail
            found = {category: len(messages) for category, messages in sorted(res.emu.failures.items())}
            print(f"self-test {config:22s} {name:50s}: {'FAIL' if res.failed else 'PASS'} "
                  f"(expected {'FAIL' if expect_fail else 'PASS'}) {found}")
    return ok


def dump_one(config, out_dir):
    """Compile one config with FlyDSL IR dumps; COMPILE_ONLY builds the kernel without launching it."""
    cfg = CONFIGS[config]
    os.environ.update(COMPILE_ONLY="1", FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=os.path.join(out_dir, config),
                      FLYDSL_RUNTIME_CACHE_DIR=tempfile.mkdtemp(prefix="gemm_audit_cache_"))
    import importlib

    import flydsl.compiler as flyc
    import flydsl.expr as fx

    name = f"gemm_a8w8_blockscale_gfx942_2x4_lds_{cfg['variant']}"
    launch = getattr(importlib.import_module(f"pyhip.contrib.flydsl.{name}"), f"compile_{name}")(cfg["N"], cfg["K"], cfg["preshuffle"])
    null = [flyc.from_c_void_p(t, 0) for t in (fx.Int32, fx.Int32, fx.Int32, fx.Float32, fx.Float32)]
    launch(*null, EMU_M)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", help="compile every config with IR dumps into this directory")
    parser.add_argument("--dump-one", nargs=2, metavar=("CONFIG", "DIR"), help=argparse.SUPPRESS)
    parser.add_argument("--isa", help="directory holding <config>/gemm_kernel_0/22_final_isa.s")
    parser.add_argument("--config", action="append", choices=sorted(CONFIGS), help="audit only these configs")
    parser.add_argument("--report", help="also write the text report to this file")
    parser.add_argument("--self-test", action="store_true", help="inject schedule defects into the KB=5 kernels and expect failures")
    args = parser.parse_args()
    if args.dump_one:
        dump_one(*args.dump_one)
        return
    configs = args.config or list(CONFIGS)
    if args.dump:
        for config in configs:
            subprocess.run([sys.executable, os.path.abspath(__file__), "--dump-one", config, args.dump], check=True, stdout=subprocess.DEVNULL)
        args.isa = args.isa or args.dump
    if not args.isa:
        parser.error("pass --isa or --dump")
    if args.self_test:
        sys.exit(0 if self_test(args.isa, [c for c in configs if "_k640_" in c] or configs) else 1)
    lines, failed = [], False
    for config in configs:
        res = audit(config, isa_path(args.isa, config))
        print("\n".join(res.lines) + "\n", flush=True)
        lines += res.lines + [""]
        failed |= res.failed
    if args.report:
        with open(args.report, "w") as f:
            f.write("\n".join(lines))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
