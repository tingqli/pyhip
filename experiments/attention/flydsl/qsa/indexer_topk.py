"""Exact per-row top-512 of QSA indexer logits for gfx942 (FlyDSL), expanded to request-local token ids.

One wave owns one query row (four rows per 256-thread CTA, no CTA barriers); block j is lane
j % 64 of slot j / 64. Every pass sweeps wave-uniform slot groups with UNROLL independent
coalesced loads in flight. Fast path: one 256-bin histogram of a monotone linear digit between
the row min and max; blocks above the threshold bin are kept and the <= 64 threshold-bin keys
are ranked exactly. Otherwise an ordered-key radix select finds the exact 512th key. Ties keep
the lowest block ids. Row order is deterministic: kept blocks outside the resolving bin
ascending, then the resolving-bin picks ascending (consumers treat order as unspecified).
Output: 4 tokens per block, causal tail, -1.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr import gpu, rocdl

from ..mha._common import _buffer, _uniform

WAVES, TOPK, RATIO = 4, 512, 4
WIDTH = TOPK * RATIO + RATIO - 1
BINS, CANDIDATES, UNROLL = 256, 64, 8
# Per wave: 257 histogram ints (sink bin last; later 64 candidate keys, then their ids), and the
# 512 chosen uint16 block ids.
SCRATCH, CHOSEN = 1040, 1024
KEYS, IDS = 0, 64 * 4
SIGN = -(1 << 31)


def _lds(address):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.Int32(address).ir_value())


def _load(address):
    return fx.Int32(llvm.load(fx.Int32.ir_type, _lds(address), alignment=4))


def _store(address, value):
    llvm.store(fx.Int32(value).ir_value(), _lds(address), alignment=4)


def _load16(address):
    value = llvm.load(ir.IntegerType.get_signless(16), _lds(address), alignment=2)
    return fx.Int32(arith.extui(fx.Int32.ir_type, value))


def _store16(address, value):
    llvm.store(arith.trunci(ir.IntegerType.get_signless(16), fx.Int32(value).ir_value()), _lds(address),
               alignment=2)


def _add(address):
    llvm.atomicrmw(llvm.AtomicBinOp.add, _lds(address), fx.Int32(1).ir_value(), llvm.AtomicOrdering.monotonic,
                   syncscope="wavefront", alignment=4)


def _put(output, index, value):
    rocdl.raw_ptr_buffer_store(fx.Int32(value).ir_value(), output, (fx.Int32(index) * 4).ir_value(),
                               fx.Int32(0).ir_value())


def _sync():
    llvm.fence(llvm.AtomicOrdering.seq_cst, syncscope="wavefront")
    rocdl.wave_barrier()


def _intrinsic(name, dtype, *args):
    return dtype(llvm.call_intrinsic(dtype.ir_type, name, [a.ir_value() for a in args], [], []))


def _ballot(predicate):
    return fx.Int64(rocdl.ballot(fx.Int64.ir_type, predicate.ir_value()))


def _below(mask):
    low = rocdl.mbcnt_lo(fx.Int32.ir_type, fx.Int32(mask).ir_value(), fx.Int32(0).ir_value())
    return fx.Int32(rocdl.mbcnt_hi(fx.Int32.ir_type, fx.Int32(mask >> 32).ir_value(), low))


def _popcount(mask):
    return fx.Int32(_intrinsic("llvm.ctpop.i64", fx.Int64, mask))


def _readlane(value, lane):
    return fx.Int32(rocdl.readlane(fx.Int32.ir_type, fx.Int32(value).ir_value(), fx.Int32(lane).ir_value()))


def _minnum(a, b):
    return _intrinsic("llvm.minnum.f32", fx.Float32, a, b)


def _maxnum(a, b):
    return _intrinsic("llvm.maxnum.f32", fx.Float32, a, b)


def _umin(a, b):
    return _intrinsic("llvm.umin.i32", fx.Int32, a, b)


def _umax(a, b):
    return _intrinsic("llvm.umax.i32", fx.Int32, a, b)


def _ugt(a, b):
    return (a ^ SIGN) > (b ^ SIGN)


def _uge(a, b):
    return (a ^ SIGN) >= (b ^ SIGN)


def _order_key(x):
    bits = fx.Float32(x).bitcast(fx.Int32)
    return (bits < 0).select(~bits, bits | SIGN)


def _finite(x):
    return (fx.Float32(x).bitcast(fx.Int32) & 0x7F800000) != 0x7F800000


def _reduce(value, op):
    for offset in (32, 16, 8, 4, 2, 1):
        value = op(value, value.shuffle_xor(offset, 64))
    return value


def _group(logits, base, group, lane):
    # UNROLL coalesced loads of slot group `group`; reads past the chunk return 0 (masked later).
    offset = base + (group * (64 * UNROLL) + lane) * 4
    return fx.Vector.from_elements(
        [fx.Float32(rocdl.raw_ptr_buffer_load(fx.Float32.ir_type, logits, (offset + i * 256).ir_value(),
                                              fx.Int32(0).ir_value(), fx.Int32(0).ir_value()))
         for i in range(UNROLL)], fx.Float32)


def _slots(value, group, lane, count):
    for i in range(UNROLL):
        j = (group * UNROLL + i) * 64 + lane
        yield j, value[i], j < count


def _digit(x, low, scale):
    digit = ((x - low) * scale).to(fx.Int32)
    return (digit < BINS - 1).select(digit, fx.Int32(BINS - 1))


def _clear(bins, lane):
    for i in range(0, BINS + 1, 64):
        _store(bins + (i + lane if i < BINS else i) * 4, 0)


def _threshold(bins, need, lane):
    """(bin, count above it, its size) for the bin holding the need-th largest element."""
    owned = [_load(bins + (lane * (BINS // 64) + i) * 4) for i in range(BINS // 64)]
    inclusive = owned[0] + owned[1] + owned[2] + owned[3]
    for offset in (1, 2, 4, 8, 16, 32):
        source = ((lane - offset) & 63) * 4
        other = fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, source.ir_value(), inclusive.ir_value()))
        inclusive = (lane >= offset).select(inclusive + other, inclusive)
    higher = _readlane(inclusive, 63) - inclusive
    found, found_above, found_size = fx.Int32(-1), fx.Int32(0), fx.Int32(0)
    for i in reversed(range(BINS // 64)):
        hit = (higher < need) & (higher + owned[i] >= need)
        found = hit.select(lane * (BINS // 64) + i, found)
        found_above = hit.select(higher, found_above)
        found_size = hit.select(owned[i], found_size)
        higher = higher + owned[i]
    source = fx.Int32(_intrinsic("llvm.cttz.i64", fx.Int64, _ballot(found >= 0), fx.Boolean(False)))
    return _readlane(found, source), _readlane(found_above, source), _readlane(found_size, source)


def _minmax_group(value, group, lane, count, low, high, nan):
    for _, x, live in _slots(value, group, lane, count):
        low = _minnum(low, live.select(x, fx.Float32(float("inf"))))
        high = _maxnum(high, live.select(x, fx.Float32(float("-inf"))))
        nan = nan | (live & (x != x)).select(fx.Int32(1), fx.Int32(0))
    return low, high, nan


@flyc.jit
def _sweep_minmax(logits, base, count, lane):
    low, high, nan = fx.Float32(float("inf")), fx.Float32(float("-inf")), fx.Int32(0)
    groups = (count + 64 * UNROLL - 1) // (64 * UNROLL)
    following = _group(logits, base, fx.Int32(0), lane)
    for index in range(fx.Int32(0), groups, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < groups:
            following = _group(logits, base, group + 1, lane)
        low, high, nan = _minmax_group(value, group, lane, count, low, high, nan)
    return low, high, nan


def _hist_group(value, group, lane, count, bins, low, scale):
    for _, x, live in _slots(value, group, lane, count):
        _add(bins + live.select(_digit(x, low, scale), fx.Int32(BINS)) * 4)


@flyc.jit
def _sweep_hist(logits, base, count, lane, bins, low, scale):
    groups = (count + 64 * UNROLL - 1) // (64 * UNROLL)
    following = _group(logits, base, fx.Int32(0), lane)
    for index in range(fx.Int32(0), groups, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < groups:
            following = _group(logits, base, group + 1, lane)
        _hist_group(value, group, lane, count, bins, low, scale)


def _gather_group(value, group, lane, count, bins, chosen, low, scale, bin, kept, gathered):
    for j, x, live in _slots(value, group, lane, count):
        digit = live.select(_digit(x, low, scale), fx.Int32(-1))
        over, inside = digit > bin, digit == bin
        over_mask, inside_mask = _ballot(over), _ballot(inside)
        _keep(over, chosen, kept, over_mask, j)
        _collect(inside, bins, gathered, inside_mask, x, j)
        kept = kept + _popcount(over_mask)
        gathered = gathered + _popcount(inside_mask)
    return kept, gathered


@flyc.jit
def _keep(condition, chosen, base, mask, j):
    if condition:
        _store16(chosen + (base + _below(mask)) * 2, j)


@flyc.jit
def _collect(condition, bins, base, mask, x, j):
    if condition:
        slot = base + _below(mask)
        _store(bins + KEYS + slot * 4, _order_key(x))
        _store(bins + IDS + slot * 4, j)


@flyc.jit
def _keep_tie(condition, chosen, tied_base, ties, mask, need, j):
    if condition:
        rank = ties + _below(mask)
        if rank < need:
            _store16(chosen + (tied_base + rank) * 2, j)


@flyc.jit
def _sweep_gather(logits, base, count, lane, bins, chosen, low, scale, bin):
    kept, gathered = fx.Int32(0), fx.Int32(0)
    groups = (count + 64 * UNROLL - 1) // (64 * UNROLL)
    following = _group(logits, base, fx.Int32(0), lane)
    for index in range(fx.Int32(0), groups, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < groups:
            following = _group(logits, base, group + 1, lane)
        kept, gathered = _gather_group(value, group, lane, count, bins, chosen, low, scale, bin, kept, gathered)


@flyc.jit
def _rank(bins, chosen, lane, size, above):
    # Candidates are in ascending id order: rank by key, ties by lower id.
    key = _load(bins + KEYS + lane * 4)
    rank = fx.Int32(0)
    for index in range(fx.Int32(0), size, fx.Int32(1)):
        m = fx.Int32(index)
        other = _load(bins + KEYS + m * 4)
        rank = rank + (_ugt(other, key) | ((other == key) & (m < lane))).select(fx.Int32(1), fx.Int32(0))
    rank = (lane < size).select(rank, fx.Int32(CANDIDATES))
    picked = rank < TOPK - above
    _keep_id(picked, chosen, above, _ballot(picked), bins + IDS + lane * 4)


@flyc.jit
def _keep_id(condition, chosen, base, mask, source):
    if condition:
        _store16(chosen + (base + _below(mask)) * 2, _load(source))


def _keys_group(value, group, lane, count, high_key, low_key):
    for _, x, live in _slots(value, group, lane, count):
        key = _order_key(x)
        high_key = _umax(high_key, live.select(key, fx.Int32(0)))
        low_key = _umin(low_key, live.select(key, fx.Int32(-1)))
    return high_key, low_key


@flyc.jit
def _sweep_keys(logits, base, count, lane):
    high_key, low_key = fx.Int32(0), fx.Int32(-1)
    groups = (count + 64 * UNROLL - 1) // (64 * UNROLL)
    following = _group(logits, base, fx.Int32(0), lane)
    for index in range(fx.Int32(0), groups, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < groups:
            following = _group(logits, base, group + 1, lane)
        high_key, low_key = _keys_group(value, group, lane, count, high_key, low_key)
    return high_key, low_key


def _radix_group(value, group, lane, count, bins, prefix, prefix_mask, shift, width):
    for _, x, live in _slots(value, group, lane, count):
        key = _order_key(x)
        hit = live & ((key & prefix_mask) == prefix)
        _add(bins + hit.select((key >> shift) & ((fx.Int32(1) << width) - 1), fx.Int32(BINS)) * 4)


@flyc.jit
def _sweep_radix(logits, base, count, lane, bins, prefix, prefix_mask, shift, width):
    groups = (count + 64 * UNROLL - 1) // (64 * UNROLL)
    following = _group(logits, base, fx.Int32(0), lane)
    for index in range(fx.Int32(0), groups, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < groups:
            following = _group(logits, base, group + 1, lane)
        _radix_group(value, group, lane, count, bins, prefix, prefix_mask, shift, width)


def _select_group(value, group, lane, count, chosen, prefix, prefix_mask, whole, need, kept, ties):
    tied_base = TOPK - need
    for j, x, live in _slots(value, group, lane, count):
        key = _order_key(x)
        over = live & (whole != 0).select(_uge(key & prefix_mask, prefix), _ugt(key, prefix))
        tie = (whole == 0) & live & (key == prefix)
        over_mask, tie_mask = _ballot(over), _ballot(tie)
        _keep(over, chosen, kept, over_mask, j)
        _keep_tie(tie, chosen, tied_base, ties, tie_mask, need, j)
        kept = kept + _popcount(over_mask)
        ties = ties + _popcount(tie_mask)
    return kept, ties


@flyc.jit
def _sweep_select(logits, base, count, lane, chosen, prefix, prefix_mask, whole, need):
    kept, ties = fx.Int32(0), fx.Int32(0)
    groups = (count + 64 * UNROLL - 1) // (64 * UNROLL)
    following = _group(logits, base, fx.Int32(0), lane)
    for index in range(fx.Int32(0), groups, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < groups:
            following = _group(logits, base, group + 1, lane)
        kept, ties = _select_group(value, group, lane, count, chosen, prefix, prefix_mask, whole, need, kept, ties)


@flyc.jit
def _radix(logits, base, count, lane, bins, chosen):
    # Ordered-key radix: live keys share bits above `top`; <= 8-bit digits refine the 512th key.
    high_key, low_key = _sweep_keys(logits, base, count, lane)
    high_key, low_key = _reduce(high_key, _umax), _reduce(low_key, _umin)
    differ = high_key ^ low_key
    top = (differ == 0).select(fx.Int32(-1), 31 - _intrinsic("llvm.ctlz.i32", fx.Int32, differ, fx.Boolean(False)))
    shifted = (fx.Int32(2) << (top & 31)) - 1
    prefix_mask = (top < 0).select(fx.Int32(-1), (top == 31).select(fx.Int32(0), ~shifted))
    prefix = high_key & prefix_mask
    need, whole = fx.Int32(TOPK), fx.Int32(0)
    while (top >= 0) & (whole == 0):
        width = (top + 1 < 8).select(top + 1, fx.Int32(8))
        shift = top - width + 1
        _sync()
        _clear(bins, lane)
        _sync()
        _sweep_radix(logits, base, count, lane, bins, prefix, prefix_mask, shift, width)
        _sync()
        bin, above, size = _threshold(bins, need, lane)
        need = need - above
        prefix = prefix | (bin << shift)
        prefix_mask = prefix_mask | (((fx.Int32(1) << width) - 1) << shift)
        whole = (size == need).select(fx.Int32(1), fx.Int32(0))
        top = (size == need).select(top, shift - 1)
    # Keys above the 512th key ascending, then the kept ties ascending.
    _sweep_select(logits, base, count, lane, chosen, prefix, prefix_mask, whole, need)


@flyc.jit
def _select(logits, base, count, lane, bins, chosen, output):
    low, high, nan = _sweep_minmax(logits, base, count, lane)
    low, high = _reduce(low, _minnum), _reduce(high, _maxnum)
    scale = fx.Float32(float(BINS)) / (high - low)
    done = fx.Int32(0)
    if ((_ballot(nan != 0) == 0) & _finite(low) & _finite(high) & (high > low) & _finite(scale)):
        # x <= y implies digit(x) <= digit(y): bins partition blocks by value.
        _clear(bins, lane)
        _sync()
        _sweep_hist(logits, base, count, lane, bins, low, scale)
        _sync()
        bin, above, size = _threshold(bins, fx.Int32(TOPK), lane)
        if size <= CANDIDATES:
            _sync()
            _sweep_gather(logits, base, count, lane, bins, chosen, low, scale, bin)
            _sync()
            _rank(bins, chosen, lane, size, above)
            done = fx.Int32(1)
    if done == 0:
        _radix(logits, base, count, lane, bins, chosen)
    _sync()
    for i in range(TOPK * RATIO // 64):
        token = lane + i * 64
        value = _load16(chosen + (token >> 2) * 2) * RATIO + (token & (RATIO - 1))
        _put(output, token, value)


@flyc.jit
def _row(LOGITS, stride, row0, POSITIONS, ROW_INFO, OUT, VALID, local, lane, bins, chosen):
    row = row0 + local
    expected, length = _uniform(ROW_INFO[2 * row]), _uniform(ROW_INFO[2 * row + 1])
    if (lane == 0) & (fx.Int64(POSITIONS[row]) != fx.Int64(expected)):
        VALID[0] = fx.Int32(0)
    visible = expected + 1
    count = visible // RATIO
    count = (count < length // RATIO).select(count, length // RATIO)
    output = _buffer(fx.make_view(fx.get_iter(OUT) + fx.Int64(row) * WIDTH, fx.make_layout(1, 1)), WIDTH * 4)
    blocks = count
    if count <= TOPK:
        for token in range(lane, count * RATIO, fx.Int32(64)):
            _put(output, token, token)
    else:
        blocks = fx.Int32(TOPK)
        logits = _buffer(LOGITS, fx.Int32(0x7FFFFFF0))
        _select(logits, local * stride * 4, count, lane, bins, chosen, output)
    tail_start = visible // RATIO * RATIO
    tail_count = visible - tail_start
    for index in range(blocks * RATIO + lane, fx.Int32(WIDTH), fx.Int32(64)):
        column = fx.Int32(index)
        offset = column - blocks * RATIO
        token = tail_start + offset
        value = ((offset < RATIO - 1) & (offset < tail_count) & (token < length)).select(token, fx.Int32(-1))
        _put(output, column, value)


@flyc.kernel(name="qsa_indexer_topk", known_block_size=[256, 1, 1])
def _kernel(LOGITS: fx.Tensor, stride: fx.Int32, row0: fx.Int32, rows: fx.Int32, POSITIONS: fx.Tensor,
            ROW_INFO: fx.Tensor, OUT: fx.Tensor, VALID: fx.Tensor):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, WAVES * (SCRATCH + CHOSEN), 16]).peek()
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(WAVES * (SCRATCH + CHOSEN), 1)))))
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    local = _uniform(fx.Int32(gpu.block_id("x")) * WAVES + wave)
    if local < rows:
        _row(LOGITS, stride, row0, POSITIONS, ROW_INFO, OUT, VALID, local, lane, shared + wave * SCRATCH,
             shared + WAVES * SCRATCH + wave * CHOSEN)


@flyc.jit
def _launch(LOGITS: fx.Tensor, stride: fx.Int32, row0: fx.Int32, rows: fx.Int32, POSITIONS: fx.Tensor,
            ROW_INFO: fx.Tensor, OUT: fx.Tensor, VALID: fx.Tensor, ctas: fx.Int32, stream: fx.Stream):
    _kernel(LOGITS, stride, row0, rows, POSITIONS, ROW_INFO, OUT, VALID).launch(
        grid=(ctas, 1, 1), block=(256, 1, 1), stream=stream)


_COMPILED = {}


def launch(logits, stride, row0, rows, positions, row_info, out, valid):
    """Rows [row0, row0 + rows) of out from logits rows [0, rows) (row stride `stride`); row_info holds
    (query position, sequence length) per row; valid is cleared if positions disagree."""
    stream = torch.cuda.current_stream(logits.device)
    args = (logits.view(-1), stride, row0, rows, positions, row_info.view(-1), out.view(-1), valid,
            -(-rows // WAVES), stream)
    with torch.cuda.device(logits.device):
        compiled = _COMPILED.get(logits.device)
        if compiled is None:
            _COMPILED[logits.device] = flyc.compile(_launch, *args)
        else:
            compiled(*args)
