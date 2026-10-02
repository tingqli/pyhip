"""Exact per-row top-512 of QSA indexer logits for gfx942 (FlyDSL), expanded to request-local token ids.

Prefill: one wave owns one query row (four rows per 256-thread CTA, no CTA barriers), and the logits
kernel supplies the row's (min, max) bits. Decode: one 8-wave CTA owns a row; waves take contiguous
slot-group ranges and per-wave histogram counts place their outputs, so results match the one-wave
selection. Block j is lane j % 64 of slot j / 64; every pass sweeps wave-uniform slot groups with
UNROLL independent coalesced loads in flight. Fast path: one 256-bin histogram of a monotone linear
digit between the row min and max; blocks above the threshold bin are kept and the <= 64
threshold-bin keys are ranked exactly. Otherwise an ordered-key radix select finds the exact 512th
key. Ties keep the lowest block ids. The two ascending runs (kept, then resolving picks) are merged:
rows hold ascending blocks, 4 tokens each, then the causal tail, then -1.
"""

from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr import gpu, rocdl

from pyhip.codegen.flydsl.helpers import rocdl_aux
from pyhip.ops.mha.flydsl._common import _buffer, _min, _uniform

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
                               fx.Int32(0).ir_value(), aux=rocdl_aux(0))


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
                                              fx.Int32(0).ir_value(), aux=rocdl_aux(0)))
         for i in range(UNROLL)], fx.Float32)


def _groups(count):
    return (count + 64 * UNROLL - 1) // (64 * UNROLL)


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
    return _threshold_owned([_load(bins + (lane * (BINS // 64) + i) * 4) for i in range(BINS // 64)], need, lane)


def _threshold_owned(owned, need, lane):
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
def _sweep_minmax(logits, base, count, lane, first, last):
    low, high, nan = fx.Float32(float("inf")), fx.Float32(float("-inf")), fx.Int32(0)
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
            following = _group(logits, base, group + 1, lane)
        low, high, nan = _minmax_group(value, group, lane, count, low, high, nan)
    return low, high, nan


def _hist_group(value, group, lane, count, bins, low, scale):
    for _, x, live in _slots(value, group, lane, count):
        _add(bins + live.select(_digit(x, low, scale), fx.Int32(BINS)) * 4)


@flyc.jit
def _sweep_hist(logits, base, count, lane, bins, low, scale, first, last):
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
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
def _sweep_gather(logits, base, count, lane, bins, chosen, low, scale, bin, first, last, kept, gathered):
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
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
def _sweep_keys(logits, base, count, lane, first, last):
    high_key, low_key = fx.Int32(0), fx.Int32(-1)
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
            following = _group(logits, base, group + 1, lane)
        high_key, low_key = _keys_group(value, group, lane, count, high_key, low_key)
    return high_key, low_key


def _radix_group(value, group, lane, count, bins, prefix, prefix_mask, shift, width):
    for _, x, live in _slots(value, group, lane, count):
        key = _order_key(x)
        hit = live & ((key & prefix_mask) == prefix)
        _add(bins + hit.select((key >> shift) & ((fx.Int32(1) << width) - 1), fx.Int32(BINS)) * 4)


@flyc.jit
def _sweep_radix(logits, base, count, lane, bins, prefix, prefix_mask, shift, width, first, last):
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
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
def _sweep_select(logits, base, count, lane, chosen, prefix, prefix_mask, whole, need, first, last, kept, ties):
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
            following = _group(logits, base, group + 1, lane)
        kept, ties = _select_group(value, group, lane, count, chosen, prefix, prefix_mask, whole, need, kept, ties)


def _key_range(high_key, low_key):
    """(top differing bit or -1, prefix mask, prefix) shared by all live ordered keys."""
    differ = high_key ^ low_key
    top = (differ == 0).select(fx.Int32(-1), 31 - _intrinsic("llvm.ctlz.i32", fx.Int32, differ, fx.Boolean(False)))
    shifted = (fx.Int32(2) << (top & 31)) - 1
    prefix_mask = (top < 0).select(fx.Int32(-1), (top == 31).select(fx.Int32(0), ~shifted))
    return top, prefix_mask, high_key & prefix_mask


def _space(waves, wave, lane, hist, cand, chosen, merged, hist_base=None, red=None):
    """LDS of one row's selection, shared by `waves` waves (a trace-time int; 1 = wave-private row).

    hist: this wave's 257-int histogram (wave w at hist_base + w * HIST when waves > 1); cand: 64
    candidate keys, then their ids (may alias hist); chosen: 512 uint16 block ids; merged: 1 KiB of
    finished scratch for the ascending merge; red: 4 reduction words per wave (waves > 1).
    """
    return SimpleNamespace(waves=waves, wave=wave, lane=lane, thread=lane if waves == 1 else wave * 64 + lane,
                           threads=64 * waves, hist=hist, cand=cand, chosen=chosen, merged=merged,
                           hist_base=hist_base, red=red)


def _sync_of(row):
    return _sync if row.waves == 1 else _cta_sync


def _row_sync(row):
    _sync_of(row)()


def _span(row, count):
    """This wave's slot groups [first, last): all of them, or a contiguous 1/waves share."""
    groups = _groups(count)
    if row.waves == 1:
        return fx.Int32(0), groups
    per = (groups + row.waves - 1) // row.waves
    first = _min(row.wave * per, groups)
    return first, _min(first + per, groups)


def _exchange(row, values):
    """Per-wave values of every wave of the row (after one CTA barrier)."""
    for slot, value in enumerate(values):
        _store(row.red + row.wave * 16 + slot * 4, value)
    _cta_sync()
    return [[_load(row.red + w * 16 + slot * 4) for w in range(row.waves)] for slot in range(len(values))]


def _below_wave(values, wave):
    total = fx.Int32(0)
    for w, value in enumerate(values):
        total = total + (fx.Int32(w) < wave).select(value, fx.Int32(0))
    return total


def _bases(row, above, inside):
    """Where this wave's outputs start: exclusive prefix over the row's waves."""
    if row.waves == 1:
        return fx.Int32(0), fx.Int32(0)
    aboves, insides = _exchange(row, [above, inside])
    return _below_wave(aboves, row.wave), _below_wave(insides, row.wave)


def _row_minmax(row, low, high, nan):
    """Row (min, max, any NaN) from this wave's partial sweep."""
    low, high = _reduce(low, _minnum), _reduce(high, _maxnum)
    nan = (_ballot(nan != 0) != 0).select(fx.Int32(1), fx.Int32(0))
    if row.waves == 1:
        return low, high, nan
    lows, highs, nans = _exchange(row, [low.bitcast(fx.Int32), high.bitcast(fx.Int32), nan])
    low, high, nan = fx.Float32(float("inf")), fx.Float32(float("-inf")), fx.Int32(0)
    for value in lows:
        low = _minnum(low, value.bitcast(fx.Float32))
    for value in highs:
        high = _maxnum(high, value.bitcast(fx.Float32))
    for value in nans:
        nan = nan | value
    return low, high, nan


def _key_bounds(row, high_key, low_key):
    high_key, low_key = _reduce(high_key, _umax), _reduce(low_key, _umin)
    if row.waves == 1:
        return high_key, low_key
    highs, lows = _exchange(row, [high_key, low_key])
    high_key, low_key = fx.Int32(0), fx.Int32(-1)
    for value in highs:
        high_key = _umax(high_key, value)
    for value in lows:
        low_key = _umin(low_key, value)
    return high_key, low_key


def _row_threshold(row, need):
    """(bin, count above it, its size) of the need-th largest element over the row's waves."""
    if row.waves == 1:
        return _threshold(row.hist, need, row.lane)
    owned = []
    for i in range(BINS // 64):
        total = fx.Int32(0)
        for w in range(row.waves):
            total = total + _load(row.hist_base + w * HIST + (row.lane * (BINS // 64) + i) * 4)
        owned.append(total)
    return _threshold_owned(owned, need, row.lane)


def _split(row, bin):
    """This wave's (count above bin, count in bin)."""
    above = fx.Int32(0)
    for i in range(BINS // 64):
        index = row.lane * (BINS // 64) + i
        above = above + (index > bin).select(_load(row.hist + index * 4), fx.Int32(0))
    return _reduce(above, lambda a, b: a + b), _load(row.hist + bin * 4)


def _gather_bases(row, bin):
    if row.waves == 1:
        _sync()
        return fx.Int32(0), fx.Int32(0)
    return _bases(row, *_split(row, bin))


@flyc.jit
def _radix(logits, base, count, row):
    # Ordered-key radix: live keys share bits above `top`; <= 8-bit digits refine the 512th key.
    first, last = _span(row, count)
    high_key, low_key = _sweep_keys(logits, base, count, row.lane, first, last)
    top, prefix_mask, prefix = _key_range(*_key_bounds(row, high_key, low_key))
    need, whole = fx.Int32(TOPK), fx.Int32(0)
    # This wave's keys above the resolved prefix, and in its current bin (all live keys at first).
    mine = fx.Int32(0)
    inside = _min(last * (64 * UNROLL), count) - first * (64 * UNROLL)
    inside = (inside > 0).select(inside, fx.Int32(0))
    while (top >= 0) & (whole == 0):
        width = (top + 1 < 8).select(top + 1, fx.Int32(8))
        shift = top - width + 1
        _row_sync(row)
        _clear(row.hist, row.lane)
        _sync()
        _sweep_radix(logits, base, count, row.lane, row.hist, prefix, prefix_mask, shift, width, first, last)
        _row_sync(row)
        bin, above, size = _row_threshold(row, need)
        higher, inside = _split(row, bin)
        mine = mine + higher
        need = need - above
        prefix = prefix | (bin << shift)
        prefix_mask = prefix_mask | (((fx.Int32(1) << width) - 1) << shift)
        whole = (size == need).select(fx.Int32(1), fx.Int32(0))
        top = (size == need).select(top, shift - 1)
    # Keys above the 512th key ascending, then the kept ties ascending.
    _row_sync(row)
    kept, ties = _bases(row, mine + (whole != 0).select(inside, fx.Int32(0)), (whole != 0).select(fx.Int32(0), inside))
    _sweep_select(logits, base, count, row.lane, row.chosen, prefix, prefix_mask, whole, need, first, last,
                  kept, ties)
    return TOPK - need


def _max(a, b):
    return (a > b).select(a, b)


def _emit_ascending(output, chosen, merged, split, lane, threads, sync):
    # chosen[:split] (A) and chosen[split:] (B) are ascending. Merge path: each thread binary
    # searches where its output diagonal crosses the runs, merges its outputs into LDS `merged`
    # (1 KiB of finished scratch), and coalesced 16-byte stores then write four tokens per block.
    per = TOPK // threads
    diagonal = lane * per
    count = fx.Int32(TOPK) - split
    low = _max(diagonal - count, fx.Int32(0))
    high = _min(diagonal, split)
    for _ in range(10):
        active = low < high
        mid = (low + high) >> 1
        a = _load16(chosen + _min(mid, fx.Int32(TOPK - 1)) * 2)
        b = _load16(chosen + _min(_max(split + diagonal - mid - 1, fx.Int32(0)), fx.Int32(TOPK - 1)) * 2)
        right = a < b
        low = (active & right).select(mid + 1, low)
        high = active.select(right.select(high, mid), high)
    i, j = low, diagonal - low
    for k in range(per):
        a = (i < split).select(_load16(chosen + _min(i, fx.Int32(TOPK - 1)) * 2), fx.Int32(1 << 16))
        b = (j < count).select(_load16(chosen + _min(split + j, fx.Int32(TOPK - 1)) * 2), fx.Int32(1 << 16))
        first = a < b
        _store16(merged + (diagonal + k) * 2, first.select(a, b))
        i = first.select(i + 1, i)
        j = first.select(j, j + 1)
    sync()
    for k in range(TOPK // threads):
        index = lane + k * threads
        value = _load16(merged + index * 2)
        tokens = fx.Vector.from_elements([value * RATIO + r for r in range(RATIO)], fx.Int32)
        rocdl.raw_ptr_buffer_store(tokens.ir_value(), output, (index * (RATIO * 4)).ir_value(),
                                   fx.Int32(0).ir_value(), aux=rocdl_aux(0))


@flyc.jit
def _rank_row(row, size, above):
    if fx.const_expr(row.waves == 1):
        _rank(row.cand, row.chosen, row.lane, size, above)
    else:
        if row.wave == 0:
            _rank(row.cand, row.chosen, row.lane, size, above)


@flyc.jit
def _select(logits, base, count, row, low, high, nan):
    """Top-512 block ids of a row whose logits lie in [low, high] (nan: any NaN) into row.chosen, as
    two ascending runs; returns where the second run starts."""
    first, last = _span(row, count)
    scale = fx.Float32(float(BINS)) / (high - low)
    done = fx.Int32(0)
    split = fx.Int32(0)
    if ((nan == 0) & _finite(low) & _finite(high) & (high > low) & _finite(scale)):
        # x <= y implies digit(x) <= digit(y): bins partition blocks by value.
        _clear(row.hist, row.lane)
        _sync()
        _sweep_hist(logits, base, count, row.lane, row.hist, low, scale, first, last)
        _row_sync(row)
        bin, above, size = _row_threshold(row, fx.Int32(TOPK))
        if size <= CANDIDATES:
            kept, gathered = _gather_bases(row, bin)
            _sweep_gather(logits, base, count, row.lane, row.cand, row.chosen, low, scale, bin, first, last,
                          kept, gathered)
            _row_sync(row)
            _rank_row(row, size, above)
            done = fx.Int32(1)
            split = above
    if done == 0:
        split = _radix(logits, base, count, row)
    return split


def _bounds(logits, base, count, row, STATS, index):
    """The row's (low, high, any NaN): from the logits kernel's stats if given, else swept here."""
    if STATS is None:
        first, last = _span(row, count)
        return _row_minmax(row, *_sweep_minmax(logits, base, count, row.lane, first, last))
    # (min bits, ~max bits) of logits >= 0 over a superset of this row's causal keys: any NaN
    # orders above +inf. Wider bounds keep the digit monotone, so selection stays exact.
    low = fx.Int32(_uniform(STATS[2 * index]))
    high = ~fx.Int32(_uniform(STATS[2 * index + 1]))
    return (low.bitcast(fx.Float32), high.bitcast(fx.Float32),
            _ugt(high, fx.Int32(0x7F800000)).select(fx.Int32(1), fx.Int32(0)))


@flyc.jit
def _write_row(LOGITS, stride, local, out_row, count, visible, length, OUT, row, STATS):
    # Logits row `local` (row stride `stride`) selects output row `out_row`.
    output = _buffer(fx.make_view(fx.get_iter(OUT) + fx.Int64(out_row) * WIDTH, fx.make_layout(1, 1)), WIDTH * 4)
    blocks = count
    if count <= TOPK:
        for token in range(row.thread, count * RATIO, fx.Int32(row.threads)):
            _put(output, token, token)
    else:
        blocks = fx.Int32(TOPK)
        logits = _buffer(LOGITS, fx.Int32(0x7FFFFFF0))
        base = local * stride * 4
        low, high, nan = _bounds(logits, base, count, row, STATS, out_row)
        split = _select(logits, base, count, row, low, high, nan)
        _row_sync(row)
        _emit_ascending(output, row.chosen, row.merged, split, row.thread, row.threads, _sync_of(row))
    tail_start = visible // RATIO * RATIO
    tail_count = visible - tail_start
    for index in range(blocks * RATIO + row.thread, fx.Int32(WIDTH), fx.Int32(row.threads)):
        column = fx.Int32(index)
        offset = column - blocks * RATIO
        token = tail_start + offset
        value = ((offset < RATIO - 1) & (offset < tail_count) & (token < length)).select(token, fx.Int32(-1))
        _put(output, column, value)


@flyc.jit
def _row(LOGITS, stride, row0, POSITIONS, ROW_INFO, OUT, STATS, local, row):
    index = row0 + local
    expected, length = _uniform(ROW_INFO[2 * index]), _uniform(ROW_INFO[2 * index + 1])
    if fx.Int64(POSITIONS[index]) != fx.Int64(expected):
        # s_trap 2: the queue enters the error state and the process aborts.
        llvm.intr_trap()
    visible = expected + 1
    count = visible // RATIO
    count = (count < length // RATIO).select(count, length // RATIO)
    _write_row(LOGITS, stride, local, index, count, visible, length, OUT, row, STATS)


@flyc.kernel(name="qsa_indexer_topk", known_block_size=[256, 1, 1])
def _kernel(LOGITS: fx.Tensor, stride: fx.Int32, row0: fx.Int32, rows: fx.Int32, POSITIONS: fx.Tensor,
            ROW_INFO: fx.Tensor, OUT: fx.Tensor, STATS: fx.Tensor):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, WAVES * (SCRATCH + CHOSEN), 16]).peek()
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(WAVES * (SCRATCH + CHOSEN), 1)))))
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    local = _uniform(fx.Int32(gpu.block_id("x")) * WAVES + wave)
    # One wave per row: the histogram scratch also holds the candidates and the merge.
    bins = shared + wave * SCRATCH
    row = _space(1, wave, lane, bins, bins, shared + WAVES * SCRATCH + wave * CHOSEN, bins)
    if local < rows:
        _row(LOGITS, stride, row0, POSITIONS, ROW_INFO, OUT, STATS, local, row)


@flyc.jit
def _launch(LOGITS: fx.Tensor, stride: fx.Int32, row0: fx.Int32, rows: fx.Int32, POSITIONS: fx.Tensor,
            ROW_INFO: fx.Tensor, OUT: fx.Tensor, STATS: fx.Tensor, ctas: fx.Int32, stream: fx.Stream):
    if ctas > 0:
        _kernel(LOGITS, stride, row0, rows, POSITIONS, ROW_INFO, OUT, STATS).launch(
            grid=(ctas, 1, 1), block=(256, 1, 1), stream=stream)


_COMPILED = {}


def _compiled(cache, key, launcher, args, count_index):
    # Compile with zero work (no kernel runs), then call: first use never hides inside a capture.
    compiled = cache.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA indexer before graph capture")
        compiled = cache[key] = flyc.compile(launcher, *args[:count_index], 0, *args[count_index + 1:])
    return compiled


def launch(logits, stride, row0, rows, positions, row_info, out, stats):
    """Rows [row0, row0 + rows) of out from logits rows [0, rows) (row stride `stride`); row_info holds
    (query position, sequence length) per row, and a row whose position disagrees traps. stats [rows, 2]
    int32 holds (min bits, ~max bits) of each row's logits from the logits kernel."""
    stream = torch.cuda.current_stream(logits.device)
    args = (logits.view(-1), stride, row0, rows, positions, row_info.view(-1), out.view(-1), stats.view(-1),
            -(-rows // WAVES), stream)
    with torch.cuda.device(logits.device):
        _compiled(_COMPILED, logits.device, _launch, args, 8)(*args)


# Decode: one CTA of DECODE_WAVES waves per row. Waves own contiguous slot-group ranges, so each
# wave's histogram counts give its output offsets: row order and bits match the one-wave selection.
# LDS: per-wave histograms, per-wave reduction words, 64 candidates (keys, ids), chosen ids.
DECODE_WAVES = 8
HIST = (BINS + 1) * 4
D_RED = DECODE_WAVES * HIST
D_CAND = D_RED + DECODE_WAVES * 16
D_CHOSEN = D_CAND + 2 * CANDIDATES * 4
D_BYTES = D_CHOSEN + TOPK * 2


def _cta_sync():
    llvm.fence(llvm.AtomicOrdering.seq_cst, syncscope="workgroup")
    rocdl.s_barrier()
    llvm.fence(llvm.AtomicOrdering.seq_cst, syncscope="workgroup")


@flyc.kernel(name="qsa_indexer_decode_topk", known_block_size=[64 * DECODE_WAVES, 1, 1])
def _decode_kernel(LOGITS: fx.Tensor, stride: fx.Int32, rows: fx.Int32, LENGTHS: fx.Tensor,
                   POSITIONS: fx.Tensor, SEQUENCES: fx.Tensor, OUT: fx.Tensor):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, D_BYTES, 16]).peek()
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(D_BYTES, 1)))))
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    row = fx.Int32(gpu.block_id("x"))
    count = _uniform(LENGTHS[row])
    visible = _uniform(POSITIONS[row]) + 1
    length = _uniform(SEQUENCES[row])
    space = _space(DECODE_WAVES, wave, lane, shared + wave * HIST, shared + D_CAND, shared + D_CHOSEN, shared,
                   hist_base=shared, red=shared + D_RED)
    _write_row(LOGITS, stride, row, row, count, visible, length, OUT, space, None)


@flyc.jit
def _launch_decode(LOGITS: fx.Tensor, stride: fx.Int32, rows: fx.Int32, LENGTHS: fx.Tensor,
                   POSITIONS: fx.Tensor, SEQUENCES: fx.Tensor, OUT: fx.Tensor,
                   ctas: fx.Int32, stream: fx.Stream):
    if ctas > 0:
        _decode_kernel(LOGITS, stride, rows, LENGTHS, POSITIONS, SEQUENCES, OUT).launch(
            grid=(ctas, 1, 1), block=(64 * DECODE_WAVES, 1, 1), stream=stream)


_COMPILED_DECODE = {}


def launch_decode(logits, lengths, positions, sequence_lengths, out):
    rows, stride = logits.shape
    stream = torch.cuda.current_stream(logits.device)
    args = (logits.view(-1), stride, rows, lengths, positions, sequence_lengths, out.view(-1), rows, stream)
    key = (logits.device, lengths.dtype, positions.dtype, sequence_lengths.dtype)
    with torch.cuda.device(logits.device):
        _compiled(_COMPILED_DECODE, key, _launch_decode, args, 7)(*args)
