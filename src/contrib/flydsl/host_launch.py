"""Host-side launch of FlyDSL kernels: ``@flydsl_kernel`` turns a compile function into a
Triton-style kernel object.

A compile function (the "builder") takes the compile-time values as keywords and returns
a ``@flyc.jit`` launcher; the kernel object is called with the runtime arguments
positionally and the compile-time key as keywords, like ``triton_kernel[grid](...)``::

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from pyhip.contrib.flydsl.host_launch import flydsl_kernel, kernel_name, ptr

    @flydsl_kernel(heuristics={"rows_per_wave": lambda k: 4 if k["cols"] >= 4096 else 16})
    def row_max(*, cols, rows_per_wave):              # compile-time values only
        rows_per_block = 4 * rows_per_wave            # derive constants here, not in the kernel

        @flyc.kernel(name=kernel_name("row_max", c=cols, rpw=rows_per_wave),
                     known_block_size=(256, 1, 1))
        def kernel(src: fx.Int64, out: fx.Int64, i32_rows: fx.Int32):
            ...                                       # fx.inttoptr(...) turns Int64 into pointers

        @flyc.jit
        def launch(src: fx.Int64, out: fx.Int64, i32_rows: fx.Int32, stream: fx.Stream):
            grid = (i32_rows + (rows_per_block - 1)) // rows_per_block
            kernel(src, out, i32_rows).launch(grid=(fx.Int64(grid), 1, 1), block=(256, 1, 1),
                                              stream=stream)

        return launch

    def row_max_op(x):                                # the op is plain Python
        rows, cols = x.shape
        out = torch.empty(rows, device=x.device, dtype=torch.int32)
        if rows:
            row_max(ptr(x, torch.uint8), ptr(out, torch.int32), rows, cols=cols)
        return out

Calling ``kern(*args, **key)``:

- ``args`` are the launch parameters except the trailing stream: a tensor is passed as its
  ``data_ptr()``, ``None`` as a null pointer, ints / floats as they are.
- ``key`` selects the specialization. Missing names are filled by ``heuristics`` (in
  order; each entry sees the key so far). Values must be compile-time scalars or tuples
  (lists become tuples, ``__index__`` types become ints); tensors and other objects are
  rejected, because FlyDSL's disk-cache key ignores closure objects. Keep runtime sizes
  out of the key: pass them as ``fx.Int32`` launch parameters and derive the grid inside
  ``launch``. The kernel object warns once it holds ``warn_after`` variants
  (``warn_after=None`` disables the warning, e.g. for tuning sweeps).
- The launch goes to the device of the first tensor argument, on that device's current
  stream (raw handles, ~0.1 us, instead of ``torch.cuda.current_stream()``, ~4 us). A
  compiled FlyDSL artifact only runs on the device it was first called on, so each
  further device gets its own launcher (a cache hit, not a recompile). CUDA graph capture
  works: the launch is a plain kernel launch on the current stream.

The launcher signature is checked on the first compile: every parameter must be a FlyDSL
scalar (``fx.Int64`` for pointers, ``fx.Int32``, ``fx.Float32``, ...) and the last one
``stream: fx.Stream``. ``fx.Pointer`` / ``fx.Tensor`` / ``fx.Constexpr`` parameters are not
supported (keep calling those launchers directly).

Other entry points:

- ``kern.get(**key)``: the specialization, callable with the runtime arguments only
  (skips the key lookup: bind it once in hot loops or persistent objects).
- ``kern.preload(**key)``: compile into the disk cache without launching anything.
- ``kern.variants`` (normalized key -> specialization), ``kern.clear()``,
  ``kern.builder`` (the undecorated compile function).

Debugging: ``PYHIP_FLYDSL_CHECK=1`` (or ``host_launch.CHECK = True``) validates every
launch against the launcher's annotations and the ``ptr(t, dtype, align, numel)``
annotations: argument count, tensor vs scalar positions, Int32 range, GPU / same device /
contiguous / dtype / alignment / size. Off, ``ptr()`` returns the tensor and costs
nothing. Compiles are logged at INFO level on this module's logger.

Host cost per launch (MI300X, 5 runtime arguments; ``python tests/flydsl/test_host_launch.py``):
~6 us, vs ~4.3 us for a bare CompiledFunction with a raw stream and ~9 us with
``torch.cuda.current_stream()`` (the usual ``_run_compiled`` pattern). Calling a
``@flyc.jit`` function directly costs ~70 us (it rebuilds the full cache key on every
call) and always launches on the current device.

Tests: ``tests/flydsl/test_host_launch.py``. Conventions for writing kernels with it:
the ``flydsl-host-launch`` skill (``.claude/skills/flydsl-host-launch/SKILL.md``).
"""

import enum
import inspect
import logging
import os
import re
import threading
import time
import warnings
from typing import NamedTuple, Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.compiler.jit_function import JitFunction

__all__ = ["CHECK", "Ptr", "cdiv", "flydsl_kernel", "kernel_name", "ptr"]

log = logging.getLogger(__name__)

# Read on every launch, so tests can flip it at runtime.
CHECK = os.environ.get("PYHIP_FLYDSL_CHECK", "0") == "1"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def cdiv(a, b):
    return -(-a // b)


def _token(v):
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, tuple):
        return "x".join(_token(x) for x in v)
    return str(v)


def kernel_name(base, **key):
    """Fold compile-time values into the kernel symbol so ISA dumps and profiles tell the
    variants apart: ``kernel_name("a8w4_chan_exp", c=4096, rpw=4)``."""
    name = "_".join([base, *(f"{k}{_token(v)}" for k, v in key.items())])
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


_SCALARS = (bool, int, float, str, type(None), enum.Enum)


def _freeze(name, v):
    if isinstance(v, _SCALARS):
        return v
    if isinstance(v, (tuple, list)):
        return tuple(_freeze(name, x) for x in v)
    if not isinstance(v, torch.Tensor) and hasattr(v, "__index__"):
        return int(v.__index__())
    raise TypeError(f"compile key {name!r}: {type(v).__name__} is not a compile-time value "
                    "(use int / float / bool / str / None / Enum / tuple)")


def _placeholder(ann):
    if ann is fx.Stream:
        return torch.cuda.current_stream()
    return 0.0 if issubclass(ann, fx.Float) else 0


# Raw handles, as the Triton / Inductor launchers use: torch.cuda.current_stream() costs
# ~4 us per call (~1.7 us with a device index), these ~0.1 us; fx.Stream takes an int.
try:
    _get_device = torch._C._cuda_getDevice
    _raw_stream = torch._C._cuda_getCurrentRawStream
except AttributeError:  # pragma: no cover
    _get_device = torch.cuda.current_device

    def _raw_stream(idx):
        return torch.cuda.current_stream(idx).cuda_stream


# --------------------------------------------------------------------------------------
# Compiling
# --------------------------------------------------------------------------------------

_COMPILE_ONLY_LOCK = threading.RLock()


def _drop_ir_contexts():
    # A failed compile can leave its MLIR context entered; the next compile would nest
    # inside it.
    while ir.Context.current is not None:
        ir.Context.current.__exit__(None, None, None)


def _compile_and_run(launch, args):
    """First call of a launcher: compile (or load from the disk cache) and execute.
    Returns the CompiledFunction, or None when ``COMPILE_ONLY=1`` is set."""
    try:
        cf = flyc.compile(launch, *args)
    except Exception:
        _drop_ir_contexts()
        raise
    if cf is not None:
        launch._cf = cf  # what a8w4 kernels/_launch._run_compiled looks for
    return cf


def _compile_only(launch, args):
    preload = getattr(launch, "preload", None)
    if callable(preload):  # newer FlyDSL
        preload(*args)
        return
    with _COMPILE_ONLY_LOCK:  # FlyDSL 0.3.2: the process-wide COMPILE_ONLY mode
        old = os.environ.get("COMPILE_ONLY")
        os.environ["COMPILE_ONLY"] = "1"
        try:
            flyc.compile(launch, *args)
        except Exception:
            _drop_ir_contexts()
            raise
        finally:
            if old is None:
                os.environ.pop("COMPILE_ONLY", None)
            else:
                os.environ["COMPILE_ONLY"] = old


# --------------------------------------------------------------------------------------
# Runtime arguments: annotations, CHECK mode, launching
# --------------------------------------------------------------------------------------

class Ptr(NamedTuple):
    """A pointer argument plus what the kernel assumes about it (see ``ptr``)."""

    t: Optional[torch.Tensor]
    dtype: Optional[torch.dtype] = None
    align: int = 0  # bytes; 0 -> element size
    numel: int = 0  # minimum element count the kernel touches; 0 -> unchecked


def ptr(t, dtype=None, align=0, numel=0):
    """Annotate a pointer argument (``t=None`` passes a null pointer). Outside CHECK mode
    this returns ``t`` itself, so the annotations cost nothing on the hot path."""
    return Ptr(t, dtype, align, numel) if CHECK else t


def _check(exe, args):
    params = exe.params()
    if len(params) != len(args) + 1:
        raise TypeError(f"{exe.name}: called with {len(args)} args, its launch takes "
                        f"{len(params) - 1} plus the stream")
    dev = None
    for i, (a, ann) in enumerate(zip(args, params)):
        where = f"{exe.name}: arg {i}"
        p = a if a.__class__ is Ptr else Ptr(a) if isinstance(a, torch.Tensor) else None
        if p is None:
            if ann is fx.Int32 and isinstance(a, int) and not -(2**31) <= a < 2**31:
                raise OverflowError(f"{where}: {a} does not fit the Int32 parameter")
            if not isinstance(a, (int, float, type(None))):
                raise TypeError(f"{where}: unsupported launch value {type(a).__name__}")
            continue
        if ann is not fx.Int64:
            raise TypeError(f"{where}: tensor passed to a {ann.__name__} parameter "
                            "(argument order?)")
        t = p.t
        if t is None:
            continue
        if not t.is_cuda:
            raise ValueError(f"{where}: tensor on {t.device}, expected a GPU tensor")
        dev = dev or t.device
        if t.device != dev:
            raise ValueError(f"{where}: tensor on {t.device}, earlier tensors on {dev}")
        if not t.is_contiguous():
            raise ValueError(f"{where}: not contiguous (shape {tuple(t.shape)}, "
                             f"stride {t.stride()})")
        if p.dtype is not None and t.dtype != p.dtype:
            raise TypeError(f"{where}: dtype {t.dtype}, kernel reads {p.dtype}")
        align = p.align or t.element_size()
        if t.data_ptr() % align:
            raise ValueError(f"{where}: address {t.data_ptr():#x} not {align}-byte aligned")
        if t.numel() < p.numel:
            raise ValueError(f"{where}: {t.numel()} elements, kernel touches {p.numel}")


_Tensor = torch.Tensor


def _launch(exe, args):
    if CHECK:
        _check(exe, args)
        args = [a.t if a.__class__ is Ptr else a for a in args]
    raw = [a.data_ptr() if isinstance(a, _Tensor) else 0 if a is None else a for a in args]
    dev = -1
    for a in args:
        if isinstance(a, _Tensor):
            dev = a.get_device()
            break
    cur = _get_device()
    if dev < 0 or dev == cur:
        raw.append(_raw_stream(cur))
        if exe.dev == cur:
            exe.cf(*raw)  # hot path: straight into the CompiledFunction
        else:
            exe.run(cur, raw)
    else:
        with torch.cuda.device(dev):
            raw.append(_raw_stream(dev))
            exe.run(dev, raw)


# --------------------------------------------------------------------------------------
# One specialization
# --------------------------------------------------------------------------------------

class _Exe:
    """One specialization: the ``@flyc.jit`` launcher plus its CompiledFunction.

    Calling it launches with the runtime arguments, like the kernel object without the
    key. ``launch`` is the JitFunction (bound to device ``dev`` once called), ``cf`` its
    CompiledFunction.
    """

    __slots__ = ("_build", "name", "launch", "dev", "cf", "_others", "_params", "_lock")

    def __init__(self, build, name):
        self._build, self.name, self.launch = build, name, build()
        if not isinstance(self.launch, JitFunction):
            raise TypeError(f"{name}: the builder must return a @flyc.jit function, got "
                            f"{type(self.launch).__name__}")
        self.dev, self.cf, self._others, self._params = None, None, {}, None
        self._lock = threading.Lock()

    def __call__(self, *args):
        _launch(self, args)

    def __repr__(self):
        return f"<flydsl specialization {self.name}>"

    def run(self, dev, raw):
        """Launch lowered arguments (stream included) on device ``dev``, which the caller
        has made current."""
        if dev == self.dev:
            self.cf(*raw)
            return
        other = self._others.get(dev)
        if other is None:
            with self._lock:
                if self.cf is None:
                    self.params()
                    t0 = time.perf_counter()
                    cf = _compile_and_run(self.launch, raw)
                    if cf is not None:
                        self.cf, self.dev = cf, dev
                    log.info("compiled %s on cuda:%d in %.0f ms", self.name, dev,
                             (time.perf_counter() - t0) * 1e3)
                    return
                if dev == self.dev:  # another thread finished the first call meanwhile
                    other = self
                else:
                    other = self._others.get(dev)
                    if other is None:
                        other = self._others[dev] = _Exe(self._build, self.name)
        other.run(dev, raw)

    def params(self):
        """Annotations of the launch parameters, validated on first use."""
        if self._params is None:
            # JitFunction._original_func is rebuilt without __annotations__; .func keeps
            # them. eval_str resolves `from __future__ import annotations`.
            sig = inspect.signature(self.launch.func, eval_str=True)
            params = [p.annotation for p in sig.parameters.values()]
            ok = (params and params[-1] is fx.Stream
                  and all(isinstance(a, type) and issubclass(a, fx.Numeric) for a in params[:-1]))
            if not ok:
                raise TypeError(
                    f"{self.name}: launch{sig} is not supported. Annotate every parameter "
                    "with a FlyDSL scalar type (fx.Int64 for pointers, then fx.inttoptr in "
                    "the kernel; fx.Int32; fx.Float32; ...) and end with `stream: "
                    "fx.Stream`. fx.Pointer / fx.Tensor / fx.Constexpr parameters are not "
                    "supported, and an unannotated parameter would be passed as Int32.")
            self._params = params
        return self._params

    def preload(self):
        """Compile (and persist to the disk cache) without dispatching any kernel."""
        if self.cf is None:
            _compile_only(self.launch, [_placeholder(a) for a in self.params()])


# --------------------------------------------------------------------------------------
# @flydsl_kernel
# --------------------------------------------------------------------------------------

def flydsl_kernel(fn=None, *, heuristics=None, warn_after=32):
    """Turn ``fn(**key) -> @flyc.jit launcher`` into a kernel object called as
    ``kern(*runtime_args, **key)`` (see the module docstring).

    ``heuristics`` maps a key name to ``f(key_dict)``, used when the caller omits that
    name; entries see the key given so far, in order. The hot path is one dict lookup on
    the caller's ``tuple(key.items())``; unhashable values (lists) are normalized on every
    call, so pass tuples on hot paths. ``warn_after``: warn when this many variants exist
    (``None`` disables it).
    """

    def deco(fn):
        fast, variants = {}, {}
        lock = threading.Lock()
        prefix = fn.__name__.strip("_")
        derived = tuple((heuristics or {}).items())

        def build(key):
            key = dict(key)
            for k, f in derived:
                if k not in key:
                    key[k] = f(key)
            norm = tuple(sorted((k, _freeze(k, v)) for k, v in key.items()))
            with lock:
                exe = variants.get(norm)
                if exe is None:
                    label = ",".join(f"{k}={v}" for k, v in norm)
                    full = dict(norm)
                    exe = variants[norm] = _Exe(lambda: fn(**full), f"{prefix}[{label}]")
                    if warn_after and len(variants) == warn_after:
                        warnings.warn(f"{prefix}: {warn_after} compiled variants; is a runtime "
                                      f"value in the compile key? last: {label}", stacklevel=4)
            return exe

        def lookup(key):
            fk = tuple(key.items())
            try:
                return fast[fk]
            except KeyError:
                exe = fast[fk] = build(key)
                return exe
            except TypeError:  # unhashable caller key (a list): normalize every time
                return build(key)

        def kern(*args, **key):
            if not args:
                raise TypeError(f"{prefix}: no runtime arguments; for the launcher of a "
                                f"specialization use {fn.__name__}.get(**key)")
            try:
                exe = fast[tuple(key.items())]
            except (KeyError, TypeError):
                exe = lookup(key)
            _launch(exe, args)

        def get(**key):
            return lookup(key)

        def preload(**key):
            lookup(key).preload()

        def clear():
            with lock:
                fast.clear()
                variants.clear()

        for attr in ("__module__", "__name__", "__qualname__", "__doc__"):
            setattr(kern, attr, getattr(fn, attr))
        kern.builder = fn
        kern.variants = variants
        kern.get = get
        kern.preload = preload
        kern.clear = clear
        return kern

    return deco if fn is None else deco(fn)
