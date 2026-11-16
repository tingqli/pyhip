# Vendored from aiter/ops/flydsl/kernels (aiter 7cd70aca8) so the a8w4 package
# does not depend on aiter's private FlyDSL helpers. Imports made relative.
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import threading

import flydsl.compiler as flyc
from flydsl._mlir import ir

_PRELOAD_COMPILE_LOCK = threading.RLock()


def _run_compiled(exe, *args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.
    """
    cf = getattr(exe, "_cf", None)
    if cf is not None:
        cf(*args)
        return
    try:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    except Exception:
        # flyc.compile leaks ir.Context on failure; pop it so a retry takes the right path.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001, S110
            pass
        raise


def _preload_compiled(exe, *args):
    """Materialize a JIT artifact without dispatching its GPU kernels.

    Newer FlyDSL versions expose ``JitFunction.preload()`` for this operation.
    Older supported versions provide the same no-dispatch behavior through the
    public ``COMPILE_ONLY`` mode.  Calling ``flyc.compile()`` without either
    guard executes the launcher and is unsafe for preload callers that pass
    placeholder pointers.
    """
    preload = getattr(exe, "preload", None)
    if callable(preload):
        return preload(*args)

    with _PRELOAD_COMPILE_LOCK:
        old_compile_only = os.environ.get("COMPILE_ONLY")
        os.environ["COMPILE_ONLY"] = "1"
        try:
            return flyc.compile(exe, *args)
        finally:
            if old_compile_only is None:
                os.environ.pop("COMPILE_ONLY", None)
            else:
                os.environ["COMPILE_ONLY"] = old_compile_only


def _to_raw(v):
    """Convert ArithValue / Numeric (Int32, Boolean, …) to raw ir.Value."""
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return _to_raw(v.ir_value())
    return ir.Value._CAPICreate(v._CAPIPtr)


