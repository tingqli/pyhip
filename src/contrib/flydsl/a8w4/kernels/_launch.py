"""Host side: launch / preload a ``@flyc.jit`` launcher with raw arguments.

Thin wrappers over ``pyhip.contrib.flydsl.host_launch``, which owns the compile details
(FlyDSL 0.3.2 has no ``JitFunction.preload``: compile-only goes through the process-wide
``COMPILE_ONLY`` mode under a lock; a failed compile leaves its MLIR context entered). New
host code uses ``@flydsl_kernel`` from the same module instead.
"""

from pyhip.contrib.flydsl.host_launch import _compile_and_run, _compile_only


def _run_compiled(launch, *args):
    """``launch(*args)``: the first call compiles (or loads from the disk cache) and runs,
    later calls go straight to the cached CompiledFunction."""
    cf = getattr(launch, "_cf", None)
    if cf is not None:
        cf(*args)
        return
    _compile_and_run(launch, args)


def _preload_compiled(launch, *args):
    """Compile into the disk cache without dispatching (``args`` may be placeholders)."""
    _compile_only(launch, args)
