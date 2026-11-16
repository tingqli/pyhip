"""``_run_compiled`` for the frozen experiment scripts (the package itself launches through
``@flydsl_kernel`` from ``pyhip.contrib.flydsl.host_launch``; was ``kernels/_launch.py``)."""

from pyhip.contrib.flydsl.host_launch import _compile_and_run


def _run_compiled(launch, *args):
    """``launch(*args)`` with raw arguments (pointers as ints, stream last): the first call
    compiles (or loads from the disk cache) and runs, later calls go straight to the cached
    CompiledFunction."""
    cf = getattr(launch, "_cf", None)
    if cf is not None:
        cf(*args)
        return
    _compile_and_run(launch, args)
