"""PyHIP: AMDGPU kernels and language-specific development tools."""

from .testing.misc import *

# when user call from pyhip import *
__all__ = [
    'module', 'jit', 'JIT'
]


def __getattr__(name):
    # Preserve the small root API without importing unused compiler backends.
    if name in ("jit", "JIT"):
        from .codegen.asm import asmjit
        value = getattr(asmjit, name)
    elif name == "module":
        from .runtime.hiptools import module
        value = module
    elif name in ("fly", "printv"):
        from .codegen.flydsl import utils
        value = getattr(utils, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
