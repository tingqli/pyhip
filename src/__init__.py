"""PyHIP: Python AMDGPU kernel development toolkit"""

# core API
from .core.hiptools import module
from .core.asmjit import jit, JIT
from .core.perf import cudaPerf, torchPerf, calc_diff, div_up

# when user call from pyhip import *
__all__ = [
    'module', 'jit', 'JIT'
]
