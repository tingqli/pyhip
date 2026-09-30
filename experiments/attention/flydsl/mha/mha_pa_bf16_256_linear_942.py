"""Bridge for existing MHA experiments and the unchanged temporary QSA plugin."""

import sys
from pyhip.ops.mha.flydsl import mha_pa_bf16_256_linear_942 as _implementation

sys.modules[__name__] = _implementation