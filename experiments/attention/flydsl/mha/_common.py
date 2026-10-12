"""Bridge for existing MHA experiments and the unchanged temporary QSA plugin."""

import sys
from pyhip.ops.mha.flydsl import _common as _implementation

sys.modules[__name__] = _implementation