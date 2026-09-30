"""Temporary bridge for the unchanged SGLang plugin; implementation lives in pyhip."""

import sys
from pyhip.ops.qsa.flydsl import indexer as _implementation

sys.modules[__name__] = _implementation