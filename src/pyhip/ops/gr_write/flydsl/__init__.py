# SPDX-License-Identifier: MIT
"""gfx942 GR write prefill: MFMA gate plus fused combine and per-branch RMSNorm."""

from .common import prepare_weights
from .host import gr_write

__all__ = ['gr_write', 'prepare_weights']
