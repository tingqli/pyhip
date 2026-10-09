"""BF16 GR write (hc_combine) fused with the next per-branch RMSNorm, prefill path."""

from .flydsl import gr_write, prepare_weights

__all__ = ['gr_write', 'prepare_weights']
