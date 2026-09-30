"""Packed sparse GQA inputs, reference, and frozen SGLang baseline.

GPU implementations are imported explicitly so input-contract tests stay CPU-only.
"""

from .contract import AttentionInputs, CaseSpec, ModelShape, load_model_shape
from .inputs import default_spec, make_inputs, validate_inputs

__all__ = [
    "AttentionInputs",
    "CaseSpec",
    "ModelShape",
    "default_spec",
    "load_model_shape",
    "make_inputs",
    "validate_inputs",
]
