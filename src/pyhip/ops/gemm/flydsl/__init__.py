"""FlyDSL GEMM implementations (CDNA4/gfx950).

Modules use ``gemm_<format>_<waves>w``; tests use the same stem with ``test_``.
``gemm_fp8_blockscale_8w`` uses FP32 block scales; ``gemm_mxfp8_4w`` supports
MXFP8 activations with FP8 or MXFP4 weights.

Import individual kernel modules explicitly to keep FlyDSL an optional backend.
"""