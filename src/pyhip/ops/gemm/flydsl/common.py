"""Host-side target checks shared by the CDNA4 GEMM factories."""

import flydsl.compiler as flyc


def require_cdna4():
    """Reject unsupported compilation targets, including in compile-only mode.

    Use the compiler's resolved target rather than querying Torch at import
    time, so explicit offline gfx950 compilation remains possible.
    """
    target = flyc.get_backend().target
    arch = target.arch.lower().split(":", 1)[0]
    if target.backend != "rocm" or arch != "gfx950":
        raise RuntimeError(
            "FlyDSL FP8/MXFP8 GEMM requires CDNA4 (gfx950); "
            f"got {target.backend}/{target.arch}"
        )