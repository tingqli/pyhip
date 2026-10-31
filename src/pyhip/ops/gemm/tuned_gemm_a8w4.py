"""Aiter-compatible MXFP8 x MXFP4 GEMM with the same autotuner as tuned_moe.

On gfx950, FlyDSL autotune validates and compares Aiter's native MXFP kernels
and PyHIP's four-wave kernel. It reuses its normal winner cache and optional
FLYDSL_AUTOTUNE_CONFIG_DIR artifacts; a miss tunes, never guesses a winner.
Other architectures forward every argument to Aiter's public operator.

gfx950 operands use Aiter's gfx950 MX layout and neither backend copies A or B:

    A       [M, K]             fp8 e4m3, row-major (``a_preshuffle=False``)
    B       [N, K/2]           fp4x2, ``aiter.ops.shuffle.shuffle_mxfp8fp4_b``
                               (the same bytes as ``shuffle_weight(B, (16, 16))``)
    ScaleA  [pad256(M), pad8(K/32)] e8m0, ``per_1x32_mx_quant_hip(..., shuffle=True)``
    ScaleB  [pad256(N), pad8(K/32)] e8m0, ``aiter.ops.shuffle.shuffle_scale``

The PyHIP scale repack and Aiter split-K reduction are inside the timed
candidate. Output allocation is common to every candidate. Graph capture
requires prior eager tuning and warmup. See README.md for cache/export usage.
"""

import functools
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import pyhip
import torch
from torch import Tensor
from flydsl.autotune import Config, autotune

from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.gemm_op_a8w4 import gemm_a8w4_mxfp8 as _aiter_gemm_a8w4_mxfp8

__all__ = [
    "gemm_a8w4_mxfp8",
    "last_dispatch",
    "record_dispatch",
]

logger = logging.getLogger(__name__)

_DIFF_LIMIT = 1e-5
_AUTOTUNE_KEYS = ["M", "N", "K", "model_key"]
_ARTIFACT_NAME = "pyhip_gemm_a8w4_mxfp8_v1"
_SCALES = dict(
    a_scale_block="1x32", b_scale_block="1x32",
    a_scale_layout="mfma_tile", b_scale_layout="mfma_tile",
)

# Serial diagnostics: read after a successful eager call; graph replay does not update it.
record_dispatch = False
last_dispatch = None
_native_device = None
_WARMED_KERNELS = set()


def _pad256(rows: int) -> int:
    return (rows + 255) // 256 * 256


def _scale_columns(K: int) -> int:
    # shuffle_scale pads K/32 to whole 8-column (256-K) MFMA scale words.
    return (K // 32 + 7) // 8 * 8


@dataclass(frozen=True)
class PyhipKernel:
    """A PyHIP gfx950 4-wave MXFP8 x MXFP4 configuration (TILE 256x256x128)."""

    permlane_epilogue: bool = True
    store_overlap: bool = False
    lds_swizzle: bool = False

    @property
    def name(self) -> str:
        return (
            "pyhip_mxfp8_4w_256x256x128"
            f"_pl{int(self.permlane_epilogue)}_so{int(self.store_overlap)}"
            f"_as{int(self.lds_swizzle)}"
        )

    split_k = 1

    @classmethod
    def parse(cls, name: str):
        match = _PYHIP_NAME.match(name.strip())
        return None if match is None else cls(*(flag == "1" for flag in match.groups()))

    def supports(self, M: int, N: int, K: int) -> bool:
        # 32-bit buffer offsets cover A and C; the A LDS swizzle with preshuffled B
        # keeps the kernel's power-of-two K restriction.
        return (
            M > 0 and N > 0 and K > 0
            and K % 256 == 0
            and N % 16 == 0
            and M * K < 2**32
            and N * K // 2 < 2**32
            and M * N * 2 < 2**32
            and (not self.lds_swizzle or K & (K - 1) == 0)
        )


_PYHIP_NAME = re.compile(r"^pyhip_mxfp8_4w_256x256x128_pl([01])_so([01])_as([01])$")
PYHIP_KERNELS = (
    PyhipKernel(),
    PyhipKernel(store_overlap=True),
    PyhipKernel(permlane_epilogue=False),
    PyhipKernel(lds_swizzle=True),
)


def _aiter_catalog():
    from aiter.ops.flydsl.gemm_tune import flydsl_gemm_mxscale_preshuffle_common

    return flydsl_gemm_mxscale_preshuffle_common


def aiter_kernel(name: str, split_k: Optional[int] = None):
    """Aiter FlyDSL ``kernelInstance`` for an MXFP8 x MXFP4 BF16 kernelName, else None."""
    catalog = _aiter_catalog()
    params = catalog.parse_kernel_name(name)
    if params is None or params["fused_split_k"]:
        return None
    if (params["a_dtype"], params["b_dtype"], params["out_dtype"]) != ("fp8", "fp4", "bf16"):
        return None
    kernel = catalog.kernelInstance(**params)
    return kernel if split_k is None else replace(kernel, split_k=split_k)


def _aiter_fits(kernel, M: int, N: int, K: int) -> bool:
    # Aiter bounds the mfma_tile ScaleB to N // 32 whole 32-row chunks, so the
    # rows of a ragged final chunk would read zero scales.
    catalog = _aiter_catalog()
    return (
        M > 0 and N > 0 and K > 0
        and M * K < 2**32
        and N * K // 2 < 2**32
        and M * N * 2 < 2**32
        and N % 32 == 0
        and catalog.instance_valid(kernel)
        and catalog.fits_shape(kernel, M, N, K, **_SCALES)
    )


@functools.lru_cache(maxsize=4096)
def _explicit(name, M, N, K, splitk):
    kernel = PyhipKernel.parse(name)
    if kernel is not None:
        if splitk not in (None, 1):
            raise ValueError(f"{name} does not split K; got splitk={splitk}")
        if not kernel.supports(M, N, K):
            raise NotImplementedError(f"{name} does not support M={M} N={N} K={K}")
        return "pyhip", kernel
    kernel = aiter_kernel(name, splitk)
    if kernel is None:
        raise ValueError(f"unknown gfx950 MXFP8 x MXFP4 kernelName {name!r}")
    if not _aiter_fits(kernel, M, N, K):
        raise NotImplementedError(
            f"{name} (split_k={kernel.split_k}) does not support M={M} N={N} K={K}"
        )
    return "aiter", kernel


@functools.lru_cache(maxsize=1024)
def _candidate_names(M, N, K, splitk):
    catalog = _aiter_catalog()
    max_m = max(64, 1 << (M - 1).bit_length()) if M <= 128 else 256
    candidates = []
    # The catalog's combos omit fp8/fp4, but its kernel and legality checks support it.
    for tm, tn, tk, wpe, xcd, split in itertools.product(
        (32, 64, 96, 128, 256), (16, 32, 64, 128, 256, 512),
        (128, 256), (0, 2, 4), (0, 4), (splitk,) if splitk is not None else range(1, 33),
    ):
        if tm > max_m:
            continue
        kernel = catalog.kernelInstance(
            tm, tn, tk, "fp8", "fp4", "bf16", wpe, xcd, split,
        )
        if _aiter_fits(kernel, M, N, K):
            candidates.append(("aiter", kernel.name))
    if splitk in (None, 1):
        candidates.extend(
            ("pyhip", kernel.name) for kernel in PYHIP_KERNELS if kernel.supports(M, N, K)
        )
    return tuple(candidates)


def _configs(A, B, ScaleA, ScaleB, *, output, M, N, K, splitk, model_key, aiter_only):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("A8W4 autotuning must run before graph capture; warm up in eager mode")
    return [
        Config(_impl=impl, kernel_name=name)
        for impl, name in _candidate_names(M, N, K, splitk)
        if not aiter_only or impl == "aiter"
    ]


def _check_gfx950(A, B, ScaleA, ScaleB, dtype, a_preshuffle):
    if dtype != dtypes.bf16:
        raise NotImplementedError(f"gfx950 A8W4 MXFP8 x MXFP4 GEMM: unsupported output dtype {dtype}")
    if a_preshuffle:
        raise NotImplementedError("gfx950 A8W4 kernels read row-major A; pass a_preshuffle=False")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2-D, got {tuple(A.shape)} and {tuple(B.shape)}")
    if not A.is_cuda:
        raise ValueError(f"gfx950 A8W4 GEMM requires GPU tensors, got {A.device}")
    M, K = A.shape
    N = B.shape[0]
    if K <= 0 or N <= 0:
        raise ValueError(f"gfx950 A8W4 requires positive N and K, got N={N} K={K}")
    if max(M * K, N * K // 2, M * N * 2) >= 2**32:
        raise NotImplementedError("gfx950 A8W4 operands must fit the kernels' 32-bit buffer bounds")
    if A.dtype != dtypes.fp8:
        raise TypeError(f"A must be {dtypes.fp8}, got {A.dtype}")
    if B.dtype not in (dtypes.fp4x2, torch.uint8) or B.shape[1] * 2 != K:
        raise ValueError(f"B must be packed MXFP4 [N, K/2]=[{N}, {K // 2}], got {B.dtype} {tuple(B.shape)}")
    if K % 128 or N % 16:
        raise NotImplementedError(f"gfx950 A8W4 GEMM requires K%128==0 and N%16==0, got N={N} K={K}")
    for name, scale, rows in (("ScaleA", ScaleA, M), ("ScaleB", ScaleB, N)):
        if (
            scale.dtype not in (dtypes.fp8_e8m0, torch.uint8)
            or scale.ndim != 2
            or scale.shape[0] < _pad256(rows)
            or scale.shape[1] != _scale_columns(K)
        ):
            raise ValueError(
                f"{name} must be shuffled e8m0 [>={_pad256(rows)}, {_scale_columns(K)}], "
                f"got {scale.dtype} {tuple(scale.shape)}"
            )
    for name, tensor in (("A", A), ("B", B), ("ScaleA", ScaleA), ("ScaleB", ScaleB)):
        if not tensor.is_contiguous() or tensor.device != A.device:
            raise ValueError(f"{name} must be contiguous on {A.device}")
        if tensor.requires_grad:
            raise NotImplementedError("gfx950 A8W4 backends support inference only")
        alignment = 16 if name in ("A", "B") else 4
        if tensor.data_ptr() % alignment:
            raise ValueError(f"{name} must be {alignment}-byte aligned")
    return M, N, K


def _run_aiter(kernel, A, B, ScaleA, ScaleB, out):
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import flydsl_mxscale_preshuffle_gemm

    flydsl_mxscale_preshuffle_gemm(
        A, B, ScaleA, ScaleB, out, a_dtype="fp8", b_dtype="fp4",
        tile_m=kernel.tile_m, tile_n=kernel.tile_n, tile_k=kernel.tile_k,
        waves_per_eu=kernel.waves_per_eu, xcd_swizzle=kernel.xcd_swizzle,
        split_k=kernel.split_k, **_SCALES,
    )


def _pyhip_scale_reference(scale, rows, K):
    """Aiter's [r32][g8][g4][r16][g2][r2] e8m0 bytes -> PyHIP's [g][m128][m0][m1] dwords."""
    groups = K // 32
    tiles = scale.view(torch.uint8)[:rows].view(rows // 128, 4, groups // 8, 4, 16, 2, 2)
    return tiles.permute(2, 5, 3, 0, 6, 4, 1).contiguous().view(torch.int32).view(-1)


@functools.cache
def _repack_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def repack(src_a, src_b, dst_a, dst_b, a_row_blocks, b_row_blocks, group_blocks,
               BLOCK_G8: tl.constexpr):
        # One program: a 128-row block of ScaleA or ScaleB x BLOCK_G8 8-group blocks.
        # Each source dword (r32, g8, g4, r16) holds bytes [g2][r2]; each destination
        # dword (g, m128, m0 = r2*16 + r16) holds bytes [m1] for r32 = m128*4 + m1,
        # so four source dwords become four destination dwords by a 4x4 byte transpose.
        block = tl.program_id(0)
        if block < a_row_blocks:
            src, dst, m128, row_blocks = src_a, dst_a, block, a_row_blocks
        else:
            src, dst, m128, row_blocks = src_b, dst_b, block - a_row_blocks, b_row_blocks
        g8 = tl.program_id(1) * BLOCK_G8 + tl.arange(0, BLOCK_G8)
        m1 = tl.arange(0, 4)
        g4 = tl.arange(0, 4)
        r16 = tl.arange(0, 16)
        r32 = m128 * 4 + m1
        source = (
            (r32[:, None, None, None] * group_blocks + g8[None, :, None, None]) * 4
            + g4[None, None, :, None]
        ) * 16 + r16[None, None, None, :]
        valid = g8 < group_blocks
        words = tl.load(src + source, mask=valid[None, :, None, None], other=0)
        shifts = (m1 * 8)[:, None, None, None]
        for byte in tl.static_range(4):
            # Disjoint byte lanes: the sum is the bitwise OR of the four bytes.
            packed = tl.sum(((words >> (byte * 8)) & 0xFF) << shifts, axis=0)
            group = g8[:, None, None] * 8 + (byte // 2) * 4 + g4[None, :, None]
            target = (group * row_blocks + m128) * 32 + (byte % 2) * 16 + r16[None, None, :]
            tl.store(dst + target, packed, mask=valid[:, None, None])

    return repack


def _pyhip_scales(ScaleA, ScaleB, M, N, K):
    """Repack both of Aiter's gfx950 e8m0 scales into PyHIP's layout in one launch."""
    groups = K // 32
    rows_a, rows_b = _pad256(M), _pad256(N)
    out = torch.empty((rows_a + rows_b) * groups // 4, dtype=torch.int32, device=ScaleA.device)
    dst_a, dst_b = out.split((rows_a * groups // 4, rows_b * groups // 4))
    block = 8
    grid = ((rows_a + rows_b) // 128, (groups // 8 + block - 1) // block)
    _repack_kernel()[grid](
        ScaleA.view(torch.uint8)[:rows_a].view(torch.int32),
        ScaleB.view(torch.uint8)[:rows_b].view(torch.int32),
        dst_a, dst_b, rows_a // 128, rows_b // 128, groups // 8, BLOCK_G8=block,
    )
    return dst_a, dst_b


_PYHIP_COMPILED = {}


def _run_pyhip(kernel, A, B, ScaleA, ScaleB, out, M, N, K):
    args = (
        A.view(torch.int8), B.view(torch.int8), *_pyhip_scales(ScaleA, ScaleB, M, N, K),
        out.view(-1), M, torch.cuda.current_stream(A.device),
    )
    key = (kernel, N, K)
    compiled = _PYHIP_COMPILED.get(key)
    if compiled is not None:
        compiled(*args)
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up the selected A8W4 kernel before graph capture")
    import flydsl.compiler as flyc

    from .flydsl.gemm_mxfp8_4w import compile_gemm_fp8

    launcher = compile_gemm_fp8(
        256, 256, 128, N, K, lds_swizzle=kernel.lds_swizzle, b_lds_swizzle=True,
        preshuffle_b=True, permlane_epilogue=kernel.permlane_epilogue,
        store_overlap=kernel.store_overlap, with_scale=True, b_mxfp4=True,
    )
    # flyc.compile launches once while compiling; later calls reuse the binary.
    _PYHIP_COMPILED[key] = flyc.compile[{"opt_level": 2}](launcher, *args)


def _gemm_wrapper(
    A, B, ScaleA, ScaleB, *, output, M, N, K, splitk, model_key, aiter_only,
    _impl: str = "aiter", kernel_name: str = "",
):
    """Config supplies only the backend and kernel; tensors always come from this call."""
    global last_dispatch
    impl, kernel = _explicit(kernel_name, M, N, K, splitk)
    if impl != _impl or (aiter_only and impl != "aiter"):
        raise ValueError(f"Backend {_impl!r} does not match kernel {kernel_name!r} or candidate scope")
    warm_key = (kernel.name, M, N, K)
    if torch.cuda.is_current_stream_capturing() and warm_key not in _WARMED_KERNELS:
        raise RuntimeError("Warm up the selected A8W4 kernel before graph capture")
    if impl == "aiter":
        _run_aiter(kernel, A, B, ScaleA, ScaleB, output)
    else:
        _run_pyhip(kernel, A, B, ScaleA, ScaleB, output, M, N, K)
    _WARMED_KERNELS.add(warm_key)
    if record_dispatch:
        last_dispatch = dict(
            _impl=impl, libtype=impl, kernelName=kernel.name,
            splitK=kernel.split_k, M=M, N=N, K=K,
        )
    return output


def _torch_reference(A, B, ScaleA, ScaleB):
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    M, K = A.shape
    N = B.shape[0]

    def unshuffle(scale, rows):
        padded, groups = _pad256(rows), _scale_columns(K)
        plain = (
            scale.view(torch.uint8)[:padded]
            .view(padded // 32, groups // 8, 4, 16, 2, 2)
            .permute(0, 5, 3, 1, 4, 2).contiguous().view(padded, groups)
        )
        return e8m0_to_f32(plain[:rows, :K // 32].contiguous())

    weight = (
        B.view(torch.uint8).view(N // 16, K // 32, 16, 16)
        .permute(0, 2, 1, 3).contiguous().view(N, K // 2)
    )
    b_ref = mxfp4_to_f32(weight) * unshuffle(ScaleB, N).repeat_interleave(32, dim=1)
    a_scales = unshuffle(ScaleA, M)
    reference = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    for row in range(0, M, 4096):
        a_ref = A[row:row + 4096].float() * a_scales[row:row + 4096].repeat_interleave(32, dim=1)
        reference[row:row + 4096].copy_((a_ref @ b_ref.T).bfloat16())
    return reference


def _prune_invalid_configs(configs, sig_args):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("A8W4 autotuning must run before graph capture; warm up in eager mode")
    reference = _torch_reference(*(sig_args[name] for name in ("A", "B", "ScaleA", "ScaleB")))
    if not torch.isfinite(reference).all().item():
        raise RuntimeError("A8W4 reference contains nonfinite values; refusing to tune")
    output = sig_args["output"]
    valid = []
    for config in configs:
        output.fill_(float("nan"))
        try:
            result = _gemm_wrapper(**sig_args, **config.all_kwargs())
        except Exception as error:
            # A sticky GPU fault must stop the search, not become a rejected candidate.
            torch.cuda.synchronize(output.device)
            logger.warning("A8W4 candidate %s rejected: %s: %s", config, type(error).__name__, error)
            continue
        torch.cuda.synchronize(output.device)
        if result is not output or not torch.isfinite(output).all().item():
            logger.warning("A8W4 candidate %s rejected: output identity/finite check failed", config)
            continue
        diff = pyhip.calc_diff(output, reference)
        if not math.isfinite(diff) or diff > _DIFF_LIMIT:
            logger.warning("A8W4 candidate %s rejected: calc_diff=%g > %g", config, diff, _DIFF_LIMIT)
            continue
        valid.append(config)
    if not valid:
        raise RuntimeError("No A8W4 candidate passed validation; check shape, weight and scale layouts")
    return valid


@functools.cache
def _implementation_key():
    from .flydsl import gemm_mxfp8_4w
    from aiter.ops.flydsl import mxscale_preshuffle_kernels
    from aiter.ops.flydsl.kernels.gemm import mxscale_preshuffle

    digest = hashlib.sha256()
    for path in (
        __file__, gemm_mxfp8_4w.__file__, mxscale_preshuffle_kernels.__file__,
        mxscale_preshuffle.__file__, _aiter_catalog().__file__,
    ):
        digest.update(Path(path).read_bytes())
    return {
        "source": digest.hexdigest(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "flydsl", "triton")
        },
    }


def _model_key(A, B, ScaleA, ScaleB, splitk, aiter_only):
    props = torch.cuda.get_device_properties(A.device)
    values = dict(
        version=1, implementation=_implementation_key(),
        layout="gfx950_b16_scale_mfma_tile_1x32", out_dtype="bf16",
        splitk=splitk, aiter_only=aiter_only,
        tensors=[
            (name, tuple(t.shape), str(t.dtype), tuple(t.stride()))
            for name, t in (("A", A), ("B", B), ("ScaleA", ScaleA), ("ScaleB", ScaleB))
        ],
        device=(props.name, props.gcnArchName, get_cu_num()),
        precision=(torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32),
        environment=sorted(
            (name, value) for name, value in os.environ.items()
            if name.startswith(("AITER_", "PYHIP_"))
        ),
    )
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _make_tune_args(A, B, ScaleA, ScaleB, output, splitk=None, aiter_only=False):
    return dict(
        A=A, B=B, ScaleA=ScaleA, ScaleB=ScaleB, output=output,
        M=A.shape[0], N=B.shape[0], K=A.shape[1], splitk=splitk, aiter_only=aiter_only,
        model_key=_model_key(A, B, ScaleA, ScaleB, splitk, aiter_only),
    )


_autotuned_gemm = autotune(
    configs=_configs,
    key=_AUTOTUNE_KEYS,
    prune_configs_by=_prune_invalid_configs,
    artifact_name=_ARTIFACT_NAME,
)(_gemm_wrapper)


def _gemm_a8w4_mxfp8(
    A, B, ScaleA, ScaleB, dtype, a_preshuffle, kernelName, splitk, *, aiter_only=False,
):
    global last_dispatch, _native_device
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        result = _aiter_gemm_a8w4_mxfp8(
            A, B, ScaleA, ScaleB, dtype, a_preshuffle, kernelName, splitk,
        )
        if record_dispatch:
            last_dispatch = dict(_impl="aiter", libtype="aiter", source="passthrough", gfx=gfx)
        return result
    M, N, K = _check_gfx950(A, B, ScaleA, ScaleB, dtype, a_preshuffle)
    if splitk is not None and (type(splitk) is not int or splitk < 1):
        raise ValueError(f"splitk must be a positive integer, got {splitk!r}")
    if M == 0:
        if record_dispatch:
            last_dispatch = dict(_impl=None, libtype=None, source="empty", M=M, N=N, K=K)
        return torch.empty((M, N), dtype=dtype, device=A.device)
    # Both libraries cache device-loaded FlyDSL modules process-wide.
    if _native_device is not None and _native_device != A.device:
        raise RuntimeError("A8W4 FlyDSL backends require one GPU per process")
    _native_device = A.device
    with torch.cuda.device(A.device), torch.no_grad():
        output = torch.empty((M, N), dtype=dtype, device=A.device)
        if kernelName:
            impl, kernel = _explicit(kernelName, M, N, K, splitk)
            result = _gemm_wrapper(
                A, B, ScaleA, ScaleB, output=output, M=M, N=N, K=K,
                splitk=splitk, model_key="", aiter_only=aiter_only,
                _impl=impl, kernel_name=kernel.name,
            )
        else:
            result = _autotuned_gemm(
                **_make_tune_args(A, B, ScaleA, ScaleB, output, splitk, aiter_only)
            )
    if record_dispatch:
        last_dispatch["source"] = "explicit" if kernelName else "autotune"
    return result


@torch.library.custom_op("pyhip::gemm_a8w4_mxfp8", mutates_args=())
def _gemm_a8w4_mxfp8_op(
    A: Tensor, B: Tensor, ScaleA: Tensor, ScaleB: Tensor, dtype: torch.dtype,
    a_preshuffle: bool, kernelName: str, splitk: Optional[int],
) -> Tensor:
    return _gemm_a8w4_mxfp8(A, B, ScaleA, ScaleB, dtype, a_preshuffle, kernelName, splitk)


@_gemm_a8w4_mxfp8_op.register_fake
def _gemm_a8w4_mxfp8_fake(A, B, ScaleA, ScaleB, dtype, a_preshuffle, kernelName, splitk):
    return A.new_empty((A.shape[0], B.shape[0]), dtype=dtype)


def gemm_a8w4_mxfp8(
    A: Tensor,
    B: Tensor,
    ScaleA: Tensor,
    ScaleB: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    a_preshuffle: bool = True,
    kernelName: str = "",
    splitk: int | None = None,
) -> Tensor:
    """MXFP8 (activation) x MXFP4 (weight) GEMM: D[M, N] bf16 = A @ B^T.

    Drop-in for ``aiter.gemm_a8w4_mxfp8``. Off gfx950 every argument goes to Aiter.
    On gfx950 the operands follow the module's layout contract (``a_preshuffle``
    must be False). An explicit ``kernelName`` bypasses autotuning: an Aiter
    ``flydsl_mxpsh_*_F8_F4_B16_*`` name or a PyHIP ``pyhip_mxfp8_4w_*`` name.
    Otherwise FlyDSL autotune validates and selects a configuration, cached by
    exact M/N/K, layouts, implementation identity, device and numerical options.
    FLYDSL_AUTOTUNE=1 forces a search; FLYDSL_AUTOTUNE_CONFIG_DIR enables config
    artifact lookup/export. A miss tunes even when FLYDSL_AUTOTUNE=0: prewarm
    before latency-sensitive calls and graph capture.

    ``splitk`` is a literal count restricting the search or overriding an
    explicit Aiter kernel's split count. PyHIP kernels do not split K.
    Returns a new [M, N] BF16 tensor. Native backends are inference-only and
    currently require one GPU per process.
    """
    if torch.compiler.is_compiling():
        return _gemm_a8w4_mxfp8_op(A, B, ScaleA, ScaleB, dtype, a_preshuffle, kernelName, splitk)
    return _gemm_a8w4_mxfp8(A, B, ScaleA, ScaleB, dtype, a_preshuffle, kernelName, splitk)
