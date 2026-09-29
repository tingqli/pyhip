"""兼容不同 Aiter 版本的 MoE 参数和 caller-owned output。"""

from functools import wraps
from inspect import signature

from aiter.fused_moe import fused_moe as _fused_moe, moe_sorting as _moe_sorting


_fused_moe_parameters = signature(_fused_moe).parameters
_unsupported_kwargs = tuple(
    name for name in ("stage2_scatter", "quant_type_a", "quant_dtype_a", "quant_dtype_a2")
    if name not in _fused_moe_parameters
)


@wraps(_moe_sorting)
def moe_sorting(*args, output=None, **kwargs):
    result = _moe_sorting(*args, **kwargs)
    if output is None:
        return result
    # Aiter 只清零内部 moe_buf；本地 atomic kernel 写 caller 的 out，需要单独清零。
    # 与排序处在同一 stream，graph replay 也会执行，不做主机同步。
    output.zero_()
    return (*result[:4], output, *result[5:])


@wraps(_fused_moe)
def fused_moe(*args, output=None, **kwargs):
    # Only omit unsupported defaults; explicit values still reach Aiter and fail.
    for name in _unsupported_kwargs:
        if kwargs.get(name) is None:
            kwargs.pop(name, None)
    result = _fused_moe(*args, **kwargs)
    if output is not None:
        if (output.shape != result.shape or output.dtype != result.dtype
                or output.device != result.device):
            raise RuntimeError("output must match Aiter result's shape/dtype/device")
        output.copy_(result)
        return output
    return result
