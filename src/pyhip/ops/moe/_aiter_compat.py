"""补齐当前 Aiter 尚未提供的 caller-owned output 接口。"""

from functools import wraps

from aiter.fused_moe import fused_moe as _fused_moe, moe_sorting as _moe_sorting


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
def fused_moe(*args, output=None, stage2_scatter=None, **kwargs):
    if stage2_scatter is not None:
        raise NotImplementedError("installed Aiter fused_moe does not support stage2_scatter")
    result = _fused_moe(*args, **kwargs)
    if output is not None:
        if (output.shape != result.shape or output.dtype != result.dtype
                or output.device != result.device):
            raise RuntimeError("output must match Aiter result's shape/dtype/device")
        output.copy_(result)
        return output
    return result
