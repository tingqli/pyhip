# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""8x1编译期VMEM账本：同时约束B提交和本拍打包的scale消费者。"""

from functools import cache


def packing_events(k, stage, first=False, k_widths=None):
    """返回(packet, previous-N, super-record)，不是当前MFMA的输出分片。"""
    assert k != 192, "K192 uses the independent BK192 helper"
    assert k in (256, 320, 384, 512, 640), "shared 8x1 schedule不支持该K"
    assert k_widths == (128, 192) if k == 320 else k_widths is None
    ks = (k + 127) // 128
    if k == 320:
        if stage == 0 and not first:
            return ((0, True, 2), (1, True, 3))
        if stage == 2:
            return ((0, False, 0), (1, False, 1))
    else:
        if stage == 0 and not first:
            return ((0, True, 3),)
        if stage == ks - 1:
            return ((1, False, 0),)
        if stage == ks:
            return ((0, False, 1),)
        if stage == 2 * ks - 1:
            return ((1, False, 2),)
    return ()


def output_quarter(k, stage, k_widths=None):
    assert k != 192, "K192 uses the independent BK192 helper"
    assert k in (256, 320, 384, 512, 640), "shared 8x1 schedule不支持该K"
    assert k_widths == (128, 192) if k == 320 else k_widths is None
    return stage if stage < 4 else None


@cache
def vmem_wait_schedule(k, n_tiles, ptpc=True, rolling=True, relax=True, k_widths=None):
    """普通buffer/global每条指令一个事件；wait后才issue下一条B。

    预算取所有即将消费的B/scale的年龄最小值。不能只计算B的年龄：
    K256某些packet用的是上一拍scale，会把9收紧到7。
    pure保留原保守阈值；其scale在退休点显式wait(0)，不按rolling推断。
    K192使用独立BK192账本；K320必须显式传入唯一分块(128, 192)。
    """
    assert k in (256, 320, 384, 512, 640) and n_tiles >= 1, "shared 8x1 schedule不支持该K或N块数"
    assert k_widths == (128, 192) if k == 320 else k_widths is None
    widths = k_widths or tuple(min(128, k - 128 * i) for i in range((k + 127) // 128))
    ks, events, requests, scales = len(widths), [], {}, {}

    def valid(q):
        n, stage = divmod(q, 2 * ks)
        return 0 <= q < n_tiles * 2 * ks and n * ks + stage % ks + 1 < n_tiles * ks

    def request(q):
        if valid(q):
            # 末块192每lane24B，实际16B+8B两条VMEM；以最后一条约束提交。
            target_k = (q % ks + 1) % ks
            events.extend([("B", q)] * (2 if widths[target_k] == 192 else 1))
            requests[q] = len(events) - 1

    request(0)
    request(1)
    result = []
    for q in range(n_tiles * 2 * ks):
        n, stage = divmod(q, 2 * ks)
        scale_count = 2 if rolling and ptpc and stage < 4 else 0
        events.extend([("scale", n, stage)] * scale_count)
        if scale_count:
            scales[n, stage] = len(events) - 1
        store_count = (2 if rolling and n > 0 and output_quarter(k, stage, k_widths) is not None
                       else 8 if not rolling and n > 0 and stage == 0 else 0)
        events.extend([("store", n, stage)] * store_count)
        required = [requests[q]] if valid(q) else []
        if rolling and ptpc:
            for _, previous, record in packing_events(k, stage, n == 0, k_widths):
                required.append(scales[n - int(previous), record])
        old = scale_count + store_count + int(valid(q + 1))
        if k == 512 and ptpc and rolling and n > 0 and 0 < stage <= 4 and valid(q + 1):
            old += 4
        budget = min((len(events) - 1 - index for index in required), default=63)
        if k_widths is not None:
            # 新末块路径pure也按native B保护，scale在pure pack点另行wait(0)。
            result.append(budget if relax else 0)
        else:
            result.append(budget if relax and rolling else old)
        request(q + 2)
    return tuple(result)