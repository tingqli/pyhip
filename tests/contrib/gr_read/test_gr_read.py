# SPDX-License-Identifier: MIT
"""GRRead回归；直接执行默认先检查全部16档batch，再输出Down/Up/Total性能。

pytest只收集下面的功能用例，不自动运行性能矩阵。
"""

# 必须先解析CLI并选卡，再导入Torch/PyHIP；直接执行不依赖pytest。
if __name__ == "__main__":
    if __package__:
        from .benchmark import main
    else:
        from benchmark import main
    raise SystemExit(main())

import pytest

from pyhip.contrib.flydsl.gr_read.common import H, K, R, select_n_splits


@pytest.mark.parametrize("k,splits", ((1,8),(2,8),(4,4),(8,2),(10,2),(12,8),(16,8),(20,2),
                                    (24,4),(28,2),(30,2),(32,8),(36,2),(48,2),(60,2),(64,4)))
def test_batch_dispatch(k, splits):
    assert select_n_splits(k * 1024, 80) == splits


@pytest.mark.parametrize("rows", (-1, 65537, True, 1.5, "1024"))
def test_invalid_rows(rows):
    with pytest.raises(ValueError):
        select_n_splits(rows, 80)


def test_dispatch_boundaries():
    assert select_n_splits(0, 80) == 8
    for rows in range(1, 65537):
        assert select_n_splits(rows, 80) in (2, 4, 8)
    for cu in (0, -1, True):
        with pytest.raises(ValueError):
            select_n_splits(1, cu)


def require_gpu():
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("gfx942 required")
    return torch


@pytest.mark.parametrize("rows", (0, 1, 129, 257, 2561, 5121))
def test_prepared_reader(rows):
    torch = require_gpu()
    from pyhip.contrib.flydsl.gr_read import CombinedPaddedGRRead
    from .reference import DOWN_TOLERANCE, OUTPUT_TOLERANCE, check_close, reference_bf16

    gen = torch.Generator(device="cuda").manual_seed(131)
    x = torch.randn((rows, K), device="cuda", dtype=torch.bfloat16, generator=gen)
    wd = torch.randn((R, K), device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
    wu = torch.randn((K, R), device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
    reader = CombinedPaddedGRRead(rows, wd, wu)
    assert reader.n_splits == select_n_splits(rows, torch.cuda.get_device_properties().multi_processor_count)
    assert reader.partial.dtype == reader.output.dtype == torch.bfloat16
    pointer = reader.output.data_ptr()
    if rows == 0:
        assert reader(x) is reader.output and reader.run_down(x).numel() == 0
        assert reader.run_up(x).shape == (0, H)
        assert reader.down is reader.up is None
        return
    p_expected, expected = reference_bf16(x, wd, wu)
    reader.partial.fill_(torch.nan); reader.output.fill_(torch.nan)
    assert reader(x) is reader.output
    check_close(reader.partial.view(-1, R)[:rows], p_expected, DOWN_TOLERANCE)
    check_close(reader.output, expected, OUTPUT_TOLERANCE)
    assert (reader.partial.view(-1, R)[rows:] == 0).all().item()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reader(x); reader(x)
    for _ in range(3):
        x.copy_(torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen))
        p_expected, expected = reference_bf16(x, wd, wu)
        reader.output.fill_(torch.nan); graph.replay()
        check_close(reader.partial.view(-1, R)[:rows], p_expected, DOWN_TOLERANCE)
        check_close(reader.output, expected, OUTPUT_TOLERANCE)
        assert reader.output.data_ptr() == pointer
    with pytest.raises(ValueError):
        reader(x[:, :-1])
    with pytest.raises(ValueError):
        reader(x.float())
    noncontiguous = torch.empty((rows, K * 2), device=x.device, dtype=x.dtype)[:, ::2]
    with pytest.raises(ValueError):
        reader(noncontiguous)