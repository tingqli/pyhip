# SPDX-License-Identifier: MIT
"""按 rows 查看 GR read 算法，并用 torch.allclose 检查普通调用的精度。

    python3 tests/ops/gr_read/test_gr_read_algorithms.py
    python3 tests/ops/gr_read/test_gr_read_algorithms.py --gpu 2 --rows 1 24 32 33 64 512 4k

算法速查（gfx942 / 80CU；实际 prefill 配置由 common.py 选择）：
  T1..32    decode：Down split-K4 写 FP32 partial；Up 归约 + SiLU high/low + GEMM。
  T33..128  prefill：Down M16/W2/BK1024 + SiLU；Up M64。
  T129..256 prefill：Down M16/W2/BK512 + SiLU；Up M128。
  T257..512 prefill：Down M32/W4/BK512 + SiLU；Up M256。
  T>512     prefill：沿用现有 Down/Up 配置选择，打印对象实际使用的配置。

仅用于阅读算法和检查精度，不测性能、不捕获 CUDA Graph，不依赖 SGLang。
"""
import argparse

if __package__:
    from . import test_gr_read as checks
else:
    import test_gr_read as checks


# 代表性 rows 覆盖 decode 特化与 prefill 配置边界；可用 --rows 自选。
DEFAULT_ROWS = (1, 8, 16, 17, 24, 32,
                33, 48, 64, 512, 513, 2048)


def prepare_reader(rows, packed_down, packed_up):
    """唯一的算法路由；复用正式对象，不在本文件实现 kernel。"""
    from pyhip.ops.gr_read.flydsl import GRReadDecode, GRReadPrefill

    if rows <= 0:
        raise ValueError("rows must be positive")
    if rows <= 32:
        return GRReadDecode(rows, packed_down, packed_up)
    return GRReadPrefill(rows, packed_down, packed_up)


def describe_reader(reader):
    rows = reader.rows
    if rows <= 32:
        # 对应 down.py / up.py 当前的编译期配置，仅用于说明，不参与 launch。
        unroll = 2 if rows <= 25 else 1
        up_bk = 32 if 9 <= rows <= 16 or 29 <= rows <= 31 else 160
        skip_padding = not 9 <= rows <= 16
        return (f"decode split-K4 | Down M{reader.padded_rows}/W4/BK128/U{unroll} | "
                f"Up M16/BN128/BK{up_bk}/skip_padding={skip_padding}/preload_W={rows <= 16} | "
                f"P=FP32[4,{reader.padded_rows},320]")
    dm, dw, dn, dk = reader.down_config
    um, un = reader.up_config
    return (f"prefill | Down+SiLU M{dm}/W{dw}/Nsplit{dn}/BK{dk} | "
            f"Up M{um}/Nsplit{un} | swizzle={reader.config[-1]} | P=BF16[{rows},320]")


def check_case(rows, w_down, w_up, packed, seed):
    import torch

    generator = torch.Generator(device=w_down.device).manual_seed(seed + rows)
    x = torch.randn(rows, checks.C * checks.H, dtype=torch.bfloat16,
                    device=w_down.device, generator=generator)
    reader = prepare_reader(rows, *packed)
    print(f"T={rows}: {describe_reader(reader)}", flush=True)
    assert reader.w_down.data_ptr() == packed[0].data_ptr()
    assert reader.w_up.data_ptr() == packed[1].data_ptr()

    # 两条路径都直接调用。先预填 NaN，避免未写出的 P/Y 被误判通过。
    reader.partial.fill_(float("nan"))
    reader.output.fill_(float("nan"))
    actual = reader(x)
    if rows <= 32:
        expected = checks.decode_reference(x, w_down, w_up)
        partial = reader.partial.view(4, reader.padded_rows, checks.R)
        # Decode 的 P 是四份线性 partial，SiLU 尚未执行。
        for split in range(4):
            begin, end = split * checks.H, (split + 1) * checks.H
            reference = x[:, begin:end].double() @ w_down[:, begin:end].double().T
            assert torch.allclose(partial[split, :rows].double(), reference,
                                  rtol=2e-5, atol=1e-5), f"T={rows}: Down split {split}"
        assert torch.count_nonzero(partial[:, rows:]) == 0, f"T={rows}: P padding"
        reference_name = "FP64"
    else:
        # Prefill 保留原 BF16 舍入边界及分块参考，不换成 decode 的参考。
        expected_p, expected = checks.reference_bf16(x, w_down, w_up)
        assert torch.allclose(reader.partial.double(), expected_p.double(),
                              **checks.DOWN_TOLERANCE), f"T={rows}: Down+SiLU"
        reference_name = "BF16 Torch compile"
    assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
    assert torch.allclose(actual.double(), expected.double(),
                          **checks.OUTPUT_TOLERANCE), f"T={rows}: output vs {reference_name}"
    print(f"  PASS: P and Y; reference={reference_name}; ordinary call, no CUDA Graph", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", nargs="+", type=checks.parse_batch, default=list(DEFAULT_ROWS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=131)
    args = parser.parse_args(argv)
    if args.gpu < 0 or not __debug__:
        parser.error("GPU must be nonnegative; do not use python -O")
    checks.prepare_cli_environment(args)
    torch, _, _, _ = checks.dependencies()
    from pyhip.ops.gr_read.flydsl import prepare_weights

    with torch.no_grad():
        # 一对原始权重只打包一次，所有 rows、decode/prefill 共用相同指针。
        _, w_down, w_up = checks.make_inputs(torch, 1, args.seed)
        packed = prepare_weights(w_down, w_up)
        for rows in args.rows:
            try:
                check_case(rows, w_down, w_up, packed, args.seed)
            finally:
                checks.release_buffers()
    print(f"All {len(args.rows)} algorithm cases passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
