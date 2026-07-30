import pyhip
import torch
from typing import Any, cast

INT_POINTER = "int*"


def _rotate_wave(J, value, steps, wait_states):
    current = value
    if wait_states:
        J.s_nop(mod=str(wait_states - 1))
    for _ in range(steps):
        rotated = J.gpr("vu32")
        J.v_mov_b32(rotated, current, mod="wave_rol:1")
        if wait_states:
            J.s_nop(mod=str(wait_states - 1))
        current = rotated
    return current


def test_wave_rol_column_reduce():
    @pyhip.jit()  # pyright: ignore[reportAttributeAccessIssue]
    def kernel(
        J,
        output: INT_POINTER,  # pyright: ignore[reportInvalidTypeForm]
        wait_states,
    ):
        lane = J.gpr("vu32", J.threadIdx.x[0])
        value = J.gpr("vu32", lane)
        rotated16 = _rotate_wave(J, value, 16, wait_states)
        rotated32 = _rotate_wave(J, rotated16, 16, wait_states)
        rotated48 = _rotate_wave(J, rotated32, 16, wait_states)

        max_value = J.gpr("vu32", value)
        J.v_max_u32(max_value, max_value, rotated16)
        J.v_max_u32(max_value, max_value, rotated32)
        J.v_max_u32(max_value, max_value, rotated48)

        sum_value = J.gpr("vu32", value)
        J.v_add_u32(sum_value, sum_value, rotated16)
        J.v_add_u32(sum_value, sum_value, rotated32)
        J.v_add_u32(sum_value, sum_value, rotated48)

        offset = J.gpr("vu32", lane * 4)
        J.global_store_dword(offset, rotated16, output)
        J.global_store_dword(offset, rotated32, output, mod="offset:256")
        J.global_store_dword(offset, rotated48, output, mod="offset:512")
        J.global_store_dword(offset, max_value, output, mod="offset:768")
        J.global_store_dword(offset, sum_value, output, mod="offset:1024")
        J.s_waitcnt(mod="vmcnt(0)")

    output = torch.empty(5, 64, dtype=torch.int32, device="cuda")
    compiled_kernel = cast(Any, kernel)
    compiled_kernel([1], [64], output.data_ptr(), 2)
    torch.cuda.synchronize()

    lane = torch.arange(64, dtype=torch.int32, device="cuda")
    expected16 = (lane + 16) % 64
    expected32 = (lane + 32) % 64
    expected48 = (lane + 48) % 64
    assert torch.equal(output[0], expected16), (output[0].cpu(), expected16.cpu())
    assert torch.equal(output[1], expected32), (output[1].cpu(), expected32.cpu())
    assert torch.equal(output[2], expected48), (output[2].cpu(), expected48.cpu())

    column = lane % 16
    expected_max = column + 48
    expected_sum = 4 * column + 96
    assert torch.equal(output[3], expected_max)
    assert torch.equal(output[4], expected_sum)


def _run_three_wave_shr_then_readlane():
    @pyhip.jit()  # pyright: ignore[reportAttributeAccessIssue]
    def kernel(
        J,
        output: INT_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    ):
        lane = J.gpr("vu32", J.threadIdx.x[0])
        row = J.gpr("vu32", lane >> 4)
        column = J.gpr("vu32", lane & 15)
        maximum = J.gpr("vu32", row * 100 + column)
        offset = J.gpr("vu32", lane * 4)

        # 精确实现：每次wave_shr后立即与当前maximum合并，共执行三次。
        stages = []
        for _ in range(3):
            J.s_nop(mod="1")
            shifted = J.gpr("vu32")
            J.v_mov_b32(shifted, maximum, mod="wave_shr:1 bound_ctrl:0")
            J.s_nop(mod="1")
            J.v_max_u32(maximum, maximum, shifted)
            stages.append(J.gpr("vu32", maximum))

        scalar_maximum = J.gpr("su32")
        J.v_readlane_b32(scalar_maximum, maximum, 63)
        broadcast_maximum = J.gpr("vu32", scalar_maximum)
        J.global_store_dword(offset, stages[0], output)
        J.global_store_dword(offset, stages[1], output, mod="offset:256")
        J.global_store_dword(offset, stages[2], output, mod="offset:512")
        J.global_store_dword(offset, broadcast_maximum, output, mod="offset:768")
        J.s_waitcnt(mod="vmcnt(0)")

    output = torch.empty(4, 64, dtype=torch.int32, device="cuda")
    compiled_kernel = cast(Any, kernel)
    compiled_kernel([1], [64], output.data_ptr())
    torch.cuda.synchronize()

    lane = torch.arange(64, dtype=torch.int32, device="cuda")
    column = lane % 16
    expected_column_maximum = 300 + column

    # readlane读取一个lane到SGPR，再广播写回；它只能产生一个全wave标量。
    assert torch.all(output[3] == output[3, 0])
    assert output[3, 0].item() == output[2, 63].item()

    # 三次wave_shr合并的是相邻lane窗口，不是{q,q+16,q+32,q+48}列组。
    assert not torch.equal(output[2], expected_column_maximum), (
        output[2].cpu(),
        expected_column_maximum.cpu(),
    )

    actual = output[2]
    # 初值按100*row+column编码；每个row内递增，边界处受bound_ctrl补0影响。
    encoded = 100 * (lane // 16) + column
    adjacent_encoded = torch.stack(
        [
            encoded,
            torch.roll(encoded, 1),
            torch.roll(encoded, 2),
            torch.roll(encoded, 3),
        ]
    ).amax(dim=0)
    adjacent_encoded[:3] = encoded[:3]
    assert torch.equal(actual, adjacent_encoded)

    return {
        "x_lane_20": output[0, 20].item(),
        "y_lane_20": output[1, 20].item(),
        "z_lane_20": output[2, 20].item(),
        "column_max_lane_20": expected_column_maximum[20].item(),
        "readlane_63": output[3, 0].item(),
    }


def test_three_wave_shr_then_readlane():
    result = _run_three_wave_shr_then_readlane()
    assert result == {
        "x_lane_20": 104,
        "y_lane_20": 104,
        "z_lane_20": 104,
        "column_max_lane_20": 304,
        "readlane_63": 315,
    }


if __name__ == "__main__":
    test_wave_rol_column_reduce()
    print(_run_three_wave_shr_then_readlane())
