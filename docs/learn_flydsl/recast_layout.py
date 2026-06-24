import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

import pyhip.contrib.flydsl.utils as fxu
# fxu.enable_dump_ir(True)

@flyc.jit
def test_recast_layout(input:fx.Tensor):
    print(input.dtype, type(input.dtype), input.dtype.width)
    print(fx.BFloat16, type(fx.BFloat16), fx.BFloat16.width)

    # layoutRecast(old=16bits, new=32bits)
    #   layoutUpcastImpl(factor=2)
    """
        Layout<(64,32):(1,64)> 16bits -> Layout<(32,32):(1,32)> 32bits
        shape changed on dimensions with stride < 2, because that dimension is reinterpreted as 32bits,
        which is 2x of 16bits. every 2 elements in that dimension will be merged into 1 element in the new layout.
    """
    A = fx.Tensor(fx.make_view(fx.get_iter(input), fx.make_layout((64,32), (1, 64))))
    new_layout = fx.recast_layout(A.layout, input.dtype.width, fx.Float32.width)
    print(f"{A.layout} {A.dtype.width}bits -> {new_layout} {fx.Float32.width}bits ===== shape changed")

    """
        Layout<(64,32):(32,2)> 16bits -> Layout<(64,32):(16,1)> 32bits
        shape unchanged on dimensions with stride >= 2, because that dimension is reinterpreted as 32bits,
        which requires no need for merging elements in that dimension.
    """
    A = fx.Tensor(fx.make_view(fx.get_iter(input), fx.make_layout((64,32), (32, 2))))
    new_layout = fx.recast_layout(A.layout, input.dtype.width, fx.Float32.width)
    print(f"{A.layout} {A.dtype.width}bits -> {new_layout} {fx.Float32.width}bits ===== shape unchanged")
    
    """
        dynamic case:
    """
    new_layout = fx.recast_layout(input.layout, input.dtype.width, fx.Float32.width)
    print(f"{input.layout} {input.dtype.width}bits -> {new_layout} {fx.Float32.width}bits")
    fx.printf("[runtime] layout {} => {}", input.layout, new_layout)
    
    #
    #fx.printf("[runtime] new layout {}", new_layout)

test_recast_layout(torch.zeros(64, 32, dtype=torch.bfloat16))
