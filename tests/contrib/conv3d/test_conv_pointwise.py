import torch
from pyhip.contrib.conv_pointwise import *
import pyhip

torch.set_default_device("cuda")
torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
# torch.cuda.set_device(0)
torch.manual_seed(0)

def test_conv_pointwise_gl():
    group_size = 4
    B,C,D,H,W = 1, 2048, 65,45,80
    input = torch.randn([B, C, D,H,W], dtype=torch.bfloat16)
    weight = torch.randn([C//group_size, group_size, group_size], dtype=torch.bfloat16)
    output = torch.zeros([B, C, D,H,W], dtype=torch.bfloat16)
    bias = torch.randn([C], dtype=torch.bfloat16)

    #input[...] = 1
    #weight[...] = 1
    #bias[...] = 0
    
    #ref = torch.matmul(weight.reshape(group_size, group_size), input.reshape(group_size, D*H*W)) + bias.reshape(C, 1)

    ref = torch.nn.functional.conv1d(input.view(B, C, D*H*W), weight.view(C, group_size, 1), 
                            bias=bias,
                            stride=1, 
                            padding=0, 
                            dilation=1,
                            groups=C//group_size).view(output.shape)

    grid = (B, C//group_size)
    for _ in range(10):
        with pyhip.cudaPerf(rw_bytes=B*C*D*H*W*2, name="conv_pointwise_gl"):
            output = conv_pointwise(input, weight, bias, groups=C//group_size, use_gluon=True)

    all_diff_gl = pyhip.calc_diff(ref, output)
    print(f"{all_diff_gl=:.6f}")

    output[...] = 0
    for _ in range(10):
        with pyhip.cudaPerf(rw_bytes=B*C*D*H*W*2, name="conv_pointwise_jit"):
            output = conv_pointwise(input, weight, bias, groups=C//group_size, use_gluon=False)

    all_diff_jit = pyhip.calc_diff(ref, output)
    print(f"{all_diff_jit=:.6f}")

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @gluon.jit
    def test_gl(output_ptr):
        pid = gl.program_id(0)

        layout0: gl.constexpr = gl.BlockedLayout([4, 1], [1, 64], [1, 1], order=[0, 1])
        layout1: gl.constexpr = gl.BlockedLayout([4, 1], [1, 64], [1, 1], order=[1, 0])
        v = gl.arange(0, 64, layout=gl.SliceLayout(1, layout1))

        layout2: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        tid = gl.arange(0, 64, layout=layout2)

        """
        打印 to_linear_layout 对于理解 BlockedLayout/SliceLayout 非常有用
        SliceLayout(dim=1, layout1) 就是拿掉layout1的第1维，
        """
        gl.static_print(">>>>>> layout0 ", gl.to_linear_layout(layout0, [4,64]))
        gl.static_print(">>>>>> layout1 ", gl.to_linear_layout(layout1, [4,64]))
        gl.static_print()
        gl.static_print(">>>>>> SliceLayout(0, layout0) ", gl.to_linear_layout(gl.SliceLayout(0, layout0), [4]))
        gl.static_print(">>>>>> SliceLayout(0, layout1) ", gl.to_linear_layout(gl.SliceLayout(0, layout1), [4]))
        gl.static_print()
        v_l0 = gl.zeros([64, 64], gl.float32, layout=layout0)
        v_l1 = gl.convert_layout(v_l0, layout1, assert_trivial=True)

        slice_layout0: gl.constexpr = gl.SliceLayout(0, layout1)
        slice_layout1: gl.constexpr = gl.SliceLayout(1, layout1)
        gl.static_print(">>>>>> slice_layout0 ", gl.to_linear_layout(slice_layout0, [4]))
        gl.static_print(">>>>>> slice_layout0 ", gl.to_linear_layout(slice_layout0, [8]))
        gl.static_print(">>>>>> slice_layout0 ", gl.to_linear_layout(slice_layout0, [64]))
        gl.static_print(">>>>>> slice_layout0 ", gl.to_linear_layout(slice_layout0, [64*2]))
        gl.static_print(">>>>>> slice_layout0 ", gl.to_linear_layout(slice_layout0, [64*4]))
        gl.static_print(">>>>>> slice_layout0 ", gl.to_linear_layout(slice_layout0, [64*8]))
        gl.static_print()
        gl.static_print(">>>>>> slice_layout1 ", gl.to_linear_layout(slice_layout1, [4]))
        gl.static_print(">>>>>> slice_layout1 ", gl.to_linear_layout(slice_layout1, [8]))
        gl.static_print(">>>>>> slice_layout1 ", gl.to_linear_layout(slice_layout1, [64]))
        gl.static_print(">>>>>> slice_layout1 ", gl.to_linear_layout(slice_layout1, [128]))

        gl.static_print(tid)
        # gl.amd.cdna3.buffer_store(v, output_ptr, tid)
        return
except:
    pass

if __name__ == "__main__":
    # TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=gl 
    test_conv_pointwise_gl()

    if 0:
        debug = torch.zeros(64, dtype=torch.int)
        grid = (1,)
        test_gl[grid](debug, num_warps=1)
        print(debug)