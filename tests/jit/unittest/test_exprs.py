import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_expr():

    exprs = [
        lambda a,b : a + b,
        lambda a,b : a - b,
        lambda a,b : a * b,
        lambda a,b : a >> b,
        lambda a,b : a << b,
        lambda a,b : a & b,
        lambda a,b : a | b,
        lambda a,b : a ^ b,
        lambda a,b : a == b,
        lambda a,b : a != b,
        lambda a,b : a > b,
        lambda a,b : a >= b,
        lambda a,b : a < b,
        lambda a,b : a <= b,
        lambda a,b : a + b*8,
        lambda a,b : a + b>>6 + (128 - a),
    ]

    @pyhip.jit("(int*, int, int)")
    def kernel(J):
        s_pkargs = J.new_gpr('s',[0,1],name="s_pkargs")
        threadIdx_x = J.new_gpr('v',[0,0],name="threadIdx_x")
        s_pout = J.new_gpr('s', 2, align=2, name="s_pout")
        s_inputs = J.new_gpr('s', 2, dtype="u32", name="s_inputs")
        s_output = J.new_gpr('s', 1, dtype="u32", name="s_output")

        J.s_load_dwordx2(s_pout, s_pkargs, 0)
        J.s_load_dword(s_inputs[0], s_pkargs, 8)
        J.s_load_dword(s_inputs[1], s_pkargs, 8+4)
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        for idx, func in enumerate(exprs):
            s_output[0] = func(s_inputs[0], s_inputs[1])
            J.s_store_dword(s_output, s_pout, idx*4, mod="glc")

        J.s_waitcnt(mod=f"lgkmcnt({0})")

    OUT = torch.ones(32, dtype=torch.int)
    a = 123
    b = 3
    kernel([1],[64], OUT.data_ptr(), a, b)
    torch.cuda.synchronize()
    for idx, func in enumerate(exprs):
        assert OUT[idx] == func(a, b)
    print(OUT)

test_expr()