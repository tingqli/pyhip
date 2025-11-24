import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

exprs = [
    lambda a,b : a + b,
    lambda a,b : a - b,
    lambda a,b : a * b,
    lambda a,b : a >> (b & 31),
    lambda a,b : a << (b & 31),
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
    lambda a,b : a + (b>>6) + (128 - a),
    lambda a,b : (a + b)>>2 < (89 - a),
    lambda a,b : (a > 1) | (b < 10) & (a < 899),
]

def test_sexpr():
    @pyhip.jit("(int*, int, int)")
    def kernel(J):
        s_pkargs = J.new_gpr('s',[0,1],name="s_pkargs")
        threadIdx_x = J.new_gpr('v',[0,0],name="threadIdx_x")
        s_pout = J.new_gpr('s', 2, align=2, name="s_pout")
        s_inputs = J.new_gpr('s', 2, dtype="i32", name="s_inputs")
        s_output = J.new_gpr('s', 1, dtype="i32", name="s_output")

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
        ref = int(func(a, b))
        res = OUT[idx] 
        assert ref == res, f"expression {idx} failed: ref {ref} != res {res}"
    print(OUT)

def test_vexpr():
    @pyhip.jit("(int*, int*, int*)")
    def kernel(J):
        s_pkargs = J.new_gpr('s',[0,1],name="s_pkargs")
        threadIdx_x = J.new_gpr('v',[0,0],name="threadIdx_x")
        s_pout = J.new_gpr('s', 2, align=2, name="s_pout")
        s_pin0 = J.new_gpr('s', 2, align=2, name="s_pin0")
        s_pin1 = J.new_gpr('s', 2, align=2, name="s_pin1")
        J.s_load_dwordx2(s_pout, s_pkargs, 0)
        J.s_load_dwordx2(s_pin0, s_pkargs, 8)
        J.s_load_dwordx2(s_pin1, s_pkargs, 16)
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        v_inputs = J.new_gpr('v', 2, dtype="i32", name="v_inputs")
        v_output = J.new_gpr('v', 1, dtype="i32", name="v_output")
        vaddr = J.new_gpr('v', 1, dtype="i32", name="v_output")

        vaddr[0] = threadIdx_x[0] << 2
        J.global_load_dword(v_inputs[0], vaddr, s_pin0)
        J.global_load_dword(v_inputs[1], vaddr, s_pin1)
        J.s_waitcnt(mod=f"vmcnt({0})")
        for idx, func in enumerate(exprs):
            v_output[0] = func(v_inputs[0], v_inputs[1])
            #  vaddr,    vdata,       saddr       offset13s sc0 nt sc1
            J.global_store_dword(vaddr, v_output, s_pout)
            vaddr[0] = vaddr[0] + 64*4

        J.s_waitcnt(mod=f"vmcnt({0})")

    OUT = torch.zeros(32, 64, dtype=torch.int)
    low = -65536*100
    high = 65536*100
    a = torch.randint(low, high, (64,), dtype=torch.int)
    b = torch.randint(0, high, (64,), dtype=torch.int)
    kernel([1],[64], OUT.data_ptr(), a.data_ptr(), b.data_ptr())
    torch.cuda.synchronize()
    for idx, func in enumerate(exprs):
        ref = func(a, b).to(dtype=torch.int)
        res = OUT[idx, :]
        assert torch.allclose(ref, res), f"expression {idx} failed: ref {ref[:8]} != res {res[:8]}"
    
if __name__ == "__main__":
    test_sexpr()
    test_vexpr()