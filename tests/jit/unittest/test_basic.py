import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_basic():
    @pyhip.jit()
    def kernel(J, pA:"int*", cnt:"int"):
        acc = J.new_gpr("a", 4,name="acc")
        s_idx = J.new_gpr('s',1, dtype="u32", name="s_idx")
        s_temp0 = J.new_gpr('s',1,name="s_temp0")
        s_temp = J.new_gpr('s',2, align=2,name="s_temp")
        s_temp2 = J.new_gpr('s',2, align=2,name="s_temp2")
        vtemp = J.new_gpr('v',2, dtype="u32", align=2,name="vtemp")

        for i in range(4):
            J.v_accvgpr_write_b32(acc[i], 0)

        s_idx[0] = 0
        J.Label("bb0")

        #J.s_lshl_b32(s_temp[1],1, s_idx)
        J.s_lshl_b32(s_temp0,s_idx,2)
        #s_temp[0] = s_idx[0] << 2

        s_temp[:] = pA[:] + s_temp0[0]
        J.s_store_dword(cnt, s_temp, 0, mod="glc")

        J.s_add_u32(s_temp2[0], pA[0], s_temp0)

        J.s_addk_i32(s_idx, 1)
        J.s_cmp_lt_i32(s_idx, cnt)
        J.s_cbranch_scc0(mod="bb1")
        J.s_branch(mod="bb0")

        J.Label("bb1")

    A = torch.ones(64, dtype=torch.int)
    CNT = 31
    kernel([1],[64], A.data_ptr(), CNT)
    torch.cuda.synchronize()
    ref = torch.ones(64, dtype=torch.int)
    ref[:CNT] = CNT
    torch.testing.assert_close(A, ref)

def test_basic2():
    @pyhip.jit()
    def kernel(J, pA:"int*", cnt:"int"):
        s_idx = J.new_gpr('s',1, dtype="u32", name="s_idx")
        s_offset = J.new_gpr('s',1,dtype="u32", name="s_offset")
        s_temp = J.new_gpr('s',2, dtype="u32", align=2,name="s_temp")

        J.s_waitcnt(mod=f"lgkmcnt({0})")

        s_idx[0] = 0
        J.Label("bb0")

        s_offset[0] = s_idx[0] << 2

        s_temp[0] = pA[0] + s_offset[0]
        J.s_addc_u32(s_temp[1], pA[1], 0)

        J.s_store_dword(cnt, s_temp, 0, mod="glc")

        J.s_addk_i32(s_idx, 1)
        J.s_cmp_lt_i32(s_idx, cnt)
        J.s_cbranch_scc0(mod="bb1")
        J.s_branch(mod="bb0")

        J.Label("bb1")
        J.s_waitcnt(mod=f"lgkmcnt({0})")

    A = torch.ones(64, dtype=torch.int)
    CNT = 56
    kernel([1],[64], A.data_ptr(), CNT)
    torch.cuda.synchronize()
    ref = torch.ones(64, dtype=torch.int)
    ref[:CNT] = CNT
    torch.testing.assert_close(A, ref)

class Buffer:
    def __init__(self, J):
        self.J = J
        self.desc = J.new_gpr('s', 4, align=4)
        self.base = self.desc[0:1]
        self.range = self.desc[2]
        self.config = self.desc[3]
        J.s_mov_b32(self.config, 0x00020000)

    def setup(self, base, range):
        self.base[0] = base[0]
        self.base[1] = base[1]
        self.range[0] = range[0]

    def load_dwordx4(self, vdst, voffset, soffset, offset12=0):
        # vdst,     vaddr,           srsrc, soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset12:{offset12}"
        self.J.buffer_load_dwordx4(vdst, voffset, self.desc, soffset, mod=mod)

    def store_dwordx4(self, vdata, voffset, soffset, offset12=0):
        # vdata,    vaddr,        srsrc,  soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset12:{offset12}"
        self.J.buffer_store_dwordx4(vdata, voffset, self.desc, soffset, mod=mod)

def test_vadd():
    @pyhip.jit()
    def kernel(J, pA:"int*", pB:"int*", cnt:"int"):
        vtemp =J.new_gpr('v',1) 
        v4 = J.new_gpr('v', 4, dtype="u32",align=2)

        buff_a = Buffer(J)
        buff_b = Buffer(J)

        size = J.gpr(cnt[0] * 4)
        buff_a.setup(pA, size)
        buff_b.setup(pB, size)
        
        vtemp[0] = J.threadIdx.x[0] << 4
        buff_a.load_dwordx4(v4, vtemp[0], 0)
        J.s_waitcnt(mod=f"vmcnt(0)")

        for i in range(4):
            # J.v_add_u32(v4[i], 102, v4[i])
            v4[i] = 102 + v4[i]

        buff_b.store_dwordx4(v4, vtemp[0], 0)
        J.s_waitcnt(mod=f"vmcnt(0)")

    A = torch.ones(64*4*8, dtype=torch.int)
    B = torch.ones(64*4*8, dtype=torch.int)
    kernel([1],[64], A.data_ptr(), B.data_ptr(), 64*4)
    torch.cuda.synchronize()
    ref = torch.full((256,), 103, dtype=torch.int)
    if not torch.allclose(ref, B[:256]):
        print(B)
    torch.testing.assert_close(B[:256], ref)

if __name__ == "__main__":
    test_basic()
    test_basic2()
    test_vadd()
