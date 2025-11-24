import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_basic():
    @pyhip.jit("(int*, int)")
    def kernel(J):
        p_kargs = J.new_gpr('s',[0,1],name="p_kargs")
        threadIdx_x = J.new_gpr('v',[0,0],name="threadIdx_x")
        pA = J.new_gpr('s',2,align=4,name="pA")
        cnt = J.new_gpr('s',1,name="K")
        acc = J.new_gpr("a", 4,name="acc")
        s_idx = J.new_gpr('s',1, dtype="u32", name="s_idx")
        s_temp0 = J.new_gpr('s',1,name="s_temp0")
        s_temp = J.new_gpr('s',2, align=2,name="s_temp")
        s_temp2 = J.new_gpr('s',2, align=2,name="s_temp2")
        vtemp = J.new_gpr('v',2, dtype="u32", align=2,name="vtemp")

        J.s_load_dwordx2(pA, p_kargs, 0)

        for i in range(4):
            J.v_accvgpr_write_b32(acc[i], 0)

        J.s_load_dword(cnt, p_kargs, 8)
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        s_idx[0] = 0
        J.Label("bb0")

        #J.s_lshl_b32(s_temp[1],1, s_idx)
        J.s_lshl_b32(s_temp0,s_idx,2)
        #s_temp[0] = s_idx[0] << 2

        J.s_add_u32(s_temp[0], pA[0], s_temp0)
        J.s_addc_u32(s_temp[1], pA[1], 0)

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
    @pyhip.jit("(int*, int)")
    def kernel(J):
        p_kargs = J.new_gpr('s',[0,1],name="p_kargs")
        threadIdx_x = J.new_gpr('v',[0,0],name="threadIdx_x")
        pA = J.new_gpr('s',2,align=4,name="pA")
        cnt = J.new_gpr('s',1,name="K")
        s_idx = J.new_gpr('s',1, dtype="u32", name="s_idx")
        s_offset = J.new_gpr('s',1,dtype="u32", name="s_offset")
        s_temp = J.new_gpr('s',2, dtype="u32", align=2,name="s_temp")

        J.s_load_dwordx2(pA, p_kargs, 0)

        J.s_load_dword(cnt, p_kargs, 8)
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

    def setup(self, sgpr_args_base, offset_base:int, offset_count:int, sizeof_ele:int):
        J = self.J
        J.s_load_dwordx2(self.base, sgpr_args_base, offset_base)
        J.s_load_dword(self.range, sgpr_args_base, offset_count)
        J.s_waitcnt(mod=f"lgkmcnt({0})")
        if sizeof_ele != 1:
            J.s_mulk_i32(self.range, sizeof_ele)

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
    @pyhip.jit("(int*, int*, int)")
    def kernel(J):
        p_kargs = J.new_gpr('s',[0,1])
        threadId_X = J.new_gpr('v', [0,0], dtype="u32")
        vtemp =J.new_gpr('v',1) 
        v4 = J.new_gpr('v', 4, dtype="u32",align=2)

        buff_a = Buffer(J)
        buff_b = Buffer(J)
        
        buff_a.setup(p_kargs, 0, 16, sizeof_ele=4)
        buff_b.setup(p_kargs, 8, 16, sizeof_ele=4)
        
        vtemp[0] = threadId_X[0] << 4
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
    #test_basic2()
    #test_vadd()
