import pyhip
import torch
torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

def extract_inst_pattern(artifact, inst_opcode):
    # return a instruction occurrence pattern string for each BB
    ret = []
    ipattern = ""
    for asm in artifact["asm"][1:]:
        print(asm)
        opcode = asm.split()[0]
        if opcode.endswith(":"):
            # a new block starts,
            ret.append(ipattern)
            ipattern = ""
        if opcode == inst_opcode:
            ipattern += "1"
        else:
            ipattern += "0"
    ret.append(ipattern)
    return ret

def test_lds_tensor():
    sizeof_bf16 = 2
    M,K,N = 16,32*4,32

    @pyhip.jit(force_recompile = True)
    def kernel(J, pA:"void*", pB:"void*", p_output:"void*"):
        Creg = J.gpr(2, 4, "vf32") # 2 x [16 x 16] vf32
        Areg = J.gpr(2, 2, "vbf16x2") # 8-bf16 == DWORDx4
        Breg = J.gpr(2, 2, 2, "vbf16x2") # 8-bf16 == DWORDx4        
        Creg[:] = 0.0

        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)
        stride_AB = K*sizeof_bf16

        buff_a = J.Buffer(pA, 16*K*sizeof_bf16)
        buff_b = J.Buffer(pB, 32*K*sizeof_bf16)

        voffset_ab = J.gpr((J.threadIdx.x % 16)*stride_AB + (J.threadIdx.x//16) * 16)

        for k in range(0, K, 4*32):
            soffset = J.gpr("su32")
            soffset[0] = k*sizeof_bf16
            buff_a.load_dwordx4(Areg, voffset_ab, soffset)
            buff_b.load_dwordx4(Breg[0], voffset_ab, soffset)
            buff_b.load_dwordx4(Breg[1], voffset_ab, soffset + (16*stride_AB))
            J.s_waitcnt(mod=f"vmcnt(0)")

            J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[0,0], Areg[0], Creg[0])
            J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[1,0], Areg[0], Creg[1])

            J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[0,1], Areg[1], Creg[0])
            J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[1,1], Areg[1], Creg[1])

        lds_buff = J.LDSTensor([4*16, 32], torch.float)

        lane = J.threadIdx.x[0] % 64

        # 4 waves each store a 16x32 vf32 : vertically stacked got [4x16,32]
        lds_buff.write("b128", Creg[0], (lane % 16) + warp_id*16, (lane//16)*4)
        lds_buff.write("b128", Creg[1], (lane % 16) + warp_id*16, (lane//16)*4 + 16)

        J.s_barrier()

        # each wave read 4 [4,32] vf32 (from all wave's own Creg results), reduce sum them
        # into a single [4,32] vf32 and store to output buffer
        result = J.gpr(4, 2, "vf32")
        lds_buff.read("b64", result[0], (lane//16) + warp_id*4 + 0*16, (lane % 16)*2)
        lds_buff.read("b64", result[1], (lane//16) + warp_id*4 + 1*16, (lane % 16)*2)
        lds_buff.read("b64", result[2], (lane//16) + warp_id*4 + 2*16, (lane % 16)*2)
        lds_buff.read("b64", result[3], (lane//16) + warp_id*4 + 3*16, (lane % 16)*2)

        J.s_waitcnt(mod=f"lgkmcnt(2)")
        J.v_pk_add_f32(result[0], result[0], result[1])
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.v_pk_add_f32(result[2], result[2], result[3])
        J.v_pk_add_f32(result[0], result[0], result[2])

        act_bf16 = J.gpr("vbf16x2")
        act_bf16[0] = (result[0,1] & 0xFFFF0000)|(result[0,0] >> 16)
        # each wave stores 4x32 bf16, WG stores 16x32 bf16
        J.global_store_dword((lane//16 + warp_id*4) * (32*sizeof_bf16) + (lane % 16)*(2*sizeof_bf16),
                             act_bf16[0],
                             p_output)
        J.s_waitcnt(mod=f"vmcnt(0)")
    A = (torch.randn(M, K, dtype=torch.bfloat16) + 1)*0.1
    B = torch.randn(N, K, dtype=torch.bfloat16)
    C = torch.ones(M, N, dtype=torch.bfloat16)
    C0 = A @ B.t()
    artifact = kernel([1], [256],  A.data_ptr(), B.data_ptr(), C.data_ptr())

    # check if LDSTensor generats correct read & write code
    # (with CSE's help, only offsets are different between ds_write/ds_read instructions)
    ip_ds_write_b128 = extract_inst_pattern(artifact,"ds_write_b128")[-1]
    ip_ds_write_b128 = ip_ds_write_b128.strip("0")
    assert ip_ds_write_b128 == "11", ip_ds_write_b128

    ip_ds_read_b64 = extract_inst_pattern(artifact,"ds_read_b64")[-1]
    ip_ds_read_b64 = ip_ds_read_b64.strip("0")
    assert ip_ds_read_b64 == "1111", ip_ds_read_b64

    if not torch.allclose(C, C0, atol=0.01, rtol=0.01):
        print(C)
        print(C0)
        assert 0

if __name__ == "__main__":
    test_lds_tensor()