import pyhip
import torch
torch.set_printoptions(linewidth=300)
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

def test_dce():
    @pyhip.jit(force_recompile = True)
    def kernel(J, p:"int*",
                  v1:"int*",
                  v2:"int*",
                  v3:"int*",
                  K:"int"):
        vdst = J.gpr(4,4,"vf32")
        lane = J.threadIdx.x[0] % 64
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + 0, p)
        J.global_load_dwordx4(vdst[1], (lane % 16)*K*4 + (lane//16)*16 + K*2, p)
        J.global_load_dwordx4(vdst[2], (lane % 16)*K*4 + (lane//16)*16 + K*2 + 8, p)
        J.global_load_dwordx4(vdst[3], (lane % 16)*K*4 + (lane//16)*16 + 8, p)

    A = torch.zeros(64*1024, dtype=torch.int32)
    artifact = kernel([1],[64], A.data_ptr(), A.data_ptr(),A.data_ptr(),A.data_ptr(), 1)
    ipattern = extract_inst_pattern(artifact,"s_load_dwordx2")[-1]
    ipattern = ipattern.strip("0")
    assert ipattern == "", ipattern

    ipattern = extract_inst_pattern(artifact,"global_load_dwordx4")[-1]
    ipattern = ipattern.strip("0")
    assert ipattern == "", ipattern

    for asm_line in artifact["asm"][1:]:
        assert "v_" not in asm_line

if __name__ == "__main__":
    test_dce()