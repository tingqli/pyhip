import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

def extract_inst_pattern(kernel, inst_opcode):
    # return a instruction occurrence pattern string for each BB
    ret = []
    ipattern = ""
    for asm in kernel.artifact["asm"][1:]:
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

def test_cse_1bb():
    @pyhip.jit(force_recompile = True)
    def kernel(J, p:"int*", K:"int"):
        vdst = J.gpr(4,4,"vf32")
        lane = J.threadIdx.x[0] % 64
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + 0, p)
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + K*2, p)
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + K*2 + 8, p)
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + 8, p)

    ipattern = extract_inst_pattern(kernel,"global_load_dwordx4")[-1]
    ipattern = ipattern.strip("0")
    assert ipattern == "10010101"
    print(ipattern)

def test_cse_2bb():
    @pyhip.jit(force_recompile = True)
    def kernel(J, p:"int*", K:"int"):
        vdst = J.gpr(4,4,"vf32")
        lane = J.threadIdx.x[0] % 64
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + 0, p)
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + K*2, p)
        J.Label()
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + K*2 + 8, p)
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + 8, p)

    ipattern0, ipattern1 = extract_inst_pattern(kernel,"global_load_dwordx4")
    ipattern0 = ipattern0.strip("0")
    ipattern1 = ipattern1.strip("0")
    assert ipattern0 == "1001", ipattern0
    assert ipattern1 == "101", ipattern1

if __name__ == "__main__":
    test_cse_1bb()
    test_cse_2bb()