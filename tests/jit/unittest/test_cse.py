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

def test_cse_1bb():
    @pyhip.jit(force_recompile = True)
    def kernel(J, p:"int*", K:"int"):
        vdst = J.gpr(4,4,"vf32")
        lane = J.threadIdx.x[0] % 64
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + 0, p)
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + K*2, p)
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + K*2 + 8, p)
        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + 8, p)

    A = torch.zeros(64*1024, dtype=torch.int32)
    artifact = kernel([1],[64], A.data_ptr(), 1)
    ipattern = extract_inst_pattern(artifact,"global_load_dwordx4")[-1]
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

    A = torch.zeros(64*1024, dtype=torch.int32)
    artifact = kernel([1],[64], A.data_ptr(), 1)
    ipattern0, ipattern1 = extract_inst_pattern(artifact,"global_load_dwordx4")
    ipattern0 = ipattern0.strip("0")
    ipattern1 = ipattern1.strip("0")
    assert ipattern0 == "1001", ipattern0
    assert ipattern1 == "101", ipattern1

def test_cse_inloop():
    @pyhip.jit(force_recompile = True)
    def kernel(J, p:"int*", K:"int"):
        vdst = J.gpr(4,4,"vf32")
        lane = J.threadIdx.x[0] % 64

        J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + 0, p)

        si = J.gpr("su32")
        si[0] = 0
        with J.While(si < K) as loop:
            J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + 0, p)
            J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + K*2, p)
            J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + K*2 + 8, p)
            J.global_load_dwordx4(vdst[0], (lane % 16)*K*4 + (lane//16)*16 + 8, p)
            si[0] += 1

    A = torch.zeros(64*1024, dtype=torch.int32)
    artifact = kernel([1],[64], A.data_ptr(), 1)
    ipattern = extract_inst_pattern(artifact,"global_load_dwordx4")[-2]
    ipattern = ipattern.strip("0")
    assert ipattern == "10010101", ipattern
    print(ipattern)


def test_cse_global():
    @pyhip.jit(force_recompile = True)
    def kernel(J, p:"int*", K:"int"):
        vdst = J.gpr(4,4,"vf32")
        lane = J.threadIdx.x[0] % 64

        # these offsets are all SSA, they will all reuse off0
        # although they are read in another BB
        off0 = J.gpr((lane % 16)*K*4 + (lane//16)*16)
        off1 = J.gpr((lane % 16)*K*4 + (lane//16)*16)
        off2 = J.gpr((lane % 16)*K*4 + (lane//16)*16)
        off3 = J.gpr((lane % 16)*K*4 + (lane//16)*16)

        si = J.gpr("su32")
        si[0] = 0
        with J.While(si < K) as loop:
            J.global_load_dwordx4(vdst[0], off0 + 8, p)
            J.global_load_dwordx4(vdst[0], off1 + 8, p)
            J.global_load_dwordx4(vdst[0], off2 + 8, p)
            J.global_load_dwordx4(vdst[0], off3 + 8, p)
            si[0] += 1

    A = torch.zeros(64*1024, dtype=torch.int32)
    artifact = kernel([1],[64], A.data_ptr(), 1)
    ipattern = extract_inst_pattern(artifact,"v_lshrrev_b32")
    assert ipattern[0].strip("0") == "1", ipattern
    assert ipattern[-2].strip("0") == "", ipattern

    ipattern = extract_inst_pattern(artifact,"global_load_dwordx4")[-2]
    ipattern = ipattern.strip("0")
    assert ipattern == "1111", ipattern
    print(ipattern)

if __name__ == "__main__":
    test_cse_1bb()
    test_cse_2bb()
    test_cse_inloop()
    test_cse_global()