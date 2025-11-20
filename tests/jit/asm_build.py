import torch
import pyhip
import sys

torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

def build(asm):
    args = []
    used_gprs = []
    kernel_name = None
    for a in asm.splitlines():
        if a.startswith(";kernel "):
            kernel_name = a.split()[1]
        elif a.startswith(";arg "):
            arg_type, arg_name = a.split()[1:]
            args.append([arg_type, arg_name])
        else:
            for op in a.replace(","," ").split()[1:]:
                if (op.startswith("s") or op.startswith("v") or op.startswith("a")) and (op[1].isdigit()):
                    gpr_type = op[0]
                    str_gpr = f"{gpr_type}{int(op[1:])}"
                    if str_gpr not in used_gprs: used_gprs.append(str_gpr)
                elif op.startswith("s[") or op.startswith("v[") or op.startswith("a["):
                    gpr_type = op[0]
                    idx0, idx1 = op[2:-1].split(":")
                    idx0 = int(idx0)
                    idx1 = int(idx1)
                    for idx in range(idx0, idx1 + 1):
                        str_gpr = f"{gpr_type}{idx}"
                        if str_gpr not in used_gprs: used_gprs.append(str_gpr)

    str_arg_c_del = ",".join([f"{a[0]} {a[1]}" for a in args])
    str_used_gprs = ",".join([f'\"{s}\"' for s in used_gprs])
    inline_asm = "\n".join([ f'"{line}\\n"' if len(line) else "" for line in asm.splitlines()])
    signature = f"{kernel_name}({str_arg_c_del})"

    print(f" kernel: {signature}  used_gprs={used_gprs}")


    hip_src =r'''
    #include <hip/hip_fp16.h> // for __fp16
    #include <hip/hip_bf16.h> // for bfloat16
    #include "hip/hip_runtime.h"
    #include <vector>
    #include <iostream>
    #include <cstdio>
    #include <cstdlib>
    __global__ void __launch_bounds__(256, 1) ''' + signature +  r''' {
        //A[threadIdx.x] = K;
        asm volatile("\n"
    ''' + "\n" + inline_asm + \
    r'''
                    ::
                    :"memory",''' + str_used_gprs + r''');
    }
    '''
    with open('asmgen.gen.cpp', 'w', encoding='utf-8') as f:
        f.write(hip_src)
    hip = pyhip.module("asmgen.gen.cpp")
    return getattr(hip, kernel_name)


# 类似xbayk一样的JIT,每个语句就是emit一条汇编:
# 基本feature:
#
#   - 指令助记符号例如
#       v2 = s2             =>   v_mov_b32      v2,s2
#       v2 = (v0<<2)+v2     =>   v_lshl_add_u32 v2, v0, 2, v2
#   - 寄存器命名伪指令，例如
#       A = alloc_sgpr(2)  分配两个 sgpr起名为A, 可以使用 A[0], A[1] 或者 A[0:1] 或者直接 A
#       del A 就可以释放寄存器
#   - 
#

if __name__ == "__main__":

    asm = '''
    ;kernel test
    ;arg int* A
    ;arg int K

    s_load_dwordx2 s[2:3], s[0:1] 0x0
    s_load_dword s4, s[0:1] 0x8

    s_waitcnt lgkmcnt(0)


    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshl_add_u32 v2, v0, 2, v2

    ;flat_store_dword v[2:3], v0

    s_store_dword s4, s[2:3], 0 glc

    s_waitcnt lgkmcnt(0)

    '''

    kernel = build(asm)

    A = torch.ones(64, dtype=torch.int)
    print(A)
    kernel([1],[64], A.data_ptr(), 64)
    torch.cuda.synchronize()
    print(A)