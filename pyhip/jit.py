from typing import List, Optional, Set
from .hiptools import module

# https://llvm.org/docs/AMDGPUInstructionSyntax.html#amdgpu-syn-instructions
class Instruction:
    def __init__(self, parent_bb:'BasicBlock', opcode):
        self.opcode = opcode
        self.parent_bb = parent_bb

    def __call__(self, *operands, mod:str=""):
        self.operands = operands
        self.mod = mod
        for i, op in enumerate(operands):
            assert isinstance(op, GPRExpr) or isinstance(op, int) or isinstance(op, GPRs), f"arg {i} type is {type(op)}"
        self.parent_bb.add_instruction(self)

    def __repr__(self):
        return f"{self.opcode} {','.join([repr(op) for op in self.operands])} {self.mod}"

class GPRExpr:
    def __init__(self, op:str, src0=None, src1=None, src2=None):
        self.op=op
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.depth = 1
        if isinstance(src0, GPRExpr):
            self.depth = max(self.depth, src0.depth + 1)
        if isinstance(src1, GPRExpr):
            self.depth = max(self.depth, src1.depth + 1)
        if isinstance(src2, GPRExpr):
            self.depth = max(self.depth, src2.depth + 1)

    def match(self, other: 'GPRExpr'):
        if self.op == "pattern":
            # pattern node handles the rest
            return self.src0.match(other)
        if self.op != other.op:
            return False
        if (self.src0 is not other.src0) and (not self.src0.match(other.src0)):
            return False
        if (self.src1 is not other.src1) and (not self.src1.match(other.src1)):
            return False
        return True

    def __add__(self, other):
        return GPRExpr("+", self, other)
    def __sub__(self, other):
        return GPRExpr("-", self, other)
    def __mul__(self, other):
        return GPRExpr("*", self, other)
    def __lshift__(self, other):
        return GPRExpr("<<", self, other)
    def __rshift__(self, other):
        return GPRExpr(">>", self, other)

    def __repr__(self):
        if self.op == "getitem":
            base_id = self.src0.start_id
            first = self.src1
            last = self.src2
            if last > first:
                return f"{self.src0.rtype}[{base_id + first}:{base_id + last}]"  # s[0:1] v[16:20]
            else:
                return f"{self.src0.rtype}{base_id + first}"  # v9, a9, s9
        elif (self.src1 is None) and (self.src2 is None):
            return f"{self.op} {self.src0}"
        elif (self.src2 is None):
            return f"{self.src0} {self.op} {self.src1}"
        else:
            return f"{self.op}({self.src0},{self.src1},{self.src2})"

class GPRItemPat:
    def __init__(self, rtype:str, count:int = 1):
        self.rtype = rtype
        self.count = count

    def match(self, other):
        if self.rtype == "i":
            return isinstance(other, int)
        self.other = other
        if not isinstance(other, GPRExpr): return False
        if other.op != "getitem": return False
        if self.rtype != other.src0.rtype: return False
        other_count = other.src2 - other.src1 + 1
        return self.count == other_count

# a continous region of GPRs allocated
# to reference GPR in it, we need getitem:
#   v[0:1]
#   v[2]
# we cann't use v2 as pure-asm does
# GPRs should be allocated by JIT
class GPRs:
    patterns = None
    def __init__(self, jit, rtype, start_id, count):
        self.jit = jit
        self.rtype = rtype
        self.start_id = start_id
        self.count = count

    def __getitem__(self, key):
        if isinstance(key, slice):
            s = key
            assert s.step == 1 or s.step is None, f"slice {s} with step not eq 1"
            return GPRExpr("getitem", self, s.start, s.stop)
        elif isinstance(key, int):
            idx = key
            return GPRExpr("getitem", self, idx, idx)
        else:
            assert 0, f"unsupported key {key}"

    def __setitem__(self, key, value:GPRExpr):
        dst = self[key]
        i0 = GPRExpr("pattern", GPRItemPat("i", 1))
        s1 = GPRExpr("pattern", GPRItemPat("s", 1))
        s2 = GPRExpr("pattern", GPRItemPat("s", 1))
        ds1 = GPRExpr("pattern", GPRItemPat("s", 2))
        ds2 = GPRExpr("pattern", GPRItemPat("s", 2))

        if (s1 << s2).match(value):
            inst = Instruction(self.jit.current_bb, "s_lshl_b32")
            inst(dst, s1.src0.other, s2.src0.other)
        elif (s1 << i0).match(value):
            inst = Instruction(self.jit.current_bb, "s_lshl_b32")
            inst(dst, s1.src0.other, s2.src0.other)
        elif (s1 + s2).match(value):
            inst = Instruction(self.jit.current_bb, "s_add_u32")
            inst(dst, s1.src0.other, s2.src0.other)
        else:
            assert 0

    def __repr__(self):
        if self.count == 1:
            return f"{self.rtype}{self.start_id}"
        else:
            return f"{self.rtype}[{self.start_id}:{self.start_id + self.count - 1}]"

# every basic block has a Label in asm
class BasicBlock:
    def __init__(self, jit, label_name):
        self.jit = jit
        self.label_name = label_name
        self.instructions: List[Instruction] = []
        self.predecessors: List['BasicBlock'] = []
        self.successors: List['BasicBlock'] = []

    def add_instruction(self, instr: Instruction):
        self.instructions.append(instr)

    def add_successor(self, successor: 'BasicBlock'):
        if successor not in self.successors:
            self.successors.append(successor)
        if self not in successor.predecessors:
            successor.predecessors.append(self)

    def __repr__(self):
        # https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html#Special-format-strings
        asm = f"{self.label_name}%=:\n"
        for inst in self.instructions:
            asm += f"\t{inst}\n"
        return asm

# JIT emits instructions into BBs
class JIT:
    def __init__(self):
        self.current_bb = BasicBlock(self, "main")
        self.blocks = []
        # increased freely
        self.free_gpr_id = {'s':0, 'v':0, 'a': 0}

    def __getattr__(self, instruction):
        return Instruction(self.current_bb, instruction)

    def _finish_bb(self, bb, next_bb_name=""):
        assert bb is self.current_bb
        self.blocks.append(bb)
        next_bb = BasicBlock(self, next_bb_name)
        bb.add_successor(next_bb)
        self.current_bb = next_bb

    def Label(self, name=""):
        # creat new basic block, archieve current bb
        self._finish_bb(self.current_bb, name)
        return self.current_bb

    def _align_up(self, a, align):
        return ((a + align - 1)//align) * align

    def new_gpr(self, reg_type, count_range, align=1):
        assert reg_type == 's' or reg_type == 'v' or reg_type == 'a'
        if isinstance(count_range, int):
            # allocate do not reuse, just increase the index
            count = count_range
            start_id = self._align_up(self.free_gpr_id[reg_type], align)
            self.free_gpr_id[reg_type] = start_id + count
            return GPRs(self, reg_type, start_id, count)
        elif isinstance(count_range, tuple) or isinstance(count_range, list):
            assert len(count_range) == 2
            assert align == 1
            first_id, last_id = count_range
            assert self.free_gpr_id[reg_type] <= first_id, "specified reg has been allocated"
            assert self.free_gpr_id[reg_type] <= last_id, "specified reg has been allocated"
            self.free_gpr_id[reg_type] = last_id + 1
            return GPRs(self, reg_type, first_id, last_id - first_id + 1)
        else:
            assert 0

    def build(self, signature):
        self._finish_bb(self.current_bb)
        # generate asm
        asm=""
        for bb in self.blocks:
            asm += repr(bb)

        used_gprs = []
        for a in asm.splitlines():
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
        '''
        str_arg_c_del = ",".join([f"{a[0]} {a[1]}" for a in args])
        signature = f"{kernel_name}({str_arg_c_del})"
        '''
        inline_asm = "\n".join([ f'"{line}\\n"' if len(line) else "" for line in asm.splitlines()])
        kernel_name = signature.split("(")[0]
        str_used_gprs = ",".join([f'\"{s}\"' for s in used_gprs])
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
        import os
        cpp_src_fpath = os.path.join(os.getcwd(),f'.jit-gen-{kernel_name}.cpp')
        with open(cpp_src_fpath, 'w', encoding='utf-8') as f:
            f.write(hip_src)
        hip = module(cpp_src_fpath)
        return getattr(hip, kernel_name)
